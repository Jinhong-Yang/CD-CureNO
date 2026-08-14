"""Evaluation-only access to the frozen P4 true-2-D benchmark.

This module is deliberately separate from :mod:`cdcureno.data.target_2d`,
whose public data path exposes only target training and validation labels.
An evaluation manifest must match a pre-registered byte hash, role, exact case
order, source manifest, pre-label plan, and array checksums before a dataset can
be constructed.

Metadata validation never opens, memory-maps, or hashes either monolithic
temperature/cure label container.  Once a release split is authorized, a
bounded reader extracts only that split's exact case-row byte ranges, verifies
every row against the frozen per-case semantic hash, and publishes physically
isolated split-order label shards.  Evaluation opens only those shards.

Both source-range extraction and shard-slice evaluation use fsynced,
hash-chained journals.  An attempt is durable before each read and a completion
is durable afterward.  Evaluation loaders must use ``num_workers=0``; worker
process access is rejected inside ``__getitem__`` so journal ordering remains
authoritative.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import struct
import threading
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset, get_worker_info

from cdcureno.data.target_2d import (
    TARGET_INPUT_CHANNELS,
    BaselineBuilder,
    FrozenSourceNormalization,
    build_label_free_coarse_1d_baseline,
    build_target_2d_input,
    load_source_normalization_contract,
)


FROZEN_P4_SOURCE_MANIFEST_SHA256 = (
    "983ced5f4bcb8bfb06de88b6f17ad46dccd479932e1ae18b942e693957dbd445"
)
FROZEN_P4_PLAN_FILE_SHA256 = (
    "12106526576ae73e2419079afea6ddbfba28cd40dac7a2be5d65e097dd23d808"
)

LABEL_ARRAY_NAMES = ("temperature_K", "alpha")
LABEL_SHARD_SCHEMA_VERSION = 1
LABEL_SHARD_ARTIFACT_ROLE = "target_2d_authorized_label_shard_access_receipt"
LABEL_SHARD_PENDING_ROLE = "target_2d_authorized_label_shard_pending"
LABEL_SHARD_JOURNAL_SCHEMA_VERSION = 1
P6_V2_RELEASE_ID = "p6_v2_complete_population_release_v1"
P6_V2_RELEASE_STAGE_SPLITS = {
    "stage1": ("id_test",),
    "stage2": (
        "cycle_ood",
        "htc_ood",
        "pattern_ood",
        "combined_ood",
    ),
}
P6_V2_HARD_INTERRUPTION_RESUME_POLICY = {
    "allowed": True,
    "explicit_opt_in_required": True,
    "pending_without_failure_or_receipt_required": True,
    "same_frozen_authorities_required": True,
    "completed_split_prefix_is_revalidated_and_skipped": True,
    "partial_split_adapter_flag_required": True,
    "scientific_exception_retry_allowed": False,
    "overwrite_allowed": False,
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_NPY_MAGIC = b"\x93NUMPY"


@dataclass(frozen=True)
class FrozenEvaluationManifestSpec:
    """Pre-registered identity of one permitted evaluation manifest."""

    name: str
    relative_path: str
    manifest_sha256: str
    role: str
    case_ids: tuple[int, ...]
    source_split_key: str
    source_manifest_relative_path: str
    source_manifest_sha256: str
    plan_file_sha256: str

    def validated(self) -> "FrozenEvaluationManifestSpec":
        if not self.name:
            raise ValueError("Evaluation manifest spec needs a name.")
        for label, value in (
            ("relative_path", self.relative_path),
            (
                "source_manifest_relative_path",
                self.source_manifest_relative_path,
            ),
        ):
            if (
                not value
                or "\\" in value
                or Path(value).is_absolute()
                or ".." in Path(value).parts
            ):
                raise ValueError(
                    f"Evaluation manifest spec {label} must be a safe "
                    "repository-relative POSIX path."
                )
        for label, value in (
            ("manifest_sha256", self.manifest_sha256),
            ("source_manifest_sha256", self.source_manifest_sha256),
            ("plan_file_sha256", self.plan_file_sha256),
        ):
            if (
                len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(
                    f"Evaluation manifest spec {label} must be lowercase SHA256."
                )
        if self.role not in {
            "target_id",
            "cycle_ood",
            "htc_ood",
            "pattern_ood",
            "combined_ood",
            "geometry_ood",
            "external_validation",
        }:
            raise ValueError("Evaluation manifest spec has an unknown role.")
        if not self.source_split_key:
            raise ValueError("Evaluation manifest spec needs a source split key.")
        if any(
            isinstance(case_id, bool)
            or not isinstance(case_id, int)
            or case_id < 0
            for case_id in self.case_ids
        ):
            raise ValueError(
                "Evaluation manifest spec case IDs must be nonnegative integers."
            )
        if len(self.case_ids) != len(set(self.case_ids)):
            raise ValueError("Evaluation manifest spec contains duplicate IDs.")
        return self


def _canonical_spec(
    *,
    name: str,
    relative_path: str,
    manifest_sha256: str,
    role: str,
    case_ids: range,
    source_split_key: str,
) -> FrozenEvaluationManifestSpec:
    return FrozenEvaluationManifestSpec(
        name=name,
        relative_path=relative_path,
        manifest_sha256=manifest_sha256,
        role=role,
        case_ids=tuple(case_ids),
        source_split_key=source_split_key,
        source_manifest_relative_path="splits/p4_2d_core_v1.json",
        source_manifest_sha256=FROZEN_P4_SOURCE_MANIFEST_SHA256,
        plan_file_sha256=FROZEN_P4_PLAN_FILE_SHA256,
    ).validated()


CANONICAL_EVALUATION_MANIFESTS = {
    "id_test": _canonical_spec(
        name="id_test",
        relative_path="splits/2d_id_v1.json",
        manifest_sha256=(
            "0628acabce004df2d9d3bb1ac6928e2439dfdbc636575e52ffb1afd9103417f0"
        ),
        role="target_id",
        case_ids=range(288, 352),
        source_split_key="id_test",
    ),
    "cycle_ood": _canonical_spec(
        name="cycle_ood",
        relative_path="splits/2d_cycle_ood_v1.json",
        manifest_sha256=(
            "7f30813d9bb6dd83e6df4a4aedf5254a730d5ba203e4af065324934841a18274"
        ),
        role="cycle_ood",
        case_ids=range(352, 392),
        source_split_key="cycle_ood",
    ),
    "htc_ood": _canonical_spec(
        name="htc_ood",
        relative_path="splits/2d_htc_ood_v1.json",
        manifest_sha256=(
            "44a9df372c8ebe9272673b780679bf83296b6d9338f9aa8178c78bfdaead7c84"
        ),
        role="htc_ood",
        case_ids=range(392, 432),
        source_split_key="htc_ood",
    ),
    "pattern_ood": _canonical_spec(
        name="pattern_ood",
        relative_path="splits/2d_pattern_ood_v1.json",
        manifest_sha256=(
            "6e717ec7644f862678fca7c515e6a0d0224743a6bb14a8aa83a9627767669053"
        ),
        role="pattern_ood",
        case_ids=range(432, 472),
        source_split_key="pattern_ood",
    ),
    "combined_ood": _canonical_spec(
        name="combined_ood",
        relative_path="splits/2d_combined_ood_v1.json",
        manifest_sha256=(
            "90a53f27c93e712cc02f470016592670065f45389650fe76371b5839d68db5c2"
        ),
        role="combined_ood",
        case_ids=range(472, 512),
        source_split_key="combined_ood",
    ),
}


CANONICAL_DEFERRED_MANIFESTS = {
    "splits/2d_geometry_ood_v1.json": {
        "sha256": (
            "263bdf78ecb870d536315a856d57753f424ee5e6697c4b502d501443df55964a"
        ),
        "role": "geometry_ood",
        "status": "deferred_until_F3",
    },
    "splits/external_validation_v1.json": {
        "sha256": (
            "e079bfdf4a30a7472ab5c9c564497894bbd4e3deb7361747c59ed355b6eef6f8"
        ),
        "role": "external_validation",
        "status": "deferred_until_P7_compatibility_audit",
    },
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_json_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _strict_sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be one lowercase SHA-256 digest.")
    return value


def _fsync_directory(path: Path) -> bool:
    """Best-effort directory fsync, including a Windows directory handle."""

    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            create_file = ctypes.WinDLL(
                "kernel32", use_last_error=True
            ).CreateFileW
            create_file.argtypes = [
                wintypes.LPCWSTR,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.LPVOID,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.HANDLE,
            ]
            create_file.restype = wintypes.HANDLE
            flush = ctypes.WinDLL(
                "kernel32", use_last_error=True
            ).FlushFileBuffers
            flush.argtypes = [wintypes.HANDLE]
            flush.restype = wintypes.BOOL
            close = ctypes.WinDLL(
                "kernel32", use_last_error=True
            ).CloseHandle
            close.argtypes = [wintypes.HANDLE]
            close.restype = wintypes.BOOL
            handle = create_file(
                str(path),
                0x80000000,
                0x00000001 | 0x00000002 | 0x00000004,
                None,
                3,
                0x02000000,
                None,
            )
            invalid = wintypes.HANDLE(-1).value
            if handle == invalid:
                return False
            try:
                return bool(flush(handle))
            finally:
                close(handle)
        except (AttributeError, OSError, ValueError):
            return False
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return False
    try:
        os.fsync(descriptor)
        return True
    except OSError:
        return False
    finally:
        os.close(descriptor)


def _write_once_bytes(path: Path, payload: bytes) -> str:
    """Create one durable file without a check-then-replace race."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or path.is_symlink():
        raise PermissionError(f"Unsafe write-once path: {path}.")
    for interrupted in path.parent.glob(f".tmp-{path.name}-*"):
        if (
            interrupted.parent.resolve(strict=True)
            != path.parent.resolve(strict=True)
            or interrupted.is_symlink()
            or not interrupted.is_file()
        ):
            raise PermissionError(
                f"Unsafe interrupted write-once artifact: {interrupted}."
            )
        interrupted.unlink()
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(
                f"Existing write-once artifact differs: {path}."
            )
        return "verified_existing"
    temporary = (
        path.parent / f".tmp-{path.name}-{os.getpid()}-{uuid.uuid4().hex}"
    )
    try:
        with temporary.open("xb", buffering=0) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if (
                path.is_symlink()
                or not path.is_file()
                or path.read_bytes() != payload
            ):
                raise FileExistsError(
                    f"Existing write-once artifact differs: {path}."
                )
            return "verified_existing"
        except OSError as error:
            raise RuntimeError(
                "Atomic no-replace publication is unavailable."
            ) from error
        _fsync_directory(path.parent)
        return "created"
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()


def _artifact_reference(path: Path, project_root: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    root = project_root.resolve(strict=True)
    if (
        path.is_symlink()
        or not resolved.is_file()
        or not resolved.is_relative_to(root)
    ):
        raise PermissionError(f"Artifact is missing or unsafe: {path}.")
    return {
        "path": resolved.relative_to(root).as_posix(),
        "sha256": _sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _verify_small_artifact_reference(
    reference: Any,
    *,
    project_root: Path,
    label: str,
    expected_path: Path | None = None,
) -> Path:
    if not isinstance(reference, Mapping) or set(reference) != {
        "path",
        "sha256",
        "bytes",
    }:
        raise ValueError(f"{label} artifact reference differs.")
    path = _resolve_repo_path(
        project_root,
        reference["path"],
        label=f"{label} path",
    )
    if expected_path is not None and path.resolve(strict=True) != (
        expected_path.resolve(strict=True)
    ):
        raise ValueError(f"{label} path differs.")
    digest = _strict_sha256(reference["sha256"], label=f"{label} SHA-256")
    if (
        path.is_symlink()
        or not path.is_file()
        or type(reference["bytes"]) is not int
        or reference["bytes"] < 0
        or path.stat().st_size != reference["bytes"]
        or _sha256_file(path) != digest
    ):
        raise ValueError(f"{label} hash/byte count differs.")
    return path


def _read_canonical_json(path: Path, *, label: str) -> dict[str, Any]:
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read {label}: {error}") from error
    if not isinstance(payload, dict) or _canonical_json_bytes(payload) != raw:
        raise ValueError(f"{label} must be one canonical JSON object.")
    return payload


def _semantic_array_sha256(
    raw: bytes | memoryview,
    *,
    dtype: np.dtype[Any],
    shape: tuple[int, ...],
) -> str:
    """Match ``scripts/generate_p4_2d.py::_sha256_array`` exactly."""

    digest = hashlib.sha256()
    digest.update(dtype.str.encode("ascii"))
    digest.update(_canonical_json_bytes(list(shape)))
    digest.update(raw)
    return digest.hexdigest()


def _journal_records(path: Path, *, journal_id: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.is_symlink() or not path.is_file():
        raise PermissionError(f"Journal is unsafe: {path}.")
    try:
        lines = path.read_bytes().splitlines(keepends=True)
    except OSError as error:
        raise ValueError(f"Cannot read journal {path}: {error}") from error
    records: list[dict[str, Any]] = []
    prior = "0" * 64
    for sequence, raw in enumerate(lines, start=1):
        if not raw.endswith(b"\n"):
            raise ValueError("Journal has a non-durable partial final line.")
        try:
            record = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Journal line {sequence} is invalid.") from error
        if (
            not isinstance(record, dict)
            or _canonical_json_bytes(record) != raw
        ):
            raise ValueError(f"Journal line {sequence} is noncanonical.")
        unsigned = {
            key: value
            for key, value in record.items()
            if key != "event_sha256"
        }
        if (
            record.get("schema_version")
            != LABEL_SHARD_JOURNAL_SCHEMA_VERSION
            or record.get("journal_id") != journal_id
            or record.get("sequence") != sequence
            or record.get("previous_event_sha256") != prior
            or record.get("event_sha256")
            != _canonical_json_sha256(unsigned)
        ):
            raise ValueError(f"Journal line {sequence} hash chain differs.")
        prior = record["event_sha256"]
        records.append(record)
    return records


class _DurableJournal:
    """Small append-only, fsynced, hash-chained JSONL journal."""

    def __init__(self, path: Path, *, journal_id: str) -> None:
        self.path = path
        self.journal_id = journal_id
        self._lock = threading.Lock()
        self._records = _journal_records(path, journal_id=journal_id)

    def _refresh_locked(self) -> None:
        observed = _journal_records(
            self.path,
            journal_id=self.journal_id,
        )
        if (
            len(observed) < len(self._records)
            or observed[: len(self._records)] != self._records
        ):
            raise PermissionError(
                "Durable journal history changed or was truncated."
            )
        self._records = observed

    @property
    def records(self) -> tuple[dict[str, Any], ...]:
        with self._lock:
            self._refresh_locked()
            return tuple(dict(item) for item in self._records)

    def append(self, event: str, **payload: Any) -> dict[str, Any]:
        with self._lock:
            self._refresh_locked()
            prior = (
                self._records[-1]["event_sha256"]
                if self._records
                else "0" * 64
            )
            unsigned = {
                "schema_version": LABEL_SHARD_JOURNAL_SCHEMA_VERSION,
                "journal_id": self.journal_id,
                "sequence": len(self._records) + 1,
                "previous_event_sha256": prior,
                "event": event,
                **payload,
            }
            record = {
                **unsigned,
                "event_sha256": _canonical_json_sha256(unsigned),
            }
            raw = _canonical_json_bytes(record)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.parent.is_symlink() or self.path.is_symlink():
                raise PermissionError("Journal path is unsafe.")
            created = not self.path.exists()
            with self.path.open("ab", buffering=0) as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            if created:
                _fsync_directory(self.path.parent)
            self._records.append(record)
            return dict(record)

    def reference(self, project_root: Path) -> dict[str, Any]:
        return _artifact_reference(self.path, project_root)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read {label} JSON at {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object.")
    return payload


def _resolve_repo_path(
    project_root: Path,
    value: Any,
    *,
    label: str,
) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(
            f"{label} must be a nonempty repository-relative POSIX path."
        )
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must stay inside the project root.")
    root = project_root.resolve()
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{label} escapes the project root.")
    return resolved


def _integer_ids(values: Any, *, label: str) -> tuple[int, ...]:
    if not isinstance(values, list):
        raise ValueError(f"{label} must be a JSON list.")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in values
    ):
        raise ValueError(f"{label} must contain nonnegative integer IDs.")
    result = tuple(int(value) for value in values)
    if len(result) != len(set(result)):
        raise ValueError(f"{label} contains duplicate IDs.")
    return result


def _spec_for_path(
    manifest_path: Path,
    project_root: Path,
    supplied: FrozenEvaluationManifestSpec | None,
) -> FrozenEvaluationManifestSpec:
    root = project_root.resolve()
    resolved = manifest_path.resolve()
    if supplied is not None:
        spec = supplied.validated()
        canonical_source_exists = (
            root / "splits" / "p4_2d_core_v1.json"
        ).is_file()
        if canonical_source_exists:
            canonical = next(
                (
                    candidate
                    for candidate in CANONICAL_EVALUATION_MANIFESTS.values()
                    if resolved
                    == (root / candidate.relative_path).resolve()
                ),
                None,
            )
            if canonical is None or spec != canonical:
                raise PermissionError(
                    "A repository containing the canonical P4 source cannot "
                    "override the frozen evaluation-manifest registry."
                )
            return canonical
        expected_path = (root / spec.relative_path).resolve()
        if resolved != expected_path:
            raise PermissionError(
                "Evaluation manifest path differs from its pre-registered path."
            )
        return spec

    for relative_path, deferred in CANONICAL_DEFERRED_MANIFESTS.items():
        if resolved == (root / relative_path).resolve():
            actual_sha = _sha256_file(resolved)
            if actual_sha != deferred["sha256"]:
                raise ValueError("Deferred evaluation manifest SHA256 drifted.")
            raise PermissionError(
                f"{deferred['role']} is empty and {deferred['status']}; "
                "evaluation is forbidden."
            )
    for spec in CANONICAL_EVALUATION_MANIFESTS.values():
        if resolved == (root / spec.relative_path).resolve():
            return spec
    raise PermissionError(
        "Evaluation manifest is not one of the pre-registered frozen manifests."
    )


@dataclass(frozen=True)
class FrozenTarget2DEvaluationContract:
    """Validated metadata needed by an evaluation-only dataset."""

    project_root: Path
    artifact_root: Path
    manifest_path: Path
    source_manifest_path: Path
    plan_path: Path
    metadata_path: Path
    spec: FrozenEvaluationManifestSpec
    manifest: dict[str, Any]
    source_manifest: dict[str, Any]
    plan: dict[str, Any]
    definitions: dict[int, dict[str, Any]]
    geometry: dict[str, float]
    array_paths: dict[str, Path]
    checksums: dict[str, Any]


@dataclass(frozen=True)
class Target2DEvaluationPhysicsInputs:
    """Label-free case definitions, coordinates, geometry, and material mask."""

    definitions: dict[int, dict[str, Any]]
    time_s: NDArray[np.float64]
    z_m: NDArray[np.float64]
    x_m: NDArray[np.float64]
    composite_mask: NDArray[np.bool_]
    geometry: dict[str, float]
    array_root: Path


@dataclass(frozen=True)
class FrozenTarget2DLabelShard:
    """Validated, physically isolated labels for one authorized split."""

    project_root: Path
    root: Path
    pending_path: Path
    receipt_path: Path
    source_journal_path: Path
    slice_journal_path: Path
    stage: str
    split: str
    role: str
    case_ids: tuple[int, ...]
    row_shape: tuple[int, int, int]
    array_paths: dict[str, Path]
    expected_slice_sha256: dict[int, dict[str, str]]
    receipt: dict[str, Any]
    access_session: dict[str, Any]
    resume_attempt: dict[str, Any] | None = None

    @property
    def access_session_sha256(self) -> str:
        """Identify this one authorized evaluator invocation."""

        return _strict_sha256(
            self.access_session["session_authority_sha256"],
            label="label-access session authority SHA-256",
        )

    @property
    def access_session_kind(self) -> str:
        return str(self.access_session["invocation_kind"])

    def evidence(self) -> dict[str, Any]:
        """Return only small control artifacts plus declared shard references."""

        receipt_reference = _artifact_reference(
            self.receipt_path,
            self.project_root,
        )
        pending_reference = _artifact_reference(
            self.pending_path,
            self.project_root,
        )
        source_journal = _artifact_reference(
            self.source_journal_path,
            self.project_root,
        )
        slice_journal = _artifact_reference(
            self.slice_journal_path,
            self.project_root,
        )
        return {
            "schema_version": LABEL_SHARD_SCHEMA_VERSION,
            "stage": self.stage,
            "split": self.split,
            "role": self.role,
            "ordered_global_case_ids": list(self.case_ids),
            "pending": pending_reference,
            "access_receipt": receipt_reference,
            "access_receipt_payload_sha256": self.receipt[
                "receipt_payload_sha256"
            ],
            "source_access_journal": source_journal,
            "shard_slice_access_journal": slice_journal,
            "label_access_session": dict(self.access_session),
            "resume_attempt": (
                dict(self.resume_attempt)
                if self.resume_attempt is not None
                else None
            ),
            # These are declarations copied from the authenticated receipt.
            # Downstream receipt validation must not reopen/hash payload shards.
            "label_shards": dict(self.receipt["label_shards"]),
        }


def _validate_stage_split(stage: str, split: str) -> None:
    if stage == "stage1":
        if split != "id_test":
            raise PermissionError("Stage 1 may authorize only the ID-test shard.")
        return
    if stage == "stage2":
        if split not in {
            "cycle_ood",
            "htc_ood",
            "pattern_ood",
            "combined_ood",
        }:
            raise PermissionError(
                "Stage 2 may authorize only a registered OOD shard."
            )
        return
    raise ValueError("Label-shard stage must be stage1 or stage2.")


def _label_shard_paths(
    project_root: Path,
    *,
    stage: str,
    split: str,
    shard_root: Path | None,
) -> dict[str, Path]:
    base = (
        shard_root.resolve()
        if shard_root is not None
        else (
            project_root
            / "outputs"
            / "p6_v2"
            / "evaluation"
            / "release_v1"
            / "label_shards"
        ).resolve()
    )
    control = base / "control" / stage / split
    final = base / stage / split
    staging = base / ".building" / stage / split
    return {
        "base": base,
        "control": control,
        "final": final,
        "staging": staging,
        "pending": control / "pending.json",
        "source_journal": control / "source_access.jsonl",
        "slice_journal": control / "shard_slice_access.jsonl",
        "receipt": final / "label_shard_access_receipt.json",
    }


def _validate_label_shard_authorization(
    authorization: Any,
    *,
    contract: FrozenTarget2DEvaluationContract,
    stage: str,
    split: str,
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "artifact_role",
        "stage",
        "split",
        "prelabel_payload_sha256",
        "population_sha256",
        "implementation_sha256",
        "split_start",
        "stage1_receipt",
        "authorization_payload_sha256",
    }
    if not isinstance(authorization, Mapping):
        raise TypeError("Label-shard authorization must be a mapping.")
    value = dict(authorization)
    unsigned = {
        key: item
        for key, item in value.items()
        if key != "authorization_payload_sha256"
    }
    if (
        set(value) != expected_keys
        or value["schema_version"] != LABEL_SHARD_SCHEMA_VERSION
        or value["artifact_role"]
        != "p6_v2_release_label_shard_authorization"
        or value["stage"] != stage
        or value["split"] != split
        or value["authorization_payload_sha256"]
        != _canonical_json_sha256(unsigned)
    ):
        raise PermissionError("Label-shard authorization identity differs.")
    for key in (
        "prelabel_payload_sha256",
        "population_sha256",
        "implementation_sha256",
    ):
        _strict_sha256(value[key], label=f"authorization {key}")
    split_start = _verify_small_artifact_reference(
        value["split_start"],
        project_root=contract.project_root,
        label=f"{stage}/{split} split-start authority",
    )
    split_start_payload = _read_canonical_json(
        split_start,
        label=f"{stage}/{split} split-start authority",
    )
    if (
        split_start_payload.get("artifact_role")
        != "p6_v2_release_split_start"
        or split_start_payload.get("stage") != stage
        or split_start_payload.get("split") != split
        or split_start_payload.get("prelabel_payload_sha256")
        != value["prelabel_payload_sha256"]
        or split_start_payload.get("population_sha256")
        != value["population_sha256"]
    ):
        raise PermissionError("Split-start authority does not authorize shard.")
    if stage == "stage1":
        if value["stage1_receipt"] is not None:
            raise PermissionError(
                "Stage-1 label shard cannot bind a Stage-1 prerequisite."
            )
    else:
        if not isinstance(value["stage1_receipt"], Mapping):
            raise PermissionError(
                "Stage-2 label shard lacks a Stage-1 prerequisite."
            )
        stage1_path = _verify_small_artifact_reference(
            value["stage1_receipt"],
            project_root=contract.project_root,
            label="Stage-2 label-shard Stage-1 prerequisite",
        )
        stage1_payload = _read_canonical_json(
            stage1_path,
            label="Stage-1 release receipt",
        )
        if (
            stage1_payload.get("artifact_role")
            != "p6_v2_release_stage_receipt"
            or stage1_payload.get("stage") != "stage1"
            or stage1_payload.get("status") != "passed"
            or stage1_payload.get("prelabel_payload_sha256")
            != value["prelabel_payload_sha256"]
            or stage1_payload.get("population_sha256")
            != value["population_sha256"]
        ):
            raise PermissionError(
                "Stage 2 lacks a matching completed Stage-1 authority."
            )
    return value


def _release_control_root(project_root: Path) -> Path:
    return (
        project_root
        / "outputs"
        / "p6_v2"
        / "control"
        / "release_v1"
    )


def _release_split_start_path(
    project_root: Path,
    *,
    stage: str,
    split: str,
) -> Path:
    splits = P6_V2_RELEASE_STAGE_SPLITS[stage]
    if split not in splits:
        raise PermissionError("Release label-access stage/split differs.")
    return _release_control_root(project_root) / (
        f"{stage}_split_{splits.index(split) + 1:02d}_{split}_started.json"
    )


def _validate_release_pending_reference(
    reference: Any,
    *,
    project_root: Path,
    stage: str,
    prelabel_payload_sha256: str,
    population_sha256: str,
) -> dict[str, Any]:
    expected_path = _release_control_root(project_root) / (
        "stage1_pending.json" if stage == "stage1" else "stage2_pending.json"
    )
    path = _verify_small_artifact_reference(
        reference,
        project_root=project_root,
        label=f"{stage} release pending transaction",
        expected_path=expected_path,
    )
    pending = _read_canonical_json(
        path,
        label=f"{stage} release pending transaction",
    )
    unsigned = {
        key: value
        for key, value in pending.items()
        if key != "pending_payload_sha256"
    }
    if (
        pending.get("schema_version") != 1
        or pending.get("phase") != "P6V2"
        or pending.get("release_id") != P6_V2_RELEASE_ID
        or pending.get("artifact_role") != "p6_v2_release_stage_pending"
        or pending.get("stage") != stage
        or pending.get("status")
        != "started_write_once_hard_interruption_resume_only"
        or pending.get("prelabel_payload_sha256")
        != prelabel_payload_sha256
        or pending.get("population_sha256") != population_sha256
        or pending.get("hard_interruption_resume_allowed") is not True
        or pending.get("retry_or_overwrite_allowed") is not False
        or pending.get("pending_payload_sha256")
        != _canonical_json_sha256(unsigned)
    ):
        raise PermissionError("Release pending transaction identity differs.")
    return dict(reference)


def _validate_release_split_start(
    reference: Any,
    *,
    project_root: Path,
    stage: str,
    split: str,
    prelabel_payload_sha256: str,
    population_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_path = _release_split_start_path(
        project_root,
        stage=stage,
        split=split,
    )
    path = _verify_small_artifact_reference(
        reference,
        project_root=project_root,
        label=f"{stage}/{split} release split start",
        expected_path=expected_path,
    )
    start = _read_canonical_json(
        path,
        label=f"{stage}/{split} release split start",
    )
    unsigned = {
        key: value
        for key, value in start.items()
        if key != "split_start_payload_sha256"
    }
    expected_keys = {
        "schema_version",
        "phase",
        "release_id",
        "artifact_role",
        "stage",
        "split",
        "split_order",
        "status",
        "started_at",
        "prelabel_payload_sha256",
        "population_sha256",
        "pending_transaction",
        "hard_interruption_resume_allowed",
        "retry_or_overwrite_allowed",
        "split_start_payload_sha256",
    }
    split_order = P6_V2_RELEASE_STAGE_SPLITS[stage].index(split) + 1
    if (
        set(start) != expected_keys
        or start["schema_version"] != 1
        or start["phase"] != "P6V2"
        or start["release_id"] != P6_V2_RELEASE_ID
        or start["artifact_role"] != "p6_v2_release_split_start"
        or start["stage"] != stage
        or start["split"] != split
        or start["split_order"] != split_order
        or start["status"] != "started_write_once"
        or type(start["started_at"]) is not str
        or not start["started_at"]
        or start["prelabel_payload_sha256"] != prelabel_payload_sha256
        or start["population_sha256"] != population_sha256
        or start["hard_interruption_resume_allowed"] is not True
        or start["retry_or_overwrite_allowed"] is not False
        or start["split_start_payload_sha256"]
        != _canonical_json_sha256(unsigned)
    ):
        raise PermissionError("Release split-start identity differs.")
    pending_reference = _validate_release_pending_reference(
        start["pending_transaction"],
        project_root=project_root,
        stage=stage,
        prelabel_payload_sha256=prelabel_payload_sha256,
        population_sha256=population_sha256,
    )
    return dict(reference), {
        **start,
        "pending_transaction": pending_reference,
    }


def _validate_release_resume_attempt(
    reference: Any,
    *,
    project_root: Path,
    stage: str,
    split: str,
    authorization: Mapping[str, Any],
    split_start: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _verify_small_artifact_reference(
        reference,
        project_root=project_root,
        label=f"{stage} release resume attempt",
    )
    control_root = _release_control_root(project_root).resolve(strict=True)
    match = re.fullmatch(
        rf"{re.escape(stage)}_resume_attempt_([0-9]{{4}})\.json",
        path.name,
    )
    if (
        path.parent.resolve(strict=True) != control_root
        or match is None
        or int(match.group(1)) < 1
    ):
        raise PermissionError("Release resume-attempt path differs.")
    attempt = _read_canonical_json(
        path,
        label=f"{stage} release resume attempt",
    )
    unsigned = {
        key: value
        for key, value in attempt.items()
        if key != "resume_attempt_payload_sha256"
    }
    expected_keys = {
        "schema_version",
        "phase",
        "release_id",
        "artifact_role",
        "stage",
        "attempt_number",
        "status",
        "requested_at",
        "prelabel_payload_sha256",
        "prelabel_authority_commit_sha",
        "adapter_implementation_sha256",
        "population_sha256",
        "pending_transaction",
        "stage1_receipt",
        "completed_prefix_splits",
        "completed_prefix_split_receipts",
        "partial_split",
        "resume_interrupted_adapter_call_required",
        "hard_interruption_resume_allowed",
        "hard_interruption_resume_policy",
        "outcome_interpreted",
        "retry_or_overwrite_allowed",
        "resume_attempt_payload_sha256",
    }
    splits = P6_V2_RELEASE_STAGE_SPLITS[stage]
    completed = attempt.get("completed_prefix_splits")
    completed_receipts = attempt.get("completed_prefix_split_receipts")
    completed_count = len(completed) if isinstance(completed, list) else -1
    next_split = (
        splits[completed_count]
        if 0 <= completed_count < len(splits)
        else None
    )
    if (
        set(attempt) != expected_keys
        or attempt["schema_version"] != 1
        or attempt["phase"] != "P6V2"
        or attempt["release_id"] != P6_V2_RELEASE_ID
        or attempt["artifact_role"]
        != "p6_v2_release_hard_interruption_resume_attempt"
        or attempt["stage"] != stage
        or attempt["attempt_number"] != int(match.group(1))
        or attempt["status"] != "hard_interruption_resume_requested"
        or type(attempt["requested_at"]) is not str
        or not attempt["requested_at"]
        or attempt["prelabel_payload_sha256"]
        != authorization["prelabel_payload_sha256"]
        or attempt["population_sha256"]
        != authorization["population_sha256"]
        or attempt["adapter_implementation_sha256"]
        != authorization["implementation_sha256"]
        or attempt["pending_transaction"]
        != split_start["pending_transaction"]
        or attempt["stage1_receipt"]
        != authorization["stage1_receipt"]
        or not isinstance(completed, list)
        or not isinstance(completed_receipts, list)
        or completed != list(splits[:completed_count])
        or len(completed_receipts) != completed_count
        or split in completed
        or attempt["partial_split"] not in {None, next_split}
        or attempt["resume_interrupted_adapter_call_required"]
        != (attempt["partial_split"] is not None)
        or attempt["hard_interruption_resume_allowed"] is not True
        or attempt["hard_interruption_resume_policy"]
        != P6_V2_HARD_INTERRUPTION_RESUME_POLICY
        or attempt["outcome_interpreted"] is not False
        or attempt["retry_or_overwrite_allowed"] is not False
        or attempt["resume_attempt_payload_sha256"]
        != _canonical_json_sha256(unsigned)
    ):
        raise PermissionError("Release resume-attempt identity differs.")
    if (
        type(attempt["prelabel_authority_commit_sha"]) is not str
        or re.fullmatch(
            r"[0-9a-f]{40,64}",
            attempt["prelabel_authority_commit_sha"],
        )
        is None
    ):
        raise PermissionError(
            "Release resume pre-label authority commit differs."
        )
    for completed_split, completed_reference in zip(
        completed,
        completed_receipts,
        strict=True,
    ):
        completed_order = splits.index(completed_split) + 1
        _verify_small_artifact_reference(
            completed_reference,
            project_root=project_root,
            label=f"{stage}/{completed_split} completed split receipt",
            expected_path=control_root
            / (
                f"{stage}_split_{completed_order:02d}_"
                f"{completed_split}_receipt.json"
            ),
        )
    return dict(reference), attempt


def _validate_label_access_session(
    access_session: Any,
    *,
    project_root: Path,
    stage: str,
    split: str,
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "artifact_role",
        "stage",
        "split",
        "invocation_kind",
        "adapter_resume_interrupted",
        "split_start",
        "release_resume_attempt",
        "prelabel_payload_sha256",
        "population_sha256",
        "session_authority_sha256",
        "session_payload_sha256",
    }
    if not isinstance(access_session, Mapping):
        raise TypeError("Label-access session must be a mapping.")
    value = dict(access_session)
    unsigned = {
        key: item
        for key, item in value.items()
        if key != "session_payload_sha256"
    }
    if (
        set(value) != expected_keys
        or value["schema_version"] != 1
        or value["artifact_role"] != "p6_v2_release_label_access_session"
        or value["stage"] != stage
        or value["split"] != split
        or type(value["adapter_resume_interrupted"]) is not bool
        or value["prelabel_payload_sha256"]
        != authorization["prelabel_payload_sha256"]
        or value["population_sha256"]
        != authorization["population_sha256"]
        or value["split_start"] != authorization["split_start"]
        or value["session_payload_sha256"]
        != _canonical_json_sha256(unsigned)
    ):
        raise PermissionError("Label-access session identity differs.")
    split_start_reference, split_start = _validate_release_split_start(
        value["split_start"],
        project_root=project_root,
        stage=stage,
        split=split,
        prelabel_payload_sha256=authorization[
            "prelabel_payload_sha256"
        ],
        population_sha256=authorization["population_sha256"],
    )
    if split_start_reference != dict(authorization["split_start"]):
        raise PermissionError(
            "Label-access split-start authority differs from shard authority."
        )
    resume_reference = value["release_resume_attempt"]
    if resume_reference is None:
        if (
            value["invocation_kind"]
            != "initial_release_stage_invocation"
            or value["adapter_resume_interrupted"] is not False
            or value["session_authority_sha256"]
            != split_start_reference["sha256"]
        ):
            raise PermissionError(
                "Initial label-access session authority differs."
            )
    else:
        checked_reference, attempt = _validate_release_resume_attempt(
            resume_reference,
            project_root=project_root,
            stage=stage,
            split=split,
            authorization=authorization,
            split_start=split_start,
        )
        if (
            checked_reference != dict(resume_reference)
            or value["invocation_kind"]
            != "hard_interruption_resume_stage_invocation"
            or value["adapter_resume_interrupted"]
            != (attempt["partial_split"] == split)
            or value["session_authority_sha256"]
            != checked_reference["sha256"]
        ):
            raise PermissionError(
                "Resumed label-access session authority differs."
            )
    _strict_sha256(
        value["session_authority_sha256"],
        label="label-access session authority SHA-256",
    )
    _strict_sha256(
        value["session_payload_sha256"],
        label="label-access session payload SHA-256",
    )
    return value


def _expected_slice_hashes(
    contract: FrozenTarget2DEvaluationContract,
) -> tuple[dict[int, dict[str, str]], str]:
    case_artifacts = contract.source_manifest.get("case_artifact_hashes")
    if not isinstance(case_artifacts, dict):
        raise ValueError("Source case-artifact hashes are missing.")
    expected: dict[int, dict[str, str]] = {}
    for case_id in contract.spec.case_ids:
        definition = contract.definitions[case_id]
        case_key = definition.get("case_key")
        entry = case_artifacts.get(case_key)
        output = (
            entry.get("output_slice_sha256")
            if isinstance(entry, Mapping)
            else None
        )
        if not isinstance(output, Mapping) or set(output) != set(
            LABEL_ARRAY_NAMES
        ):
            raise ValueError(
                f"Source output-slice hashes differ for case {case_id}."
            )
        expected[case_id] = {
            name: _strict_sha256(
                output[name],
                label=f"case {case_id} {name} output-slice SHA-256",
            )
            for name in LABEL_ARRAY_NAMES
        }
    snapshot = [
        {
            "global_case_id": case_id,
            "case_key": contract.definitions[case_id]["case_key"],
            "output_slice_sha256": expected[case_id],
        }
        for case_id in contract.spec.case_ids
    ]
    return expected, _canonical_json_sha256(snapshot)


def _label_shard_pending_payload(
    contract: FrozenTarget2DEvaluationContract,
    *,
    stage: str,
    split: str,
    authorization: Mapping[str, Any],
    expected_slice_hashes_sha256: str,
) -> dict[str, Any]:
    source = contract.source_manifest
    shape = source["array_shape_case_time_z_x"]
    unsigned = {
        "schema_version": LABEL_SHARD_SCHEMA_VERSION,
        "phase": "P6V2",
        "artifact_role": LABEL_SHARD_PENDING_ROLE,
        "status": "authorized_before_any_monolithic_label_read",
        "stage": stage,
        "split": split,
        "role": contract.spec.role,
        "ordered_global_case_ids": list(contract.spec.case_ids),
        "row_shape": list(shape[1:]),
        "label_dtypes": {
            name: np.dtype("float32").str for name in LABEL_ARRAY_NAMES
        },
        "source_manifest": _artifact_reference(
            contract.source_manifest_path,
            contract.project_root,
        ),
        "source_dataset_id": source["dataset_id"],
        "source_array_artifact_root": source["array_artifact_root"],
        "source_array_shape_case_time_z_x": list(shape),
        "source_expected_full_label_sha256_metadata_only": {
            name: source["array_sha256"][name]
            for name in LABEL_ARRAY_NAMES
        },
        "expected_output_slice_hashes_sha256": (
            expected_slice_hashes_sha256
        ),
        "authorization": dict(authorization),
        "monolithic_label_payload_opened_before_pending_fsync": False,
        "whole_monolithic_label_hashing_authorized": False,
        "monolithic_label_mmap_authorized": False,
        "write_once": True,
    }
    return {
        **unsigned,
        "pending_payload_sha256": _canonical_json_sha256(unsigned),
    }


def _source_journal_state(
    journal: _DurableJournal,
) -> dict[str, dict[str, dict[str, Any]]]:
    state: dict[str, dict[str, dict[str, Any]]] = {}
    for record in journal.records:
        event = record["event"]
        if event not in {
            "source_range_read_attempt",
            "source_range_read_completion",
            "source_range_recovered_completion",
        }:
            continue
        token = record.get("read_token")
        if type(token) is not str or not token:
            raise ValueError("Source journal read token is invalid.")
        entry = state.setdefault(token, {})
        key = (
            "attempt"
            if event == "source_range_read_attempt"
            else "completion"
        )
        if key in entry:
            raise PermissionError(
                f"Source byte range was replayed or completed twice: {token}."
            )
        entry[key] = record
    for token, entry in state.items():
        if "completion" in entry and "attempt" not in entry:
            raise ValueError(
                f"Source completion lacks a durable attempt: {token}."
            )
        if (
            "completion" in entry
            and entry["completion"]["sequence"]
            <= entry["attempt"]["sequence"]
        ):
            raise ValueError(
                f"Source completion precedes its attempt: {token}."
            )
    return state


class _SourceRangeSession:
    """Unbuffered source-container session with durable open evidence."""

    def __init__(
        self,
        path: Path,
        *,
        label_name: str,
        journal: _DurableJournal,
        session_id: str,
    ) -> None:
        self.path = path
        self.label_name = label_name
        self.journal = journal
        self.session_id = session_id
        self.stream: BinaryIO | None = None

    def open(self) -> BinaryIO:
        if self.stream is None:
            if self.path.is_symlink() or not self.path.is_file():
                raise PermissionError("Monolithic label container is unsafe.")
            self.journal.append(
                "source_container_open_attempt",
                session_id=self.session_id,
                label_name=self.label_name,
                access_mode="unbuffered_binary_bounded_ranges_only",
                whole_file_hashing=False,
                mmap=False,
            )
            self.stream = self.path.open("rb", buffering=0)
            self.journal.append(
                "source_container_open_completion",
                session_id=self.session_id,
                label_name=self.label_name,
                access_mode="unbuffered_binary_bounded_ranges_only",
                whole_file_hashing=False,
                mmap=False,
            )
        return self.stream

    def close(self) -> None:
        if self.stream is not None:
            self.stream.close()
            self.stream = None

    def __enter__(self) -> "_SourceRangeSession":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def exact_read(
        self,
        *,
        offset: int,
        length: int,
        read_token: str,
        purpose: str,
        case_id: int | None,
    ) -> bytes:
        state = _source_journal_state(self.journal)
        if read_token in state:
            raise PermissionError(
                f"Source range replay is forbidden: {read_token}."
            )
        self.journal.append(
            "source_range_read_attempt",
            session_id=self.session_id,
            read_token=read_token,
            label_name=self.label_name,
            purpose=purpose,
            global_case_id=case_id,
            byte_range=[offset, offset + length],
            byte_range_convention="half_open_[start,end)",
            requested_bytes=length,
        )
        stream = self.open()
        stream.seek(offset, os.SEEK_SET)
        raw = stream.read(length)
        if len(raw) != length:
            raise OSError(
                f"Short bounded read for {read_token}: "
                f"{len(raw)} != {length}."
            )
        return raw

    def complete_read(
        self,
        *,
        read_token: str,
        purpose: str,
        case_id: int | None,
        offset: int,
        raw: bytes,
        observed_semantic_sha256: str | None = None,
        recovered: bool = False,
    ) -> None:
        state = _source_journal_state(self.journal)
        entry = state.get(read_token)
        if entry is None or "attempt" not in entry or "completion" in entry:
            raise PermissionError(
                f"Source completion state differs: {read_token}."
            )
        self.journal.append(
            (
                "source_range_recovered_completion"
                if recovered
                else "source_range_read_completion"
            ),
            session_id=self.session_id,
            read_token=read_token,
            label_name=self.label_name,
            purpose=purpose,
            global_case_id=case_id,
            byte_range=[offset, offset + len(raw)],
            byte_range_convention="half_open_[start,end)",
            observed_bytes=len(raw),
            observed_bytes_sha256=hashlib.sha256(raw).hexdigest(),
            observed_semantic_sha256=observed_semantic_sha256,
            source_range_replayed=False,
        )


def _header_segment(
    session: _SourceRangeSession,
    *,
    cache_path: Path,
    offset: int,
    length: int,
    purpose: str,
) -> bytes:
    token = f"{session.label_name}:header:{offset}:{length}:{purpose}"
    state = _source_journal_state(session.journal)
    entry = state.get(token)
    cache_size = cache_path.stat().st_size if cache_path.exists() else 0
    end = offset + length
    if cache_path.is_symlink():
        raise PermissionError("Source-header cache is unsafe.")
    if entry is not None:
        if cache_size < end:
            raise PermissionError(
                "Interrupted source-header read has no complete durable cache; "
                "replay is forbidden."
            )
        with cache_path.open("rb", buffering=0) as stream:
            stream.seek(offset)
            raw = stream.read(length)
        if len(raw) != length:
            raise ValueError("Source-header cache is truncated.")
        if "completion" not in entry:
            session.complete_read(
                read_token=token,
                purpose=purpose,
                case_id=None,
                offset=offset,
                raw=raw,
                recovered=True,
            )
        return raw
    if cache_size != offset:
        raise ValueError(
            "Source-header cache length does not match the next exact range."
        )
    raw = session.exact_read(
        offset=offset,
        length=length,
        read_token=token,
        purpose=purpose,
        case_id=None,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "ab" if cache_path.exists() else "xb"
    with cache_path.open(mode, buffering=0) as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    if offset == 0:
        _fsync_directory(cache_path.parent)
    session.complete_read(
        read_token=token,
        purpose=purpose,
        case_id=None,
        offset=offset,
        raw=raw,
    )
    return raw


def _parse_source_npy_header(
    session: _SourceRangeSession,
    *,
    cache_path: Path,
    expected_shape: tuple[int, ...],
) -> dict[str, Any]:
    magic = _header_segment(
        session,
        cache_path=cache_path,
        offset=0,
        length=6,
        purpose="npy_magic",
    )
    if magic != _NPY_MAGIC:
        raise ValueError("Monolithic label container has invalid NPY magic.")
    version_raw = _header_segment(
        session,
        cache_path=cache_path,
        offset=6,
        length=2,
        purpose="npy_version",
    )
    version = tuple(version_raw)
    if version == (1, 0):
        length_size = 2
        encoding = "latin1"
    elif version in {(2, 0), (3, 0)}:
        length_size = 4
        encoding = "utf-8" if version == (3, 0) else "latin1"
    else:
        raise ValueError(f"Unsupported NPY version: {version}.")
    length_raw = _header_segment(
        session,
        cache_path=cache_path,
        offset=8,
        length=length_size,
        purpose="npy_header_length",
    )
    header_length = (
        struct.unpack("<H", length_raw)[0]
        if length_size == 2
        else struct.unpack("<I", length_raw)[0]
    )
    if header_length < 1 or header_length > 1_000_000:
        raise ValueError("NPY header length is invalid.")
    header_offset = 8 + length_size
    header_raw = _header_segment(
        session,
        cache_path=cache_path,
        offset=header_offset,
        length=header_length,
        purpose="npy_header_payload",
    )
    try:
        header = ast.literal_eval(header_raw.decode(encoding).strip())
    except (UnicodeDecodeError, SyntaxError, ValueError) as error:
        raise ValueError("Cannot parse monolithic NPY header.") from error
    if not isinstance(header, dict) or set(header) != {
        "descr",
        "fortran_order",
        "shape",
    }:
        raise ValueError("Monolithic NPY header mapping differs.")
    dtype = np.dtype(header["descr"])
    shape = tuple(header["shape"])
    if (
        dtype != np.dtype("float32")
        or dtype.str != np.dtype("<f4").str
        or header["fortran_order"] is not False
        or shape != expected_shape
    ):
        raise ValueError("Monolithic label NPY header shape/dtype/order differs.")
    data_offset = header_offset + header_length
    expected_bytes = data_offset + int(np.prod(shape, dtype=np.int64)) * 4
    if session.path.stat().st_size != expected_bytes:
        raise ValueError("Monolithic label container byte count differs.")
    return {
        "version": list(version),
        "shape": list(shape),
        "dtype": dtype.str,
        "fortran_order": False,
        "data_offset": data_offset,
        "container_bytes": expected_bytes,
        "header_byte_ranges_read": [
            [0, 6],
            [6, 8],
            [8, 8 + length_size],
            [header_offset, data_offset],
        ],
    }


def _create_empty_npy(
    path: Path,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
) -> int:
    """Create one fixed-size C-order NPY file without replacing a target."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or path.exists() or path.is_symlink():
        raise FileExistsError(f"Staging NPY already exists: {path}.")
    with path.open("xb", buffering=0) as stream:
        np.lib.format.write_array_header_2_0(
            stream,
            {
                "descr": np.lib.format.dtype_to_descr(dtype),
                "fortran_order": False,
                "shape": shape,
            },
        )
        data_offset = stream.tell()
        stream.truncate(
            data_offset + int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        )
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(path.parent)
    return data_offset


def _inspect_shard_npy(
    path: Path,
    *,
    expected_shape: tuple[int, ...],
) -> tuple[int, int]:
    if path.is_symlink() or not path.is_file():
        raise PermissionError("Label shard is missing or unsafe.")
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot inspect label shard {path}: {error}") from error
    try:
        if (
            array.shape != expected_shape
            or array.dtype != np.dtype("float32")
            or not array.flags.c_contiguous
        ):
            raise ValueError("Label shard header differs.")
        data_offset = int(array.offset)
    finally:
        del array
    expected_bytes = (
        data_offset
        + int(np.prod(expected_shape, dtype=np.int64))
        * np.dtype("float32").itemsize
    )
    if path.stat().st_size != expected_bytes:
        raise ValueError("Label shard byte count differs.")
    return data_offset, expected_bytes


def _write_exact_shard_row(
    path: Path,
    *,
    data_offset: int,
    local_row: int,
    row_bytes: int,
    raw: bytes,
) -> None:
    if len(raw) != row_bytes:
        raise ValueError("Shard row byte count differs.")
    with path.open("r+b", buffering=0) as stream:
        stream.seek(data_offset + local_row * row_bytes)
        written = stream.write(raw)
        if written != row_bytes:
            raise OSError("Short label-shard row write.")
        stream.flush()
        os.fsync(stream.fileno())


def _read_exact_shard_row(
    path: Path,
    *,
    data_offset: int,
    local_row: int,
    row_bytes: int,
) -> bytes:
    with path.open("rb", buffering=0) as stream:
        stream.seek(data_offset + local_row * row_bytes)
        raw = stream.read(row_bytes)
    if len(raw) != row_bytes:
        raise ValueError("Label shard row is truncated.")
    return raw


def _source_case_read_token(
    label_name: str,
    *,
    case_id: int,
    offset: int,
    length: int,
) -> str:
    return f"{label_name}:case:{case_id}:{offset}:{length}"


def _extract_one_label_shard(
    contract: FrozenTarget2DEvaluationContract,
    *,
    label_name: str,
    staging_path: Path,
    header_cache_path: Path,
    journal: _DurableJournal,
    expected_hashes: Mapping[int, Mapping[str, str]],
    session_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_shape = tuple(
        int(value)
        for value in contract.source_manifest[
            "array_shape_case_time_z_x"
        ]
    )
    row_shape = source_shape[1:]
    shard_shape = (len(contract.spec.case_ids), *row_shape)
    row_bytes = int(np.prod(row_shape, dtype=np.int64)) * 4
    if staging_path.exists():
        try:
            _inspect_shard_npy(
                staging_path,
                expected_shape=shard_shape,
            )
        except (OSError, PermissionError, ValueError):
            state = _source_journal_state(journal)
            attempted_label_rows = [
                item
                for item in state.values()
                if item["attempt"].get("label_name") == label_name
                and item["attempt"].get("purpose")
                == "authorized_case_label_slice"
            ]
            if attempted_label_rows or staging_path.is_symlink():
                raise PermissionError(
                    "Interrupted staging shard differs after a durable "
                    "source-range attempt; overwrite is forbidden."
                )
            staging_path.unlink()
    if not staging_path.exists():
        _create_empty_npy(
            staging_path,
            shape=shard_shape,
            dtype=np.dtype("<f4"),
        )
    shard_data_offset, _ = _inspect_shard_npy(
        staging_path,
        expected_shape=shard_shape,
    )
    source_path = contract.array_paths[label_name]
    with _SourceRangeSession(
        source_path,
        label_name=label_name,
        journal=journal,
        session_id=session_id,
    ) as session:
        source_header = _parse_source_npy_header(
            session,
            cache_path=header_cache_path,
            expected_shape=source_shape,
        )
        source_data_offset = int(source_header["data_offset"])
        ranges: list[dict[str, Any]] = []
        for local_row, case_id in enumerate(contract.spec.case_ids):
            offset = source_data_offset + case_id * row_bytes
            token = _source_case_read_token(
                label_name,
                case_id=case_id,
                offset=offset,
                length=row_bytes,
            )
            state = _source_journal_state(journal)
            prior = state.get(token)
            expected = expected_hashes[case_id][label_name]
            if prior is not None:
                raw = _read_exact_shard_row(
                    staging_path,
                    data_offset=shard_data_offset,
                    local_row=local_row,
                    row_bytes=row_bytes,
                )
                observed = _semantic_array_sha256(
                    raw,
                    dtype=np.dtype("<f4"),
                    shape=row_shape,
                )
                if observed != expected:
                    raise PermissionError(
                        "Interrupted source-range attempt cannot be replayed "
                        f"and its durable shard row differs: {token}."
                    )
                if "completion" not in prior:
                    session.complete_read(
                        read_token=token,
                        purpose="authorized_case_label_slice",
                        case_id=case_id,
                        offset=offset,
                        raw=raw,
                        observed_semantic_sha256=observed,
                        recovered=True,
                    )
            else:
                raw = session.exact_read(
                    offset=offset,
                    length=row_bytes,
                    read_token=token,
                    purpose="authorized_case_label_slice",
                    case_id=case_id,
                )
                observed = _semantic_array_sha256(
                    raw,
                    dtype=np.dtype("<f4"),
                    shape=row_shape,
                )
                if observed != expected:
                    raise ValueError(
                        f"Frozen {label_name} slice hash differs for "
                        f"case {case_id}; shard commit is forbidden."
                    )
                _write_exact_shard_row(
                    staging_path,
                    data_offset=shard_data_offset,
                    local_row=local_row,
                    row_bytes=row_bytes,
                    raw=raw,
                )
                session.complete_read(
                    read_token=token,
                    purpose="authorized_case_label_slice",
                    case_id=case_id,
                    offset=offset,
                    raw=raw,
                    observed_semantic_sha256=observed,
                )
            ranges.append(
                {
                    "global_case_id": case_id,
                    "local_shard_row": local_row,
                    "source_byte_range": [offset, offset + row_bytes],
                    "byte_range_convention": "half_open_[start,end)",
                    "bytes": row_bytes,
                    "expected_semantic_sha256": expected,
                    "observed_semantic_sha256": observed,
                }
            )
    return source_header, ranges


def _source_journal_summary(
    journal: _DurableJournal,
    *,
    contract: FrozenTarget2DEvaluationContract,
    source_headers: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    records = journal.records
    state = _source_journal_state(journal)
    if any(
        "attempt" not in item or "completion" not in item
        for item in state.values()
    ):
        raise PermissionError(
            "Source-range journal contains an unresolved attempt."
        )
    expected_tokens: set[str] = set()
    expected_case_ranges: dict[str, list[list[int]]] = {}
    row_bytes = (
        int(
            np.prod(
                contract.source_manifest[
                    "array_shape_case_time_z_x"
                ][1:],
                dtype=np.int64,
            )
        )
        * 4
    )
    for label_name in LABEL_ARRAY_NAMES:
        header = source_headers[label_name]
        data_offset = int(header["data_offset"])
        ranges: list[list[int]] = []
        for case_id in contract.spec.case_ids:
            offset = data_offset + case_id * row_bytes
            token = _source_case_read_token(
                label_name,
                case_id=case_id,
                offset=offset,
                length=row_bytes,
            )
            expected_tokens.add(token)
            entry = state.get(token)
            expected_range = [offset, offset + row_bytes]
            case_key = contract.definitions[case_id]["case_key"]
            expected_semantic = contract.source_manifest[
                "case_artifact_hashes"
            ][case_key]["output_slice_sha256"][label_name]
            if (
                entry is None
                or entry["attempt"].get("label_name") != label_name
                or entry["attempt"].get("purpose")
                != "authorized_case_label_slice"
                or entry["attempt"].get("global_case_id") != case_id
                or entry["attempt"].get("byte_range") != expected_range
                or entry["attempt"].get("requested_bytes") != row_bytes
                or entry["completion"].get("label_name") != label_name
                or entry["completion"].get("purpose")
                != "authorized_case_label_slice"
                or entry["completion"].get("global_case_id") != case_id
                or entry["completion"].get("byte_range") != expected_range
                or entry["completion"].get("observed_bytes") != row_bytes
                or entry["completion"].get(
                    "observed_semantic_sha256"
                )
                != expected_semantic
                or entry["completion"].get("source_range_replayed")
                is not False
            ):
                raise PermissionError(
                    f"Source journal range evidence differs: {token}."
                )
            ranges.append(list(entry["completion"]["byte_range"]))
        expected_case_ranges[label_name] = ranges
    observed_case_tokens = {
        token
        for token, item in state.items()
        if item["attempt"].get("purpose") == "authorized_case_label_slice"
    }
    if observed_case_tokens != expected_tokens:
        raise PermissionError(
            "Source journal does not contain the exact authorized case ranges."
        )
    expected_header_tokens = {
        (
            f"{label_name}:header:{start}:{end - start}:"
            f"{purpose}"
        )
        for label_name in LABEL_ARRAY_NAMES
        for (start, end), purpose in zip(
            source_headers[label_name]["header_byte_ranges_read"],
            (
                "npy_magic",
                "npy_version",
                "npy_header_length",
                "npy_header_payload",
            ),
            strict=True,
        )
    }
    if set(state) != expected_tokens | expected_header_tokens:
        raise PermissionError(
            "Source journal contains an unregistered or missing byte range."
        )
    for token in expected_header_tokens:
        entry = state[token]
        if (
            entry["attempt"].get("global_case_id") is not None
            or entry["completion"].get("global_case_id") is not None
            or entry["attempt"].get("byte_range")
            != entry["completion"].get("byte_range")
            or entry["completion"].get("source_range_replayed") is not False
        ):
            raise PermissionError(
                f"Source header-range journal differs: {token}."
            )
    unauthorized_intersections: list[dict[str, Any]] = []
    total_cases = int(
        contract.source_manifest["array_shape_case_time_z_x"][0]
    )
    authorized = set(contract.spec.case_ids)
    for label_name in LABEL_ARRAY_NAMES:
        data_offset = int(source_headers[label_name]["data_offset"])
        observed = expected_case_ranges[label_name]
        for case_id in range(total_cases):
            if case_id in authorized:
                continue
            forbidden = [
                data_offset + case_id * row_bytes,
                data_offset + (case_id + 1) * row_bytes,
            ]
            for current in observed:
                if max(current[0], forbidden[0]) < min(
                    current[1], forbidden[1]
                ):
                    unauthorized_intersections.append(
                        {
                            "label_name": label_name,
                            "global_case_id": case_id,
                            "forbidden_range": forbidden,
                            "observed_range": current,
                        }
                    )
    if unauthorized_intersections:
        raise PermissionError("Unauthorized monolithic case bytes were read.")
    open_attempts = [
        item
        for item in records
        if item["event"] == "source_container_open_attempt"
    ]
    open_completions = [
        item
        for item in records
        if item["event"] == "source_container_open_completion"
    ]
    if len(open_attempts) != len(open_completions) or not open_completions:
        raise PermissionError("Source-container open journal is incomplete.")
    recovered = sum(
        item["event"] == "source_range_recovered_completion"
        for item in records
    )
    return {
        "journal_record_count": len(records),
        "journal_tail_sha256": (
            records[-1]["event_sha256"] if records else "0" * 64
        ),
        "source_container_open_count": len(open_completions),
        "source_container_opened_for_bounded_extraction": True,
        "whole_source_container_hashed": False,
        "source_container_mmap_used": False,
        "authorized_case_ids": list(contract.spec.case_ids),
        "authorized_case_byte_ranges_read": expected_case_ranges,
        "unauthorized_case_byte_ranges_read": [],
        "unauthorized_case_bytes_read": False,
        "source_range_replay_count": 0,
        "recovered_without_source_replay_completion_count": recovered,
        "all_source_read_attempts_reconciled": True,
    }


def _resume_attempt_reference(
    paths: Mapping[str, Path],
    *,
    contract: FrozenTarget2DEvaluationContract,
    stage: str,
    split: str,
    pending: Mapping[str, Any],
) -> dict[str, Any]:
    control = paths["control"]
    candidates = sorted(control.glob("resume_attempt_*.json"))
    pattern = re.compile(r"^resume_attempt_([0-9]{4})\.json$")
    numbers: list[int] = []
    for path in candidates:
        match = pattern.fullmatch(path.name)
        if match is None or path.is_symlink() or not path.is_file():
            raise PermissionError("Malformed label-shard resume artifact.")
        numbers.append(int(match.group(1)))
    if numbers != list(range(1, len(numbers) + 1)):
        raise PermissionError(
            "Label-shard resume-attempt numbering is not contiguous."
        )
    number = len(numbers) + 1
    unsigned = {
        "schema_version": LABEL_SHARD_SCHEMA_VERSION,
        "artifact_role": "target_2d_label_shard_resume_attempt",
        "stage": stage,
        "split": split,
        "attempt_number": number,
        "pending_payload_sha256": pending["pending_payload_sha256"],
        "source_container_range_replay_allowed": False,
        "completed_source_ranges_may_be_adopted": True,
        "incomplete_attempt_requires_exact_durable_shard_row": True,
        "isolated_shard_slice_read_scope": (
            "exactly_once_per_authorized_evaluation_invocation"
        ),
        "prior_invocation_journal_entries_remain_immutable": True,
    }
    document = {
        **unsigned,
        "resume_attempt_payload_sha256": _canonical_json_sha256(unsigned),
    }
    path = control / f"resume_attempt_{number:04d}.json"
    _write_once_bytes(path, _canonical_json_bytes(document))
    return _artifact_reference(path, contract.project_root)


def _declared_reference(
    current_path: Path,
    *,
    final_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    resolved_final = final_path.resolve(strict=False)
    if not resolved_final.is_relative_to(root):
        raise PermissionError("Declared shard path escapes the project.")
    return {
        "path": resolved_final.relative_to(root).as_posix(),
        "sha256": _sha256_file(current_path),
        "bytes": current_path.stat().st_size,
    }


def _publish_file_no_replace(
    source: Path,
    target: Path,
    *,
    expected: Mapping[str, Any],
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if (
            target.is_symlink()
            or not target.is_file()
            or target.stat().st_size != expected["bytes"]
            or _sha256_file(target) != expected["sha256"]
        ):
            raise FileExistsError(
                f"Partial/tampered published label shard differs: {target}."
            )
        return
    try:
        os.link(source, target)
    except FileExistsError:
        raise
    except OSError as error:
        raise RuntimeError(
            "Atomic no-replace label-shard publication is unavailable."
        ) from error
    _fsync_directory(target.parent)
    if (
        target.stat().st_size != expected["bytes"]
        or _sha256_file(target) != expected["sha256"]
    ):
        raise ValueError("Published label shard differs from staging.")


def _label_shard_receipt_payload(
    contract: FrozenTarget2DEvaluationContract,
    *,
    stage: str,
    split: str,
    pending: Mapping[str, Any],
    pending_reference: Mapping[str, Any],
    authorization: Mapping[str, Any],
    expected_hashes: Mapping[int, Mapping[str, str]],
    source_headers: Mapping[str, Mapping[str, Any]],
    source_ranges: Mapping[str, list[dict[str, Any]]],
    source_journal: _DurableJournal,
    paths: Mapping[str, Path],
    resume_attempt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    row_shape = tuple(
        int(value)
        for value in contract.source_manifest[
            "array_shape_case_time_z_x"
        ][1:]
    )
    shard_shape = (len(contract.spec.case_ids), *row_shape)
    label_shards = {
        name: {
            **_declared_reference(
                paths["staging"] / f"{name}.npy",
                final_path=paths["final"] / f"{name}.npy",
                project_root=contract.project_root,
            ),
            "shape": list(shard_shape),
            "dtype": np.dtype("<f4").str,
            "global_case_id_to_local_row": {
                str(case_id): local_row
                for local_row, case_id in enumerate(contract.spec.case_ids)
            },
        }
        for name in LABEL_ARRAY_NAMES
    }
    cases = [
        {
            "local_shard_row": local_row,
            "global_case_id": case_id,
            "case_key": contract.definitions[case_id]["case_key"],
            "expected_output_slice_sha256": dict(expected_hashes[case_id]),
            "observed_output_slice_sha256": {
                name: source_ranges[name][local_row][
                    "observed_semantic_sha256"
                ]
                for name in LABEL_ARRAY_NAMES
            },
            "source_byte_ranges": {
                name: source_ranges[name][local_row]["source_byte_range"]
                for name in LABEL_ARRAY_NAMES
            },
        }
        for local_row, case_id in enumerate(contract.spec.case_ids)
    ]
    journal_summary = _source_journal_summary(
        source_journal,
        contract=contract,
        source_headers=source_headers,
    )
    unsigned = {
        "schema_version": LABEL_SHARD_SCHEMA_VERSION,
        "phase": "P6V2",
        "artifact_role": LABEL_SHARD_ARTIFACT_ROLE,
        "status": "complete_write_once_manifest_last",
        "stage": stage,
        "split": split,
        "role": contract.spec.role,
        "ordered_global_case_ids": list(contract.spec.case_ids),
        "row_shape": list(row_shape),
        "label_dtypes": {
            name: np.dtype("<f4").str for name in LABEL_ARRAY_NAMES
        },
        "pending": dict(pending_reference),
        "pending_payload_sha256": pending["pending_payload_sha256"],
        "authorization": dict(authorization),
        "source_manifest": _artifact_reference(
            contract.source_manifest_path,
            contract.project_root,
        ),
        "source_expected_full_label_sha256_metadata_only": {
            name: contract.source_manifest["array_sha256"][name]
            for name in LABEL_ARRAY_NAMES
        },
        "source_containers": {
            name: {
                "path": contract.array_paths[name]
                .resolve(strict=True)
                .relative_to(contract.project_root.resolve(strict=True))
                .as_posix(),
                "expected_full_sha256_metadata_only": (
                    contract.source_manifest["array_sha256"][name]
                ),
                "header": dict(source_headers[name]),
                "header_cache": _artifact_reference(
                    paths["control"] / f"{name}_header.bin",
                    contract.project_root,
                ),
                "authorized_case_ranges": list(source_ranges[name]),
                "source_container_opened_for_bounded_extraction": True,
                "whole_source_container_hashed": False,
                "source_container_mmap_used": False,
                "unauthorized_case_bytes_read": False,
            }
            for name in LABEL_ARRAY_NAMES
        },
        "cases": cases,
        "label_shards": label_shards,
        "source_access_journal": source_journal.reference(
            contract.project_root
        ),
        "source_access_summary": journal_summary,
        "shard_slice_access_journal_path": (
            paths["slice_journal"]
            .resolve(strict=False)
            .relative_to(contract.project_root.resolve(strict=True))
            .as_posix()
        ),
        "shard_slice_access_journal_initial": _artifact_reference(
            paths["slice_journal"],
            contract.project_root,
        ),
        "resume_attempt": (
            dict(resume_attempt) if resume_attempt is not None else None
        ),
        "manifest_written_after_both_shards_fsynced": True,
        "whole_monolithic_label_hashing_performed": False,
        "monolithic_label_mmap_performed": False,
        "unauthorized_case_bytes_read": False,
        "write_once_no_overwrite": True,
    }
    return {
        **unsigned,
        "receipt_payload_sha256": _canonical_json_sha256(unsigned),
    }


def validate_target_2d_evaluation_metadata(
    evaluation_manifest: Path,
    *,
    project_root: Path,
    manifest_spec: FrozenEvaluationManifestSpec | None = None,
    artifact_root: Path | None = None,
    verify_array_checksums: bool = True,
) -> FrozenTarget2DEvaluationContract:
    """Validate frozen evaluation metadata without indexing any label value."""

    root = project_root.resolve()
    manifest_path = evaluation_manifest.resolve()
    spec = _spec_for_path(manifest_path, root, manifest_spec)
    actual_manifest_sha = _sha256_file(manifest_path)
    if actual_manifest_sha != spec.manifest_sha256:
        raise ValueError("Evaluation manifest SHA256 differs from pre-registration.")
    manifest = _read_json(manifest_path, label="evaluation manifest")
    if manifest.get("status") is not None:
        raise PermissionError(
            "Deferred evaluation manifests cannot construct a dataset."
        )
    if manifest.get("schema_version") != 2 or manifest.get("phase") != "P4":
        raise ValueError("Evaluation manifest must be a frozen P4 schema v2 file.")
    if manifest.get("role") != spec.role:
        raise ValueError("Evaluation manifest role differs from pre-registration.")
    if spec.role == "target_id":
        split_payload = manifest.get("splits")
        if not isinstance(split_payload, dict):
            raise ValueError("ID evaluation manifest has no split mapping.")
        id_manifest_train = _integer_ids(
            split_payload.get("train_pool"),
            label="evaluation manifest train pool IDs",
        )
        id_manifest_validation = _integer_ids(
            split_payload.get("validation"),
            label="evaluation manifest validation IDs",
        )
        case_ids = _integer_ids(
            split_payload.get("test"),
            label="evaluation manifest ID-test IDs",
        )
        manifest_hash_case_ids = set(
            id_manifest_train + id_manifest_validation + case_ids
        )
    else:
        id_manifest_train = ()
        id_manifest_validation = ()
        case_ids = _integer_ids(
            manifest.get("case_ids"),
            label="evaluation manifest case IDs",
        )
        manifest_hash_case_ids = set(case_ids)
    if not case_ids:
        raise PermissionError("Empty evaluation manifests are not evaluable.")
    if case_ids != spec.case_ids:
        raise PermissionError(
            "Evaluation case IDs differ from the exact pre-registered order."
        )

    expected_source_path = (root / spec.source_manifest_relative_path).resolve()
    source_path = _resolve_repo_path(
        root,
        manifest.get("source_manifest"),
        label="source_manifest",
    )
    if source_path != expected_source_path:
        raise ValueError("Evaluation manifest points to an unexpected source.")
    if manifest.get("source_manifest_sha256") != spec.source_manifest_sha256:
        raise ValueError("Evaluation manifest source SHA256 binding drifted.")
    if _sha256_file(source_path) != spec.source_manifest_sha256:
        raise ValueError("Frozen P4 source manifest SHA256 drifted.")
    source = _read_json(source_path, label="P4 source manifest")
    if source.get("schema_version") != 2 or source.get("phase") != "P4":
        raise ValueError("P4 source manifest must be frozen schema v2 metadata.")
    if manifest.get("dataset_id") != source.get("dataset_id"):
        raise ValueError("Evaluation and source dataset IDs differ.")
    if manifest.get("dataset_array_sha256") != source.get("array_sha256"):
        raise ValueError("Evaluation manifest array binding drifted.")
    if manifest.get("dataset_metadata_sha256") != source.get("metadata_sha256"):
        raise ValueError("Evaluation manifest metadata binding drifted.")
    if manifest.get("dataset_plan_sha256") != source.get("plan_sha256"):
        raise ValueError("Evaluation manifest plan binding drifted.")

    source_splits = source.get("splits")
    if not isinstance(source_splits, dict):
        raise ValueError("P4 source manifest has no split mapping.")
    train_ids = set(_integer_ids(source_splits.get("train"), label="source train"))
    validation_ids = set(
        _integer_ids(source_splits.get("validation"), label="source validation")
    )
    if spec.role == "target_id" and (
        id_manifest_train
        != _integer_ids(source_splits.get("train"), label="source train")
        or id_manifest_validation
        != _integer_ids(
            source_splits.get("validation"), label="source validation"
        )
    ):
        raise ValueError(
            "ID evaluation manifest train/validation metadata drifted."
        )
    if set(case_ids) & (train_ids | validation_ids):
        raise PermissionError(
            "Evaluation manifest attempts to expose train or validation IDs."
        )
    source_role_ids = _integer_ids(
        source_splits.get(spec.source_split_key),
        label=f"source {spec.source_split_key}",
    )
    if case_ids != source_role_ids:
        raise PermissionError(
            "Evaluation IDs differ from the canonical source split."
        )
    all_source_ids: list[int] = []
    for split_name, values in source_splits.items():
        all_source_ids.extend(
            _integer_ids(values, label=f"source split {split_name}")
        )
    if len(all_source_ids) != len(set(all_source_ids)):
        raise ValueError("P4 source split mappings overlap.")

    plan_path = _resolve_repo_path(
        root, source.get("plan_path"), label="source plan_path"
    )
    if _sha256_file(plan_path) != spec.plan_file_sha256:
        raise ValueError("Frozen pre-label plan file SHA256 drifted.")
    plan = _read_json(plan_path, label="P4 pre-label plan")
    semantic_sha = plan.get("plan_sha256")
    if semantic_sha != source.get("plan_sha256"):
        raise ValueError("Pre-label plan semantic SHA differs from source.")
    plan_without_sha = {
        key: value for key, value in plan.items() if key != "plan_sha256"
    }
    if _canonical_json_sha256(plan_without_sha) != semantic_sha:
        raise ValueError("Pre-label plan semantic SHA256 is invalid.")
    if (
        plan.get("plan_role") != "pre_label_case_plan"
        or plan.get("dataset_id") != source.get("dataset_id")
    ):
        raise ValueError("Source plan is not the frozen pre-label case plan.")
    plan_splits = plan.get("splits")
    if (
        not isinstance(plan_splits, dict)
        or _integer_ids(
            plan_splits.get(spec.source_split_key),
            label=f"plan {spec.source_split_key}",
        )
        != case_ids
    ):
        raise ValueError("Evaluation IDs differ from the pre-label plan.")

    cases = plan.get("cases")
    case_count = source.get("case_count")
    if (
        isinstance(case_count, bool)
        or not isinstance(case_count, int)
        or case_count < 1
        or not isinstance(cases, list)
        or len(cases) != case_count
    ):
        raise ValueError("P4 plan and source case counts differ.")
    definitions: dict[int, dict[str, Any]] = {}
    source_hashes = source.get("case_definition_hashes")
    if not isinstance(source_hashes, dict):
        raise ValueError("Source case-definition hashes are missing.")
    expected_evaluation_hashes: dict[str, str] = {}
    for position, entry in enumerate(cases):
        if not isinstance(entry, dict) or not isinstance(
            entry.get("definition"), dict
        ):
            raise ValueError("Every plan case needs a definition.")
        definition = entry["definition"]
        case_id = definition.get("case_id")
        case_key = definition.get("case_key")
        if (
            isinstance(case_id, bool)
            or not isinstance(case_id, int)
            or case_id != position
            or not isinstance(case_key, str)
            or not case_key
        ):
            raise ValueError("Plan case IDs and keys must be canonical.")
        definition_sha = _canonical_json_sha256(definition)
        if source_hashes.get(case_key) != definition_sha:
            raise ValueError("Source case-definition hash drifted.")
        definitions[case_id] = definition
        if case_id in manifest_hash_case_ids:
            expected_evaluation_hashes[case_key] = definition_sha
    if manifest.get("case_definition_hashes") != expected_evaluation_hashes:
        raise ValueError("Evaluation case-definition hash mapping drifted.")

    generation = source.get("generation_status")
    if (
        not isinstance(generation, dict)
        or generation.get("failed_case_ids") != []
        or generation.get("silent_failure_case_ids") != []
        or set(generation.get("passed_case_ids", ())) != set(definitions)
    ):
        raise ValueError("P4 generation status does not pass every case.")

    case_artifacts = source.get("case_artifact_hashes")
    all_case_keys = {
        definition["case_key"] for definition in definitions.values()
    }
    if (
        not isinstance(case_artifacts, dict)
        or set(case_artifacts) != all_case_keys
    ):
        raise ValueError(
            "Source case-artifact hashes are not the exact full case registry."
        )
    case_output_hash_snapshot: list[dict[str, Any]] = []
    for case_id in range(case_count):
        definition = definitions[case_id]
        case_key = definition["case_key"]
        artifact = case_artifacts[case_key]
        if not isinstance(artifact, dict) or set(artifact) != {
            "input_slice_sha256",
            "output_slice_sha256",
        }:
            raise ValueError(
                f"Source case-artifact record differs for case {case_id}."
            )
        inputs = artifact["input_slice_sha256"]
        outputs = artifact["output_slice_sha256"]
        if (
            not isinstance(inputs, dict)
            or not inputs
            or any(
                _SHA256_RE.fullmatch(value) is None
                for value in inputs.values()
                if isinstance(value, str)
            )
            or any(type(value) is not str for value in inputs.values())
            or not isinstance(outputs, dict)
            or set(outputs) != set(LABEL_ARRAY_NAMES)
        ):
            raise ValueError(
                f"Source case slice hashes differ for case {case_id}."
            )
        normalized_outputs = {
            name: _strict_sha256(
                outputs[name],
                label=f"case {case_id} {name} output-slice SHA-256",
            )
            for name in LABEL_ARRAY_NAMES
        }
        case_output_hash_snapshot.append(
            {
                "global_case_id": case_id,
                "case_key": case_key,
                "output_slice_sha256": normalized_outputs,
            }
        )

    shape = source.get("array_shape_case_time_z_x")
    if (
        not isinstance(shape, list)
        or len(shape) != 4
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in shape
        )
        or shape[0] != case_count
    ):
        raise ValueError("P4 source array shape is invalid.")
    resolved_artifact_root = (
        artifact_root.resolve()
        if artifact_root is not None
        else _resolve_repo_path(
            root,
            source.get("array_artifact_root"),
            label="array_artifact_root",
        )
    )
    required_shapes = {
        "air_temperature_K": (shape[0], shape[1]),
        "alpha": tuple(shape),
        "composite_mask": (shape[2], shape[3]),
        "temperature_K": tuple(shape),
        "time_s": (shape[1],),
        "top_h_W_m2_K": (shape[0], shape[3]),
        "x_m": (shape[3],),
        "z_m": (shape[2],),
    }
    required_dtypes = {
        "air_temperature_K": np.dtype("float32"),
        "alpha": np.dtype("float32"),
        "composite_mask": np.dtype("bool"),
        "temperature_K": np.dtype("float32"),
        "time_s": np.dtype("float64"),
        "top_h_W_m2_K": np.dtype("float32"),
        "x_m": np.dtype("float64"),
        "z_m": np.dtype("float64"),
    }
    array_hashes = source.get("array_sha256")
    if not isinstance(array_hashes, dict) or set(array_hashes) != set(
        required_shapes
    ):
        raise ValueError("P4 source array checksums are incomplete.")
    array_paths: dict[str, Path] = {}
    for name, expected_shape in required_shapes.items():
        path = resolved_artifact_root / f"{name}.npy"
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Frozen array {name} is missing or unsafe.")
        if name not in LABEL_ARRAY_NAMES:
            try:
                array = np.load(path, mmap_mode="r", allow_pickle=False)
            except (OSError, ValueError) as error:
                raise ValueError(
                    f"Cannot inspect frozen array {path}: {error}"
                ) from error
            if (
                array.shape != expected_shape
                or array.dtype != required_dtypes[name]
            ):
                raise ValueError(f"Frozen array {name} header drifted.")
            del array
            if (
                verify_array_checksums
                and _sha256_file(path) != array_hashes[name]
            ):
                raise ValueError(f"Frozen array {name} SHA256 drifted.")
        else:
            # Critical firewall: even header inspection through np.load(...,
            # mmap_mode="r") opens/maps the monolith.  Shape, dtype, and full
            # expected hash remain metadata-only until authorized sharding.
            _strict_sha256(
                array_hashes[name],
                label=f"source {name} expected full SHA-256",
            )
        array_paths[name] = path

    metadata_path = _resolve_repo_path(
        root, source.get("metadata_path"), label="metadata_path"
    )
    if _sha256_file(metadata_path) != source.get("metadata_sha256"):
        raise ValueError("Frozen P4 metadata SHA256 drifted.")

    resolved = plan.get("resolved_config")
    geometry_payload = (
        resolved.get("geometry") if isinstance(resolved, dict) else None
    )
    if not isinstance(geometry_payload, dict):
        raise ValueError("P4 plan has no resolved geometry.")
    geometry = {
        key: float(geometry_payload[key])
        for key in (
            "width_m",
            "tool_thickness_m",
            "composite_thickness_m",
        )
    }
    if any(
        not np.isfinite(value) or value <= 0.0 for value in geometry.values()
    ):
        raise ValueError("P4 evaluation geometry is invalid.")

    return FrozenTarget2DEvaluationContract(
        project_root=root,
        artifact_root=resolved_artifact_root,
        manifest_path=manifest_path,
        source_manifest_path=source_path,
        plan_path=plan_path,
        metadata_path=metadata_path,
        spec=spec,
        manifest=manifest,
        source_manifest=source,
        plan=plan,
        definitions={
            case_id: definitions[case_id] for case_id in case_ids
        },
        geometry=geometry,
        array_paths=array_paths,
        checksums={
            "evaluation_manifest_sha256": actual_manifest_sha,
            "source_manifest_sha256": spec.source_manifest_sha256,
            "pre_label_plan_file_sha256": spec.plan_file_sha256,
            "pre_label_plan_semantic_sha256": semantic_sha,
            "metadata_sha256": source["metadata_sha256"],
            "dataset_array_sha256": dict(array_hashes),
            "exogenous_array_files_verified": bool(verify_array_checksums),
            "monolithic_label_containers_opened_during_metadata_validation": (
                False
            ),
            "monolithic_label_containers_hashed_during_metadata_validation": (
                False
            ),
            "monolithic_label_containers_mapped_during_metadata_validation": (
                False
            ),
            "source_case_output_slice_hash_registry_sha256": (
                _canonical_json_sha256(case_output_hash_snapshot)
            ),
            "label_values_indexed_during_metadata_validation": False,
        },
    )


def _validate_label_shard_receipt(
    contract: FrozenTarget2DEvaluationContract,
    *,
    stage: str,
    split: str,
    paths: Mapping[str, Path],
    authorization: Mapping[str, Any],
    access_session: Mapping[str, Any],
    expected_hashes: Mapping[int, Mapping[str, str]],
    resume_attempt: Mapping[str, Any] | None,
    verify_shard_payloads: bool,
) -> FrozenTarget2DLabelShard:
    receipt = _read_canonical_json(
        paths["receipt"],
        label=f"{stage}/{split} label-shard access receipt",
    )
    unsigned = {
        key: value
        for key, value in receipt.items()
        if key != "receipt_payload_sha256"
    }
    row_shape = tuple(
        int(value)
        for value in contract.source_manifest[
            "array_shape_case_time_z_x"
        ][1:]
    )
    if (
        receipt.get("schema_version") != LABEL_SHARD_SCHEMA_VERSION
        or receipt.get("phase") != "P6V2"
        or receipt.get("artifact_role") != LABEL_SHARD_ARTIFACT_ROLE
        or receipt.get("status") != "complete_write_once_manifest_last"
        or receipt.get("stage") != stage
        or receipt.get("split") != split
        or receipt.get("role") != contract.spec.role
        or receipt.get("ordered_global_case_ids")
        != list(contract.spec.case_ids)
        or receipt.get("row_shape") != list(row_shape)
        or receipt.get("authorization") != dict(authorization)
        or receipt.get("receipt_payload_sha256")
        != _canonical_json_sha256(unsigned)
        or receipt.get("whole_monolithic_label_hashing_performed")
        is not False
        or receipt.get("monolithic_label_mmap_performed") is not False
        or receipt.get("unauthorized_case_bytes_read") is not False
    ):
        raise PermissionError("Label-shard access receipt identity differs.")
    bound_resume_attempt = receipt.get("resume_attempt")
    if (
        bound_resume_attempt is not None
        and not isinstance(bound_resume_attempt, Mapping)
    ) or (
        resume_attempt is not None
        and dict(resume_attempt) != bound_resume_attempt
    ):
        raise PermissionError("Label-shard source recovery binding differs.")
    initial_slice_journal = receipt.get(
        "shard_slice_access_journal_initial"
    )
    if (
        not isinstance(initial_slice_journal, Mapping)
        or set(initial_slice_journal) != {"path", "sha256", "bytes"}
        or initial_slice_journal.get("path")
        != paths["slice_journal"]
        .resolve(strict=False)
        .relative_to(contract.project_root.resolve(strict=True))
        .as_posix()
        or initial_slice_journal.get("sha256")
        != hashlib.sha256(b"").hexdigest()
        or initial_slice_journal.get("bytes") != 0
    ):
        raise PermissionError(
            "Label-shard receipt does not bind a prior empty slice journal."
        )
    pending_path = _verify_small_artifact_reference(
        receipt.get("pending"),
        project_root=contract.project_root,
        label=f"{stage}/{split} label-shard pending",
        expected_path=paths["pending"],
    )
    pending = _read_canonical_json(
        pending_path,
        label=f"{stage}/{split} label-shard pending",
    )
    if (
        receipt.get("pending_payload_sha256")
        != pending.get("pending_payload_sha256")
        or pending.get("authorization") != dict(authorization)
    ):
        raise PermissionError("Label-shard pending/receipt binding differs.")
    _verify_small_artifact_reference(
        receipt.get("source_manifest"),
        project_root=contract.project_root,
        label="label-shard source manifest",
        expected_path=contract.source_manifest_path,
    )
    source_journal_path = _verify_small_artifact_reference(
        receipt.get("source_access_journal"),
        project_root=contract.project_root,
        label="label-shard source-access journal",
        expected_path=paths["source_journal"],
    )
    source_journal = _DurableJournal(
        source_journal_path,
        journal_id=f"target2d-source:{stage}:{split}",
    )
    source_containers = receipt.get("source_containers")
    if not isinstance(source_containers, Mapping) or set(
        source_containers
    ) != set(LABEL_ARRAY_NAMES):
        raise ValueError("Label-shard source-container evidence differs.")
    source_headers = {
        name: dict(source_containers[name]["header"])
        for name in LABEL_ARRAY_NAMES
    }
    if receipt.get("source_access_summary") != _source_journal_summary(
        source_journal,
        contract=contract,
        source_headers=source_headers,
    ):
        raise PermissionError("Label-shard source journal summary differs.")
    cases = receipt.get("cases")
    if not isinstance(cases, list) or len(cases) != len(
        contract.spec.case_ids
    ):
        raise ValueError("Label-shard case evidence differs.")
    for local_row, (case_id, case) in enumerate(
        zip(contract.spec.case_ids, cases, strict=True)
    ):
        if (
            not isinstance(case, Mapping)
            or case.get("local_shard_row") != local_row
            or case.get("global_case_id") != case_id
            or case.get("case_key")
            != contract.definitions[case_id]["case_key"]
            or case.get("expected_output_slice_sha256")
            != dict(expected_hashes[case_id])
            or case.get("observed_output_slice_sha256")
            != dict(expected_hashes[case_id])
        ):
            raise PermissionError(
                f"Label-shard case evidence differs for case {case_id}."
            )
    shard_records = receipt.get("label_shards")
    shard_shape = (len(contract.spec.case_ids), *row_shape)
    if not isinstance(shard_records, Mapping) or set(shard_records) != set(
        LABEL_ARRAY_NAMES
    ):
        raise ValueError("Label-shard artifact records differ.")
    array_paths: dict[str, Path] = {}
    for name in LABEL_ARRAY_NAMES:
        record = shard_records[name]
        if (
            not isinstance(record, Mapping)
            or set(record)
            != {
                "path",
                "sha256",
                "bytes",
                "shape",
                "dtype",
                "global_case_id_to_local_row",
            }
            or record["shape"] != list(shard_shape)
            or record["dtype"] != np.dtype("<f4").str
            or record["global_case_id_to_local_row"]
            != {
                str(case_id): local_row
                for local_row, case_id in enumerate(contract.spec.case_ids)
            }
            or type(record["bytes"]) is not int
            or record["bytes"] < 1
        ):
            raise ValueError(f"{name} shard declaration differs.")
        _strict_sha256(
            record["sha256"],
            label=f"{name} label-shard SHA-256",
        )
        path = _resolve_repo_path(
            contract.project_root,
            record["path"],
            label=f"{name} label-shard path",
        )
        if path.resolve(strict=False) != (
            paths["final"] / f"{name}.npy"
        ).resolve(strict=False):
            raise PermissionError(f"{name} label-shard path differs.")
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != record["bytes"]
        ):
            raise PermissionError(f"{name} label shard is missing or unsafe.")
        if verify_shard_payloads:
            if (
                _sha256_file(path) != record["sha256"]
            ):
                raise PermissionError(f"{name} label shard is tampered.")
            data_offset, _ = _inspect_shard_npy(
                path,
                expected_shape=shard_shape,
            )
            row_bytes = int(np.prod(row_shape, dtype=np.int64)) * 4
            for local_row, case_id in enumerate(contract.spec.case_ids):
                raw = _read_exact_shard_row(
                    path,
                    data_offset=data_offset,
                    local_row=local_row,
                    row_bytes=row_bytes,
                )
                if _semantic_array_sha256(
                    raw,
                    dtype=np.dtype("<f4"),
                    shape=row_shape,
                ) != expected_hashes[case_id][name]:
                    raise PermissionError(
                        f"{name} label-shard row {local_row} is tampered."
                    )
        array_paths[name] = path
    allowed = {
        "temperature_K.npy",
        "alpha.npy",
        "label_shard_access_receipt.json",
    }
    if (
        paths["final"].is_symlink()
        or not paths["final"].is_dir()
        or {child.name for child in paths["final"].iterdir()} != allowed
        or any(
            child.is_symlink() or not child.is_file()
            for child in paths["final"].iterdir()
        )
    ):
        raise PermissionError("Label-shard final namespace differs.")
    if (
        paths["slice_journal"].is_symlink()
        or not paths["slice_journal"].is_file()
    ):
        raise PermissionError(
            "Committed label-shard receipt lacks its prior durable "
            "slice-access journal."
        )
    _journal_records(
        paths["slice_journal"],
        journal_id=f"target2d-shard-slice:{stage}:{split}",
    )
    return FrozenTarget2DLabelShard(
        project_root=contract.project_root,
        root=paths["final"],
        pending_path=paths["pending"],
        receipt_path=paths["receipt"],
        source_journal_path=paths["source_journal"],
        slice_journal_path=paths["slice_journal"],
        stage=stage,
        split=split,
        role=contract.spec.role,
        case_ids=contract.spec.case_ids,
        row_shape=row_shape,
        array_paths=array_paths,
        expected_slice_sha256={
            case_id: dict(expected_hashes[case_id])
            for case_id in contract.spec.case_ids
        },
        receipt=receipt,
        access_session=dict(access_session),
        resume_attempt=(
            dict(bound_resume_attempt)
            if bound_resume_attempt is not None
            else None
        ),
    )


def materialize_target_2d_label_shard(
    contract: FrozenTarget2DEvaluationContract,
    *,
    stage: str,
    split: str,
    authorization: Mapping[str, Any],
    access_session: Mapping[str, Any],
    shard_root: Path | None = None,
) -> FrozenTarget2DLabelShard:
    """Create/adopt one stage-authorized split shard without scanning labels."""

    _validate_stage_split(stage, split)
    if (
        contract.spec.source_split_key != split
        or contract.spec.case_ids != tuple(contract.definitions)
    ):
        raise PermissionError("Label-shard contract split identity differs.")
    checked_authorization = _validate_label_shard_authorization(
        authorization,
        contract=contract,
        stage=stage,
        split=split,
    )
    checked_access_session = _validate_label_access_session(
        access_session,
        project_root=contract.project_root,
        stage=stage,
        split=split,
        authorization=checked_authorization,
    )
    expected_hashes, expected_hashes_sha = _expected_slice_hashes(contract)
    paths = _label_shard_paths(
        contract.project_root,
        stage=stage,
        split=split,
        shard_root=shard_root,
    )
    root = contract.project_root.resolve(strict=True)
    if any(
        not path.resolve(strict=False).is_relative_to(root)
        for path in paths.values()
    ):
        raise PermissionError("Label-shard output path escapes the project.")
    pending = _label_shard_pending_payload(
        contract,
        stage=stage,
        split=split,
        authorization=checked_authorization,
        expected_slice_hashes_sha256=expected_hashes_sha,
    )
    pending_state = _write_once_bytes(
        paths["pending"],
        _canonical_json_bytes(pending),
    )
    pending_reference = _artifact_reference(
        paths["pending"],
        contract.project_root,
    )
    resume_attempt: dict[str, Any] | None = None
    if (
        pending_state == "verified_existing"
        and not paths["receipt"].is_file()
    ):
        resume_attempt = _resume_attempt_reference(
            paths,
            contract=contract,
            stage=stage,
            split=split,
            pending=pending,
        )
    if paths["receipt"].is_file():
        return _validate_label_shard_receipt(
            contract,
            stage=stage,
            split=split,
            paths=paths,
            authorization=checked_authorization,
            access_session=checked_access_session,
            expected_hashes=expected_hashes,
            resume_attempt=resume_attempt,
            # Existing receipts are adopted without reopening or hashing
            # held-out shard payloads.  Actual evaluation verifies every
            # authorized slice semantically under its durable access session.
            verify_shard_payloads=False,
        )
    if paths["receipt"].exists() or paths["receipt"].is_symlink():
        raise PermissionError("Label-shard receipt path is unsafe.")
    paths["staging"].mkdir(parents=True, exist_ok=True)
    if paths["staging"].is_symlink():
        raise PermissionError("Label-shard staging directory is unsafe.")
    permitted_staging = {
        f"{name}.npy" for name in LABEL_ARRAY_NAMES
    }
    if any(
        child.name not in permitted_staging
        or child.is_symlink()
        or not child.is_file()
        for child in paths["staging"].iterdir()
    ):
        raise PermissionError("Label-shard staging namespace differs.")
    source_journal = _DurableJournal(
        paths["source_journal"],
        journal_id=f"target2d-source:{stage}:{split}",
    )
    session_id = uuid.uuid4().hex
    source_headers: dict[str, dict[str, Any]] = {}
    source_ranges: dict[str, list[dict[str, Any]]] = {}
    for name in LABEL_ARRAY_NAMES:
        header, ranges = _extract_one_label_shard(
            contract,
            label_name=name,
            staging_path=paths["staging"] / f"{name}.npy",
            header_cache_path=paths["control"] / f"{name}_header.bin",
            journal=source_journal,
            expected_hashes=expected_hashes,
            session_id=session_id,
        )
        source_headers[name] = header
        source_ranges[name] = ranges
    # This future evaluation journal is durable before receipt assembly and
    # therefore before the receipt's manifest-last commit.
    _write_once_bytes(paths["slice_journal"], b"")
    receipt = _label_shard_receipt_payload(
        contract,
        stage=stage,
        split=split,
        pending=pending,
        pending_reference=pending_reference,
        authorization=checked_authorization,
        expected_hashes=expected_hashes,
        source_headers=source_headers,
        source_ranges=source_ranges,
        source_journal=source_journal,
        paths=paths,
        resume_attempt=resume_attempt,
    )
    paths["final"].parent.mkdir(parents=True, exist_ok=True)
    try:
        paths["final"].mkdir()
        _fsync_directory(paths["final"].parent)
    except FileExistsError:
        if paths["final"].is_symlink() or not paths["final"].is_dir():
            raise PermissionError("Label-shard final path is unsafe.")
    allowed_partial = {
        "temperature_K.npy",
        "alpha.npy",
    }
    if any(
        child.name not in allowed_partial
        or child.is_symlink()
        or not child.is_file()
        for child in paths["final"].iterdir()
    ):
        raise PermissionError("Partial label-shard namespace differs.")
    for name in LABEL_ARRAY_NAMES:
        _publish_file_no_replace(
            paths["staging"] / f"{name}.npy",
            paths["final"] / f"{name}.npy",
            expected=receipt["label_shards"][name],
        )
    _write_once_bytes(
        paths["receipt"],
        _canonical_json_bytes(receipt),
    )
    shard = _validate_label_shard_receipt(
        contract,
        stage=stage,
        split=split,
        paths=paths,
        authorization=checked_authorization,
        access_session=checked_access_session,
        expected_hashes=expected_hashes,
        resume_attempt=resume_attempt,
        verify_shard_payloads=True,
    )
    for name in LABEL_ARRAY_NAMES:
        staging_file = paths["staging"] / f"{name}.npy"
        if staging_file.exists() and not staging_file.is_symlink():
            staging_file.unlink()
    try:
        paths["staging"].rmdir()
    except OSError:
        pass
    return shard


def validate_target_2d_label_shard_evidence(
    evidence: Any,
    *,
    project_root: Path,
    stage: str,
    split: str,
    expected_case_ids: tuple[int, ...],
    require_complete_slice_reads: bool,
    allow_prior_complete_session_adoption: bool = False,
) -> dict[str, Any]:
    """Validate only small receipts/journals; never open/hash shard payloads."""

    _validate_stage_split(stage, split)
    expected_keys = {
        "schema_version",
        "stage",
        "split",
        "role",
        "ordered_global_case_ids",
        "pending",
        "access_receipt",
        "access_receipt_payload_sha256",
        "source_access_journal",
        "shard_slice_access_journal",
        "label_access_session",
        "resume_attempt",
        "label_shards",
    }
    if not isinstance(evidence, Mapping) or set(evidence) != expected_keys:
        raise PermissionError("Label-shard evidence keys differ.")
    if (
        evidence["schema_version"] != LABEL_SHARD_SCHEMA_VERSION
        or evidence["stage"] != stage
        or evidence["split"] != split
        or evidence["ordered_global_case_ids"] != list(expected_case_ids)
    ):
        raise PermissionError("Label-shard evidence identity differs.")
    pending_path = _verify_small_artifact_reference(
        evidence["pending"],
        project_root=project_root,
        label="label-shard pending",
    )
    receipt_path = _verify_small_artifact_reference(
        evidence["access_receipt"],
        project_root=project_root,
        label="label-shard access receipt",
    )
    source_journal_path = _verify_small_artifact_reference(
        evidence["source_access_journal"],
        project_root=project_root,
        label="source-range journal",
    )
    slice_journal_path = _verify_small_artifact_reference(
        evidence["shard_slice_access_journal"],
        project_root=project_root,
        label="shard-slice journal",
    )
    pending = _read_canonical_json(
        pending_path,
        label="label-shard pending",
    )
    receipt = _read_canonical_json(
        receipt_path,
        label="label-shard access receipt",
    )
    unsigned = {
        key: value
        for key, value in receipt.items()
        if key != "receipt_payload_sha256"
    }
    if (
        receipt.get("artifact_role") != LABEL_SHARD_ARTIFACT_ROLE
        or receipt.get("stage") != stage
        or receipt.get("split") != split
        or receipt.get("role") != evidence["role"]
        or receipt.get("ordered_global_case_ids")
        != list(expected_case_ids)
        or receipt.get("pending") != dict(evidence["pending"])
        or receipt.get("pending_payload_sha256")
        != pending.get("pending_payload_sha256")
        or receipt.get("receipt_payload_sha256")
        != evidence["access_receipt_payload_sha256"]
        or receipt.get("receipt_payload_sha256")
        != _canonical_json_sha256(unsigned)
        or receipt.get("source_access_journal")
        != dict(evidence["source_access_journal"])
        or receipt.get("label_shards") != evidence["label_shards"]
        or receipt.get("whole_monolithic_label_hashing_performed")
        is not False
        or receipt.get("monolithic_label_mmap_performed") is not False
        or receipt.get("unauthorized_case_bytes_read") is not False
    ):
        raise PermissionError("Label-shard receipt/evidence binding differs.")
    authorization = receipt.get("authorization")
    if not isinstance(authorization, Mapping):
        raise PermissionError("Label-shard receipt authorization is absent.")
    current_access_session = _validate_label_access_session(
        evidence["label_access_session"],
        project_root=project_root,
        stage=stage,
        split=split,
        authorization=authorization,
    )
    current_session_sha256 = current_access_session[
        "session_authority_sha256"
    ]
    _journal_records(
        source_journal_path,
        journal_id=f"target2d-source:{stage}:{split}",
    )
    slice_records = _journal_records(
        slice_journal_path,
        journal_id=f"target2d-shard-slice:{stage}:{split}",
    )
    slice_journal = _DurableJournal(
        slice_journal_path,
        journal_id=f"target2d-shard-slice:{stage}:{split}",
    )
    resume_pattern = re.compile(r"^resume_attempt_([0-9]{4})\.json$")
    resume_paths = sorted(pending_path.parent.glob("resume_attempt_*.json"))
    resume_numbers: list[int] = []
    resume_references: list[dict[str, Any]] = []
    for resume_path in resume_paths:
        match = resume_pattern.fullmatch(resume_path.name)
        if (
            match is None
            or resume_path.is_symlink()
            or not resume_path.is_file()
        ):
            raise PermissionError("Malformed label-shard resume artifact.")
        resume_numbers.append(int(match.group(1)))
        resume = _read_canonical_json(
            resume_path,
            label="label-shard resume attempt",
        )
        resume_unsigned = {
            key: value
            for key, value in resume.items()
            if key != "resume_attempt_payload_sha256"
        }
        if (
            resume.get("artifact_role")
            != "target_2d_label_shard_resume_attempt"
            or resume.get("stage") != stage
            or resume.get("split") != split
            or resume.get("attempt_number") != int(match.group(1))
            or resume.get("pending_payload_sha256")
            != pending.get("pending_payload_sha256")
            or resume.get("source_container_range_replay_allowed")
            is not False
            or resume.get("completed_source_ranges_may_be_adopted")
            is not True
            or resume.get(
                "incomplete_attempt_requires_exact_durable_shard_row"
            )
            is not True
            or resume.get("isolated_shard_slice_read_scope")
            != "exactly_once_per_authorized_evaluation_invocation"
            or resume.get(
                "prior_invocation_journal_entries_remain_immutable"
            )
            is not True
            or resume.get("resume_attempt_payload_sha256")
            != _canonical_json_sha256(resume_unsigned)
        ):
            raise PermissionError("Label-shard resume attempt differs.")
        reference = _artifact_reference(resume_path, project_root)
        resume_references.append(reference)
    if resume_numbers != list(range(1, len(resume_numbers) + 1)):
        raise PermissionError(
            "Label-shard resume-attempt numbering is not contiguous."
        )
    resume_reference = evidence["resume_attempt"]
    if resume_reference is None:
        if resume_references:
            raise PermissionError(
                "Current label-shard evidence omits its source recovery "
                "attempt."
            )
    else:
        resume_path = _verify_small_artifact_reference(
            resume_reference,
            project_root=project_root,
            label="label-shard resume attempt",
        )
        if (
            not resume_references
            or resume_references[-1] != dict(resume_reference)
            or resume_path.name
            != f"resume_attempt_{len(resume_paths):04d}.json"
        ):
            raise PermissionError(
                "Current label-shard resume attempt is not the latest "
                "authorized invocation."
            )
    registrations = _shard_slice_access_session_registrations(
        slice_journal,
        project_root=project_root,
        stage=stage,
        split=split,
        authorization=authorization,
    )
    registered_current = registrations.get(current_session_sha256)
    if (
        registered_current is not None
        and registered_current != current_access_session
    ):
        raise PermissionError(
            "Current label-access session differs from its registration."
        )
    valid_sessions = dict(registrations)
    valid_sessions.setdefault(
        current_session_sha256,
        current_access_session,
    )
    state = _shard_slice_journal_state(
        slice_journal,
        registrations=registrations,
    )
    expected_pairs = {
        (label_name, case_id, local_row)
        for local_row, case_id in enumerate(expected_case_ids)
        for label_name in LABEL_ARRAY_NAMES
    }
    state_by_session: dict[
        str,
        dict[str, dict[str, dict[str, Any]]],
    ] = {session: {} for session in valid_sessions}
    for token, entry in state.items():
        attempt = entry["attempt"]
        session_sha256 = attempt["access_session_sha256"]
        expected_session = valid_sessions.get(session_sha256)
        observed_pair = (
            attempt.get("label_name"),
            attempt.get("global_case_id"),
            attempt.get("local_shard_row"),
        )
        if (
            expected_session is None
            or attempt.get("access_session_kind")
            != expected_session["invocation_kind"]
            or observed_pair not in expected_pairs
        ):
            raise PermissionError(
                "Shard-slice journal contains an unauthorized session or ID."
            )
        state_by_session[session_sha256][token] = entry
    expected_tokens_by_session = {
        session_sha256: {
            f"{session_sha256}:{label_name}:"
            f"case:{case_id}:local:{local_row}"
            for label_name, case_id, local_row in expected_pairs
        }
        for session_sha256 in valid_sessions
    }
    for session_sha256, session_state in state_by_session.items():
        if not set(session_state).issubset(
            expected_tokens_by_session[session_sha256]
        ):
            raise PermissionError(
                "Shard-slice journal contains unauthorized IDs."
            )
    current_state = state_by_session[current_session_sha256]
    current_expected = expected_tokens_by_session[
        current_session_sha256
    ]
    current_unresolved = {
        token
        for token, item in current_state.items()
        if "completion" not in item
    }
    complete_sessions = [
        session_sha256
        for session_sha256, session_state in state_by_session.items()
        if (
            set(session_state)
            == expected_tokens_by_session[session_sha256]
            and all("completion" in item for item in session_state.values())
        )
    ]
    selected_session_sha256 = current_session_sha256
    satisfaction = "current_authorized_invocation"
    if (
        require_complete_slice_reads
        and (
            set(current_state) != current_expected
            or current_unresolved
        )
        and allow_prior_complete_session_adoption
        and not current_state
    ):
        prior_complete = [
            session
            for session in complete_sessions
            if session != current_session_sha256
        ]
        if prior_complete:
            selected_session_sha256 = prior_complete[-1]
            satisfaction = "prior_complete_invocation_adopted"
    selected_state = state_by_session[selected_session_sha256]
    selected_expected = expected_tokens_by_session[
        selected_session_sha256
    ]
    selected_unresolved = {
        token
        for token, item in selected_state.items()
        if "completion" not in item
    }
    replay_rejections = [
        item
        for item in slice_records
        if item["event"] == "shard_label_slice_replay_rejected"
    ]
    if require_complete_slice_reads and (
        set(selected_state) != selected_expected
        or selected_unresolved
        or replay_rejections
    ):
        raise PermissionError(
            "Shard-slice journal is incomplete, replayed, or outside scope."
        )
    cumulative_unresolved = sum(
        "completion" not in item
        for session_state in state_by_session.values()
        for item in session_state.values()
    )
    return {
        "stage": stage,
        "split": split,
        "ordered_global_case_ids": list(expected_case_ids),
        "source_access_receipt_payload_sha256": receipt[
            "receipt_payload_sha256"
        ],
        "source_container_opened_for_bounded_extraction": True,
        "whole_source_container_hashed": False,
        "source_container_mmap_used": False,
        "unauthorized_case_bytes_read": False,
        "access_session_sha256": current_session_sha256,
        "access_session_satisfaction": satisfaction,
        "selected_complete_access_session_sha256": (
            selected_session_sha256
        ),
        "shard_slice_attempt_count": len(selected_state),
        "shard_slice_completion_count": (
            len(selected_state) - len(selected_unresolved)
        ),
        "shard_slice_replay_rejection_count": len(replay_rejections),
        "cumulative_access_session_count": len(valid_sessions),
        "journaled_access_session_count": len(registrations),
        "complete_access_session_count": len(complete_sessions),
        "cumulative_shard_slice_attempt_count": len(state),
        "cumulative_shard_slice_completion_count": (
            len(state) - cumulative_unresolved
        ),
        "cumulative_shard_slice_unresolved_attempt_count": (
            cumulative_unresolved
        ),
        "shard_label_payloads_reopened_or_hashed_by_validation": False,
    }


def _shard_slice_access_session_registrations(
    journal: _DurableJournal,
    *,
    project_root: Path,
    stage: str,
    split: str,
    authorization: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    registrations: dict[str, dict[str, Any]] = {}
    base_keys = {
        "schema_version",
        "journal_id",
        "sequence",
        "previous_event_sha256",
        "event",
        "event_sha256",
    }
    for record in journal.records:
        if record["event"] != "shard_label_access_session_registered":
            continue
        if set(record) != base_keys | {"label_access_session"}:
            raise PermissionError(
                "Shard label-access registration fields differ."
            )
        session = _validate_label_access_session(
            record["label_access_session"],
            project_root=project_root,
            stage=stage,
            split=split,
            authorization=authorization,
        )
        authority = session["session_authority_sha256"]
        if authority in registrations:
            raise PermissionError(
                "A label-access session was registered more than once."
            )
        registrations[authority] = session
    return registrations


def _shard_slice_journal_state(
    journal: _DurableJournal,
    *,
    registrations: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    state: dict[str, dict[str, dict[str, Any]]] = {}
    base_keys = {
        "schema_version",
        "journal_id",
        "sequence",
        "previous_event_sha256",
        "event",
        "event_sha256",
    }
    attempt_keys = {
        "read_token",
        "label_name",
        "global_case_id",
        "local_shard_row",
        "access_session_sha256",
        "access_session_kind",
        "session_payload_sha256",
        "shard_byte_range",
        "byte_range_convention",
        "expected_semantic_sha256",
    }
    completion_keys = {
        "read_token",
        "label_name",
        "global_case_id",
        "local_shard_row",
        "access_session_sha256",
        "access_session_kind",
        "session_payload_sha256",
        "shard_byte_range",
        "byte_range_convention",
        "observed_semantic_sha256",
        "observed_bytes",
        "payload_read_once",
    }
    rejection_keys = {
        "read_token",
        "label_name",
        "global_case_id",
        "local_shard_row",
        "access_session_sha256",
        "access_session_kind",
        "session_payload_sha256",
        "payload_read",
    }
    for record in journal.records:
        event = record["event"]
        if event == "shard_label_access_session_registered":
            continue
        if event not in {
            "shard_label_slice_read_attempt",
            "shard_label_slice_read_completion",
            "shard_label_slice_replay_rejected",
        }:
            raise PermissionError("Unknown shard-slice journal event.")
        expected_keys = (
            attempt_keys
            if event == "shard_label_slice_read_attempt"
            else (
                completion_keys
                if event == "shard_label_slice_read_completion"
                else rejection_keys
            )
        )
        if set(record) != base_keys | expected_keys:
            raise PermissionError("Shard-slice journal fields differ.")
        token = record.get("read_token")
        if type(token) is not str or not token:
            raise ValueError("Shard-slice journal token differs.")
        session_sha256 = _strict_sha256(
            record.get("access_session_sha256"),
            label="shard-slice access-session SHA-256",
        )
        session = registrations.get(session_sha256)
        if (
            session is None
            or record.get("access_session_kind")
            != session.get("invocation_kind")
            or record.get("session_payload_sha256")
            != session.get("session_payload_sha256")
        ):
            raise ValueError("Shard-slice access-session kind differs.")
        expected_token = (
            f"{session_sha256}:{record.get('label_name')}:"
            f"case:{record.get('global_case_id')}:"
            f"local:{record.get('local_shard_row')}"
        )
        if token != expected_token:
            raise ValueError("Shard-slice session token differs.")
        if event == "shard_label_slice_replay_rejected":
            if record.get("payload_read") is not False:
                raise PermissionError(
                    "Shard-slice replay rejection payload status differs."
                )
            continue
        entry = state.setdefault(token, {})
        key = (
            "attempt"
            if event == "shard_label_slice_read_attempt"
            else "completion"
        )
        if key in entry:
            raise PermissionError(
                f"Shard label slice was replayed/completed twice: {token}."
            )
        entry[key] = record
    for token, entry in state.items():
        if "completion" in entry and "attempt" not in entry:
            raise ValueError(
                f"Shard-slice completion lacks attempt: {token}."
            )
        if "completion" in entry:
            attempt = entry["attempt"]
            completion = entry["completion"]
            for key in (
                "access_session_sha256",
                "access_session_kind",
                "session_payload_sha256",
                "label_name",
                "global_case_id",
                "local_shard_row",
                "shard_byte_range",
                "byte_range_convention",
            ):
                if completion.get(key) != attempt.get(key):
                    raise ValueError(
                        f"Shard-slice attempt/completion differs: {token}."
                    )
    return state


class Target2DEvaluationDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
):
    """Read-only, audited dataset for one exact frozen evaluation regime."""

    required_num_workers = 0
    temperature_target_units = "K"
    alpha_target_units = "dimensionless"

    def __init__(
        self,
        *,
        contract: FrozenTarget2DEvaluationContract,
        normalization: FrozenSourceNormalization,
        baseline_builder: BaselineBuilder,
        label_shard: FrozenTarget2DLabelShard | None = None,
    ) -> None:
        self.role = contract.spec.role
        self.case_ids = contract.spec.case_ids
        self.channel_names = TARGET_INPUT_CHANNELS
        self.normalization = normalization.metadata
        self._array_paths = dict(contract.array_paths)
        self._definitions = dict(contract.definitions)
        self._geometry = dict(contract.geometry)
        self._artifact_root = contract.artifact_root
        self._source_normalization = normalization
        self._baseline_builder = baseline_builder
        self._label_shard = label_shard
        self._input_arrays: dict[str, NDArray[Any]] | None = None
        self._label_arrays: dict[str, NDArray[Any]] | None = None
        self._label_data_offsets: dict[str, int] = {}
        self._baseline_cache: dict[
            int, tuple[NDArray[np.float64], NDArray[np.float64]]
        ] = {}
        self._attempted_case_ids: set[int] = set()
        self._completed_case_ids: set[int] = set()
        self._access_session_sha256: str | None = None
        self._access_session_kind: str | None = None
        self._access_session: dict[str, Any] | None = None
        self._adopted_from_access_session_sha256: str | None = None
        self._prior_complete_access_sessions: tuple[str, ...] = ()
        self._lock = threading.Lock()
        self._slice_journal: _DurableJournal | None = None
        if label_shard is not None:
            if (
                label_shard.project_root != contract.project_root
                or label_shard.role != contract.spec.role
                or label_shard.case_ids != contract.spec.case_ids
                or label_shard.row_shape
                != tuple(
                    contract.source_manifest[
                        "array_shape_case_time_z_x"
                    ][1:]
                )
            ):
                raise PermissionError(
                    "Evaluation label shard differs from the frozen contract."
                )
            self._slice_journal = _DurableJournal(
                label_shard.slice_journal_path,
                journal_id=(
                    f"target2d-shard-slice:{label_shard.stage}:"
                    f"{label_shard.split}"
                ),
            )
            self._access_session_sha256 = (
                label_shard.access_session_sha256
            )
            self._access_session_kind = label_shard.access_session_kind
            self._access_session = dict(label_shard.access_session)
            registrations = _shard_slice_access_session_registrations(
                self._slice_journal,
                project_root=label_shard.project_root,
                stage=label_shard.stage,
                split=label_shard.split,
                authorization=label_shard.receipt["authorization"],
            )
            state = _shard_slice_journal_state(
                self._slice_journal,
                registrations=registrations,
            )
            attempted: dict[int, set[str]] = {}
            completed: dict[int, set[str]] = {}
            complete_by_session: dict[str, set[tuple[str, int]]] = {}
            for entry in state.values():
                attempt = entry["attempt"]
                session_sha256 = _strict_sha256(
                    attempt.get("access_session_sha256"),
                    label="shard-slice access-session SHA-256",
                )
                case_id = int(attempt["global_case_id"])
                label_name = str(attempt["label_name"])
                if session_sha256 == self._access_session_sha256:
                    attempted.setdefault(case_id, set()).add(label_name)
                if "completion" in entry:
                    complete_by_session.setdefault(
                        session_sha256,
                        set(),
                    ).add((label_name, case_id))
                    if session_sha256 == self._access_session_sha256:
                        completed.setdefault(case_id, set()).add(label_name)
            self._attempted_case_ids.update(attempted)
            self._completed_case_ids.update(
                case_id
                for case_id, labels in completed.items()
                if labels == set(LABEL_ARRAY_NAMES)
            )
            expected_pairs = {
                (label_name, case_id)
                for case_id in self.case_ids
                for label_name in LABEL_ARRAY_NAMES
            }
            self._prior_complete_access_sessions = tuple(
                session_sha256
                for session_sha256, pairs in complete_by_session.items()
                if (
                    session_sha256 != self._access_session_sha256
                    and pairs == expected_pairs
                )
            )

    def __len__(self) -> int:
        return len(self.case_ids)

    @property
    def attempted_case_ids(self) -> tuple[int, ...]:
        with self._lock:
            return tuple(sorted(self._attempted_case_ids))

    @property
    def completed_case_ids(self) -> tuple[int, ...]:
        with self._lock:
            return tuple(sorted(self._completed_case_ids))

    def adopt_prior_complete_label_access(self) -> str:
        """Adopt a prior complete session without reopening label payloads."""

        if self._label_shard is None:
            raise PermissionError("Stage-authorized label shard is absent.")
        with self._lock:
            if self._attempted_case_ids or self._completed_case_ids:
                raise PermissionError(
                    "Current label-access session already has activity."
                )
            if not self._prior_complete_access_sessions:
                raise PermissionError(
                    "No prior complete label-access session is available."
                )
            adopted = self._prior_complete_access_sessions[-1]
            self._attempted_case_ids.update(self.case_ids)
            self._completed_case_ids.update(self.case_ids)
            self._adopted_from_access_session_sha256 = adopted
        return adopted

    @property
    def access_audit(self) -> dict[str, Any]:
        registrations = (
            _shard_slice_access_session_registrations(
                self._slice_journal,
                project_root=self._label_shard.project_root,
                stage=self._label_shard.stage,
                split=self._label_shard.split,
                authorization=self._label_shard.receipt["authorization"],
            )
            if self._slice_journal is not None
            and self._label_shard is not None
            else {}
        )
        shard_state = (
            _shard_slice_journal_state(
                self._slice_journal,
                registrations=registrations,
            )
            if self._slice_journal is not None
            else {}
        )
        current_state = {
            token: entry
            for token, entry in shard_state.items()
            if entry["attempt"].get("access_session_sha256")
            == self._access_session_sha256
        }
        unresolved = sum(
            "completion" not in entry for entry in current_state.values()
        )
        cumulative_unresolved = sum(
            "completion" not in entry for entry in shard_state.values()
        )
        replay_records = (
            [
                item
                for item in self._slice_journal.records
                if item["event"] == "shard_label_slice_replay_rejected"
            ]
            if self._slice_journal is not None
            else []
        )
        replay_rejections = sum(
            item.get("access_session_sha256")
            == self._access_session_sha256
            for item in replay_records
        )
        audit = {
            "role": self.role,
            "allowed_case_ids": list(self.case_ids),
            "attempted_case_ids": list(self.attempted_case_ids),
            "completed_case_ids": list(self.completed_case_ids),
            "required_num_workers": self.required_num_workers,
            "monolithic_label_containers_opened_for_bounded_extraction": (
                self._label_shard is not None
            ),
            "whole_monolithic_label_containers_hashed": False,
            "monolithic_label_containers_mmap_used": False,
            "unauthorized_case_bytes_read": False,
            "evaluation_dataset_opened_only_split_label_shards": (
                self._label_shard is not None
            ),
            "shard_label_slice_read_attempt_count": len(current_state),
            "shard_label_slice_read_completion_count": (
                len(current_state) - unresolved
            ),
            "shard_label_slice_unresolved_attempt_count": unresolved,
            "shard_label_slice_replay_rejection_count": replay_rejections,
            "label_access_session_sha256": self._access_session_sha256,
            "label_access_session_kind": self._access_session_kind,
            "label_access_session_satisfaction": (
                "prior_complete_session_adopted_without_payload_reopen"
                if self._adopted_from_access_session_sha256 is not None
                else (
                    "current_session_payload_reads_complete"
                    if self._completed_case_ids == set(self.case_ids)
                    else "current_session_pending"
                )
            ),
            "adopted_from_access_session_sha256": (
                self._adopted_from_access_session_sha256
            ),
            "prior_complete_access_session_count": len(
                self._prior_complete_access_sessions
            ),
            "total_journaled_access_session_count": len(
                registrations
            ),
            "cumulative_shard_label_slice_read_attempt_count": len(
                shard_state
            ),
            "cumulative_shard_label_slice_read_completion_count": (
                len(shard_state) - cumulative_unresolved
            ),
            "cumulative_shard_label_slice_unresolved_attempt_count": (
                cumulative_unresolved
            ),
            "cumulative_shard_label_slice_replay_rejection_count": len(
                replay_records
            ),
        }
        if self._label_shard is not None:
            audit["label_shard_evidence"] = self._label_shard.evidence()
        else:
            audit["label_shard_evidence"] = None
        return audit

    def _ensure_access_session_registered(self) -> None:
        if (
            self._label_shard is None
            or self._slice_journal is None
            or self._access_session is None
            or self._access_session_sha256 is None
        ):
            raise PermissionError("Stage-authorized label session is absent.")
        with self._lock:
            registrations = _shard_slice_access_session_registrations(
                self._slice_journal,
                project_root=self._label_shard.project_root,
                stage=self._label_shard.stage,
                split=self._label_shard.split,
                authorization=self._label_shard.receipt["authorization"],
            )
            registered = registrations.get(self._access_session_sha256)
            if registered is not None:
                if registered != self._access_session:
                    raise PermissionError(
                        "Registered label-access session identity differs."
                    )
                return
            self._slice_journal.append(
                "shard_label_access_session_registered",
                label_access_session=dict(self._access_session),
            )

    def _open_input_arrays(self) -> dict[str, NDArray[Any]]:
        if self._input_arrays is None:
            names = (
                "air_temperature_K",
                "composite_mask",
                "time_s",
                "top_h_W_m2_K",
                "x_m",
                "z_m",
            )
            self._input_arrays = {
                name: np.load(
                    self._array_paths[name],
                    mmap_mode="r",
                    allow_pickle=False,
                )
                for name in names
            }
        return self._input_arrays

    def _open_label_arrays(self) -> dict[str, NDArray[Any]]:
        if self._label_shard is None:
            raise PermissionError(
                "Evaluation labels require a stage-authorized split shard; "
                "the monolithic label containers are never opened here."
            )
        if self._label_arrays is None:
            self._label_arrays = {
                name: np.load(
                    self._label_shard.array_paths[name],
                    mmap_mode="r",
                    allow_pickle=False,
                )
                for name in LABEL_ARRAY_NAMES
            }
            expected_shape = (
                len(self.case_ids),
                *self._label_shard.row_shape,
            )
            if any(
                array.shape != expected_shape
                or array.dtype != np.dtype("float32")
                for array in self._label_arrays.values()
            ):
                raise PermissionError("Opened label-shard header differs.")
            self._label_data_offsets = {
                name: int(array.offset)
                for name, array in self._label_arrays.items()
            }
        return self._label_arrays

    def _read_shard_label_slice(
        self,
        *,
        labels: Mapping[str, NDArray[Any]],
        label_name: str,
        case_id: int,
        local_row: int,
    ) -> NDArray[np.float32]:
        if self._label_shard is None or self._slice_journal is None:
            raise PermissionError("Stage-authorized label shard is absent.")
        if (
            self._access_session_sha256 is None
            or self._access_session is None
        ):
            raise PermissionError("Label-access session identity is absent.")
        token = (
            f"{self._access_session_sha256}:{label_name}:"
            f"case:{case_id}:local:{local_row}"
        )
        registrations = _shard_slice_access_session_registrations(
            self._slice_journal,
            project_root=self._label_shard.project_root,
            stage=self._label_shard.stage,
            split=self._label_shard.split,
            authorization=self._label_shard.receipt["authorization"],
        )
        state = _shard_slice_journal_state(
            self._slice_journal,
            registrations=registrations,
        )
        if token in state:
            self._slice_journal.append(
                "shard_label_slice_replay_rejected",
                read_token=token,
                label_name=label_name,
                global_case_id=case_id,
                local_shard_row=local_row,
                access_session_sha256=self._access_session_sha256,
                access_session_kind=self._access_session_kind,
                session_payload_sha256=self._access_session[
                    "session_payload_sha256"
                ],
                payload_read=False,
            )
            raise PermissionError(
                "Shard label slice replay is forbidden after an earlier "
                f"durable attempt: {token}."
            )
        row_bytes = (
            int(np.prod(self._label_shard.row_shape, dtype=np.int64)) * 4
        )
        offset = self._label_data_offsets[label_name] + local_row * row_bytes
        self._slice_journal.append(
            "shard_label_slice_read_attempt",
            read_token=token,
            label_name=label_name,
            global_case_id=case_id,
            local_shard_row=local_row,
            access_session_sha256=self._access_session_sha256,
            access_session_kind=self._access_session_kind,
            session_payload_sha256=self._access_session[
                "session_payload_sha256"
            ],
            shard_byte_range=[offset, offset + row_bytes],
            byte_range_convention="half_open_[start,end)",
            expected_semantic_sha256=(
                self._label_shard.expected_slice_sha256[case_id][label_name]
            ),
        )
        value = np.array(
            labels[label_name][local_row],
            dtype=np.float32,
            copy=True,
        )
        observed = _semantic_array_sha256(
            memoryview(np.ascontiguousarray(value)).cast("B"),
            dtype=np.dtype("<f4"),
            shape=self._label_shard.row_shape,
        )
        expected = self._label_shard.expected_slice_sha256[case_id][label_name]
        if observed != expected:
            raise PermissionError(
                f"Authorized {label_name} shard slice hash differs for "
                f"case {case_id}."
            )
        self._slice_journal.append(
            "shard_label_slice_read_completion",
            read_token=token,
            label_name=label_name,
            global_case_id=case_id,
            local_shard_row=local_row,
            access_session_sha256=self._access_session_sha256,
            access_session_kind=self._access_session_kind,
            session_payload_sha256=self._access_session[
                "session_payload_sha256"
            ],
            shard_byte_range=[offset, offset + row_bytes],
            byte_range_convention="half_open_[start,end)",
            observed_semantic_sha256=observed,
            observed_bytes=row_bytes,
            payload_read_once=True,
        )
        return value

    def _baseline(
        self,
        case_id: int,
        time: NDArray[np.float64],
        z: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        with self._lock:
            cached = self._baseline_cache.get(case_id)
        if cached is not None:
            return cached
        baseline = self._baseline_builder(
            self._definitions[case_id],
            time,
            z,
            self._geometry,
        )
        frozen = (
            np.asarray(baseline[0], dtype=np.float64),
            np.asarray(baseline[1], dtype=np.float64),
        )
        with self._lock:
            self._baseline_cache.setdefault(case_id, frozen)
            return self._baseline_cache[case_id]

    def _resolved_case_id(self, index: int) -> int:
        if get_worker_info() is not None:
            raise RuntimeError(
                "Target evaluation requires DataLoader num_workers=0 so "
                "label-access audit events remain authoritative."
            )
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise TypeError("Evaluation dataset index must be an integer.")
        position = int(index)
        if position < 0:
            position += len(self.case_ids)
        if position < 0 or position >= len(self.case_ids):
            raise IndexError("Evaluation dataset index is out of range.")
        case_id = self.case_ids[position]
        if case_id not in self._definitions:
            raise PermissionError("Case ID is outside the frozen evaluation role.")
        return case_id

    def _input_for_case(
        self,
        case_id: int,
        *,
        normalization: FrozenSourceNormalization,
    ) -> torch.Tensor:
        arrays = self._open_input_arrays()
        time = np.asarray(arrays["time_s"], dtype=np.float64)
        z = np.asarray(arrays["z_m"], dtype=np.float64)
        x = np.asarray(arrays["x_m"], dtype=np.float64)
        mask = np.asarray(arrays["composite_mask"], dtype=np.bool_)
        baseline = self._baseline(case_id, time, z)

        def use_cached_baseline(
            definition: dict[str, Any],
            times_s: NDArray[np.float64],
            z_m: NDArray[np.float64],
            geometry: dict[str, float],
        ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
            del definition, times_s, z_m, geometry
            return baseline

        inputs = build_target_2d_input(
            self._definitions[case_id],
            time,
            z,
            x,
            mask,
            normalization,
            self._geometry,
            baseline_builder=use_cached_baseline,
        )
        return torch.from_numpy(inputs)

    def input_only_item(
        self,
        index: int,
        *,
        normalization: FrozenSourceNormalization | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one model input without opening or indexing either label.

        The optional normalization is used by the frozen source-seed
        sensitivity cells.  The label-free coarse solver result is shared,
        while every normalized input channel is rebuilt under the supplied
        source checkpoint contract.
        """

        case_id = self._resolved_case_id(index)
        inputs = self._input_for_case(
            case_id,
            normalization=(
                self._source_normalization
                if normalization is None
                else normalization
            ),
        )
        return inputs, torch.tensor(case_id, dtype=torch.long)

    def label_free_physics_inputs(
        self,
    ) -> Target2DEvaluationPhysicsInputs:
        """Return copied metric inputs without opening a target-label array."""

        arrays = self._open_input_arrays()
        time = np.array(arrays["time_s"], dtype=np.float64, copy=True)
        z = np.array(arrays["z_m"], dtype=np.float64, copy=True)
        x = np.array(arrays["x_m"], dtype=np.float64, copy=True)
        mask = np.array(
            arrays["composite_mask"],
            dtype=np.bool_,
            copy=True,
        )
        for value in (time, z, x, mask):
            value.setflags(write=False)
        return Target2DEvaluationPhysicsInputs(
            definitions={
                case_id: dict(self._definitions[case_id])
                for case_id in self.case_ids
            },
            time_s=time,
            z_m=z,
            x_m=x,
            composite_mask=mask,
            geometry=dict(self._geometry),
            array_root=self._artifact_root,
        )

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        case_id = self._resolved_case_id(index)
        inputs = self._input_for_case(
            case_id,
            normalization=self._source_normalization,
        )

        # The attempted event must precede both label-array opening and slicing.
        self._ensure_access_session_registered()
        with self._lock:
            self._attempted_case_ids.add(case_id)
        labels = self._open_label_arrays()
        local_row = self.case_ids.index(case_id)
        temperature = self._read_shard_label_slice(
            labels=labels,
            label_name="temperature_K",
            case_id=case_id,
            local_row=local_row,
        )
        alpha = self._read_shard_label_slice(
            labels=labels,
            label_name="alpha",
            case_id=case_id,
            local_row=local_row,
        )
        with self._lock:
            self._completed_case_ids.add(case_id)
        return (
            inputs,
            torch.from_numpy(temperature),
            torch.from_numpy(alpha),
            torch.tensor(case_id, dtype=torch.long),
        )


@dataclass(frozen=True)
class PreparedTarget2DEvaluation:
    """Prepared evaluation-only data and its immutable access contract."""

    dataset: Target2DEvaluationDataset
    role: str
    case_ids: tuple[int, ...]
    normalization: dict[str, Any]
    checksums: dict[str, Any]
    required_num_workers: int = 0

    @property
    def access_audit(self) -> dict[str, Any]:
        return self.dataset.access_audit


def prepare_target_2d_evaluation(
    evaluation_manifest: Path,
    source_checkpoint: Path,
    *,
    expected_source_checkpoint_sha256: str,
    project_root: Path,
    manifest_spec: FrozenEvaluationManifestSpec | None = None,
    artifact_root: Path | None = None,
    verify_array_checksums: bool = True,
    baseline_builder: BaselineBuilder = build_label_free_coarse_1d_baseline,
    label_shard: FrozenTarget2DLabelShard | None = None,
) -> PreparedTarget2DEvaluation:
    """Construct an audited dataset for one pre-registered held-out role."""

    if (
        len(expected_source_checkpoint_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_source_checkpoint_sha256
        )
    ):
        raise ValueError(
            "expected_source_checkpoint_sha256 must be lowercase SHA256."
        )
    checkpoint = source_checkpoint.resolve()
    if _sha256_file(checkpoint) != expected_source_checkpoint_sha256:
        raise ValueError("Source checkpoint SHA256 differs from evaluation lock.")
    contract = validate_target_2d_evaluation_metadata(
        evaluation_manifest,
        project_root=project_root,
        manifest_spec=manifest_spec,
        artifact_root=artifact_root,
        verify_array_checksums=verify_array_checksums,
    )
    normalization = load_source_normalization_contract(checkpoint)
    if normalization.checkpoint_sha256 != expected_source_checkpoint_sha256:
        raise ValueError("Loaded source normalization checkpoint SHA256 drifted.")
    checksums = dict(contract.checksums)
    checksums.update(
        {
            "source_checkpoint_sha256": normalization.checkpoint_sha256,
            "source_normalization_sha256": normalization.normalization_sha256,
            "label_shard_evidence": (
                label_shard.evidence()
                if label_shard is not None
                else None
            ),
        }
    )
    dataset = Target2DEvaluationDataset(
        contract=contract,
        normalization=normalization,
        baseline_builder=baseline_builder,
        label_shard=label_shard,
    )
    return PreparedTarget2DEvaluation(
        dataset=dataset,
        role=contract.spec.role,
        case_ids=contract.spec.case_ids,
        normalization=normalization.metadata,
        checksums=checksums,
    )


def target_2d_evaluation_dry_run(
    evaluation_manifest: Path,
    source_checkpoint: Path,
    *,
    expected_source_checkpoint_sha256: str,
    project_root: Path,
    manifest_spec: FrozenEvaluationManifestSpec | None = None,
    artifact_root: Path | None = None,
    verify_array_checksums: bool = True,
    baseline_builder: BaselineBuilder = build_label_free_coarse_1d_baseline,
) -> dict[str, Any]:
    """Validate a release contract without calling dataset ``__getitem__``."""

    prepared = prepare_target_2d_evaluation(
        evaluation_manifest,
        source_checkpoint,
        expected_source_checkpoint_sha256=(
            expected_source_checkpoint_sha256
        ),
        project_root=project_root,
        manifest_spec=manifest_spec,
        artifact_root=artifact_root,
        verify_array_checksums=verify_array_checksums,
        baseline_builder=baseline_builder,
    )
    audit = prepared.access_audit
    passed = (
        audit["attempted_case_ids"] == []
        and audit["completed_case_ids"] == []
        and prepared.required_num_workers == 0
        and prepared.checksums[
            "label_values_indexed_during_metadata_validation"
        ]
        is False
    )
    return {
        "schema_version": 1,
        "dry_run": True,
        "role": prepared.role,
        "case_ids": list(prepared.case_ids),
        "case_count": len(prepared.case_ids),
        "normalization": prepared.normalization,
        "checksums": prepared.checksums,
        "access_audit": audit,
        "passed": passed,
    }
