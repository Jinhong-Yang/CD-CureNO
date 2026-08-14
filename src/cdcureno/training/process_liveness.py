"""Process identity helpers for managed P6 execution.

Windows does not define POSIX signal-zero semantics.  In particular, using
``os.kill(pid, 0)`` there can dispatch a console control event instead of
performing a read-only existence check.  Managed launchers therefore use the
limited-information process handle below and fail closed whenever the answer
is not conclusive.

Windows virtual environments add a second identity wrinkle: their
``python.exe`` is a redirector process.  Starting that file directly makes
``Popen.pid`` identify the redirector instead of the long-lived interpreter.
The spawn plan below starts the base interpreter with CPython's
``__PYVENV_LAUNCHER__`` handoff so the managed child keeps the frozen virtual
environment identity while remaining the launcher's direct process child.
"""

from __future__ import annotations

import ctypes
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal


WindowsProcessStatus = Literal["live", "dead", "uncertain"]
PythonChildSpawnStrategy = Literal[
    "logical_executable_direct",
    "windows_venv_base_executable",
]

_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_ERROR_INVALID_PARAMETER = 87
_STILL_ACTIVE = 259
_MAX_DWORD = (1 << 32) - 1
PYVENV_LAUNCHER_ENV = "__PYVENV_LAUNCHER__"


class PythonChildSpawnError(RuntimeError):
    """A managed Python child cannot preserve its required identity."""


@dataclass(frozen=True)
class PythonChildSpawnPlan:
    """Validated process executable and environment for a Python child."""

    logical_executable: Path
    process_executable: Path
    strategy: PythonChildSpawnStrategy
    pyvenv_launcher_injected: bool
    _environment: dict[str, str] = field(repr=False, compare=False)

    def child_environment(self) -> dict[str, str]:
        """Return a private mutable copy for launcher-specific additions."""

        return dict(self._environment)

    def secret_free_record(self) -> dict[str, Any]:
        """Return the spawn identity safe for dry-runs and durable ledgers."""

        return {
            "strategy": self.strategy,
            "logical_python_executable": str(self.logical_executable),
            "process_python_executable": str(self.process_executable),
            "pyvenv_launcher_injected": self.pyvenv_launcher_injected,
        }


def _resolved_executable(value: str | os.PathLike[str], *, label: str) -> Path:
    try:
        resolved = Path(value).resolve(strict=True)
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        raise PythonChildSpawnError(
            f"{label} is missing or cannot be resolved."
        ) from error
    if not resolved.is_file():
        raise PythonChildSpawnError(f"{label} is not a regular file.")
    return resolved


def _normalized_runtime_prefix(value: str | os.PathLike[str]) -> Path:
    try:
        return Path(value).resolve(strict=False)
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        raise PythonChildSpawnError(
            "Python runtime prefix cannot be resolved."
        ) from error


def prepare_python_child_spawn(
    logical_executable: Path,
    *,
    environment: Mapping[str, str] | None = None,
    _os_name: str | None = None,
    _current_executable: str | os.PathLike[str] | None = None,
    _prefix: str | os.PathLike[str] | None = None,
    _base_prefix: str | os.PathLike[str] | None = None,
    _base_executable: str | os.PathLike[str] | None = None,
) -> PythonChildSpawnPlan:
    """Prepare a direct, identity-preserving managed Python child.

    The private runtime overrides support deterministic unit tests. Production
    callers use the current process values.  On Windows inside a virtual
    environment, the base interpreter is the process executable and
    ``__PYVENV_LAUNCHER__`` points at the validated logical executable.  Every
    other runtime uses the logical executable and leaves the environment
    unchanged.
    """

    current = _resolved_executable(
        sys.executable
        if _current_executable is None
        else _current_executable,
        label="Current Python executable",
    )
    logical = _resolved_executable(
        logical_executable,
        label="Logical Python executable",
    )
    if logical != current:
        raise PythonChildSpawnError(
            "Managed child execution requires the current logical Python "
            "interpreter exactly."
        )

    source_environment = os.environ if environment is None else environment
    if any(
        type(key) is not str or type(value) is not str
        for key, value in source_environment.items()
    ):
        raise PythonChildSpawnError(
            "Managed child environment keys and values must be strings."
        )
    child_environment = dict(source_environment)

    runtime_os_name = os.name if _os_name is None else _os_name
    prefix = _normalized_runtime_prefix(
        sys.prefix if _prefix is None else _prefix
    )
    base_prefix = _normalized_runtime_prefix(
        sys.base_prefix if _base_prefix is None else _base_prefix
    )
    windows_venv = runtime_os_name == "nt" and prefix != base_prefix
    if not windows_venv:
        return PythonChildSpawnPlan(
            logical_executable=logical,
            process_executable=logical,
            strategy="logical_executable_direct",
            pyvenv_launcher_injected=False,
            _environment=child_environment,
        )

    runtime_base_executable = (
        getattr(sys, "_base_executable", None)
        if _base_executable is None
        else _base_executable
    )
    if runtime_base_executable is None:
        raise PythonChildSpawnError(
            "Windows virtual environment has no base Python executable."
        )
    process_executable = _resolved_executable(
        runtime_base_executable,
        label="Base Python executable",
    )

    matching_keys = [
        key
        for key in child_environment
        if key.casefold() == PYVENV_LAUNCHER_ENV.casefold()
    ]
    if len(matching_keys) > 1:
        raise PythonChildSpawnError(
            "Managed child environment has conflicting "
            "__PYVENV_LAUNCHER__ entries."
        )
    if matching_keys:
        existing_key = matching_keys[0]
        try:
            existing = _resolved_executable(
                child_environment[existing_key],
                label="Existing __PYVENV_LAUNCHER__ executable",
            )
        except PythonChildSpawnError as error:
            raise PythonChildSpawnError(
                "Managed child environment has a conflicting "
                "__PYVENV_LAUNCHER__ value."
            ) from error
        if existing != logical:
            raise PythonChildSpawnError(
                "Managed child environment has a conflicting "
                "__PYVENV_LAUNCHER__ value."
            )
        del child_environment[existing_key]
    child_environment[PYVENV_LAUNCHER_ENV] = str(logical)
    return PythonChildSpawnPlan(
        logical_executable=logical,
        process_executable=process_executable,
        strategy="windows_venv_base_executable",
        pyvenv_launcher_injected=True,
        _environment=child_environment,
    )


class _FileTime(ctypes.Structure):
    _fields_ = (
        ("low_date_time", ctypes.c_uint32),
        ("high_date_time", ctypes.c_uint32),
    )


@dataclass(frozen=True)
class WindowsProcessProbe:
    """Read-only PID status and optional PID-reuse-resistant creation time."""

    status: WindowsProcessStatus
    creation_identity: str | None


def _load_windows_kernel32() -> Any:
    """Load and type the read-only Win32 process-query functions."""

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint32,
    )
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.GetExitCodeProcess.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
    )
    kernel32.GetExitCodeProcess.restype = ctypes.c_int
    kernel32.GetProcessTimes.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(_FileTime),
        ctypes.POINTER(_FileTime),
        ctypes.POINTER(_FileTime),
        ctypes.POINTER(_FileTime),
    )
    kernel32.GetProcessTimes.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
    kernel32.CloseHandle.restype = ctypes.c_int
    return kernel32


def windows_process_probe(
    pid: int,
    *,
    _kernel32: Any | None = None,
    _get_last_error: Callable[[], int] | None = None,
) -> WindowsProcessProbe:
    """Return a signal-free Windows process probe.

    Only an invalid-parameter result from ``OpenProcess`` or a successfully
    observed terminal exit code proves that a PID is dead.  Access denial,
    unavailable APIs, query failures, and every other unknown state remain
    ``uncertain`` so stale-lock recovery cannot act on weak evidence.

    The private injectable arguments exist solely for deterministic unit
    tests; production callers always use the real Win32 API.
    """

    if type(pid) is not int or pid <= 0:
        return WindowsProcessProbe("uncertain", None)
    if pid > _MAX_DWORD:
        return WindowsProcessProbe("dead", None)
    if _get_last_error is None:
        _get_last_error = getattr(ctypes, "get_last_error", lambda: -1)
    if _kernel32 is None:
        try:
            _kernel32 = _load_windows_kernel32()
        except Exception:
            return WindowsProcessProbe("uncertain", None)

    try:
        handle = _kernel32.OpenProcess(
            _PROCESS_QUERY_LIMITED_INFORMATION,
            False,
            pid,
        )
    except Exception:
        return WindowsProcessProbe("uncertain", None)
    if not handle:
        try:
            error_code = int(_get_last_error())
        except Exception:
            error_code = -1
        status: WindowsProcessStatus = (
            "dead"
            if error_code == _ERROR_INVALID_PARAMETER
            else "uncertain"
        )
        return WindowsProcessProbe(status, None)

    try:
        exit_code = ctypes.c_uint32()
        try:
            queried = bool(
                _kernel32.GetExitCodeProcess(
                    handle,
                    ctypes.byref(exit_code),
                )
            )
        except Exception:
            queried = False
        if not queried:
            return WindowsProcessProbe("uncertain", None)
        if exit_code.value != _STILL_ACTIVE:
            return WindowsProcessProbe("dead", None)

        creation = _FileTime()
        exit_time = _FileTime()
        kernel_time = _FileTime()
        user_time = _FileTime()
        try:
            times_queried = bool(
                _kernel32.GetProcessTimes(
                    handle,
                    ctypes.byref(creation),
                    ctypes.byref(exit_time),
                    ctypes.byref(kernel_time),
                    ctypes.byref(user_time),
                )
            )
        except Exception:
            times_queried = False
        identity = None
        if times_queried:
            ticks = (
                int(creation.high_date_time) << 32
            ) | int(creation.low_date_time)
            identity = f"win32-creation-filetime:{ticks}"
        return WindowsProcessProbe("live", identity)
    finally:
        try:
            _kernel32.CloseHandle(handle)
        except Exception:
            # Liveness has already been determined.  The close was attempted,
            # and an API failure must not turn uncertainty into dead evidence.
            pass


def windows_process_status(pid: int) -> WindowsProcessStatus:
    """Return the signal-free Windows liveness status for ``pid``."""

    return windows_process_probe(pid).status


def windows_process_is_live(pid: int) -> bool:
    """Fail closed for lock recovery: unknown Windows PIDs count as live."""

    return windows_process_status(pid) != "dead"
