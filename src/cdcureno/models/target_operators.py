"""Restriction-preserving axis-factorized operators for true 2-D fields.

The target tensor order is ``[batch, time, z, x, channel]``.  The transferred
``z`` and temporal branches are deliberately the same modules used by their
respective P3 source family: either the noncausal ``FactorizedFNO`` pilot or
the structurally causal dilated-convolution source.  The new lateral branch
is a high-pass, low-rank residual: it omits the ``k_x=0`` coefficient and
starts with an exactly-zero output factor.  Consequently, an extruded source
input lies on an invariant subspace at checkpoint inflation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn.functional as functional
from torch import nn

from cdcureno.models.joint_operators import (
    CausalTemporalConv1d,
    SpatialSpectralConv1d,
    TemporalSpectralConv1d,
)


TransferStage = Literal["T0", "T1", "T2"]

P5_SOURCE_CHANNEL_NAMES = (
    "air_temperature_normalized",
    "causal_physics_temperature_baseline_normalized",
    "time_normalized",
    "through_thickness_position_normalized",
    "composite_mask",
    "signed_distance_to_interface_normalized",
    "initial_degree_of_cure",
    "tool_thickness_normalized",
    "composite_thickness_normalized",
    "lower_htc_normalized",
    "upper_htc_normalized",
    "composite_conductivity_scale_normalized",
    "heat_of_reaction_scale_normalized",
    "low_fidelity_degree_of_cure",
)
P5_TARGET_ONLY_CHANNEL_NAMES = (
    "heterogeneity_gated_in_plane_position_normalized",
    "heterogeneity_gated_signed_distance_to_left_boundary_normalized",
    "heterogeneity_gated_signed_distance_to_right_boundary_normalized",
    "top_htc_anomaly_normalized",
    "edge_htc_boundary_map_normalized",
    "lateral_conductivity_scale_delta_normalized",
)
P5_TARGET_CHANNEL_NAMES = (
    *P5_SOURCE_CHANNEL_NAMES,
    *P5_TARGET_ONLY_CHANNEL_NAMES,
)
P5_TARGET_CHANNEL_INDEX = {
    name: index for index, name in enumerate(P5_TARGET_CHANNEL_NAMES)
}


class RestrictionPreservingLift(nn.Module):
    """Lift shared and target-only channels without mixing their parameters.

    Splitting the first affine map is mathematically equivalent to an expanded
    input projection.  It also lets Stage T0 optimize only the target-only
    columns, without an optimizer or weight decay modifying copied columns.
    """

    def __init__(
        self,
        shared_channels: int,
        geometry_channels: int,
        width: int,
    ) -> None:
        super().__init__()
        if min(shared_channels, geometry_channels, width) < 1:
            raise ValueError(
                "shared_channels, geometry_channels, and width must be positive."
            )
        self.shared_channels = shared_channels
        self.geometry_channels = geometry_channels
        self.width = width
        self.shared = nn.Linear(shared_channels, width)
        self.geometry = nn.Linear(geometry_channels, width, bias=False)
        self.projection = nn.Linear(width, width)

    @property
    def input_channels(self) -> int:
        return self.shared_channels + self.geometry_channels

    def combined_weight(self) -> torch.Tensor:
        """Return the conceptual expanded first-layer matrix."""

        return torch.cat((self.shared.weight, self.geometry.weight), dim=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-1] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} channels, got {inputs.shape[-1]}."
            )
        # Match the source lift's contiguous channel layout. A strided slice
        # can select a different CPU linear kernel and lose nx=1 bitwise parity.
        shared = inputs[..., : self.shared_channels].contiguous()
        geometry = inputs[..., self.shared_channels :]
        hidden = self.shared(shared) + self.geometry(geometry)
        return self.projection(functional.gelu(hidden))


class LateralSpectralAdapter(nn.Module):
    """Low-rank Fourier residual over ``x`` with an anchored zero mode.

    The DC coefficient is never consumed, so the adapter remains zero on an
    exactly homogeneous lateral field even after training.  At inflation the
    input factor is deterministic and nonzero while the output factor is zero.
    This gives exactly zero initial output and a live first-step gradient for
    the output factor.
    """

    def __init__(
        self,
        channels: int,
        modes: int,
        rank: int,
        *,
        initialization_seed: int = 0,
    ) -> None:
        super().__init__()
        if min(channels, modes, rank) < 1:
            raise ValueError("channels, modes, and rank must be positive.")
        if rank > channels:
            raise ValueError("rank cannot exceed channels.")
        self.channels = channels
        self.modes = modes
        self.rank = rank
        self.input_factor = nn.Parameter(
            torch.empty(channels, rank, modes, dtype=torch.cfloat)
        )
        self.output_factor = nn.Parameter(
            torch.empty(rank, channels, modes, dtype=torch.cfloat)
        )
        self.reset_zero_residual(initialization_seed)

    def reset_zero_residual(self, seed: int) -> None:
        """Deterministically initialize a trainable exactly-zero residual."""

        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        shape = tuple(self.input_factor.shape)
        real = torch.randn(shape, generator=generator, dtype=torch.float32)
        imaginary = torch.randn(shape, generator=generator, dtype=torch.float32)
        scale = (2.0 * self.channels) ** -0.5
        values = torch.complex(real, imaginary) * scale
        with torch.no_grad():
            self.input_factor.copy_(
                values.to(
                    device=self.input_factor.device,
                    dtype=self.input_factor.dtype,
                )
            )
            self.output_factor.zero_()

    def effective_weight(self) -> torch.Tensor:
        return torch.einsum(
            "irm,rom->iom", self.input_factor, self.output_factor
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 5:
            raise ValueError(
                f"Expected [B,Nt,Nz,Nx,C], got {tuple(inputs.shape)}."
            )
        batch, time_count, z_count, x_count, channels = inputs.shape
        if channels != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {channels}.")
        if x_count == 1:
            return torch.zeros_like(inputs)
        values = inputs.permute(0, 1, 2, 4, 3).reshape(
            batch * time_count * z_count, channels, x_count
        )
        coefficients = torch.fft.rfft(values, dim=-1, norm="ortho")
        output_coefficients = torch.zeros_like(coefficients)
        # Mode zero is an immutable dimensional-restriction anchor.
        retained = min(self.modes, max(coefficients.shape[-1] - 1, 0))
        if retained:
            output_coefficients[:, :, 1 : retained + 1] = torch.einsum(
                "bim,iom->bom",
                coefficients[:, :, 1 : retained + 1],
                self.effective_weight()[:, :, :retained],
            )
        output = torch.fft.irfft(
            output_coefficients, n=x_count, dim=-1, norm="ortho"
        )
        return output.reshape(
            batch, time_count, z_count, channels, x_count
        ).permute(0, 1, 2, 4, 3)


def _apply_source_axis_module(
    module: nn.Module, inputs: torch.Tensor
) -> torch.Tensor:
    """Apply a P3 ``[B,Nt,Nz,C]`` module independently at each x."""

    batch, time_count, z_count, x_count, channels = inputs.shape
    flattened = inputs.permute(0, 3, 1, 2, 4).reshape(
        batch * x_count, time_count, z_count, channels
    )
    output = module(flattened)
    return output.reshape(
        batch, x_count, time_count, z_count, channels
    ).permute(0, 2, 3, 1, 4)


class AxisFactorized2DBlock(nn.Module):
    """Source-equivalent ``z``/time block plus a zero-residual x adapter."""

    def __init__(
        self,
        channels: int,
        modes_time: int,
        modes_z: int,
        modes_x: int,
        lateral_rank: int,
        *,
        adapter_seed: int,
    ) -> None:
        super().__init__()
        self.spatial = SpatialSpectralConv1d(channels, modes_z)
        self.lateral = LateralSpectralAdapter(
            channels,
            modes_x,
            lateral_rank,
            initialization_seed=adapter_seed,
        )
        self.temporal = TemporalSpectralConv1d(channels, modes_time)
        self.spatial_local = nn.Linear(channels, channels)
        self.temporal_local = nn.Linear(channels, channels)
        self.spatial_norm = nn.LayerNorm(channels)
        self.temporal_norm = nn.LayerNorm(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        spatial_update = functional.gelu(
            _apply_source_axis_module(self.spatial, inputs)
            + self.spatial_local(inputs)
            + self.lateral(inputs)
        )
        hidden = self.spatial_norm(inputs + spatial_update)
        temporal_update = functional.gelu(
            _apply_source_axis_module(self.temporal, hidden)
            + self.temporal_local(hidden)
        )
        return self.temporal_norm(hidden + temporal_update)


class CausalAxisFactorized2DBlock(nn.Module):
    """Causal source-equivalent block plus a zero-residual x adapter.

    Every operation except ``temporal`` is pointwise in time.  The temporal
    branch is the exact left-padded convolution used by the causal P3 source,
    applied independently at every lateral coordinate.  Consequently, this
    block cannot expose any output prefix to a future input suffix.
    """

    def __init__(
        self,
        channels: int,
        modes_z: int,
        modes_x: int,
        lateral_rank: int,
        *,
        temporal_kernel_size: int,
        temporal_dilation: int,
        adapter_seed: int,
    ) -> None:
        super().__init__()
        self.spatial = SpatialSpectralConv1d(channels, modes_z)
        self.lateral = LateralSpectralAdapter(
            channels,
            modes_x,
            lateral_rank,
            initialization_seed=adapter_seed,
        )
        self.temporal = CausalTemporalConv1d(
            channels,
            kernel_size=temporal_kernel_size,
            dilation=temporal_dilation,
        )
        self.spatial_local = nn.Linear(channels, channels)
        self.temporal_local = nn.Linear(channels, channels)
        self.spatial_norm = nn.LayerNorm(channels)
        self.temporal_norm = nn.LayerNorm(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        spatial_update = functional.gelu(
            _apply_source_axis_module(self.spatial, inputs)
            + self.spatial_local(inputs)
            + self.lateral(inputs)
        )
        hidden = self.spatial_norm(inputs + spatial_update)
        temporal_update = functional.gelu(
            _apply_source_axis_module(self.temporal, hidden)
            + self.temporal_local(hidden)
        )
        return self.temporal_norm(hidden + temporal_update)


class AxisFactorized2DFieldHead(nn.Module):
    """Source-equivalent temperature and monotone-cure head for 2-D fields."""

    def __init__(
        self,
        channels: int,
        *,
        baseline_channel: int,
        material_mask_channel: int,
        initial_alpha_channel: int,
    ) -> None:
        super().__init__()
        self.temperature_residual = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, 1),
        )
        self.cure_rate = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, 1),
        )
        self.baseline_channel = baseline_channel
        self.material_mask_channel = material_mask_channel
        self.initial_alpha_channel = initial_alpha_channel

    def forward(
        self, hidden: torch.Tensor, inputs: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        residual = self.temperature_residual(hidden).squeeze(-1)
        temperature = inputs[..., self.baseline_channel] + residual
        material_mask = inputs[..., self.material_mask_channel].clamp(0.0, 1.0)
        rate = functional.softplus(self.cure_rate(hidden).squeeze(-1))
        rate = rate * material_mask
        initial_alpha = inputs[
            :, 0, :, :, self.initial_alpha_channel
        ].clamp(0.0, 1.0)
        time_count = hidden.shape[1]
        delta_time = 1.0 / max(time_count - 1, 1)
        zero = torch.zeros_like(rate[:, :1])
        integrated_rate = torch.cat(
            [zero, torch.cumsum(rate[:, :-1] * delta_time, dim=1)], dim=1
        )
        alpha = 1.0 - (1.0 - initial_alpha[:, None, :, :]) * torch.exp(
            -integrated_rate
        )
        alpha = alpha * material_mask
        return {
            "field": torch.stack((temperature, alpha), dim=-1),
            "temperature": temperature,
            "temperature_residual": residual,
            "alpha": alpha,
            "cure_rate": rate,
        }


class AxisFactorized2DOperator(nn.Module):
    """True-2-D target operator with an exact P3 restriction at inflation."""

    family = "axis_factorized_2d"
    temporal_family = "spectral_noncausal"

    def __init__(
        self,
        *,
        source_channel_names: Sequence[str],
        new_channel_names: Sequence[str],
        width: int,
        depth: int,
        modes_time: int,
        modes_z: int,
        modes_x: int,
        lateral_rank: int,
        adapter_seed: int = 0,
    ) -> None:
        super().__init__()
        source_names = tuple(str(name) for name in source_channel_names)
        new_names = tuple(str(name) for name in new_channel_names)
        if not source_names or not new_names:
            raise ValueError("Source and new channel lists cannot be empty.")
        combined = (*source_names, *new_names)
        if len(set(combined)) != len(combined):
            raise ValueError("Target channel names must be unique.")
        required = {
            "causal_physics_temperature_baseline_normalized",
            "composite_mask",
            "initial_degree_of_cure",
        }
        missing = sorted(required.difference(source_names))
        if missing:
            raise ValueError(f"Source channel contract is missing {missing}.")
        if min(width, depth, modes_time, modes_z, modes_x, lateral_rank) < 1:
            raise ValueError("All target architecture sizes must be positive.")
        self.source_channel_names = source_names
        self.new_channel_names = new_names
        self.channel_names = combined
        self.input_channels = len(combined)
        self.width = width
        self.depth = depth
        self.modes_time = modes_time
        self.modes_z = modes_z
        self.modes_x = modes_x
        self.lateral_rank = lateral_rank
        self.lift = RestrictionPreservingLift(
            len(source_names), len(new_names), width
        )
        self.blocks = nn.ModuleList(
            [
                AxisFactorized2DBlock(
                    width,
                    modes_time,
                    modes_z,
                    modes_x,
                    lateral_rank,
                    adapter_seed=adapter_seed + layer_index,
                )
                for layer_index in range(depth)
            ]
        )
        self.head = AxisFactorized2DFieldHead(
            width,
            baseline_channel=source_names.index(
                "causal_physics_temperature_baseline_normalized"
            ),
            material_mask_channel=source_names.index("composite_mask"),
            initial_alpha_channel=source_names.index(
                "initial_degree_of_cure"
            ),
        )

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        if inputs.ndim != 5 or inputs.shape[-1] != self.input_channels:
            raise ValueError(
                "Expected "
                f"[B,Nt,Nz,Nx,{self.input_channels}], got {tuple(inputs.shape)}."
            )
        hidden = self.lift(inputs)
        for block in self.blocks:
            hidden = block(hidden)
        return self.head(hidden, inputs)

    def set_transfer_stage(
        self, stage: TransferStage
    ) -> dict[str, int | str]:
        """Apply the required T0/T1/T2 trainability schedule."""

        if stage not in {"T0", "T1", "T2"}:
            raise ValueError(f"Unknown transfer stage: {stage}")
        for parameter in self.parameters():
            parameter.requires_grad_(stage == "T2")
        if stage in {"T0", "T1"}:
            for parameter in self.lift.geometry.parameters():
                parameter.requires_grad_(True)
            for block in self.blocks:
                for parameter in block.lateral.parameters():
                    parameter.requires_grad_(True)
        if stage == "T1":
            for parameter in self.head.parameters():
                parameter.requires_grad_(True)
            for parameter in self.lift.projection.parameters():
                parameter.requires_grad_(True)
            for block in self.blocks:
                for module in (block.spatial_local, block.temporal_local):
                    for parameter in module.parameters():
                        parameter.requires_grad_(True)
        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        total = sum(parameter.numel() for parameter in self.parameters())
        return {
            "stage": stage,
            "trainable_parameters": trainable,
            "frozen_parameters": total - trainable,
            "total_parameters": total,
        }


class CausalAxisFactorized2DOperator(nn.Module):
    """True-2-D target with exact causal-P3 restriction at inflation.

    This model is intentionally a distinct family from
    :class:`AxisFactorized2DOperator`, whose temporal branch is a noncausal
    Fourier operator.  No temporal FFT or right padding appears in this
    architecture.
    """

    family = "causal_axis_factorized_2d"
    temporal_family = "causal_dilated_convolution"
    structurally_causal = True

    def __init__(
        self,
        *,
        source_channel_names: Sequence[str],
        new_channel_names: Sequence[str],
        width: int,
        depth: int,
        modes_z: int,
        modes_x: int,
        lateral_rank: int,
        temporal_kernel_size: int = 3,
        temporal_dilations: Sequence[int] | None = None,
        adapter_seed: int = 0,
    ) -> None:
        super().__init__()
        source_names = tuple(str(name) for name in source_channel_names)
        new_names = tuple(str(name) for name in new_channel_names)
        if not source_names or not new_names:
            raise ValueError("Source and new channel lists cannot be empty.")
        combined = (*source_names, *new_names)
        if len(set(combined)) != len(combined):
            raise ValueError("Target channel names must be unique.")
        required = {
            "causal_physics_temperature_baseline_normalized",
            "composite_mask",
            "initial_degree_of_cure",
        }
        missing = sorted(required.difference(source_names))
        if missing:
            raise ValueError(f"Source channel contract is missing {missing}.")
        if min(
            width,
            depth,
            modes_z,
            modes_x,
            lateral_rank,
            temporal_kernel_size,
        ) < 1:
            raise ValueError("All target architecture sizes must be positive.")
        if lateral_rank > width:
            raise ValueError("lateral_rank cannot exceed width.")
        if temporal_dilations is None:
            dilations = tuple(2 ** (index % 8) for index in range(depth))
        else:
            dilations = tuple(int(value) for value in temporal_dilations)
        if len(dilations) != depth or any(value < 1 for value in dilations):
            raise ValueError(
                "temporal_dilations must contain one positive value per block."
            )
        self.source_channel_names = source_names
        self.new_channel_names = new_names
        self.channel_names = combined
        self.input_channels = len(combined)
        self.width = width
        self.depth = depth
        self.modes_z = modes_z
        self.modes_x = modes_x
        self.lateral_rank = lateral_rank
        self.temporal_kernel_size = temporal_kernel_size
        self.temporal_dilations = dilations
        # P6 interface ablations set this immutable-at-run-start tuple without
        # changing the 20-channel lift or its parameter count.
        self.zero_input_channel_indices: tuple[int, ...] = ()
        self.lift = RestrictionPreservingLift(
            len(source_names), len(new_names), width
        )
        self.blocks = nn.ModuleList(
            [
                CausalAxisFactorized2DBlock(
                    width,
                    modes_z,
                    modes_x,
                    lateral_rank,
                    temporal_kernel_size=temporal_kernel_size,
                    temporal_dilation=dilations[layer_index],
                    adapter_seed=adapter_seed + layer_index,
                )
                for layer_index in range(depth)
            ]
        )
        self.head = AxisFactorized2DFieldHead(
            width,
            baseline_channel=source_names.index(
                "causal_physics_temperature_baseline_normalized"
            ),
            material_mask_channel=source_names.index("composite_mask"),
            initial_alpha_channel=source_names.index(
                "initial_degree_of_cure"
            ),
        )

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        if inputs.ndim != 5 or inputs.shape[-1] != self.input_channels:
            raise ValueError(
                "Expected "
                f"[B,Nt,Nz,Nx,{self.input_channels}], got {tuple(inputs.shape)}."
            )
        resolved_inputs = inputs
        if self.zero_input_channel_indices:
            resolved_inputs = inputs.clone()
            resolved_inputs[..., list(self.zero_input_channel_indices)] = 0.0
        hidden = self.lift(resolved_inputs)
        for block in self.blocks:
            hidden = block(hidden)
        return self.head(hidden, resolved_inputs)

    def set_transfer_stage(
        self, stage: TransferStage
    ) -> dict[str, int | str]:
        """Apply the same T0/T1/T2 schedule without changing causality."""

        if stage not in {"T0", "T1", "T2"}:
            raise ValueError(f"Unknown transfer stage: {stage}")
        for parameter in self.parameters():
            parameter.requires_grad_(stage == "T2")
        if stage in {"T0", "T1"}:
            for parameter in self.lift.geometry.parameters():
                parameter.requires_grad_(True)
            for block in self.blocks:
                for parameter in block.lateral.parameters():
                    parameter.requires_grad_(True)
        if stage == "T1":
            for parameter in self.head.parameters():
                parameter.requires_grad_(True)
            for parameter in self.lift.projection.parameters():
                parameter.requires_grad_(True)
            for block in self.blocks:
                for module in (block.spatial_local, block.temporal_local):
                    for parameter in module.parameters():
                        parameter.requires_grad_(True)
        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        total = sum(parameter.numel() for parameter in self.parameters())
        return {
            "stage": stage,
            "trainable_parameters": trainable,
            "frozen_parameters": total - trainable,
            "total_parameters": total,
        }
