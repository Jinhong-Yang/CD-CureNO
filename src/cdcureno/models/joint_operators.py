"""Joint temperature/cure operators for the P2 1+1-D benchmark."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as functional
from torch import nn


JointOperatorName = Literal[
    "noncausal_fno2d",
    "factorized_fno",
    "causal_factorized",
]


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


class SpatialSpectralConv1d(nn.Module):
    """Fourier mixing over `Nz` while preserving every time slice."""

    def __init__(self, channels: int, modes: int) -> None:
        super().__init__()
        if channels < 1 or modes < 1:
            raise ValueError("channels and modes must be positive.")
        self.channels = channels
        self.modes = modes
        scale = 1.0 / channels
        self.weight = nn.Parameter(
            scale
            * torch.randn(
                channels,
                channels,
                modes,
                dtype=torch.cfloat,
            )
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4:
            raise ValueError(f"Expected [B,Nt,Nz,C], got {tuple(inputs.shape)}.")
        batch, time_count, space_count, channels = inputs.shape
        if channels != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {channels}.")
        values = inputs.permute(0, 1, 3, 2).reshape(
            batch * time_count, channels, space_count
        )
        coefficients = torch.fft.rfft(values, dim=-1, norm="ortho")
        output_coefficients = torch.zeros_like(coefficients)
        retained = min(self.modes, coefficients.shape[-1])
        output_coefficients[:, :, :retained] = torch.einsum(
            "bim,iom->bom",
            coefficients[:, :, :retained],
            self.weight[:, :, :retained],
        )
        output = torch.fft.irfft(
            output_coefficients, n=space_count, dim=-1, norm="ortho"
        )
        return output.reshape(batch, time_count, channels, space_count).permute(
            0, 1, 3, 2
        )


class TemporalSpectralConv1d(nn.Module):
    """Noncausal Fourier mixing over `Nt` at each spatial position."""

    def __init__(self, channels: int, modes: int) -> None:
        super().__init__()
        if channels < 1 or modes < 1:
            raise ValueError("channels and modes must be positive.")
        self.channels = channels
        self.modes = modes
        scale = 1.0 / channels
        self.weight = nn.Parameter(
            scale
            * torch.randn(
                channels,
                channels,
                modes,
                dtype=torch.cfloat,
            )
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4:
            raise ValueError(f"Expected [B,Nt,Nz,C], got {tuple(inputs.shape)}.")
        batch, time_count, space_count, channels = inputs.shape
        if channels != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {channels}.")
        values = inputs.permute(0, 2, 3, 1).reshape(
            batch * space_count, channels, time_count
        )
        coefficients = torch.fft.rfft(values, dim=-1, norm="ortho")
        output_coefficients = torch.zeros_like(coefficients)
        retained = min(self.modes, coefficients.shape[-1])
        output_coefficients[:, :, :retained] = torch.einsum(
            "bim,iom->bom",
            coefficients[:, :, :retained],
            self.weight[:, :, :retained],
        )
        output = torch.fft.irfft(
            output_coefficients, n=time_count, dim=-1, norm="ortho"
        )
        return output.reshape(batch, space_count, channels, time_count).permute(
            0, 3, 1, 2
        )


class SpectralConv2d(nn.Module):
    """Noncausal joint Fourier mixing over time and through-thickness position."""

    def __init__(self, channels: int, modes_time: int, modes_space: int) -> None:
        super().__init__()
        if min(channels, modes_time, modes_space) < 1:
            raise ValueError("channels and modes must be positive.")
        self.channels = channels
        self.modes_time = modes_time
        self.modes_space = modes_space
        scale = 1.0 / channels
        shape = (channels, channels, modes_time, modes_space)
        self.weights = nn.ParameterList(
            [
                nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
                for _ in range(4)
            ]
        )

    @staticmethod
    def _mix(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bitz,iotz->botz", values, weights)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4:
            raise ValueError(f"Expected [B,Nt,Nz,C], got {tuple(inputs.shape)}.")
        batch, time_count, space_count, channels = inputs.shape
        if channels != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {channels}.")
        if time_count < 2 or space_count < 2:
            raise ValueError("Joint spectral convolution needs at least two grid points.")
        values = inputs.permute(0, 3, 1, 2)
        coefficients = torch.fft.fft2(values, dim=(-2, -1), norm="ortho")
        output_coefficients = torch.zeros_like(coefficients)
        time_modes = min(self.modes_time, time_count // 2)
        space_modes = min(self.modes_space, space_count // 2)
        time_slices = (slice(0, time_modes), slice(-time_modes, None))
        space_slices = (slice(0, space_modes), slice(-space_modes, None))
        weight_index = 0
        for time_slice in time_slices:
            for space_slice in space_slices:
                output_coefficients[:, :, time_slice, space_slice] = self._mix(
                    coefficients[:, :, time_slice, space_slice],
                    self.weights[weight_index][
                        :, :, :time_modes, :space_modes
                    ],
                )
                weight_index += 1
        output = torch.fft.ifft2(
            output_coefficients, dim=(-2, -1), norm="ortho"
        ).real
        return output.permute(0, 2, 3, 1)


class CausalTemporalConv1d(nn.Module):
    """Left-padded temporal convolution applied independently at each `z`."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        if min(channels, kernel_size, dilation) < 1:
            raise ValueError("channels, kernel_size, and dilation must be positive.")
        self.channels = channels
        self.left_padding = dilation * (kernel_size - 1)
        self.convolution = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4:
            raise ValueError(f"Expected [B,Nt,Nz,C], got {tuple(inputs.shape)}.")
        batch, time_count, space_count, channels = inputs.shape
        values = inputs.permute(0, 2, 3, 1).reshape(
            batch * space_count, channels, time_count
        )
        values = functional.pad(values, (self.left_padding, 0))
        output = self.convolution(values)
        return output.reshape(batch, space_count, channels, time_count).permute(
            0, 3, 1, 2
        )


class _JointSpectralBlock(nn.Module):
    def __init__(
        self, channels: int, modes_time: int, modes_space: int
    ) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(channels, modes_time, modes_space)
        self.local = nn.Linear(channels, channels)
        self.normalization = nn.LayerNorm(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        update = functional.gelu(self.spectral(inputs) + self.local(inputs))
        return self.normalization(inputs + update)


class _FactorizedSpectralBlock(nn.Module):
    def __init__(
        self, channels: int, modes_time: int, modes_space: int
    ) -> None:
        super().__init__()
        self.spatial = SpatialSpectralConv1d(channels, modes_space)
        self.temporal = TemporalSpectralConv1d(channels, modes_time)
        self.spatial_local = nn.Linear(channels, channels)
        self.temporal_local = nn.Linear(channels, channels)
        self.spatial_norm = nn.LayerNorm(channels)
        self.temporal_norm = nn.LayerNorm(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        spatial_update = functional.gelu(
            self.spatial(inputs) + self.spatial_local(inputs)
        )
        hidden = self.spatial_norm(inputs + spatial_update)
        temporal_update = functional.gelu(
            self.temporal(hidden) + self.temporal_local(hidden)
        )
        return self.temporal_norm(hidden + temporal_update)


class _CausalFactorizedBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        modes_space: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self.spatial = SpatialSpectralConv1d(channels, modes_space)
        self.temporal = CausalTemporalConv1d(
            channels, kernel_size=3, dilation=dilation
        )
        self.spatial_local = nn.Linear(channels, channels)
        self.temporal_local = nn.Linear(channels, channels)
        self.spatial_norm = nn.LayerNorm(channels)
        self.temporal_norm = nn.LayerNorm(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        spatial_update = functional.gelu(
            self.spatial(inputs) + self.spatial_local(inputs)
        )
        hidden = self.spatial_norm(inputs + spatial_update)
        temporal_update = functional.gelu(
            self.temporal(hidden) + self.temporal_local(hidden)
        )
        return self.temporal_norm(hidden + temporal_update)


class JointFieldHead(nn.Module):
    """Temperature residual and monotone bounded degree-of-cure heads."""

    def __init__(
        self,
        channels: int,
        baseline_channel: int = 1,
        material_mask_channel: int = 4,
        initial_alpha_channel: int = 6,
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
        initial_alpha = inputs[:, 0, :, self.initial_alpha_channel].clamp(0.0, 1.0)
        time_count = hidden.shape[1]
        delta_time = 1.0 / max(time_count - 1, 1)
        zero = torch.zeros_like(rate[:, :1])
        integrated_rate = torch.cat(
            [zero, torch.cumsum(rate[:, :-1] * delta_time, dim=1)], dim=1
        )
        alpha = 1.0 - (1.0 - initial_alpha[:, None, :]) * torch.exp(
            -integrated_rate
        )
        alpha = alpha * material_mask
        return {
            "temperature": temperature,
            "temperature_residual": residual,
            "alpha": alpha,
            "cure_rate": rate,
        }


class _JointOperator(nn.Module):
    family: JointOperatorName

    def __init__(self, input_channels: int, width: int) -> None:
        super().__init__()
        if input_channels < 1 or width < 1:
            raise ValueError("input_channels and width must be positive.")
        self.input_channels = input_channels
        self.width = width
        self.lift = nn.Sequential(
            nn.Linear(input_channels, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.blocks = nn.ModuleList()
        self.head = JointFieldHead(width)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        if inputs.ndim != 4 or inputs.shape[-1] != self.input_channels:
            raise ValueError(
                f"Expected [B,Nt,Nz,{self.input_channels}], got {tuple(inputs.shape)}."
            )
        hidden = self.lift(inputs)
        for block in self.blocks:
            hidden = block(hidden)
        return self.head(hidden, inputs)


class NoncausalFNO2d(_JointOperator):
    family: JointOperatorName = "noncausal_fno2d"

    def __init__(
        self,
        input_channels: int = 7,
        width: int = 24,
        depth: int = 4,
        modes_time: int = 16,
        modes_space: int = 12,
    ) -> None:
        super().__init__(input_channels, width)
        self.blocks = nn.ModuleList(
            [
                _JointSpectralBlock(width, modes_time, modes_space)
                for _ in range(depth)
            ]
        )


class FactorizedFNO(_JointOperator):
    family: JointOperatorName = "factorized_fno"

    def __init__(
        self,
        input_channels: int = 7,
        width: int = 32,
        depth: int = 4,
        modes_time: int = 16,
        modes_space: int = 12,
    ) -> None:
        super().__init__(input_channels, width)
        self.blocks = nn.ModuleList(
            [
                _FactorizedSpectralBlock(width, modes_time, modes_space)
                for _ in range(depth)
            ]
        )


class CausalFactorizedOperator(_JointOperator):
    family: JointOperatorName = "causal_factorized"

    def __init__(
        self,
        input_channels: int = 7,
        width: int = 48,
        depth: int = 8,
        modes_space: int = 16,
    ) -> None:
        super().__init__(input_channels, width)
        self.blocks = nn.ModuleList(
            [
                _CausalFactorizedBlock(
                    width,
                    modes_space=modes_space,
                    dilation=2 ** (layer_index % 8),
                )
                for layer_index in range(depth)
            ]
        )


def build_joint_operator(
    name: JointOperatorName,
    *,
    input_channels: int = 7,
    width: int | None = None,
    depth: int | None = None,
    modes_time: int = 16,
    modes_space: int = 12,
) -> _JointOperator:
    """Construct a named P2 model without silently changing its family."""

    if name == "noncausal_fno2d":
        return NoncausalFNO2d(
            input_channels=input_channels,
            width=width or 24,
            depth=depth or 4,
            modes_time=modes_time,
            modes_space=modes_space,
        )
    if name == "factorized_fno":
        return FactorizedFNO(
            input_channels=input_channels,
            width=width or 32,
            depth=depth or 4,
            modes_time=modes_time,
            modes_space=modes_space,
        )
    if name == "causal_factorized":
        return CausalFactorizedOperator(
            input_channels=input_channels,
            width=width or 48,
            depth=depth or 8,
            modes_space=modes_space,
        )
    raise ValueError(f"Unknown joint operator: {name}")
