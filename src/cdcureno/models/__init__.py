"""Neural operator architectures."""
"""Model families exposed by CD-CureNO."""

from cdcureno.models.joint_operators import (
    CausalFactorizedOperator,
    FactorizedFNO,
    NoncausalFNO2d,
    build_joint_operator,
    parameter_count,
)
from cdcureno.models.target_operators import (
    AxisFactorized2DOperator,
    CausalAxisFactorized2DOperator,
    LateralSpectralAdapter,
    P5_SOURCE_CHANNEL_NAMES,
    P5_TARGET_CHANNEL_NAMES,
    P5_TARGET_ONLY_CHANNEL_NAMES,
)

__all__ = [
    "AxisFactorized2DOperator",
    "CausalAxisFactorized2DOperator",
    "CausalFactorizedOperator",
    "FactorizedFNO",
    "LateralSpectralAdapter",
    "NoncausalFNO2d",
    "P5_SOURCE_CHANNEL_NAMES",
    "P5_TARGET_CHANNEL_NAMES",
    "P5_TARGET_ONLY_CHANNEL_NAMES",
    "build_joint_operator",
    "parameter_count",
]
