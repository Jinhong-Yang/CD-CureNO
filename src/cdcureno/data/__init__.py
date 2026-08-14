"""Data validation, normalization, and split utilities."""
"""Dataset preparation and normalization utilities."""

from cdcureno.data.joint_case1 import (
    INPUT_CHANNELS,
    PreparedJointCase1,
    causal_exponential_smoothing,
    prepare_joint_case1,
)
from cdcureno.data.target_2d import (
    TARGET_ADAPTER_CHANNELS,
    TARGET_INPUT_CHANNELS,
    TARGET_OUTPUT_CHANNELS,
    PreparedTarget2DTraining,
    build_label_free_coarse_1d_baseline,
    lift_homogeneous_source_input,
    load_source_normalization_contract,
    prepare_target_2d_training,
    stack_target_fields,
)

__all__ = [
    "INPUT_CHANNELS",
    "PreparedJointCase1",
    "causal_exponential_smoothing",
    "prepare_joint_case1",
    "TARGET_ADAPTER_CHANNELS",
    "TARGET_INPUT_CHANNELS",
    "TARGET_OUTPUT_CHANNELS",
    "PreparedTarget2DTraining",
    "build_label_free_coarse_1d_baseline",
    "lift_homogeneous_source_input",
    "load_source_normalization_contract",
    "prepare_target_2d_training",
    "stack_target_fields",
]
