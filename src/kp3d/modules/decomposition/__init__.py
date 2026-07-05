"""Stage 0: 구륵법 인지 선·색 분해 (v2 설계 섹션 1)."""
from kp3d.modules.decomposition.lines import detect_lines, measure_line_widths
from kp3d.modules.decomposition.split import recompose, split_layers
from kp3d.modules.decomposition.statistics import (
    WeavePeriodResult,
    estimate_noise_sigma,
    estimate_weave_period,
)
from kp3d.modules.decomposition.structure import compute_structure_image

__all__ = [
    "WeavePeriodResult",
    "estimate_noise_sigma",
    "estimate_weave_period",
    "compute_structure_image",
    "detect_lines",
    "measure_line_widths",
    "split_layers",
    "recompose",
]
