"""SSEI 2.0: Structure-first 선·색 인페인팅 (v2 설계 §3)."""
from kp3d.modules.ssei_v2.clothoid import ConnectionCurve, connect_g2
from kp3d.modules.ssei_v2.endpoints import (
    Endpoint,
    detect_break_endpoints,
    stroke_statistics,
    trace_stroke,
)
from kp3d.modules.ssei_v2.fill import ColorFillResult, fill_color
from kp3d.modules.ssei_v2.inpaint import InpaintingResult, inpaint
from kp3d.modules.ssei_v2.matching import Connection, MatchResult, match_endpoints
from kp3d.modules.ssei_v2.patchmatch import derive_patch_size, patchmatch
from kp3d.modules.ssei_v2.pool import PiecePool, build_piece_pools
from kp3d.modules.ssei_v2.render import LineCompletionResult, complete_lines

__all__ = [
    "ColorFillResult",
    "Connection",
    "ConnectionCurve",
    "Endpoint",
    "InpaintingResult",
    "LineCompletionResult",
    "MatchResult",
    "PiecePool",
    "build_piece_pools",
    "complete_lines",
    "connect_g2",
    "derive_patch_size",
    "detect_break_endpoints",
    "fill_color",
    "inpaint",
    "match_endpoints",
    "patchmatch",
    "stroke_statistics",
    "trace_stroke",
]
