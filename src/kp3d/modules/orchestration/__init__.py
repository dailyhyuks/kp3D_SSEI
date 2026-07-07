"""Stage 2 오케스트레이션 — labelme 어노테이션 기반 객체별 아모달 완성 공개 API."""
from .annotations import ObjectAnnotation, load_annotations, resolve_visibility
from .complete import ObjectCompletion, complete_object
from .graph import OcclusionEdge, build_occlusion_graph, derive_dilation_radius
from .orchestrate import OrchestrationResult, orchestrate
from .refine import refine_annotations

__all__ = [
    "ObjectAnnotation", "load_annotations", "resolve_visibility",
    "OcclusionEdge", "build_occlusion_graph", "derive_dilation_radius",
    "ObjectCompletion", "complete_object",
    "OrchestrationResult", "orchestrate",
    "refine_annotations",
]
