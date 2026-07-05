"""Paper-Aligned 4-Stage Pipeline for Korean Painting 3D Reconstruction.

This module implements the exact 4-stage pipeline described in the paper:
"Design of a 3D Reconstruction Pipeline of Occluded Objects in Korean Royal Court Paintings"

Pipeline Stages:
    Stage 1 - Restoration/De-Weaving: FFT spectral notch + spatial-adaptive NLM blending + contour enhancement
    Stage 2 - Object Segmentation: LabelMe polygon annotation + SAM refinement + per-object RGBA extraction
    Stage 3 - SSEI Inpainting: Style-consistent Self-Exemplar Inpainting using layer_order from annotation
    Stage 4 - 3D Reconstruction: InstantMesh multi-view reconstruction (default, optional)

IMPORTANT: This pipeline intentionally EXCLUDES Real-ESRGAN upscaling, as per the paper specification.
           The weave_removal module handles Stage 1 restoration directly.

Usage:
    from kp3d.pipelines.paper_pipeline import PaperPipeline, PaperPipelineConfig

    config = PaperPipelineConfig(
        weave_removal_preset="v3",
        inpaint_method="patchmatch_guided",
        use_sam_refinement=True,
        use_reconstruction=False
    )
    pipeline = PaperPipeline(config)

    result = pipeline.process(
        image_path="painting.png",
        annotation_path="painting.json"  # LabelMe annotation with layer_order
    )
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import time

import cv2
import numpy as np
from loguru import logger


@dataclass
class PaperPipelineConfig:
    """Configuration for the paper-aligned 4-stage pipeline.

    Attributes:
        # Stage 1: Weave Removal (De-Weaving/Restoration)
        weave_removal_preset: Preset for weave removal ("v3", "quality", "clean").
            - "v3" (recommended): Split Radius + NLM Adaptive + Contour Enhancement
            - "quality": Edge preservation priority (alpha=0.7)
            - "clean": Maximum grid removal (alpha=1.0)

        # Stage 2-3: Occlusion Pipeline (Segmentation + Inpainting)
        inpaint_method: Inpainting algorithm for occluded regions.
            - "patchmatch_guided" (default): V22 PatchMatch texture propagation
            - "boundary_guided": V21 boundary-guided inpainting
            - "ns": OpenCV Navier-Stokes
            - "telea": OpenCV Telea
        use_sam_refinement: Refine LabelMe polygons with SAM (default: True).
        skip_occlusion_detection: Ablation toggle - skip depth-based occlusion detection.
        skip_inpainting: Ablation toggle - skip inpainting step entirely.

        # Stage 4: 3D Reconstruction
        use_reconstruction: Enable 3D reconstruction (default: False).
        reconstruction_model: Model to use ("instantmesh" (default), "wonder3d", "lgm").

        # Output settings
        output_dir: Directory for saving outputs.
        save_intermediates: Save intermediate results per stage.
        device: Compute device ("cuda" or "cpu").
    """

    # Stage 1: Weave Removal
    weave_removal_preset: str = "v3"

    # Stage 2-3: Occlusion Pipeline
    inpaint_method: str = "patchmatch_guided"
    use_sam_refinement: bool = True
    skip_occlusion_detection: bool = False
    skip_inpainting: bool = False

    # Stage 4: Reconstruction
    use_reconstruction: bool = False
    reconstruction_model: str = "instantmesh"

    # Output
    output_dir: str = "outputs/paper_pipeline"
    save_intermediates: bool = True
    device: str = "cuda"


@dataclass
class PaperPipelineResult:
    """Result from the paper-aligned pipeline.

    Attributes:
        deweaved_image: Image after Stage 1 weave removal (RGB, numpy).
        extracted: Dict mapping object labels to RGBA images.
        inpainted: Dict mapping object labels to inpainted RGB images.
        meshes: Dict mapping object labels to 3D mesh data (if reconstruction enabled).
        metadata: Processing metadata including timing and stage info.
    """

    deweaved_image: np.ndarray
    extracted: Dict[str, np.ndarray]
    inpainted: Dict[str, np.ndarray]
    meshes: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PaperPipeline:
    """Paper-aligned 4-Stage Pipeline for Korean Painting 3D Reconstruction.

    Implements the exact pipeline from the paper, with NO upscaling (Real-ESRGAN excluded).

    Stages:
        1. Weave Removal: FFT spectral notch + NLM adaptive + contour enhancement
        2. Segmentation: LabelMe polygons + SAM refinement
        3. SSEI Inpainting: Style-consistent inpainting of occluded regions
        4. 3D Reconstruction: Optional InstantMesh reconstruction

    The pipeline reuses existing modules rather than reimplementing algorithms:
        - Stage 1: kp3d.modules.weave_removal.WeaveRemovalModule
        - Stages 2-3: kp3d.modules.occlusion.OcclusionPipeline
        - Stage 4: kp3d.modules.reconstruction.ReconstructionPipeline
    """

    def __init__(
        self,
        config: Optional[PaperPipelineConfig] = None,
    ):
        """Initialize the paper-aligned pipeline.

        Args:
            config: Pipeline configuration. Uses defaults if None.
        """
        self.config = config or PaperPipelineConfig()

        # Setup output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Lazy-loaded modules (avoid import-time heavy dependencies)
        self._weave_removal = None
        self._occlusion = None
        self._reconstruction = None

        # Determine device
        self._device = None

        logger.info(f"PaperPipeline initialized (output_dir={self.output_dir})")

    @property
    def device(self):
        """Lazy compute device determination."""
        if self._device is None:
            import torch
            if self.config.device == "cuda" and torch.cuda.is_available():
                self._device = torch.device("cuda")
            else:
                self._device = torch.device("cpu")
        return self._device

    @property
    def weave_removal(self):
        """Lazy-load Stage 1: Weave Removal module."""
        if self._weave_removal is None:
            from kp3d.modules.weave_removal import (
                WeaveRemovalModule,
                WeaveRemovalPreset,
            )

            preset_map = {
                "v3": WeaveRemovalPreset.V3,
                "quality": WeaveRemovalPreset.QUALITY,
                "clean": WeaveRemovalPreset.CLEAN,
            }
            preset = preset_map.get(
                self.config.weave_removal_preset, WeaveRemovalPreset.V3
            )

            self._weave_removal = WeaveRemovalModule(
                preset.to_config(), device=self.device
            )
            logger.info(
                f"Stage 1: WeaveRemovalModule loaded (preset={self.config.weave_removal_preset})"
            )
        return self._weave_removal

    @property
    def occlusion(self):
        """Lazy-load Stages 2-3: Occlusion Pipeline (Segmentation + Inpainting)."""
        if self._occlusion is None:
            from kp3d.modules.occlusion import OcclusionPipeline, OcclusionConfig

            occ_config = OcclusionConfig(
                inpaint_method=self.config.inpaint_method,
                use_sam_refinement=self.config.use_sam_refinement,
                skip_occlusion_detection=self.config.skip_occlusion_detection,
                skip_inpainting=self.config.skip_inpainting,
            )

            self._occlusion = OcclusionPipeline(
                config=occ_config,
                output_dir=str(self.output_dir / "occlusion"),
            )
            logger.info(
                f"Stages 2-3: OcclusionPipeline loaded (inpaint={self.config.inpaint_method})"
            )
        return self._occlusion

    @property
    def reconstruction(self):
        """Lazy-load Stage 4: 3D Reconstruction Pipeline."""
        if self._reconstruction is None and self.config.use_reconstruction:
            from kp3d.modules.reconstruction import (
                ReconstructionPipeline,
                ReconstructionModel,
                ReconstructionConfig,
            )

            model_map = {
                "wonder3d": ReconstructionModel.WONDER3D,
                "instantmesh": ReconstructionModel.INSTANTMESH,
                "lgm": ReconstructionModel.LGM,
            }
            model = model_map.get(
                self.config.reconstruction_model, ReconstructionModel.INSTANTMESH
            )

            recon_config = ReconstructionConfig(model=model)
            self._reconstruction = ReconstructionPipeline(
                model=model, config=recon_config
            )
            logger.info(
                f"Stage 4: ReconstructionPipeline loaded (model={self.config.reconstruction_model})"
            )
        return self._reconstruction

    def process(
        self,
        image_path: Union[str, Path],
        annotation_path: Union[str, Path],
        output_name: Optional[str] = None,
    ) -> PaperPipelineResult:
        """Process an image through the 4-stage paper-aligned pipeline.

        Args:
            image_path: Path to input image (PNG/JPG).
            annotation_path: Path to LabelMe JSON annotation with layer_order.
            output_name: Name for output directory (defaults to image filename stem).

        Returns:
            PaperPipelineResult with deweaved image, extracted RGBAs, and optionally meshes.

        Raises:
            ValueError: If image or annotation cannot be loaded.
        """
        start_time = time.time()
        image_path = Path(image_path)
        annotation_path = Path(annotation_path)
        output_name = output_name or image_path.stem

        logger.info(f"Processing: {image_path.name}")

        # Create output subdirectory for this sample
        sample_dir = self.output_dir / output_name
        sample_dir.mkdir(parents=True, exist_ok=True)

        metadata: Dict[str, Any] = {
            "input_image": str(image_path),
            "input_annotation": str(annotation_path),
            "stages": [],
        }

        # Load input image (BGR -> RGB)
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise ValueError(f"Failed to load image: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        h_orig, w_orig = image_rgb.shape[:2]
        metadata["original_size"] = (w_orig, h_orig)
        logger.info(f"  Original size: {w_orig}x{h_orig}")

        # Save original for reference
        if self.config.save_intermediates:
            cv2.imwrite(str(sample_dir / "00_original.png"), image_bgr)

        # ==================== Stage 1: Weave Removal (De-Weaving) ====================
        logger.info("  [Stage 1] Weave Removal (FFT + NLM Adaptive + Contour)...")
        stage1_start = time.time()

        # WeaveRemovalModule.process_bgr expects BGR uint8 and returns BGR uint8
        deweaved_bgr, confidence = self.weave_removal.process_bgr(image_bgr)
        deweaved_rgb = cv2.cvtColor(deweaved_bgr, cv2.COLOR_BGR2RGB)

        stage1_time = time.time() - stage1_start
        metadata["stages"].append(
            {
                "name": "weave_removal",
                "preset": self.config.weave_removal_preset,
                "mean_confidence": float(np.mean(confidence)),
                "time": stage1_time,
            }
        )
        logger.info(f"    Stage 1 complete ({stage1_time:.2f}s)")

        if self.config.save_intermediates:
            cv2.imwrite(str(sample_dir / "01_deweaved.png"), deweaved_bgr)

        # ==================== Stages 2-3: Segmentation + Occlusion + Inpainting ====================
        logger.info("  [Stage 2-3] Segmentation + Occlusion Detection + SSEI Inpainting...")
        stage23_start = time.time()

        # OcclusionPipeline.process_from_annotation expects RGB numpy
        occ_result = self.occlusion.process_from_annotation(
            deweaved_rgb,
            str(annotation_path),
            save_outputs=self.config.save_intermediates,
        )

        extracted = occ_result.get("extracted", {})
        inpainted = occ_result.get("inpainted", {})
        detection = occ_result.get("detection")

        stage23_time = time.time() - stage23_start
        metadata["stages"].append(
            {
                "name": "occlusion_pipeline",
                "num_objects": len(extracted),
                "num_relations": len(detection.occlusion_relations) if detection else 0,
                "inpaint_method": self.config.inpaint_method,
                "time": stage23_time,
            }
        )
        logger.info(
            f"    Stage 2-3 complete: {len(extracted)} objects, {len(detection.occlusion_relations) if detection else 0} occlusion relations ({stage23_time:.2f}s)"
        )

        # Save per-object RGBA extractions
        if self.config.save_intermediates:
            for label, rgba in extracted.items():
                # Save RGBA as BGRA for cv2
                cv2.imwrite(
                    str(sample_dir / f"02_{label}_rgba.png"),
                    cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA),
                )
                # Also save visualization on white background
                h, w = rgba.shape[:2]
                vis = np.ones((h, w, 3), dtype=np.uint8) * 255
                alpha = rgba[:, :, 3:4] / 255.0
                vis = (vis * (1 - alpha) + rgba[:, :, :3] * alpha).astype(np.uint8)
                cv2.imwrite(
                    str(sample_dir / f"02_{label}_vis.png"),
                    cv2.cvtColor(vis, cv2.COLOR_RGB2BGR),
                )

        # ==================== Stage 4: 3D Reconstruction (Optional) ====================
        meshes: Optional[Dict[str, Any]] = None
        if self.config.use_reconstruction and self.reconstruction is not None:
            logger.info(f"  [Stage 4] 3D Reconstruction ({self.config.reconstruction_model})...")
            stage4_start = time.time()

            meshes = self._apply_reconstruction(extracted, sample_dir)

            stage4_time = time.time() - stage4_start
            metadata["stages"].append(
                {
                    "name": "reconstruction",
                    "model": self.config.reconstruction_model,
                    "num_meshes": len(meshes) if meshes else 0,
                    "time": stage4_time,
                }
            )
            logger.info(f"    Stage 4 complete ({stage4_time:.2f}s)")

        # ==================== Finalize ====================
        total_time = time.time() - start_time
        metadata["total_time"] = total_time
        logger.info(f"  Total time: {total_time:.2f}s")

        # Save metadata
        with open(sample_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)

        return PaperPipelineResult(
            deweaved_image=deweaved_rgb,
            extracted=extracted,
            inpainted=inpainted,
            meshes=meshes,
            metadata=metadata,
        )

    def _apply_reconstruction(
        self,
        extracted: Dict[str, np.ndarray],
        output_dir: Path,
    ) -> Dict[str, Any]:
        """Apply 3D reconstruction to extracted RGBA objects.

        Args:
            extracted: Dict mapping label to RGBA image.
            output_dir: Directory to save mesh files.

        Returns:
            Dict mapping label to mesh data (vertices, faces, path, or error).
        """
        meshes: Dict[str, Any] = {}

        for label, rgba in extracted.items():
            # Skip background objects
            if label.lower() in ("background", "bg"):
                continue

            logger.info(f"    Reconstructing: {label}")

            try:
                rgb = rgba[:, :, :3]
                mask = rgba[:, :, 3]

                result = self.reconstruction.reconstruct_single(
                    image=rgb,
                    mask=mask,
                    output_path=str(output_dir / f"03_{label}.obj"),
                    inpaint=False,  # Already inpainted in Stage 3
                )

                meshes[label] = {
                    "vertices": result.vertices,
                    "faces": result.faces,
                    "textures": result.textures,
                    "path": str(output_dir / f"03_{label}.obj"),
                }

            except Exception as e:
                logger.warning(f"    Failed to reconstruct {label}: {e}")
                meshes[label] = {"error": str(e)}

        return meshes


def run(
    image_path: str,
    annotation_path: str,
    output_dir: str = "outputs/paper_pipeline",
    weave_removal_preset: str = "v3",
    inpaint_method: str = "patchmatch_guided",
    use_reconstruction: bool = False,
    reconstruction_model: str = "instantmesh",
) -> PaperPipelineResult:
    """Convenience function to run the paper-aligned pipeline.

    This function provides a simple interface for running the full 4-stage pipeline
    without needing to manually create config and pipeline objects.

    Args:
        image_path: Path to input image.
        annotation_path: Path to LabelMe JSON annotation.
        output_dir: Directory for saving outputs.
        weave_removal_preset: Weave removal preset ("v3", "quality", "clean").
        inpaint_method: Inpainting method for occluded regions.
        use_reconstruction: Enable Stage 4 3D reconstruction.
        reconstruction_model: 3D reconstruction model to use.

    Returns:
        PaperPipelineResult with all outputs.

    Example:
        >>> result = run(
        ...     image_path="data/paintings/court_scene.png",
        ...     annotation_path="data/annotations/court_scene.json",
        ...     use_reconstruction=True
        ... )
        >>> print(f"Extracted {len(result.extracted)} objects")
    """
    config = PaperPipelineConfig(
        weave_removal_preset=weave_removal_preset,
        inpaint_method=inpaint_method,
        use_reconstruction=use_reconstruction,
        reconstruction_model=reconstruction_model,
        output_dir=output_dir,
    )

    pipeline = PaperPipeline(config)
    return pipeline.process(image_path, annotation_path)


__all__ = [
    "PaperPipeline",
    "PaperPipelineConfig",
    "PaperPipelineResult",
    "run",
]
