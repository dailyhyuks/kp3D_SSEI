"""Integrated Pipeline - Complete E2E workflow for Korean Painting 3D.

Combines all 7 modules into a single unified pipeline:
1. Restoration - Noise/degradation removal
2. Upscaling - Real-ESRGAN 4x
3. Segmentation - Object detection (annotation or auto)
4. Occlusion Detection - Find hidden regions
5. Inpainting - Restore hidden regions (V21 boundary-guided)
6. Extraction - Per-object RGBA extraction
7. 3D Reconstruction - Generate 3D mesh (TripoSR)

Usage:
    from kp3d.pipelines.integrated import IntegratedPipeline, PipelineConfig

    config = PipelineConfig(
        use_restoration=True,
        use_upscaling=True,
        inpaint_method="boundary_guided"
    )
    pipeline = IntegratedPipeline(config)

    # From annotation file
    result = pipeline.process(
        image_path="image.png",
        annotation_path="annotation.json"
    )

    # Or auto segmentation
    result = pipeline.process(
        image_path="image.png",
        text_prompts=["ceramic vase", "wooden table"]
    )
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from enum import Enum
import json
import time

import cv2
import numpy as np
import torch
from loguru import logger


class SegmentationMode(Enum):
    """Segmentation mode selection."""
    ANNOTATION = "annotation"  # Use labelme JSON annotation
    AUTO = "auto"              # SAM automatic mask generation
    TEXT = "text"              # Grounding DINO + SAM with text prompts


@dataclass
class PipelineConfig:
    """Configuration for the integrated pipeline.

    Attributes:
        # Stage toggles
        use_restoration: Enable restoration preprocessing
        use_upscaling: Enable Real-ESRGAN upscaling
        use_reconstruction: Enable 3D reconstruction

        # Restoration settings
        restoration_method: "fading_noise" or "frequency_aware"

        # Upscaling settings
        upscale_factor: Upscaling factor (2 or 4)

        # Segmentation settings
        segmentation_mode: How to segment objects
        text_prompts: Text prompts for text-based segmentation

        # Inpainting settings
        inpaint_method: Inpainting algorithm

        # Reconstruction settings
        reconstruction_model: "wonder3d", "instantmesh", or "lgm"

        # Output settings
        output_dir: Directory for saving outputs
        save_intermediates: Save intermediate results
    """
    # Stage toggles
    use_restoration: bool = False
    use_upscaling: bool = False
    use_enhancement: bool = False  # Enhancement pipeline (replaces restoration+upscaling)
    use_reconstruction: bool = False

    # Ablation toggles (for E2E ablation study)
    skip_occlusion_detection: bool = False  # Skip depth-based occlusion detection
    skip_inpainting: bool = False           # Skip inpainting of occluded regions

    # Restoration
    restoration_method: str = "fading_noise"

    # Upscaling
    upscale_factor: int = 4

    # Segmentation
    segmentation_mode: SegmentationMode = SegmentationMode.ANNOTATION
    text_prompts: List[str] = field(default_factory=lambda: ["object"])

    # Inpainting
    inpaint_method: str = "boundary_guided"

    # Reconstruction
    reconstruction_model: str = "wonder3d"

    # Output
    output_dir: str = "outputs/integrated"
    save_intermediates: bool = True

    # Device
    device: str = "cuda"


@dataclass
class PipelineResult:
    """Result from the integrated pipeline.

    Attributes:
        image: Processed image (after restoration/upscaling)
        extracted: Dict of extracted RGBA images per object
        inpainted: Dict of inpainted images per object
        meshes: Dict of 3D meshes per object (if reconstruction enabled)
        metadata: Processing metadata
    """
    image: np.ndarray
    extracted: Dict[str, np.ndarray]
    inpainted: Dict[str, np.ndarray]
    meshes: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class IntegratedPipeline:
    """Complete E2E pipeline for Korean Painting 3D reconstruction.

    This pipeline orchestrates all processing stages from raw image
    to separated objects and optionally 3D meshes.

    The pipeline is modular - each stage can be enabled/disabled
    via configuration, and intermediate results are saved for
    debugging and visualization.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        device: Optional[torch.device] = None
    ):
        """Initialize the integrated pipeline.

        Args:
            config: Pipeline configuration
            device: Compute device (auto-detected if None)
        """
        self.config = config or PipelineConfig()

        # Setup device
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() and self.config.device == "cuda"
                else "cpu"
            )
        else:
            self.device = device

        logger.info(f"IntegratedPipeline initialized on {self.device}")

        # Lazy-loaded modules
        self._restoration = None
        self._upscaling = None
        self._enhancement = None
        self._occlusion = None
        self._reconstruction = None

        # Create output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ==================== Lazy Loading ====================

    @property
    def restoration(self):
        """Lazy load restoration module."""
        if self._restoration is None and self.config.use_restoration:
            from kp3d.modules.restoration import RestorationModule
            self._restoration = RestorationModule(
                method=self.config.restoration_method,
                device=self.device
            )
            logger.info("Restoration module loaded")
        return self._restoration

    @property
    def upscaling(self):
        """Lazy load upscaling module."""
        if self._upscaling is None and self.config.use_upscaling:
            from kp3d.modules.superres import RealESRGANModule
            self._upscaling = RealESRGANModule(device=self.device)
            logger.info("Upscaling module loaded")
        return self._upscaling

    @property
    def enhancement(self):
        """Lazy load enhancement pipeline."""
        if self._enhancement is None and self.config.use_enhancement:
            from kp3d.modules.enhancement import EnhancementPipeline, EnhancementConfig
            self._enhancement = EnhancementPipeline(
                config=EnhancementConfig(),
                device=self.device
            )
            logger.info("Enhancement pipeline loaded")
        return self._enhancement

    @property
    def occlusion(self):
        """Lazy load occlusion pipeline."""
        if self._occlusion is None:
            from kp3d.modules.occlusion import OcclusionPipeline, OcclusionConfig

            occ_config = OcclusionConfig(
                inpaint_method=self.config.inpaint_method,
                skip_occlusion_detection=self.config.skip_occlusion_detection,
                skip_inpainting=self.config.skip_inpainting
            )
            self._occlusion = OcclusionPipeline(
                config=occ_config,
                output_dir=str(self.output_dir / "occlusion")
            )
            logger.info(f"Occlusion pipeline loaded (skip_occ_det={self.config.skip_occlusion_detection}, skip_inpaint={self.config.skip_inpainting})")
        return self._occlusion

    @property
    def reconstruction(self):
        """Lazy load reconstruction pipeline."""
        if self._reconstruction is None and self.config.use_reconstruction:
            from kp3d.modules.reconstruction import (
                ReconstructionPipeline,
                ReconstructionModel,
                ReconstructionConfig
            )

            model_map = {
                "wonder3d": ReconstructionModel.WONDER3D,
                "instantmesh": ReconstructionModel.INSTANTMESH,
                "lgm": ReconstructionModel.LGM
            }

            recon_config = ReconstructionConfig(
                model=model_map.get(self.config.reconstruction_model,
                                   ReconstructionModel.WONDER3D)
            )
            self._reconstruction = ReconstructionPipeline(
                model=recon_config.model,
                config=recon_config
            )
            logger.info(f"Reconstruction pipeline loaded ({self.config.reconstruction_model})")
        return self._reconstruction

    # ==================== Main Processing ====================

    def process(
        self,
        image_path: Union[str, Path],
        annotation_path: Optional[Union[str, Path]] = None,
        text_prompts: Optional[List[str]] = None,
        output_name: Optional[str] = None
    ) -> PipelineResult:
        """Process an image through the complete pipeline.

        Args:
            image_path: Path to input image
            annotation_path: Path to labelme annotation (for annotation mode)
            text_prompts: Text prompts for text-based segmentation
            output_name: Name for output files (defaults to image filename)

        Returns:
            PipelineResult with all outputs
        """
        start_time = time.time()
        image_path = Path(image_path)
        output_name = output_name or image_path.stem

        logger.info(f"Processing: {image_path.name}")

        # Create output subdirectory
        sample_dir = self.output_dir / output_name
        sample_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "input_path": str(image_path),
            "stages": []
        }

        # Load image
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise ValueError(f"Failed to load image: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        h_orig, w_orig = image_rgb.shape[:2]
        metadata["original_size"] = (w_orig, h_orig)
        logger.info(f"  Original size: {w_orig}x{h_orig}")

        # Save original
        if self.config.save_intermediates:
            cv2.imwrite(str(sample_dir / "00_original.png"), image_bgr)

        # ==================== Enhancement Pipeline (replaces Stage 1+2) ====================
        if self.config.use_enhancement and self.enhancement is not None:
            logger.info("  [Stage 1-2] Enhancement Pipeline (Upscale2x→GridRemoval→Upscale2x)...")
            stage_start = time.time()

            # Convert to tensor for enhancement
            img_tensor = torch.from_numpy(
                image_rgb.astype(np.float32) / 255.0
            ).permute(2, 0, 1).unsqueeze(0).to(self.device)

            enh_output = self.enhancement(img_tensor)

            # Convert back to numpy RGB
            result_np = enh_output.result[0].detach().cpu().numpy()
            image_rgb = (np.transpose(result_np, (1, 2, 0)) * 255.0).clip(0, 255).astype(np.uint8)

            scale = enh_output.metadata.get("effective_scale", (4.0, 4.0))[0]
            metadata["stages"].append({
                "name": "enhancement",
                "sub_stages": enh_output.metadata.get("stages", []),
                "grid_detection": enh_output.metadata.get("grid_detection", {}),
                "time": time.time() - stage_start
            })

            if self.config.save_intermediates:
                cv2.imwrite(
                    str(sample_dir / "01_enhanced.png"),
                    cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                )

        else:
            # ==================== Stage 1: Restoration ====================
            if self.config.use_restoration and self.restoration is not None:
                logger.info("  [Stage 1] Restoration...")
                stage_start = time.time()

                image_rgb = self._apply_restoration(image_rgb)

                metadata["stages"].append({
                    "name": "restoration",
                    "method": self.config.restoration_method,
                    "time": time.time() - stage_start
                })

                if self.config.save_intermediates:
                    cv2.imwrite(
                        str(sample_dir / "01_restored.png"),
                        cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                    )

            # ==================== Stage 2: Upscaling ====================
            scale = 1.0
            if self.config.use_upscaling and self.upscaling is not None:
                logger.info(f"  [Stage 2] Upscaling {self.config.upscale_factor}x...")
                stage_start = time.time()

                image_rgb, scale = self._apply_upscaling(image_rgb)

                h_up, w_up = image_rgb.shape[:2]
                metadata["stages"].append({
                    "name": "upscaling",
                    "factor": self.config.upscale_factor,
                    "output_size": (w_up, h_up),
                    "time": time.time() - stage_start
                })
                logger.info(f"    Upscaled to: {w_up}x{h_up}")

                if self.config.save_intermediates:
                    cv2.imwrite(
                        str(sample_dir / "02_upscaled.png"),
                        cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                    )

        # ==================== Stage 3-6: Occlusion Pipeline ====================
        logger.info("  [Stage 3-6] Segmentation → Occlusion → Inpainting → Extraction...")
        stage_start = time.time()

        # Determine segmentation mode
        if annotation_path is not None:
            # Scale annotation if upscaled
            result = self._process_with_annotation(
                image_rgb, annotation_path, scale, sample_dir
            )
        elif text_prompts is not None or self.config.segmentation_mode == SegmentationMode.TEXT:
            prompts = text_prompts or self.config.text_prompts
            result = self._process_with_text_prompts(
                image_rgb, prompts, sample_dir
            )
        else:
            result = self._process_auto(image_rgb, sample_dir)

        extracted = result["extracted"]
        inpainted = result["inpainted"]
        detection = result["detection"]

        metadata["stages"].append({
            "name": "occlusion_pipeline",
            "num_objects": len(extracted),
            "num_relations": len(detection.occlusion_relations) if detection else 0,
            "inpaint_method": self.config.inpaint_method,
            "time": time.time() - stage_start
        })

        # Save extracted objects
        if self.config.save_intermediates:
            for label, rgba in extracted.items():
                # RGBA
                cv2.imwrite(
                    str(sample_dir / f"06_{label}_rgba.png"),
                    cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
                )
                # Visualization on white
                h, w = rgba.shape[:2]
                vis = np.ones((h, w, 3), dtype=np.uint8) * 255
                alpha = rgba[:, :, 3:4] / 255.0
                vis = (vis * (1 - alpha) + rgba[:, :, :3] * alpha).astype(np.uint8)
                cv2.imwrite(
                    str(sample_dir / f"06_{label}_vis.png"),
                    cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
                )

            # Save debug inpaint data (before/after/mask for feather blend analysis)
            if hasattr(self.occlusion, '_debug_inpaint_data') and self.occlusion._debug_inpaint_data:
                debug_dir = sample_dir / "debug_inpaint"
                debug_dir.mkdir(exist_ok=True)

                for label, data in self.occlusion._debug_inpaint_data.items():
                    safe_label = label.replace("/", "_").replace("\\", "_")
                    cv2.imwrite(str(debug_dir / f"{safe_label}_before.png"),
                               cv2.cvtColor(data["before"], cv2.COLOR_RGB2BGR))
                    cv2.imwrite(str(debug_dir / f"{safe_label}_after.png"),
                               cv2.cvtColor(data["after"], cv2.COLOR_RGB2BGR))
                    cv2.imwrite(str(debug_dir / f"{safe_label}_mask.png"),
                               data["mask"])
                    cv2.imwrite(str(debug_dir / f"{safe_label}_occlusion_mask.png"),
                               data["occlusion_mask"])

                logger.info(f"    Debug inpaint data saved")

        # ==================== Stage 7: 3D Reconstruction ====================
        meshes = None
        if self.config.use_reconstruction and self.reconstruction is not None:
            logger.info("  [Stage 7] 3D Reconstruction...")
            stage_start = time.time()

            meshes = self._apply_reconstruction(
                extracted, sample_dir
            )

            metadata["stages"].append({
                "name": "reconstruction",
                "model": self.config.reconstruction_model,
                "num_meshes": len(meshes) if meshes else 0,
                "time": time.time() - stage_start
            })

        # ==================== Finalize ====================
        total_time = time.time() - start_time
        metadata["total_time"] = total_time
        logger.info(f"  Total time: {total_time:.2f}s")

        # Save metadata
        with open(sample_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        return PipelineResult(
            image=image_rgb,
            extracted=extracted,
            inpainted=inpainted,
            meshes=meshes,
            metadata=metadata
        )

    # ==================== Stage Implementations ====================

    def _apply_restoration(self, image: np.ndarray) -> np.ndarray:
        """Apply restoration to remove noise/degradation."""
        # Convert to tensor
        img_tensor = torch.from_numpy(image).float()
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(self.device)

        # Process
        output = self.restoration.forward(img_tensor)

        # Convert back
        result = output.result.squeeze(0).permute(1, 2, 0)
        result = (result.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

        return result

    def _apply_upscaling(self, image: np.ndarray) -> tuple:
        """Apply Real-ESRGAN upscaling."""
        from kp3d.modules.superres import ScaleFactor

        # Convert to tensor
        img_tensor = torch.from_numpy(image).float()
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(self.device)

        # Upscale
        scale_enum = ScaleFactor.X4 if self.config.upscale_factor == 4 else ScaleFactor.X2
        output = self.upscaling.forward(img_tensor, scale=scale_enum, denoise=False)

        # Convert back
        result = output.result.squeeze(0).permute(1, 2, 0)
        result = (result.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

        h_orig, w_orig = image.shape[:2]
        h_new, w_new = result.shape[:2]
        scale = w_new / w_orig

        return result, scale

    def _process_with_annotation(
        self,
        image: np.ndarray,
        annotation_path: Union[str, Path],
        scale: float,
        output_dir: Path
    ) -> Dict[str, Any]:
        """Process using annotation file."""
        annotation_path = Path(annotation_path)

        # Scale annotation coordinates if image was upscaled
        if scale != 1.0:
            with open(annotation_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            scaled_shapes = []
            for s in data.get('shapes', []):
                scaled_points = [[p[0] * scale, p[1] * scale] for p in s['points']]
                scaled_shapes.append({
                    'label': s['label'],
                    'points': scaled_points,
                    'shape_type': s.get('shape_type', 'polygon')
                })

            # Save scaled annotation
            scaled_anno_path = output_dir / "annotation_scaled.json"
            scaled_data = {**data, 'shapes': scaled_shapes}
            with open(scaled_anno_path, 'w', encoding='utf-8') as f:
                json.dump(scaled_data, f, indent=2)

            annotation_path = scaled_anno_path

        # Process with occlusion pipeline
        result = self.occlusion.process_from_annotation(
            image,
            str(annotation_path),
            save_outputs=False
        )

        return result

    def _process_with_text_prompts(
        self,
        image: np.ndarray,
        prompts: List[str],
        output_dir: Path
    ) -> Dict[str, Any]:
        """Process using text prompts (Grounding DINO + SAM)."""
        # Use occlusion pipeline's forward method
        img_tensor = torch.from_numpy(image).float()
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(self.device)

        output = self.occlusion.forward(
            img_tensor,
            text_prompts=prompts,
            save_intermediates=False
        )

        # Extract results
        h, w = image.shape[:2]

        # Convert outputs
        fg_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        fg_rgb = output.intermediate["foreground"].squeeze(0).permute(1, 2, 0)
        fg_rgb = (fg_rgb.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        fg_mask = output.intermediate["foreground_mask"].squeeze().cpu().numpy()
        fg_rgba[:, :, :3] = fg_rgb
        fg_rgba[:, :, 3] = (fg_mask * 255).astype(np.uint8)

        bg_rgb = output.result.squeeze(0).permute(1, 2, 0)
        bg_rgb = (bg_rgb.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

        return {
            "extracted": {"foreground": fg_rgba},
            "inpainted": {"background": bg_rgb},
            "detection": None
        }

    def _process_auto(
        self,
        image: np.ndarray,
        output_dir: Path
    ) -> Dict[str, Any]:
        """Process using automatic segmentation (SAM AMG)."""
        # Similar to text prompts but with auto mode
        return self._process_with_text_prompts(
            image,
            self.config.text_prompts,
            output_dir
        )

    def _apply_reconstruction(
        self,
        extracted: Dict[str, np.ndarray],
        output_dir: Path
    ) -> Dict[str, Any]:
        """Apply 3D reconstruction to extracted objects."""
        meshes = {}

        for label, rgba in extracted.items():
            if label == "background":
                continue

            logger.info(f"    Reconstructing {label}...")

            try:
                # Extract RGB and mask
                rgb = rgba[:, :, :3]
                mask = rgba[:, :, 3]

                # Reconstruct
                result = self.reconstruction.reconstruct_single(
                    image=rgb,
                    mask=mask,
                    output_path=str(output_dir / f"07_{label}.obj"),
                    inpaint=False  # Already inpainted
                )

                meshes[label] = {
                    "vertices": result.vertices,
                    "faces": result.faces,
                    "textures": result.textures,
                    "path": str(output_dir / f"07_{label}.obj")
                }

            except Exception as e:
                logger.warning(f"    Failed to reconstruct {label}: {e}")
                meshes[label] = {"error": str(e)}

        return meshes

    # ==================== Batch Processing ====================

    def process_batch(
        self,
        image_dir: Union[str, Path],
        annotation_dir: Optional[Union[str, Path]] = None,
        pattern: str = "*.png"
    ) -> Dict[str, PipelineResult]:
        """Process multiple images in batch.

        Args:
            image_dir: Directory containing images
            annotation_dir: Directory containing annotation files
            pattern: Glob pattern for image files

        Returns:
            Dict mapping image name to PipelineResult
        """
        image_dir = Path(image_dir)
        annotation_dir = Path(annotation_dir) if annotation_dir else None

        results = {}
        image_files = sorted(image_dir.glob(pattern))

        logger.info(f"Batch processing {len(image_files)} images...")

        for i, img_path in enumerate(image_files):
            logger.info(f"\n[{i+1}/{len(image_files)}] {img_path.name}")

            # Find corresponding annotation
            anno_path = None
            if annotation_dir:
                anno_path = annotation_dir / f"{img_path.stem}.json"
                if not anno_path.exists():
                    anno_path = None

            try:
                result = self.process(
                    image_path=img_path,
                    annotation_path=anno_path
                )
                results[img_path.stem] = result
            except Exception as e:
                logger.error(f"  Failed: {e}")
                results[img_path.stem] = None

        return results

    def create_comparison_grid(
        self,
        results: Dict[str, PipelineResult],
        output_path: Union[str, Path],
        cols: int = 5
    ) -> None:
        """Create a comparison grid from batch results.

        Args:
            results: Dict of pipeline results
            output_path: Path to save the grid image
            cols: Number of columns in the grid
        """
        valid_results = {k: v for k, v in results.items() if v is not None}
        if not valid_results:
            logger.warning("No valid results to create grid")
            return

        items = list(valid_results.items())
        n_items = len(items)
        rows = (n_items + cols - 1) // cols

        # Get sample dimensions
        sample = items[0][1]
        h, w = sample.image.shape[:2]

        # Scale down if needed
        max_cell = 150
        if h > max_cell or w > max_cell:
            scale = max_cell / max(h, w)
            h = int(h * scale)
            w = int(w * scale)

        margin = 5
        label_h = 20
        cell_w = w
        cell_h = h + label_h

        grid_h = rows * (cell_h + margin) + margin
        grid_w = cols * (cell_w + margin) + margin

        grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 240

        for idx, (name, result) in enumerate(items):
            row = idx // cols
            col = idx % cols

            x = col * (cell_w + margin) + margin
            y = row * (cell_h + margin) + margin

            # Resize image
            img = cv2.resize(result.image, (w, h))
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            grid[y:y+h, x:x+w] = img_bgr

            # Label
            n_objects = len(result.extracted)
            label = f"{name[:12]} ({n_objects})"
            cv2.putText(
                grid, label, (x, y + h + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 150), 1
            )

        cv2.imwrite(str(output_path), grid)
        logger.info(f"Saved comparison grid: {output_path}")


# Convenience function
def run_integrated_pipeline(
    image_path: str,
    annotation_path: Optional[str] = None,
    output_dir: str = "outputs/integrated",
    use_restoration: bool = False,
    use_upscaling: bool = False,
    use_reconstruction: bool = False,
    inpaint_method: str = "boundary_guided"
) -> PipelineResult:
    """Convenience function to run the integrated pipeline.

    Args:
        image_path: Path to input image
        annotation_path: Path to labelme annotation
        output_dir: Output directory
        use_restoration: Enable restoration
        use_upscaling: Enable upscaling
        use_reconstruction: Enable 3D reconstruction
        inpaint_method: Inpainting method

    Returns:
        PipelineResult
    """
    config = PipelineConfig(
        use_restoration=use_restoration,
        use_upscaling=use_upscaling,
        use_reconstruction=use_reconstruction,
        inpaint_method=inpaint_method,
        output_dir=output_dir
    )

    pipeline = IntegratedPipeline(config)
    return pipeline.process(image_path, annotation_path)


__all__ = [
    "IntegratedPipeline",
    "PipelineConfig",
    "PipelineResult",
    "SegmentationMode",
    "run_integrated_pipeline"
]
