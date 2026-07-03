"""Unified occlusion inpainting pipeline.

Integrates all stages:
1. Restoration (noise/artifact removal)
2. Upscaling (Real-ESRGAN 4x)
3. Segmentation (Grounding DINO + SAM2 or annotation-based)
4. Depth estimation (MiDaS)
5. Layer ordering
6. Occlusion detection
7. Inpainting (OpenCV NS, LaMa, ControlNet)
8. Per-object RGBA extraction
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
import json
import numpy as np
import torch
from torch import Tensor
import cv2
import time

from kp3d.core.base import ModuleOutput
from kp3d.modules.occlusion.base import (
    BaseOcclusion,
    OcclusionConfig,
    OcclusionResult,
    LayerInfo
)
from kp3d.modules.occlusion.segmentation import SegmentationModule
from kp3d.modules.occlusion.auto_segmentation import AutoSegmentation
from kp3d.modules.occlusion.depth import DepthEstimatorWrapper
from kp3d.modules.occlusion.layer_ordering import SimpleLayerOrdering
from kp3d.modules.occlusion.occlusion_detection import (
    OcclusionDetector,
    detect_true_occlusion,
    estimate_table_surface,
    LayeredOcclusionDetector,
    LayeredOcclusionResult,
    OcclusionRelation,
)
from kp3d.modules.occlusion.inpainting import (
    InpaintingModule,
    inpaint_occlusion_boundary_guided,
    inpaint_occlusion_patchmatch_guided,
    inpaint_occlusion_patchmatch_v25,
    get_intersection_edge,
)
from kp3d.modules.occlusion.mask_refinement import MaskRefiner
from kp3d.modules.occlusion.auto_mask_refinement import AutoMaskRefiner
from kp3d.modules.occlusion.hybrid_inpainter import HybridInpainter


class OcclusionPipeline(BaseOcclusion):
    """Complete pipeline for occlusion-aware object separation.

    Processes images with overlapping objects to:
    1. Segment individual objects
    2. Determine depth ordering
    3. Detect occluded regions
    4. Inpaint hidden areas
    5. Generate separated object images

    Designed for 3D reconstruction of traditional paintings with
    overlapping elements (e.g., ceramic vase on wooden table).
    """

    def __init__(
        self,
        config: Optional[OcclusionConfig] = None,
        output_dir: Optional[str] = None,
        **kwargs
    ):
        """Initialize occlusion pipeline.

        Args:
            config: Pipeline configuration.
            output_dir: Directory for saving outputs.
            **kwargs: Additional arguments for BaseOcclusion.
        """
        super().__init__(config=config, **kwargs)

        self.output_dir = Path(output_dir) if output_dir else Path("outputs/occlusion")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize sub-modules (lazy loading)
        self._segmentation = None
        self._auto_segmentation = None
        self._depth = None
        self._layer_ordering = None
        self._occlusion_detector = None
        self._inpainting = None
        self._mask_refiner = None
        self._auto_mask_refiner = None
        self._hybrid_inpainter = None
        self._multi_layer_detector = None
        self._sam_refiner = None

        # Debug: store inpainting intermediate data for analysis
        # Keys: occludee_label -> {"before": image, "after": image, "mask": mask}
        self._debug_inpaint_data = {}

    @property
    def segmentation(self) -> SegmentationModule:
        """Lazy-load text-based segmentation module."""
        if self._segmentation is None:
            self._segmentation = SegmentationModule(
                sam_model_type=self.config.sam_model_type,
                box_threshold=self.config.box_threshold,
                text_threshold=self.config.text_threshold,
                device=self.device
            )
        return self._segmentation

    @property
    def auto_segmentation(self) -> AutoSegmentation:
        """Lazy-load automatic segmentation module (SAM AMG)."""
        if self._auto_segmentation is None:
            self._auto_segmentation = AutoSegmentation(
                model_type=self.config.sam_model_type,
                points_per_side=self.config.auto_seg_points_per_side,
                pred_iou_thresh=self.config.auto_seg_pred_iou_thresh,
                stability_score_thresh=self.config.auto_seg_stability_thresh,
                min_mask_region_area=self.config.auto_seg_min_mask_area,
                device=self.device
            )
        return self._auto_segmentation

    @property
    def depth(self) -> DepthEstimatorWrapper:
        """Lazy-load depth estimation module."""
        if self._depth is None:
            self._depth = DepthEstimatorWrapper(
                model_type=self.config.depth_model,
                device=self.device
            )
        return self._depth

    @property
    def layer_ordering(self) -> SimpleLayerOrdering:
        """Lazy-load layer ordering module."""
        if self._layer_ordering is None:
            self._layer_ordering = SimpleLayerOrdering()
        return self._layer_ordering

    @property
    def occlusion_detector(self) -> OcclusionDetector:
        """Lazy-load occlusion detection module."""
        if self._occlusion_detector is None:
            self._occlusion_detector = OcclusionDetector(
                dilation_kernel_size=self.config.dilation_kernel_size,
                dilation_iterations=self.config.dilation_iterations,
                use_convex_hull=self.config.use_convex_hull
            )
        return self._occlusion_detector

    @property
    def inpainting(self) -> InpaintingModule:
        """Lazy-load inpainting module."""
        if self._inpainting is None:
            self._inpainting = InpaintingModule(
                method=self.config.inpaint_method,
                radius=self.config.inpaint_radius
            )
        return self._inpainting

    @property
    def mask_refiner(self) -> MaskRefiner:
        """Lazy-load mask refinement module."""
        if self._mask_refiner is None:
            self._mask_refiner = MaskRefiner()
        return self._mask_refiner

    @property
    def auto_mask_refiner(self) -> AutoMaskRefiner:
        """Lazy-load auto mask refinement module."""
        if self._auto_mask_refiner is None:
            self._auto_mask_refiner = AutoMaskRefiner(
                n_colors=self.config.auto_refine_n_colors,
                color_tolerance=self.config.auto_refine_tolerance,
                min_cluster_ratio=self.config.auto_refine_min_cluster_ratio
            )
        return self._auto_mask_refiner

    @property
    def hybrid_inpainter(self) -> HybridInpainter:
        """Lazy-load hybrid inpainting module."""
        if self._hybrid_inpainter is None:
            self._hybrid_inpainter = HybridInpainter(
                enable_symmetry=self.config.enable_symmetry_inpainting,
                enable_diffusion=self.config.enable_reference_guided,
                fallback_to_lama=True
            )
        return self._hybrid_inpainter

    @property
    def layered_detector(self) -> LayeredOcclusionDetector:
        """Lazy-load layered occlusion detector for multi-layer scenes."""
        if self._multi_layer_detector is None:
            self._multi_layer_detector = LayeredOcclusionDetector(
                dilation_kernel_size=self.config.dilation_kernel_size,
                use_convex_hull=self.config.use_convex_hull
            )
        return self._multi_layer_detector

    def _get_sam_refiner(self):
        """Lazy-init SAM mask refiner."""
        if self._sam_refiner is None:
            from kp3d.modules.occlusion.sam_mask_refiner import SAMMaskRefiner
            seg_module = SegmentationModule(
                sam_model_type=self.config.sam_model_type,
                device=self.device,
            )
            try:
                seg_module._load_sam2()
            except ImportError:
                seg_module._load_sam1_fallback()
            self._sam_refiner = SAMMaskRefiner(
                sam_predictor=seg_module._sam_predictor,
                margin_px=self.config.sam_refine_margin_px,
                min_area_ratio=self.config.sam_refine_min_area_ratio,
                erode_px=self.config.sam_refine_erode_px,
                dilate_px=self.config.sam_refine_dilate_px,
                adaptive=self.config.sam_refine_adaptive,
            )
        return self._sam_refiner

    def process_full_pipeline(
        self,
        image: np.ndarray,
        annotation_path: str,
        upscale: bool = True,
        upscale_factor: int = 4,
        restore: bool = False,
        save_outputs: bool = True
    ) -> Dict[str, Any]:
        """Full pipeline: Upscaling → Detection → Inpainting → Extraction.

        Complete workflow for processing annotated images:
        1. (Optional) Restoration - noise/artifact removal
        2. (Optional) Upscaling - Real-ESRGAN 4x for better quality
        3. Annotation coordinate scaling (if upscaled)
        4. Occlusion detection using LayeredOcclusionDetector
        5. Per-object inpainting (OpenCV NS by default)
        6. RGBA extraction with inpainted regions

        Args:
            image: RGB image (H, W, 3)
            annotation_path: Path to labelme JSON annotation
            upscale: Whether to upscale image (default: True)
            upscale_factor: Upscaling factor (default: 4)
            restore: Whether to apply restoration first (default: False)
            save_outputs: Save intermediate results (default: True)

        Returns:
            Dict with:
            - 'image': Processed image (upscaled if enabled)
            - 'detection': LayeredOcclusionResult
            - 'extracted': Dict[label, RGBA image]
            - 'inpainted': Dict[label, inpainted RGB image]
            - 'metadata': Processing metadata (timing, sizes, etc.)
        """
        start_time = time.time()
        metadata = {
            'upscaled': upscale,
            'restored': restore,
            'original_size': (image.shape[1], image.shape[0]),
        }

        # Load annotation
        with open(annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        shapes = data.get('shapes', [])

        if not shapes:
            raise ValueError(f"No shapes found in annotation: {annotation_path}")

        h_orig, w_orig = image.shape[:2]
        processed_image = image.copy()

        print(f"[Full Pipeline] Processing {Path(annotation_path).stem}")
        print(f"  Original size: {w_orig}x{h_orig}")
        print(f"  Objects: {[s['label'] for s in shapes]}")

        # Step 1: Restoration (optional)
        if restore:
            print("\n  [Step 1] Applying restoration...")
            try:
                from kp3d.modules.restoration import RestorationModule
                restorer = RestorationModule(method='fading_noise', device=self.device)

                # Convert numpy (H,W,3) uint8 -> tensor (1,3,H,W) float [0,1]
                img_tensor = torch.from_numpy(processed_image).float()
                img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0) / 255.0

                result = restorer.forward(img_tensor)

                # Convert back to numpy
                restored_tensor = result.result.squeeze(0).permute(1, 2, 0)
                processed_image = (restored_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

                metadata['restoration_applied'] = True
                print(f"    Restoration applied successfully")
            except Exception as e:
                print(f"    Warning: Restoration failed: {e}")
                metadata['restoration_applied'] = False

        # Step 2: Upscaling (optional)
        scale = 1.0
        if upscale:
            print(f"\n  [Step 2] Upscaling {upscale_factor}x...")
            try:
                from kp3d.modules.superres.real_esrgan import RealESRGANModule, ScaleFactor

                upscaler = RealESRGANModule(device=self.device)

                # Convert to tensor
                img_tensor = torch.from_numpy(processed_image).float()
                img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0) / 255.0

                # Upscale
                scale_enum = ScaleFactor.X4 if upscale_factor == 4 else ScaleFactor.X2
                result = upscaler.forward(img_tensor, scale=scale_enum, denoise=False)

                # Convert back to numpy
                upscaled_tensor = result.result.squeeze(0).permute(1, 2, 0)
                processed_image = (upscaled_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

                h_up, w_up = processed_image.shape[:2]
                scale = w_up / w_orig
                print(f"    Upscaled to: {w_up}x{h_up}")
                metadata['upscaled_size'] = (w_up, h_up)
                metadata['scale_factor'] = scale

            except Exception as e:
                print(f"    Warning: Upscaling failed: {e}")
                scale = 1.0
                metadata['upscaled'] = False

        # Step 3: Scale annotation coordinates
        if scale != 1.0:
            print(f"\n  [Step 3] Scaling annotations (factor: {scale})...")
            scaled_shapes = []
            for s in shapes:
                scaled_points = [[p[0] * scale, p[1] * scale] for p in s['points']]
                scaled_shapes.append({
                    'label': s['label'],
                    'points': scaled_points,
                    'shape_type': s.get('shape_type', 'polygon')
                })
            shapes = scaled_shapes

        # Step 3.5: SAM Mask Refinement (optional)
        # Strategy: original polygons for inpainting (wide boundary = smooth blending),
        #           SAM-refined polygons for extraction (tight boundary = clean edges)
        detection_for_inpaint = None
        if self.config.use_sam_refinement:
            # Step 3.5a: Detect occlusion on ORIGINAL polygons
            # This detection will be used for inpainting (wider masks = better blending)
            h_pre, w_pre = processed_image.shape[:2]
            print(f"\n  [Step 3.5a] Pre-detecting occlusion (original polygons)...")
            detection_for_inpaint = self.layered_detector.detect_from_shapes(shapes, (h_pre, w_pre))

            # Build occludee_label -> list of occluder LABELS for SAM refiner
            occlusion_labels = {}
            for rel in detection_for_inpaint.occlusion_relations:
                lbl = rel.occludee_label
                if lbl not in occlusion_labels:
                    occlusion_labels[lbl] = []
                if rel.occluder_label not in occlusion_labels[lbl]:
                    occlusion_labels[lbl].append(rel.occluder_label)

            if occlusion_labels:
                for lbl, occluders in occlusion_labels.items():
                    print(f"    {lbl}: occluded by {occluders}")
            else:
                print(f"    No occlusion relationships found")

            # Step 3.5b: 2-pass SAM refinement
            print(f"\n  [Step 3.5b] Refining masks with SAM (2-pass)...")
            refiner = self._get_sam_refiner()
            shapes = refiner.refine_all_shapes(
                processed_image, shapes,
                occlusion_labels=occlusion_labels if occlusion_labels else None,
            )
            metadata['sam_refinement_applied'] = True

        # Step 3.6: Segment-Aware Restoration (per-object)
        if self.config.use_segment_aware_restoration:
            print(f"\n  [Step 3.6] Applying segment-aware restoration...")
            try:
                from kp3d.modules.restoration_v2 import (
                    SegmentAwareRestorer,
                    SegmentAwareRestorationConfig,
                    ObjectMask,
                )
                sar_config = SegmentAwareRestorationConfig(
                    **(self.config.segment_aware_restoration_config or {})
                )
                restorer = SegmentAwareRestorer(config=sar_config, device=self.device)
                object_masks = self._build_masks_from_shapes(
                    shapes, processed_image.shape[:2]
                )
                sar_result = restorer.restore_all_objects(processed_image, object_masks)
                processed_image = sar_result.restored_image
                metadata['segment_aware_restoration'] = {
                    'objects_processed': sar_result.objects_processed,
                    'objects_skipped': sar_result.objects_skipped,
                    'per_object_times': sar_result.per_object_times,
                }
                print(f"    Restored {sar_result.objects_processed} objects")
            except Exception as e:
                print(f"    Warning: Segment-aware restoration failed: {e}")
                metadata['segment_aware_restoration'] = {'error': str(e)}

        # Step 4: Detect occlusion (with refined shapes if SAM was used)
        h, w = processed_image.shape[:2]
        print(f"\n  [Step 4] Detecting occlusion relations...")
        detection = self.layered_detector.detect_from_shapes(shapes, (h, w))

        print(f"    Layers: {list(detection.layer_masks.keys())}")
        print(f"    Relations: {len(detection.occlusion_relations)}")
        for rel in detection.occlusion_relations:
            print(f"      {rel.occluder_label} → {rel.occludee_label} ({rel.occlusion_ratio:.1%})")

        metadata['num_layers'] = len(detection.layer_masks)
        metadata['num_relations'] = len(detection.occlusion_relations)

        # Step 5: Inpaint occluded objects
        # Toggle: use_refined_mask_for_inpaint determines which detection drives inpaint mask
        if self.config.use_sam_refinement and self.config.use_refined_mask_for_inpaint:
            inpaint_detection = detection  # SAM-refined tight mask
            print(f"\n  [Step 5] Inpainting with SAM-refined (tight) masks"
                  f"{' + ' + str(self.config.inpaint_mask_dilate_px) + 'px dilate' if self.config.inpaint_mask_dilate_px > 0 else ''}...")
        else:
            inpaint_detection = detection_for_inpaint if detection_for_inpaint is not None else detection
            print(f"\n  [Step 5] Inpainting with original (wide) masks...")
        inpaint_start = time.time()
        inpainted_objects = self._inpaint_all_occluded(processed_image, inpaint_detection)
        metadata['inpaint_time'] = time.time() - inpaint_start

        # Step 6: Extract all objects as RGBA
        # Base alpha: SAM-refined detection (tighter masks = cleaner edges)
        # Occlusion extension: effective inpaint_detection (respects use_refined_mask_for_inpaint toggle)
        print(f"\n  [Step 6] Extracting objects as RGBA...")
        extracted_objects = self._extract_all_objects_with_occlusion(
            processed_image, detection, inpainted_objects,
            inpaint_detection=inpaint_detection,  # Use the effective inpaint_detection (respects toggle)
        )

        # Save outputs
        if save_outputs:
            print(f"\n  [Step 7] Saving results...")
            self._save_full_pipeline_results(
                processed_image, extracted_objects, inpainted_objects,
                detection, Path(annotation_path).stem
            )

        metadata['total_time'] = time.time() - start_time
        print(f"\n  Total time: {metadata['total_time']:.2f}s")

        return {
            'image': processed_image,
            'detection': detection,
            'extracted': extracted_objects,
            'inpainted': inpainted_objects,
            'metadata': metadata,
        }

    def _build_masks_from_shapes(
        self,
        shapes: List[Dict[str, Any]],
        image_shape: Tuple[int, int],
    ) -> List:
        """Build ObjectMask list from annotation shapes for segment-aware restoration.

        Args:
            shapes: List of shape dicts with 'label', 'points', 'shape_type'.
            image_shape: (H, W) of the target image.

        Returns:
            List of ObjectMask instances.
        """
        from kp3d.modules.restoration_v2 import ObjectMask

        h, w = image_shape
        object_masks = []

        # Check if there's a "background" label
        background_labels = {"background", "bg", "배경"}

        for shape in shapes:
            label = shape.get('label', 'unknown')
            points = shape.get('points', [])

            if not points:
                continue

            # Create binary mask from polygon points
            pts = np.array(points, dtype=np.int32)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 255)

            is_bg = label.lower() in background_labels
            object_masks.append(ObjectMask(
                label=label,
                mask=mask,
                is_background=is_bg,
            ))

        return object_masks

    @staticmethod
    def _fill_interior_holes(mask: np.ndarray) -> np.ndarray:
        """Fill holes fully enclosed by the mask (not connected to image border).

        Interior 0-regions completely surrounded by the object are filled to
        255, while 0-regions connected to the image border (true background
        and edge notches) are preserved.  This removes the white-gap artifacts
        that can appear when the combined object/occlusion mask leaves small
        enclosed pockets unfilled (e.g. after SAM refinement on upscaled
        images where the refined polygon and occlusion mask disagree by a few
        pixels).

        Args:
            mask: Binary mask (H, W), uint8, values 0 or 255.

        Returns:
            Binary mask (H, W), uint8, with interior holes filled.
        """
        mask_bin = (mask > 0).astype(np.uint8)
        h, w = mask_bin.shape

        # Pad with a 1px border of background so the (0, 0) seed is guaranteed
        # to land on a background pixel even if the object touches the border.
        padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
        padded[1:-1, 1:-1] = mask_bin
        inv = (1 - padded).astype(np.uint8)  # background = 1 (incl. pad border)

        # Flood background connected to the border, marking it 0.  Any
        # background (value 1) that remains is fully enclosed by the object.
        ff = np.zeros((h + 4, w + 4), dtype=np.uint8)
        cv2.floodFill(inv, ff, (0, 0), 0)  # seed on background -> border bg = 0

        enclosed = inv[1:-1, 1:-1] > 0  # remaining bg = interior holes
        filled = mask_bin.copy()
        filled[enclosed] = 1
        return (filled * 255).astype(np.uint8)

    @staticmethod
    def _feather_mask_edge(mask: np.ndarray, radius: int = 1) -> np.ndarray:
        """Apply alpha feathering at binary mask boundary.

        Converts hard 0/255 mask edges into a smooth gradient by applying
        a distance-based falloff at the boundary.  This prevents hard
        aliasing artifacts when the extraction is composited on any
        background.

        Args:
            mask: Binary mask (H, W), uint8, values 0 or 255.
            radius: Feathering radius in pixels (1-2 recommended).

        Returns:
            Alpha channel (H, W), uint8, with gradient at boundary.
        """
        if radius <= 0:
            return mask

        # Distance transform from mask boundary (inside)
        mask_bin = (mask > 0).astype(np.uint8)
        dist = cv2.distanceTransform(mask_bin, cv2.DIST_L2, 3)

        # Create feathered alpha: pixels within `radius` of edge get gradual alpha
        alpha = np.where(dist >= radius, 255.0, (dist / radius) * 255.0)
        return alpha.clip(0, 255).astype(np.uint8)

    def _extract_all_objects_with_occlusion(
        self,
        image: np.ndarray,
        detection: LayeredOcclusionResult,
        inpainted_objects: Dict[str, np.ndarray],
        inpaint_detection: LayeredOcclusionResult = None,
    ) -> Dict[str, np.ndarray]:
        """Extract all objects as RGBA, including inpainted occlusion regions.

        For occluded objects, the mask includes both the original object area
        AND the inpainted occlusion region for complete extraction.

        IMPORTANT: Subtracts background_mask from each object mask to prevent
        background regions from being included in object extraction.

        Args:
            image: Processed RGB image
            detection: Occlusion detection result (SAM-refined masks for base alpha)
            inpainted_objects: Dict of inpainted images for occluded objects
            inpaint_detection: Optional separate detection used for inpainting.
                               When provided, its occlusion masks are used to extend
                               the extraction region so all inpainted pixels are included.

        Returns:
            Dict mapping label to RGBA image
        """
        h, w = image.shape[:2]
        extracted = {}

        # Build mapping of occludee -> combined occlusion mask
        # Use inpaint_detection if provided (wider masks that match actual inpainted area)
        occ_source = inpaint_detection if inpaint_detection is not None else detection
        occludee_occlusion_masks = defaultdict(lambda: np.zeros((h, w), dtype=np.uint8))
        for rel in occ_source.occlusion_relations:
            occludee_occlusion_masks[rel.occludee_label] = np.maximum(
                occludee_occlusion_masks[rel.occludee_label],
                rel.occlusion_mask
            )

        # Get background mask to subtract from object masks
        background_mask = detection.background_mask if detection.background_mask is not None else np.zeros((h, w), dtype=np.uint8)

        for label, masks in detection.layer_masks.items():
            if label == 'background':
                continue

            # Combine masks for this label
            combined_mask = np.zeros((h, w), dtype=np.uint8)
            for m in masks:
                combined_mask = np.maximum(combined_mask, m)

            # Subtract background mask from object mask
            # This prevents background regions from being included in object extraction
            if np.sum(background_mask) > 0:
                combined_mask = cv2.bitwise_and(
                    combined_mask,
                    cv2.bitwise_not(background_mask)
                )

            # Choose source image and extend mask for occluded objects
            if label in inpainted_objects:
                source_image = inpainted_objects[label]
                # Include occlusion region in mask for complete extraction
                occlusion_mask = occludee_occlusion_masks[label]
                full_mask = np.maximum(combined_mask, (occlusion_mask > 0).astype(np.uint8) * 255)
                # Also subtract background from full_mask
                if np.sum(background_mask) > 0:
                    full_mask = cv2.bitwise_and(
                        full_mask,
                        cv2.bitwise_not(background_mask)
                    )
            else:
                source_image = image
                full_mask = combined_mask

            # Fill interior holes (enclosed gaps) so occluded/refined masks
            # do not leave white pockets inside the extracted object.
            full_mask = self._fill_interior_holes(full_mask)

            # Apply alpha feathering at mask boundary for smooth edges
            alpha = self._feather_mask_edge(full_mask)

            # Extract as RGBA
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[:, :, :3] = source_image
            rgba[:, :, 3] = alpha

            extracted[label] = rgba
            print(f"    {label}: {np.sum(alpha > 0)} pixels")

        return extracted

    def _save_full_pipeline_results(
        self,
        processed_image: np.ndarray,
        extracted: Dict[str, np.ndarray],
        inpainted: Dict[str, np.ndarray],
        detection: LayeredOcclusionResult,
        image_name: str
    ) -> None:
        """Save full pipeline results to disk.

        Outputs:
        - processed.png: Upscaled/processed original image
        - foreground_combined.png: All foreground objects combined
        - background_separated.png: Background only (inpainted if occluded)
        - {label}_rgba.png: Individual object RGBA extractions
        - {label}_vis.png: Individual object on white background
        - comparison_grid.png: Grid comparison of all outputs

        Args:
            processed_image: Final processed image
            extracted: Dict of RGBA images
            inpainted: Dict of inpainted RGB images
            detection: Occlusion detection result
            image_name: Base name for output files
        """
        # Create subdirectory for this image
        image_dir = self.output_dir / image_name
        image_dir.mkdir(parents=True, exist_ok=True)

        h, w = processed_image.shape[:2]

        # Build mask of all background annotations
        # Background is stored separately in detection.background_mask
        background_mask = detection.background_mask.copy() if detection.background_mask is not None else np.zeros((h, w), dtype=np.uint8)

        bg_pixel_count = np.sum(background_mask > 0)
        if bg_pixel_count > 0:
            print(f"    Background mask: {bg_pixel_count} pixels detected")
            # Save background mask for reference
            cv2.imwrite(str(image_dir / "background_mask.png"), background_mask)
        else:
            print(f"    No background annotations found")

        # Build mask of all foreground objects
        foreground_mask = np.zeros((h, w), dtype=np.uint8)
        for label, masks in detection.layer_masks.items():
            if label != 'background':
                for m in masks:
                    foreground_mask = np.maximum(foreground_mask, m)

        # 1. Save processed image (original with background regions removed)
        processed_clean = processed_image.copy()
        # Remove background regions (set to white)
        processed_clean[background_mask > 0] = [255, 255, 255]

        cv2.imwrite(
            str(image_dir / "processed.png"),
            cv2.cvtColor(processed_clean, cv2.COLOR_RGB2BGR)
        )

        # Also save foreground-only version (only annotated foreground objects)
        foreground_only = np.ones((h, w, 3), dtype=np.uint8) * 255
        foreground_only[foreground_mask > 0] = processed_image[foreground_mask > 0]
        cv2.imwrite(
            str(image_dir / "foreground_only.png"),
            cv2.cvtColor(foreground_only, cv2.COLOR_RGB2BGR)
        )

        # 2. Create and save foreground/background separation
        # Foreground: combine all non-background objects
        foreground_combined = np.zeros((h, w, 4), dtype=np.uint8)
        background_separated = np.ones((h, w, 3), dtype=np.uint8) * 255

        for label, masks in detection.layer_masks.items():
            combined_mask = np.zeros((h, w), dtype=np.uint8)
            for m in masks:
                combined_mask = np.maximum(combined_mask, m)

            if label == 'background':
                # Background: use inpainted version if available
                if label in inpainted:
                    background_separated[combined_mask > 0] = inpainted[label][combined_mask > 0]
                else:
                    background_separated[combined_mask > 0] = processed_image[combined_mask > 0]
            else:
                # Foreground: accumulate all objects
                if label in extracted:
                    rgba = extracted[label]
                    alpha = rgba[:, :, 3:4] / 255.0
                    # Composite onto foreground
                    fg_rgb = foreground_combined[:, :, :3].astype(float)
                    fg_alpha = foreground_combined[:, :, 3:4].astype(float) / 255.0
                    # Over compositing
                    new_alpha = alpha + fg_alpha * (1 - alpha)
                    new_rgb = (rgba[:, :, :3] * alpha + fg_rgb * fg_alpha * (1 - alpha))
                    new_rgb = np.where(new_alpha > 0, new_rgb / np.maximum(new_alpha, 1e-6), 0)
                    foreground_combined[:, :, :3] = new_rgb.clip(0, 255).astype(np.uint8)
                    foreground_combined[:, :, 3] = (new_alpha * 255).clip(0, 255).astype(np.uint8)[:, :, 0]

        # Save foreground combined (RGBA)
        cv2.imwrite(
            str(image_dir / "foreground_combined.png"),
            cv2.cvtColor(foreground_combined, cv2.COLOR_RGBA2BGRA)
        )

        # Save foreground visualization (on white background)
        fg_vis = np.ones((h, w, 3), dtype=np.uint8) * 255
        fg_alpha = foreground_combined[:, :, 3:4] / 255.0
        fg_vis = (fg_vis * (1 - fg_alpha) + foreground_combined[:, :, :3] * fg_alpha).astype(np.uint8)
        cv2.imwrite(
            str(image_dir / "foreground_vis.png"),
            cv2.cvtColor(fg_vis, cv2.COLOR_RGB2BGR)
        )

        # Save background separated
        cv2.imwrite(
            str(image_dir / "background_separated.png"),
            cv2.cvtColor(background_separated, cv2.COLOR_RGB2BGR)
        )

        # 3. Save individual RGBA extractions
        individual_vis = {}  # For grid comparison
        for label, rgba in extracted.items():
            out_path = image_dir / f"{label}_rgba.png"
            bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
            cv2.imwrite(str(out_path), bgra)

            # Also save visualization with white background
            vis = np.ones((h, w, 3), dtype=np.uint8) * 255
            alpha = rgba[:, :, 3:4] / 255.0
            vis = (vis * (1 - alpha) + rgba[:, :, :3] * alpha).astype(np.uint8)
            cv2.imwrite(
                str(image_dir / f"{label}_vis.png"),
                cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
            )
            individual_vis[label] = vis

        # 4. Save inpainted images
        for label, img in inpainted.items():
            out_path = image_dir / f"{label}_inpainted.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        # 5. Extract and save edges for each object
        edge_images = {}
        for label, rgba in extracted.items():
            edge = self._extract_edges(rgba)
            edge_images[label] = edge

            # Save edge image
            cv2.imwrite(
                str(image_dir / f"{label}_edge.png"),
                edge
            )

        # 6. Create combined edge visualization
        combined_edge = self._create_combined_edge(
            processed_image, extracted, detection.layer_masks
        )
        cv2.imwrite(
            str(image_dir / "edges_combined.png"),
            combined_edge
        )

        # 7. Create comparison grid
        self._create_comparison_grid(
            image_dir,
            processed_image,
            fg_vis,
            background_separated,
            individual_vis
        )

        print(f"    Saved to: {image_dir}")

        # Save debug inpaint data (before/after/mask for feather blend analysis)
        if self._debug_inpaint_data:
            debug_dir = image_dir / "debug_inpaint"
            debug_dir.mkdir(exist_ok=True)

            for label, data in self._debug_inpaint_data.items():
                safe_label = label.replace("/", "_").replace("\\", "_")
                cv2.imwrite(str(debug_dir / f"{safe_label}_before.png"),
                           cv2.cvtColor(data["before"], cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(debug_dir / f"{safe_label}_after.png"),
                           cv2.cvtColor(data["after"], cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(debug_dir / f"{safe_label}_mask.png"),
                           data["mask"])
                cv2.imwrite(str(debug_dir / f"{safe_label}_occlusion_mask.png"),
                           data["occlusion_mask"])

            print(f"    Debug inpaint data saved to: {debug_dir}")

    def _extract_edges(
        self,
        rgba: np.ndarray,
        method: str = "canny"
    ) -> np.ndarray:
        """Extract edges from RGBA image.

        Uses the alpha channel to mask the object, then extracts edges
        from the RGB content.

        Args:
            rgba: RGBA image (H, W, 4)
            method: Edge detection method ("canny", "sobel", "laplacian")

        Returns:
            Edge image (H, W), uint8, white edges on black background
        """
        h, w = rgba.shape[:2]

        # Get RGB and alpha
        rgb = rgba[:, :, :3]
        alpha = rgba[:, :, 3]

        # Convert to grayscale
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        # Apply alpha mask
        gray = (gray.astype(float) * (alpha / 255.0)).astype(np.uint8)

        if method == "canny":
            # Canny edge detection
            edges = cv2.Canny(gray, 50, 150)
        elif method == "sobel":
            # Sobel edge detection
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edges = np.sqrt(sobelx**2 + sobely**2)
            edges = (edges / edges.max() * 255).astype(np.uint8) if edges.max() > 0 else edges.astype(np.uint8)
        elif method == "laplacian":
            # Laplacian edge detection
            edges = cv2.Laplacian(gray, cv2.CV_64F)
            edges = np.abs(edges)
            edges = (edges / edges.max() * 255).astype(np.uint8) if edges.max() > 0 else edges.astype(np.uint8)
        else:
            edges = cv2.Canny(gray, 50, 150)

        # Also add contour from alpha mask (object boundary)
        alpha_binary = (alpha > 127).astype(np.uint8)
        contours, _ = cv2.findContours(alpha_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on edge image
        contour_img = np.zeros_like(edges)
        cv2.drawContours(contour_img, contours, -1, 255, 1)

        # Combine internal edges with boundary
        combined = np.maximum(edges, contour_img)

        # Mask to object region only
        combined[alpha < 10] = 0

        return combined

    def _create_combined_edge(
        self,
        image: np.ndarray,
        extracted: Dict[str, np.ndarray],
        layer_masks: Dict[str, List[np.ndarray]]
    ) -> np.ndarray:
        """Create a combined edge visualization with colored edges per object.

        Each object gets a different color for its edges, making it easy
        to distinguish overlapping boundaries.

        Args:
            image: Original RGB image (H, W, 3)
            extracted: Dict of RGBA images
            layer_masks: Dict of masks per layer

        Returns:
            Combined edge visualization (H, W, 3), BGR format
        """
        h, w = image.shape[:2]

        # Color palette for different objects
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple
        ]

        # Start with grayscale version of original
        gray_bg = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        result = cv2.cvtColor(gray_bg, cv2.COLOR_GRAY2BGR)

        # Make background darker for contrast
        result = (result * 0.3).astype(np.uint8)

        # Draw edges for each object in different colors
        sorted_labels = sorted(extracted.keys())
        for idx, label in enumerate(sorted_labels):
            if label == 'background':
                continue

            rgba = extracted[label]
            edge = self._extract_edges(rgba)

            # Get color for this object
            color = colors[idx % len(colors)]

            # Draw colored edges
            edge_mask = edge > 127
            result[edge_mask] = color

        return result

    def _create_comparison_grid(
        self,
        output_dir: Path,
        original: np.ndarray,
        foreground: np.ndarray,
        background: np.ndarray,
        individual: Dict[str, np.ndarray]
    ) -> None:
        """Create a grid comparison image showing all outputs.

        Grid layout:
        Row 1: Original | Foreground Combined | Background Separated
        Row 2+: Individual objects (up to 3 per row)

        Args:
            output_dir: Directory to save the grid
            original: Original/processed image
            foreground: Combined foreground visualization
            background: Separated background
            individual: Dict of individual object visualizations
        """
        h, w = original.shape[:2]
        padding = 10
        label_height = 30

        # Calculate grid dimensions
        num_individuals = len(individual)
        cols = 3
        rows = 1 + (num_individuals + cols - 1) // cols  # 1 row for main + rows for individuals

        cell_h = h + label_height
        cell_w = w

        grid_h = rows * cell_h + (rows + 1) * padding
        grid_w = cols * cell_w + (cols + 1) * padding

        # Create white background grid
        grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 240

        def place_image(img: np.ndarray, row: int, col: int, label: str) -> None:
            """Place an image in the grid with label."""
            y = padding + row * (cell_h + padding)
            x = padding + col * (cell_w + padding)

            # Place image
            grid[y:y+h, x:x+w] = img

            # Add label background
            label_y = y + h
            grid[label_y:label_y+label_height, x:x+w] = (200, 200, 200)

            # Add label text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            text_x = x + (w - text_size[0]) // 2
            text_y = label_y + (label_height + text_size[1]) // 2
            cv2.putText(
                grid, label, (text_x, text_y),
                font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA
            )

        # Row 1: Main outputs
        place_image(original, 0, 0, "Original (Processed)")
        place_image(foreground, 0, 1, "Foreground Combined")
        place_image(background, 0, 2, "Background Separated")

        # Row 2+: Individual objects
        sorted_labels = sorted(individual.keys())
        for idx, label in enumerate(sorted_labels):
            row = 1 + idx // cols
            col = idx % cols
            place_image(individual[label], row, col, label)

        # Save grid
        cv2.imwrite(
            str(output_dir / "comparison_grid.png"),
            cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
        )

    def compare_inpainting_methods(
        self,
        image: np.ndarray,
        annotation_path: str,
        methods: List[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Compare different inpainting methods on the same image.

        Args:
            image: RGB image (H, W, 3)
            annotation_path: Path to labelme JSON annotation
            methods: List of methods to compare (default: all available)

        Returns:
            Dict mapping method name to results
        """
        if methods is None:
            methods = ['ns', 'telea', 'lama']

        # Load annotation and detect occlusion
        with open(annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        shapes = data.get('shapes', [])

        h, w = image.shape[:2]
        detection = self.layered_detector.detect_from_shapes(shapes, (h, w))

        if not detection.occlusion_relations:
            print("No occlusion relations found - nothing to compare")
            return {}

        # Get combined occlusion mask for comparison
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        for rel in detection.occlusion_relations:
            combined_mask = np.maximum(combined_mask, rel.occlusion_mask)

        inpaint_mask = (combined_mask > 0).astype(np.uint8) * 255
        base_image = image.copy()
        base_image[inpaint_mask > 127] = [255, 255, 255]

        results = {}
        comparison_dir = self.output_dir / "inpaint_comparison"
        comparison_dir.mkdir(parents=True, exist_ok=True)

        # Save base image and mask
        cv2.imwrite(
            str(comparison_dir / "base_image.png"),
            cv2.cvtColor(base_image, cv2.COLOR_RGB2BGR)
        )
        cv2.imwrite(str(comparison_dir / "inpaint_mask.png"), inpaint_mask)

        print(f"\n[Inpainting Comparison] Mask pixels: {np.sum(inpaint_mask > 127)}")

        for method in methods:
            print(f"\n  Testing: {method}...")
            start_time = time.time()

            try:
                if method in ['ns', 'telea']:
                    flag = cv2.INPAINT_NS if method == 'ns' else cv2.INPAINT_TELEA
                    inpainted = cv2.inpaint(
                        base_image, inpaint_mask,
                        inpaintRadius=10,
                        flags=flag
                    )
                elif method == 'lama':
                    inpainted = self.inpainting.lama.inpaint(base_image, inpaint_mask)
                elif method == 'controlnet':
                    inpainted = self.inpainting.controlnet.inpaint(
                        base_image, inpaint_mask,
                        prompt="traditional Korean painting texture",
                        num_inference_steps=20
                    )
                else:
                    print(f"    Unknown method: {method}")
                    continue

                elapsed = time.time() - start_time

                # Apply only to mask region
                result = base_image.copy()
                mask_region = inpaint_mask > 127
                result[mask_region] = inpainted[mask_region]

                # Calculate metrics
                diff = np.abs(result.astype(float) - base_image.astype(float))
                changed_pixels = np.sum(np.any(diff > 5, axis=2))

                results[method] = {
                    'result': result,
                    'time_ms': elapsed * 1000,
                    'changed_pixels': int(changed_pixels),
                }

                # Save result
                cv2.imwrite(
                    str(comparison_dir / f"result_{method}.png"),
                    cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                )

                print(f"    Time: {elapsed*1000:.1f}ms, Changed: {changed_pixels} pixels")

            except Exception as e:
                print(f"    Error: {e}")
                results[method] = {'error': str(e)}

        return results

    def process_from_annotation(
        self,
        image: np.ndarray,
        annotation_path: str,
        save_outputs: bool = True
    ) -> Dict[str, Any]:
        """Process image using annotation file with new label system.

        Handles: object_1, object_2_*, object_3, background

        Workflow:
        1. Load annotation and detect occlusion relations
        2. For each occluded object: inpaint hidden regions (unless skip_inpainting)
        3. For all objects: extract as RGBA

        Ablation options (from config):
        - skip_occlusion_detection: Skip depth-based detection, treat as no occlusion
        - skip_inpainting: Skip inpainting step, extract with visible regions only

        Args:
            image: RGB image (H, W, 3)
            annotation_path: Path to labelme JSON annotation
            save_outputs: Save intermediate results

        Returns:
            Dict with:
            - 'detection': LayeredOcclusionResult
            - 'extracted': Dict[label, RGBA image]
            - 'inpainted': Dict[label, inpainted RGB image]
        """
        import json
        from collections import defaultdict

        # 1. Load shapes from annotation
        with open(annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        shapes = data.get('shapes', [])

        if not shapes:
            raise ValueError(f"No shapes found in annotation: {annotation_path}")

        # 2. Detect occlusion using LayeredOcclusionDetector
        h, w = image.shape[:2]
        detection = self.layered_detector.detect_from_shapes(shapes, (h, w))

        print(f"[Occlusion Pipeline]")
        print(f"  Layers: {list(detection.layer_masks.keys())}")

        # Ablation: skip_occlusion_detection - clear occlusion relations
        if self.config.skip_occlusion_detection:
            print(f"  [ABLATION] Skipping occlusion detection")
            detection.occlusion_relations = []
        else:
            print(f"  Occlusion relations: {len(detection.occlusion_relations)}")
            for rel in detection.occlusion_relations:
                print(f"    {rel.occluder_label} -> {rel.occludee_label}: {rel.occlusion_ratio:.1%}")

        # 3. Inpaint occluded objects (unless skip_inpainting)
        if self.config.skip_inpainting:
            print(f"  [ABLATION] Skipping inpainting")
            inpainted_objects = {}
        else:
            inpainted_objects = self._inpaint_all_occluded(image, detection)

        # 4. Extract all objects as RGBA
        extracted_objects = self._extract_all_objects(image, detection, inpainted_objects)

        # 5. Save outputs
        if save_outputs:
            self._save_separated_objects(extracted_objects, inpainted_objects)

        return {
            'detection': detection,
            'extracted': extracted_objects,
            'inpainted': inpainted_objects,
        }

    def _inpaint_all_occluded(
        self,
        image: np.ndarray,
        detection: LayeredOcclusionResult
    ) -> Dict[str, np.ndarray]:
        """Inpaint all occluded objects.

        For each occludee, combines ALL occluders and inpaints once.
        This fixes the bug where multiple occluders were processed separately.

        Args:
            image: Original RGB image
            detection: Occlusion detection result

        Returns:
            Dict mapping occludee label to inpainted full image
        """
        from collections import defaultdict

        if not detection.occlusion_relations:
            return {}

        h, w = image.shape[:2]
        inpainted_results = {}

        # Group relations by occludee
        occludee_relations = defaultdict(list)
        for rel in detection.occlusion_relations:
            occludee_relations[rel.occludee_label].append(rel)

        # Process each occludee
        for occludee_label, relations in occludee_relations.items():
            print(f"  Inpainting {occludee_label} (occluded by {len(relations)} objects)...")

            # Combine ALL occluder masks for this occludee
            combined_occluder_mask = np.zeros((h, w), dtype=np.uint8)
            combined_occlusion_mask = np.zeros((h, w), dtype=np.uint8)

            for rel in relations:
                combined_occluder_mask = np.maximum(combined_occluder_mask, rel.occluder_mask)
                combined_occlusion_mask = np.maximum(combined_occlusion_mask, rel.occlusion_mask)

            # Get occludee's own mask (combine if multiple)
            occludee_masks = detection.layer_masks.get(occludee_label, [])
            if not occludee_masks:
                print(f"    Warning: No mask found for {occludee_label}")
                continue

            occludee_mask = np.zeros((h, w), dtype=np.uint8)
            for m in occludee_masks:
                occludee_mask = np.maximum(occludee_mask, m)

            # Visible part of occludee (not covered by occluders)
            visible_occludee = cv2.bitwise_and(
                occludee_mask,
                cv2.bitwise_not(combined_occluder_mask)
            )

            # Create base image: original with occlusion region removed
            # Use occlusion_mask (not occluder_mask) to match inpaint_mask
            base_image = image.copy()
            base_image[combined_occlusion_mask > 0] = [255, 255, 255]

            # Scale mask to 0/255 for inpainting methods (LaMa/ControlNet expect this range)
            # Note: dilation already applied in occlusion_detection (5x5 kernel)
            inpaint_mask = (combined_occlusion_mask > 0).astype(np.uint8) * 255

            # Optional dilation when using refined tight mask
            if self.config.use_refined_mask_for_inpaint and self.config.inpaint_mask_dilate_px > 0:
                k = self.config.inpaint_mask_dilate_px
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*k+1, 2*k+1))
                inpaint_mask = cv2.dilate(inpaint_mask, kernel, iterations=1)

            # Select inpainting method based on config
            method = self.config.inpaint_method

            if method == "lama":
                # LaMa: Deep learning based, best quality
                print(f"    Using LaMa inpainting...")
                inpainted = self.inpainting.lama.inpaint(base_image, inpaint_mask)

            elif method == "lama_guided" and np.sum(visible_occludee) > 100:
                # LaMa + context-guided color matching (best for blending)
                print(f"    Using LaMa guided inpainting...")
                inpainted = self.inpainting.inpaint_lama_guided(
                    image,  # Original image for reference
                    inpaint_mask,
                    visible_occludee
                )

            elif method == "lama_guided":
                # Fallback to LaMa if no visible reference
                print(f"    Using LaMa inpainting (no reference available)...")
                inpainted = self.inpainting.lama.inpaint(base_image, inpaint_mask)

            elif method == "controlnet":
                # ControlNet: Structure-preserving with prompt
                print(f"    Using ControlNet inpainting...")
                inpainted = self.inpainting.controlnet.inpaint(
                    base_image, inpaint_mask,
                    prompt="traditional Korean painting, wooden furniture texture, seamless",
                    negative_prompt="modern, blurry, artifact",
                    controlnet_scale=0.5,
                    guidance_scale=7.5,
                    num_inference_steps=30
                )

            elif method == "texture_clone" and np.sum(visible_occludee) > 100:
                # Patch-based texture cloning from visible region
                # IMPORTANT: Use original image for reference (not base_image with white regions)
                print(f"    Using texture clone inpainting...")
                inpainted = self.inpainting.inpaint_texture_clone(
                    image,  # Original image for texture reference
                    inpaint_mask,
                    visible_occludee
                )

            elif method == "seamless_clone" and np.sum(visible_occludee) > 100:
                # OpenCV seamlessClone for natural blending
                print(f"    Using seamless clone inpainting...")
                inpainted = self.inpainting.inpaint_seamless_clone(
                    image,  # Original image for reference
                    inpaint_mask,
                    visible_occludee
                )

            elif method == "boundary_guided" and np.sum(visible_occludee) > 100:
                # V25: Boundary neighborhood sampling with dynamic edge morphology
                print(f"    Using boundary-guided inpainting (V25: dynamic edge)...")
                inpainted = inpaint_occlusion_boundary_guided(
                    image,  # Original image
                    combined_occlusion_mask,
                    combined_occluder_mask,
                    occludee_mask,
                    visible_occludee,
                    edge_darkness=0.3,
                    max_sample_distance=10,
                    use_dynamic_edge=True,
                    min_edge_width=1,
                    max_edge_width=8,
                    width_smoothing_sigma=1.5,
                    min_safe_distance=3,
                    # V26: Adaptive Kernel Smoothing (disabled by default)
                    _v26_smoothstep_gradient=False,
                    _v26_adaptive_smooth=False,
                    _v26_adaptive_aa=False,
                    _v26_feathered_transition=False,
                )

            elif method == "patchmatch_v25" and np.sum(visible_occludee) > 100:
                # V25+PM: PatchMatch body + V25 dynamic edge (best quality)
                print(f"    Using patchmatch + V25 dynamic edge inpainting...")
                inpainted = inpaint_occlusion_patchmatch_v25(
                    image,
                    combined_occlusion_mask,
                    combined_occluder_mask,
                    occludee_mask,
                    visible_occludee,
                    edge_darkness=0.3,
                    patch_size=7,
                    min_edge_width=1,
                    max_edge_width=8,
                    width_smoothing_sigma=1.5,
                    min_safe_distance=3,
                    # V26: Adaptive Kernel Smoothing (disabled by default)
                    _v26_smoothstep_gradient=False,
                    _v26_adaptive_aa=False,
                )

            elif method == "patchmatch_guided" and np.sum(visible_occludee) > 100:
                # V22: PatchMatch texture propagation with boundary guidance
                print(f"    Using patchmatch-guided inpainting (V22)...")
                inpainted = inpaint_occlusion_patchmatch_guided(
                    image,
                    combined_occlusion_mask,
                    combined_occluder_mask,
                    occludee_mask,
                    visible_occludee,
                    edge_darkness=0.3,
                    patch_size=7,
                    iterations=5
                )

            elif method in ("ns", "telea"):
                # OpenCV NS/Telea - most reliable for traditional paintings
                print(f"    Using OpenCV {method} inpainting...")
                flag = cv2.INPAINT_NS if method == "ns" else cv2.INPAINT_TELEA
                inpainted = cv2.inpaint(
                    base_image, inpaint_mask,
                    inpaintRadius=5,
                    flags=flag
                )

            elif method == "reference" and np.sum(visible_occludee) > 100:
                # Reference-based texture matching (noise-based, experimental)
                print(f"    Using reference-based inpainting...")
                inpainted = self.inpainting.inpaint_with_reference(
                    base_image,
                    inpaint_mask,
                    visible_occludee,
                    noise_factor=0.5
                )

            else:
                # Fallback: OpenCV NS
                print(f"    Using OpenCV NS inpainting (fallback)...")
                inpainted = cv2.inpaint(
                    base_image, inpaint_mask,
                    inpaintRadius=5,
                    flags=cv2.INPAINT_NS
                )

            # Apply inpainted result only to mask region (prevent changes outside mask)
            mask_region = inpaint_mask > 127
            result = base_image.copy()
            result[mask_region] = inpainted[mask_region]

            inpainted_results[occludee_label] = result

            # Store debug data for feather blend analysis (added for pipeline improvement)
            self._debug_inpaint_data[occludee_label] = {
                "before": image.copy(),           # Original image (before inpainting)
                "after": result.copy(),           # After inpainting
                "mask": inpaint_mask.copy(),      # Inpainted region mask
                "occlusion_mask": combined_occlusion_mask.copy(),
            }

            occ_px = np.sum(combined_occlusion_mask > 0)
            print(f"    Restored {occ_px} occluded pixels")

        return inpainted_results

    def _extract_all_objects(
        self,
        image: np.ndarray,
        detection: LayeredOcclusionResult,
        inpainted_objects: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Extract all objects as RGBA images.

        For occluded objects: uses inpainted image
        For non-occluded objects: uses original image

        IMPORTANT: Subtracts background_mask from each object mask to prevent
        background regions from being included in object extraction.

        Args:
            image: Original RGB image
            detection: Occlusion detection result
            inpainted_objects: Dict of inpainted images for occluded objects

        Returns:
            Dict mapping label to RGBA image
        """
        h, w = image.shape[:2]
        extracted = {}

        # Find which objects are occluded (occludees)
        occluded_labels = set(rel.occludee_label for rel in detection.occlusion_relations)

        # Get background mask to subtract from object masks
        background_mask = detection.background_mask if detection.background_mask is not None else np.zeros((h, w), dtype=np.uint8)

        for label, masks in detection.layer_masks.items():
            if label == 'background':
                continue

            # Combine masks for this label
            combined_mask = np.zeros((h, w), dtype=np.uint8)
            for m in masks:
                combined_mask = np.maximum(combined_mask, m)

            # Subtract background mask from object mask
            # This prevents background regions from being included in object extraction
            if np.sum(background_mask) > 0:
                # Where background is annotated, remove from object mask
                combined_mask = cv2.bitwise_and(
                    combined_mask,
                    cv2.bitwise_not(background_mask)
                )

            # Choose source image
            if label in inpainted_objects:
                # Use inpainted image for occluded objects
                source_image = inpainted_objects[label]
            else:
                # Use original for non-occluded objects
                source_image = image

            # Extract as RGBA
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[:, :, :3] = source_image
            rgba[:, :, 3] = combined_mask  # Alpha from mask

            extracted[label] = rgba
            print(f"  Extracted {label}: {np.sum(combined_mask > 0)} pixels")

        return extracted

    def _save_separated_objects(
        self,
        extracted: Dict[str, np.ndarray],
        inpainted: Dict[str, np.ndarray]
    ) -> None:
        """Save all separated objects to disk.

        Args:
            extracted: Dict of RGBA images
            inpainted: Dict of inpainted RGB images
        """
        # Save RGBA extractions
        for label, rgba in extracted.items():
            out_path = self.output_dir / f"{label}_rgba.png"
            # Convert RGBA to BGRA for OpenCV
            bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
            cv2.imwrite(str(out_path), bgra)
            print(f"    Saved: {out_path}")

        # Save inpainted full images
        for label, img in inpainted.items():
            out_path = self.output_dir / f"{label}_inpainted.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f"    Saved: {out_path}")

    def apply_color_refinement(
        self,
        image_np: np.ndarray,
        foreground_mask: np.ndarray,
        background_mask: np.ndarray,
        inpainted_image: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply color-based mask refinement.

        Args:
            image_np: Original RGB image.
            foreground_mask: SAM foreground (ceramic) mask.
            background_mask: SAM background (soban) mask.
            inpainted_image: Optional inpainted image for soban extraction.

        Returns:
            Tuple of (refined_foreground_mask, refined_background_mask).
        """
        refined_fg = foreground_mask
        refined_bg = background_mask

        if self.config.color_refine_ceramic:
            refined_fg = self.mask_refiner.refine_ceramic_mask(
                image_np, foreground_mask
            )

        if self.config.color_refine_soban:
            refined_bg = self.mask_refiner.refine_soban_mask(
                image_np, inpainted_image
            )

        return refined_fg, refined_bg

    def _extract_red_region(self, image: np.ndarray) -> np.ndarray:
        """Extract only red-colored region from inpainted image.

        Uses HSV color detection to isolate the red soban and remove
        any non-red artifacts from ControlNet inpainting.

        Args:
            image: RGB image from ControlNet inpainting.

        Returns:
            Image with only red region preserved, rest is white.
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Red color detection (H around 0-15 and 165-180)
        lower_red1 = np.array([0, 50, 30])
        upper_red1 = np.array([15, 255, 200])
        lower_red2 = np.array([165, 50, 30])
        upper_red2 = np.array([180, 255, 200])

        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask_red1, mask_red2)

        # Morphological cleanup
        kernel = np.ones((3, 3), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

        # Keep only largest connected component
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(red_mask)
        if num_labels > 2:
            areas = stats[1:, cv2.CC_STAT_AREA]
            main_label = 1 + np.argmax(areas)
            red_mask = (labels == main_label).astype(np.uint8) * 255

        # Create output: white background with red region
        result = np.ones_like(image) * 255
        result[red_mask > 0] = image[red_mask > 0]

        return result

    def apply_auto_color_refinement(
        self,
        image_np: np.ndarray,
        foreground_mask: np.ndarray,
        background_mask: np.ndarray,
        inpainted_image: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply automatic color-based mask refinement.

        Uses K-means clustering to automatically extract dominant colors
        from each object and removes cross-contamination.

        Args:
            image_np: Original RGB image.
            foreground_mask: SAM foreground mask.
            background_mask: SAM background mask.
            inpainted_image: Optional inpainted image for background extraction.

        Returns:
            Tuple of (refined_foreground_mask, refined_background_mask).
        """
        return self.auto_mask_refiner.refine_both_masks(
            image_np,
            foreground_mask,
            background_mask,
            inpainted_for_b=inpainted_image
        )

    def forward(
        self,
        image: Tensor,
        text_prompts: Optional[List[str]] = None,
        save_intermediates: bool = True,
        **kwargs
    ) -> ModuleOutput:
        """Process image through full occlusion pipeline.

        Args:
            image: Input image tensor (C, H, W) or (B, C, H, W).
            text_prompts: Override config text prompts.
            save_intermediates: Save intermediate results to disk.
            **kwargs: Additional parameters.

        Returns:
            ModuleOutput with:
            - result: Inpainted background tensor
            - intermediate: Dict with all stage outputs
            - metadata: Timing and configuration info
        """
        start_time = time.time()
        prompts = text_prompts or self.config.text_prompts

        # Convert tensor to numpy
        image_np = self._tensor_to_numpy(image)

        # Stage 1: Segmentation (text-based or automatic)
        seg_start = time.time()

        if self.config.segmentation_mode == "auto":
            # Automatic segmentation using SAM AMG
            fg_mask, bg_mask, all_masks = self.auto_segmentation.segment_auto(
                image_np,
                depth_map=None,  # Depth will be estimated later
                min_area_ratio=self.config.auto_seg_min_area_ratio,
                max_area_ratio=self.config.auto_seg_max_area_ratio,
                top_n=5
            )

            # Create LayerInfo from auto segmentation results
            layers = []
            if np.sum(fg_mask) > 0:
                layers.append(LayerInfo(
                    label="foreground",
                    mask=fg_mask,
                    bbox=(0, 0, image_np.shape[1], image_np.shape[0]),
                    is_foreground=True
                ))
            if np.sum(bg_mask) > 0:
                layers.append(LayerInfo(
                    label="background",
                    mask=bg_mask,
                    bbox=(0, 0, image_np.shape[1], image_np.shape[0]),
                    is_foreground=False
                ))
        else:
            # Text-based segmentation using Grounding DINO + SAM
            layers = self.segmentation.segment(image_np, prompts)

        seg_time = time.time() - seg_start

        if len(layers) < 2:
            # Not enough layers for occlusion processing
            return self._create_single_layer_output(image_np, layers, seg_time)

        # Stage 2: Depth estimation
        depth_start = time.time()
        depth_map = self.depth.estimate(image_np)
        depth_time = time.time() - depth_start

        # Stage 3: Layer ordering
        order_start = time.time()
        foreground, background = self.layer_ordering.order_combined(
            layers[:2], depth_map
        )
        order_time = time.time() - order_start

        # Stage 4: Occlusion detection - TRUE occlusion (hidden region)
        detect_start = time.time()

        # Basic boundary overlap analysis
        occlusion_info = self.occlusion_detector.analyze_occlusion(
            foreground.mask, background.mask
        )

        # TRUE occlusion: estimate hidden table surface under foreground
        # Use edge information to predict full background structure
        true_occlusion_mask = detect_true_occlusion(
            foreground.mask,
            background.mask,
            table_shape="edge",  # Use Canny edge + ellipse fitting
            margin=5,
            image=image_np
        )

        # Also estimate full table surface for visualization
        estimated_table = estimate_table_surface(background.mask, method="ellipse")

        print(f"  Boundary occlusion: {np.sum(occlusion_info['occlusion_mask'] > 0):,} px")
        print(f"  TRUE occlusion: {np.sum(true_occlusion_mask > 0):,} px")

        detect_time = time.time() - detect_start

        # Stage 5: Inpainting - Remove foreground to reveal background only
        inpaint_start = time.time()

        # Detect RED pixels in image (soban color) for extended mask
        hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        lower_red1, upper_red1 = np.array([0, 50, 30]), np.array([15, 255, 200])
        lower_red2, upper_red2 = np.array([165, 50, 30]), np.array([180, 255, 200])
        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, lower_red1, upper_red1),
            cv2.inRange(hsv, lower_red2, upper_red2)
        )

        # Red pixels outside foreground = visible table surface
        red_outside_fg = np.logical_and(red_mask > 0, foreground.mask <= 127)

        # Extended soban mask: auto soban + red outside foreground
        extended_soban_mask = np.logical_or(
            background.mask > 127,
            red_outside_fg
        ).astype(np.uint8) * 255

        # Dilated foreground mask for inpainting
        fg_mask_dilated = cv2.dilate(
            foreground.mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=1
        )

        print(f"  Inpainting foreground region: {np.sum(fg_mask_dilated > 0):,} px")

        if self.config.inpaint_method == "lama":
            # LaMa: Inpaint entire foreground for best texture extension
            lama_result = self.inpainting.lama.inpaint(image_np, fg_mask_dilated)

            # Combine: visible soban + inpainted foreground area
            combined_mask = cv2.bitwise_or(extended_soban_mask, fg_mask_dilated)
            soban_only = np.ones_like(image_np) * 255
            soban_only[combined_mask > 127] = lama_result[combined_mask > 127]

            # Apply red extraction to remove non-soban artifacts
            background_inpainted = self._extract_red_region(soban_only)

        elif self.config.inpaint_method == "controlnet":
            # ControlNet: Use EXACT original working method from inpaint_controlnet.py
            # Key: Create soban-only image first, then inpaint only occlusion region

            # 1. Create soban-only image (white background + visible soban)
            soban_image = np.ones_like(image_np) * 255
            soban_image[extended_soban_mask > 127] = image_np[extended_soban_mask > 127]

            # 2. Combine boundary overlap + TRUE occlusion for full coverage
            combined_occlusion = cv2.bitwise_or(
                occlusion_info["occlusion_mask"],
                true_occlusion_mask.astype(np.uint8) * 255 if true_occlusion_mask.max() <= 1 else true_occlusion_mask
            )
            # Dilate for smooth edges
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            inpaint_mask = cv2.dilate(
                combined_occlusion,
                kernel,
                iterations=2
            )

            print(f"  ControlNet: inpainting occlusion region only ({np.sum(inpaint_mask > 0):,} px)")

            # 3. Inpaint soban image with occlusion mask
            inpainted = self.inpainting.controlnet.inpaint(
                soban_image, inpaint_mask,
                prompt="red wooden traditional Korean soban table with decorative rim pattern, carved wood texture, antique furniture",
                negative_prompt="white, ceramic, vase, jar, modern, blurry",
                controlnet_scale=0.5,
                guidance_scale=7.5,
                num_inference_steps=30,
                seed=42  # Fixed seed for consistent results
            )

            # 4. Include both visible soban AND inpainted occlusion region
            # Combine extended_soban_mask with the inpaint_mask to get full area
            full_soban_area = cv2.bitwise_or(extended_soban_mask, inpaint_mask)
            background_inpainted = np.ones_like(image_np) * 255
            background_inpainted[full_soban_area > 127] = inpainted[full_soban_area > 127]

        else:
            # OpenCV NS/Telea inpainting: extends texture from boundary
            # Combined mask: boundary overlap + TRUE occlusion
            combined_occlusion = cv2.bitwise_or(
                occlusion_info["occlusion_mask"],  # boundary overlap
                true_occlusion_mask  # TRUE occlusion
            )
            # Dilate for smooth edges
            combined_occlusion = cv2.dilate(
                combined_occlusion,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1
            )

            # Create base image: keep soban, white where ceramic occludes
            soban_base = image_np.copy()
            soban_base[combined_occlusion > 127] = [255, 255, 255]

            # NS inpaint: extends soban texture into occlusion area
            inpainted = cv2.inpaint(
                soban_base,
                combined_occlusion,
                inpaintRadius=10,  # Larger radius for smoother fill
                flags=cv2.INPAINT_NS
            )

            # Result: visible soban + inpainted occlusion area
            result_mask = cv2.bitwise_or(extended_soban_mask, combined_occlusion)
            background_inpainted = np.ones_like(image_np) * 255
            background_inpainted[result_mask > 127] = inpainted[result_mask > 127]

        # Update occlusion_info with TRUE occlusion data
        occlusion_info["true_occlusion_mask"] = true_occlusion_mask
        occlusion_info["estimated_table"] = estimated_table
        inpaint_time = time.time() - inpaint_start

        # Stage 5.5: Auto mask refinement (optional)
        refine_time = 0.0
        if self.config.use_auto_refinement:
            refine_start = time.time()
            foreground.mask, background.mask = self.apply_auto_color_refinement(
                image_np,
                foreground.mask,
                background.mask,
                inpainted_image=background_inpainted
            )
            refine_time = time.time() - refine_start

        # Stage 6: Extract foreground
        foreground_rgba = self.inpainting.extract_object(
            image_np, foreground.mask, return_rgba=True
        )

        total_time = time.time() - start_time

        # Create result
        result = OcclusionResult(
            foreground_image=foreground_rgba,
            background_inpainted=background_inpainted,
            foreground_mask=foreground.mask,
            background_mask=background.mask,
            occlusion_mask=occlusion_info["occlusion_mask"],
            depth_map=depth_map,
            layers=[foreground, background],
            metadata={
                "occlusion_ratio": occlusion_info["occlusion_ratio"],
                "has_occlusion": occlusion_info["has_occlusion"],
                "true_occlusion_mask": true_occlusion_mask,
                "boundary_occlusion_px": int(np.sum(occlusion_info["occlusion_mask"] > 0)),
                "true_occlusion_px": int(np.sum(true_occlusion_mask > 0))
            }
        )

        # Save outputs
        if save_intermediates:
            self._save_outputs(result, image_np)

        # Convert back to tensors for ModuleOutput
        result_tensor = self._numpy_to_tensor(background_inpainted)

        intermediate = {
            "foreground": self._numpy_to_tensor(foreground_rgba[:, :, :3]),
            "foreground_alpha": torch.from_numpy(foreground_rgba[:, :, 3]).unsqueeze(0),
            "background_original": self._numpy_to_tensor(image_np),
            "depth_map": torch.from_numpy(depth_map).unsqueeze(0),
            "foreground_mask": torch.from_numpy(foreground.mask.astype(np.float32)).unsqueeze(0),
            "background_mask": torch.from_numpy(background.mask.astype(np.float32)).unsqueeze(0),
            "occlusion_mask": torch.from_numpy(occlusion_info["occlusion_mask"].astype(np.float32)).unsqueeze(0),
            "true_occlusion_mask": torch.from_numpy(true_occlusion_mask.astype(np.float32)).unsqueeze(0)
        }

        metadata = {
            "total_time": total_time,
            "segmentation_mode": self.config.segmentation_mode,
            "segmentation_time": seg_time,
            "depth_time": depth_time,
            "ordering_time": order_time,
            "detection_time": detect_time,
            "refinement_time": refine_time,
            "inpainting_time": inpaint_time,
            "foreground_label": foreground.label,
            "background_label": background.label,
            "occlusion_ratio": occlusion_info["occlusion_ratio"],
            "config": self.config.model_dump()
        }

        return ModuleOutput(
            result=result_tensor,
            intermediate=intermediate,
            metadata=metadata
        )

    def process_with_manual_masks(
        self,
        image: np.ndarray,
        foreground_mask: np.ndarray,
        background_mask: np.ndarray,
        foreground_label: str = "foreground",
        background_label: str = "background",
        save_outputs: bool = True
    ) -> OcclusionResult:
        """Process with manually provided masks (skip segmentation).

        Useful for testing or when automatic segmentation fails.

        Args:
            image: RGB image (H, W, 3), uint8.
            foreground_mask: Binary mask of foreground.
            background_mask: Binary mask of background.
            foreground_label: Label for foreground object.
            background_label: Label for background object.
            save_outputs: Save results to disk.

        Returns:
            OcclusionResult with all outputs.
        """
        # Create layer info
        foreground = LayerInfo(
            label=foreground_label,
            mask=foreground_mask.astype(np.uint8),
            bbox=(0, 0, image.shape[1], image.shape[0]),
            is_foreground=True
        )
        background = LayerInfo(
            label=background_label,
            mask=background_mask.astype(np.uint8),
            bbox=(0, 0, image.shape[1], image.shape[0]),
            is_foreground=False
        )

        # Depth estimation
        depth_map = self.depth.estimate(image)
        foreground.mean_depth = self.depth.get_mean_depth(depth_map, foreground_mask)
        background.mean_depth = self.depth.get_mean_depth(depth_map, background_mask)

        # Occlusion detection
        occlusion_info = self.occlusion_detector.analyze_occlusion(
            foreground_mask, background_mask
        )

        # Inpainting - Remove foreground to reveal background only
        background_inpainted = self.inpainting.inpaint_with_dilated_mask(
            image, foreground_mask  # Inpaint entire foreground region
        )

        # Extract foreground
        foreground_rgba = self.inpainting.extract_object(
            image, foreground_mask, return_rgba=True
        )

        result = OcclusionResult(
            foreground_image=foreground_rgba,
            background_inpainted=background_inpainted,
            foreground_mask=foreground_mask,
            background_mask=background_mask,
            occlusion_mask=occlusion_info["occlusion_mask"],
            depth_map=depth_map,
            layers=[foreground, background],
            metadata={
                "occlusion_ratio": occlusion_info["occlusion_ratio"],
                "has_occlusion": occlusion_info["has_occlusion"],
                "manual_masks": True
            }
        )

        if save_outputs:
            self._save_outputs(result, image)

        return result

    def _tensor_to_numpy(self, tensor: Tensor) -> np.ndarray:
        """Convert tensor to numpy RGB image."""
        if tensor.dim() == 4:
            tensor = tensor[0]  # Remove batch dim

        # (C, H, W) -> (H, W, C)
        img = tensor.cpu().numpy().transpose(1, 2, 0)

        # Scale to uint8 if needed
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)

        return img

    def _numpy_to_tensor(self, image: np.ndarray) -> Tensor:
        """Convert numpy RGB image to tensor."""
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        # (H, W, C) -> (C, H, W) -> (1, C, H, W)
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def _create_single_layer_output(
        self,
        image: np.ndarray,
        layers: List[LayerInfo],
        seg_time: float
    ) -> ModuleOutput:
        """Create output when only single layer detected."""
        result_tensor = self._numpy_to_tensor(image)

        return ModuleOutput(
            result=result_tensor,
            intermediate={},
            metadata={
                "warning": "Less than 2 layers detected, no occlusion processing",
                "layers_detected": len(layers),
                "segmentation_time": seg_time
            }
        )

    def _save_outputs(
        self,
        result: OcclusionResult,
        original_image: np.ndarray
    ) -> None:
        """Save all output images to disk."""
        # Foreground (RGBA -> PNG)
        fg_path = self.output_dir / "ceramic_separated.png"
        cv2.imwrite(str(fg_path), cv2.cvtColor(result.foreground_image, cv2.COLOR_RGBA2BGRA))

        # Background inpainted
        bg_path = self.output_dir / "soban_inpainted.png"
        cv2.imwrite(str(bg_path), cv2.cvtColor(result.background_inpainted, cv2.COLOR_RGB2BGR))

        # Segmentation overlay
        overlay = self._create_segmentation_overlay(
            original_image,
            result.foreground_mask,
            result.background_mask
        )
        overlay_path = self.output_dir / "segmentation_overlay.png"
        cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        # Depth map (colorized)
        depth_colored = self._colorize_depth(result.depth_map)
        depth_path = self.output_dir / "depth_map.png"
        cv2.imwrite(str(depth_path), cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR))

        # Occlusion mask (boundary overlap)
        occlusion_path = self.output_dir / "occlusion_mask.png"
        cv2.imwrite(str(occlusion_path), result.occlusion_mask * 255)

        # TRUE occlusion mask (hidden region under foreground)
        if "true_occlusion_mask" in result.metadata:
            true_occ_path = self.output_dir / "true_occlusion_mask.png"
            cv2.imwrite(str(true_occ_path), result.metadata["true_occlusion_mask"])

    def _create_segmentation_overlay(
        self,
        image: np.ndarray,
        fg_mask: np.ndarray,
        bg_mask: np.ndarray,
        fg_color: Tuple[int, int, int] = (255, 0, 0),
        bg_color: Tuple[int, int, int] = (0, 0, 255),
        alpha: float = 0.4
    ) -> np.ndarray:
        """Create visualization overlay of segmentation masks."""
        overlay = image.copy()

        # Foreground in red
        fg_region = fg_mask > 0
        overlay[fg_region] = (
            overlay[fg_region] * (1 - alpha) +
            np.array(fg_color) * alpha
        ).astype(np.uint8)

        # Background in blue
        bg_region = bg_mask > 0
        overlay[bg_region] = (
            overlay[bg_region] * (1 - alpha) +
            np.array(bg_color) * alpha
        ).astype(np.uint8)

        return overlay

    def _colorize_depth(self, depth: np.ndarray) -> np.ndarray:
        """Colorize depth map using a colormap."""
        # Normalize to 0-255
        depth_norm = (depth * 255).astype(np.uint8)

        # Apply colormap
        depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_VIRIDIS)

        # Convert BGR to RGB
        return cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)


def run_pipeline(
    image_path: str,
    text_prompts: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    config: Optional[OcclusionConfig] = None
) -> OcclusionResult:
    """Convenience function to run full pipeline on an image file.

    Args:
        image_path: Path to input image.
        text_prompts: Text prompts for object detection.
        output_dir: Output directory path.
        config: Pipeline configuration.

    Returns:
        OcclusionResult with all outputs.
    """
    import cv2

    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create pipeline
    pipeline = OcclusionPipeline(
        config=config,
        output_dir=output_dir
    )

    # Convert to tensor
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    # Run pipeline
    output = pipeline(image_tensor, text_prompts=text_prompts)

    print(f"Pipeline completed in {output.metadata['total_time']:.2f}s")
    print(f"  Segmentation: {output.metadata.get('segmentation_time', 0):.2f}s")
    print(f"  Depth: {output.metadata.get('depth_time', 0):.2f}s")
    print(f"  Inpainting: {output.metadata.get('inpainting_time', 0):.2f}s")

    if 'foreground_label' in output.metadata:
        print(f"  Foreground: {output.metadata['foreground_label']}")
        print(f"  Background: {output.metadata['background_label']}")
        print(f"  Occlusion ratio: {output.metadata['occlusion_ratio']:.1%}")

    return output
