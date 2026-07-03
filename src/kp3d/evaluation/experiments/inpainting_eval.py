"""Inpainting (Stage 4) evaluation: SSEI.

Evaluates our PatchMatch-Guided inpainting with Dynamic Edge Morphology
against traditional and deep-learning baselines.

Pipeline V2.3 context:
- Stage 4 = Inpainting (SSEI: PatchMatch body fill + Dynamic Edge rendering)
- "Ours" = inpaint_occlusion_patchmatch_v25()

Evaluation strategy:
- Inter-object occlusion using real annotation overlap
- Object1 = occluder (front), Object2 = occludee (back)
- Occlusion mask = actual intersection of Object1 and Object2 polygons
- Metrics computed within occludee (Object2) mask region
- V25 receives proper pipeline-matching masks (distinct occluder/occludee)
- GT-based: PSNR, SSIM, LPIPS (within Object2 mask)
- GT-free: COR, BS, TC
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from kp3d.evaluation.config import EvalConfig
from kp3d.evaluation.datasets import (
    create_synthetic_occlusion,
    extract_masks_from_annotation,
    find_images,
    get_annotation_for_image,
    load_annotation,
)
from kp3d.metrics.inpainting_metrics import (
    InpaintingMetrics,
    boundary_smoothness,
    color_outlier_rate,
    texture_coherence,
)


@dataclass
class InpaintingResult:
    """Result for a single image + occlusion + method."""

    image_name: str
    method: str
    occlusion_type: str
    # GT-based
    psnr: float = 0.0
    ssim: float = 0.0
    lpips: float = 0.0
    # GT-free
    cor: float = 0.0
    bs: float = 0.0
    tc: float = 0.0


@dataclass
class MaskInfo:
    """Inter-object occlusion mask with full semantic metadata.

    Real overlap strategy:
    - mask: Actual intersection of occluder and occludee polygons
    - label: Occlusion relationship description
    - occluder_mask: Full mask of front object (Object1)
    - occludee_full_mask: Full mask of back object (Object2)
    - occludee_visible_mask: Visible part of back object (Object2 - Object1)
    """

    mask: np.ndarray  # Occlusion region (real overlap)
    label: str
    occluder_mask: Optional[np.ndarray] = None
    occludee_full_mask: Optional[np.ndarray] = None
    occludee_visible_mask: Optional[np.ndarray] = None


class InpaintingExperiment:
    """Inpainting evaluation using real inter-object occlusion from annotations.

    Workflow:
    1. Load images with annotation (object1, object2, background)
    2. Object1 = occluder (front), Object2 = occludee (back)
    3. Occlusion mask = intersection of Object1 and Object2 polygons
    4. Zero out the overlap region from original image
    5. All methods fill the overlap region
    6. Metrics computed within Object2's mask region
    7. V25 receives pipeline-matching masks with edge rendering enabled
    """

    def __init__(self, config: EvalConfig):
        self.config = config
        self.results: List[InpaintingResult] = []
        self._inpainting_metrics = InpaintingMetrics()
        self._lpips_calc = None

    def run(self) -> List[InpaintingResult]:
        """Run the full inpainting evaluation."""
        images = find_images(self.config.data_dir)
        if self.config.max_images > 0:
            images = images[: self.config.max_images]

        if not images:
            raise FileNotFoundError(
                f"No images found in {self.config.data_dir}"
            )

        # Load inpainting baselines
        from kp3d.evaluation.baselines import get_inpainting_baseline

        baselines = {}
        for name in self.config.inpainting_baselines:
            try:
                baseline = get_inpainting_baseline(name)
                if hasattr(baseline, "available") and not baseline.available:
                    print(f"  [WARN] Baseline '{name}' dependencies missing, skipping")
                    continue
                baselines[name] = baseline
            except (KeyError, RuntimeError) as e:
                print(f"  [WARN] Baseline '{name}' not available: {e}")

        for img_idx, img_path in enumerate(images):
            img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_name = img_path.stem
            print(f"  [{img_idx+1}/{len(images)}] Processing: {img_name}")

            mask_list = self._get_masks(img_path, img_rgb)

            if not mask_list:
                print(f"    [SKIP] No masks for {img_name}")
                continue

            for minfo in mask_list:
                mask = minfo.mask
                occ_type = minfo.label

                if self.config.dry_run:
                    print(
                        f"  [DRY RUN] {img_name}/{occ_type}: "
                        f"mask coverage={mask.sum() / 255 / mask.size:.1%}"
                    )
                    continue

                # Ground truth = original unoccluded image
                gt_rgb = img_rgb.copy()

                # Create occluded input (zero out overlap region)
                occluded = img_rgb.copy()
                occluded[mask > 127] = 0

                # Metrics computed within occludee (Object2) mask region
                crop_bbox = self._get_object_bbox(
                    minfo.occludee_full_mask, padding=10
                )

                # Evaluate each baseline
                for name, baseline in baselines.items():
                    try:
                        result = baseline.inpaint(occluded, mask)
                        metrics = self._compute_metrics(
                            gt_rgb, result, mask, crop_bbox
                        )
                        self.results.append(
                            InpaintingResult(
                                image_name=img_name,
                                method=name,
                                occlusion_type=occ_type,
                                **metrics,
                            )
                        )
                    except Exception as e:
                        print(f"    [ERROR] {name} on {img_name}: {e}")

                # Evaluate ours (V25) with pipeline-matching masks
                result_ours = self._run_ours(gt_rgb, minfo)
                if result_ours is not None:
                    metrics = self._compute_metrics(
                        gt_rgb, result_ours, mask, crop_bbox
                    )
                    self.results.append(
                        InpaintingResult(
                            image_name=img_name,
                            method="ours_v25",
                            occlusion_type=occ_type,
                            **metrics,
                        )
                    )

                # Free GPU memory between iterations
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

        return self.results

    def _get_masks(
        self, img_path: Path, img_rgb: np.ndarray
    ) -> List[MaskInfo]:
        """Get inter-object occlusion masks from real annotation overlap.

        Depth ordering: Object1 = occluder (front), Object2 = occludee (back).
        Occlusion mask = intersection of Object1 and Object2 polygons.
        """
        if not self.config.use_annotation_masks:
            return [
                MaskInfo(
                    mask=create_synthetic_occlusion(
                        img_rgb.shape[:2], occlusion_type=occ_type
                    ),
                    label=occ_type,
                )
                for occ_type in self.config.occlusion_types
            ]

        annot_path = get_annotation_for_image(img_path)
        if annot_path is None:
            print(f"    [SKIP] No annotation for {img_path.stem}")
            return []
        annot = load_annotation(str(annot_path))
        objects = extract_masks_from_annotation(annot, img_rgb.shape[:2])
        if not objects:
            print(f"    [SKIP] No shapes in annotation for {img_path.stem}")
            return []

        # Find object1 (occluder/front) and object2 (occludee/back)
        obj1_mask = None
        obj2_mask = None
        for obj in objects:
            label = obj["label"].lower()
            if label == "object1":
                obj1_mask = obj["mask"]
            elif label == "object2":
                obj2_mask = obj["mask"]

        if obj1_mask is None or obj2_mask is None:
            print(f"    [SKIP] Need both object1 and object2 in annotation")
            return []

        # Compute real overlap
        obj1_binary = obj1_mask > 127
        obj2_binary = obj2_mask > 127
        overlap = obj1_binary & obj2_binary
        overlap_area = np.sum(overlap)

        if overlap_area < 50:
            print(f"    [SKIP] Overlap too small ({overlap_area}px)")
            return []

        # Build masks matching pipeline.py semantics
        occlusion_mask = overlap.astype(np.uint8) * 255
        occluder_mask = obj1_mask.copy()  # Object1 = front
        occludee_full_mask = obj2_mask.copy()  # Object2 = back (full extent)
        occludee_visible = obj2_binary & ~obj1_binary
        occludee_visible_mask = occludee_visible.astype(np.uint8) * 255

        obj2_area = np.sum(obj2_binary)
        visible_px = np.sum(occludee_visible)
        overlap_ratio = overlap_area / max(obj2_area, 1)

        print(
            f"    object1(front)={np.sum(obj1_binary)}px, "
            f"object2(back)={obj2_area}px, "
            f"overlap={overlap_area}px ({overlap_ratio:.0%} of object2), "
            f"visible={visible_px}px"
        )

        return [MaskInfo(
            mask=occlusion_mask,
            label="object1_covers_object2",
            occluder_mask=occluder_mask,
            occludee_full_mask=occludee_full_mask,
            occludee_visible_mask=occludee_visible_mask,
        )]

    def _run_ours(
        self,
        image_rgb: np.ndarray,
        minfo: MaskInfo,
    ) -> Optional[np.ndarray]:
        """Run V25 with pipeline-matching inter-object masks.

        Uses distinct occluder/occludee masks matching production pipeline:
        - occluder_mask = Object1 (front object, NOT same as occlusion_mask)
        - occludee_full_mask = Object2 full polygon
        - occludee_visible_mask = Object2 minus Object1
        - edge_darkness = 0.3 (matching production)
        """
        try:
            from kp3d.modules.occlusion.inpainting import (
                inpaint_occlusion_patchmatch_v25,
            )

            occ_mask = (minfo.mask > 127).astype(np.uint8) * 255

            if minfo.occluder_mask is not None:
                occluder_mask = minfo.occluder_mask
                occludee_full = minfo.occludee_full_mask
                occludee_visible = minfo.occludee_visible_mask

                visible_px = np.sum(occludee_visible > 0)
                print(
                    f"    V25: occluder={np.sum(occluder_mask > 0)}px, "
                    f"occludee={np.sum(occludee_full > 0)}px, "
                    f"overlap={np.sum(occ_mask > 0)}px, "
                    f"visible={visible_px}px"
                )

                if visible_px < 50:
                    print("    [WARN] Too few visible occludee pixels")
                    return None

                result = inpaint_occlusion_patchmatch_v25(
                    image_rgb=image_rgb,
                    occlusion_mask=occ_mask,
                    occluder_mask=occluder_mask,
                    occludee_full_mask=occludee_full,
                    occludee_visible_mask=occludee_visible,
                    edge_darkness=0.3,
                    patch_size=7,
                    iterations=5,
                    min_edge_width=1,
                    max_edge_width=8,
                    width_smoothing_sigma=1.5,
                    min_safe_distance=3,
                )
            else:
                # Fallback: no annotation
                dist_from_mask = cv2.distanceTransform(
                    255 - occ_mask, cv2.DIST_L2, 5
                )
                nearby_context = (dist_from_mask > 0) & (dist_from_mask <= 50)
                visible_mask = nearby_context.astype(np.uint8) * 255
                full_mask = np.maximum(occ_mask, visible_mask)

                result = inpaint_occlusion_patchmatch_v25(
                    image_rgb=image_rgb,
                    occlusion_mask=occ_mask,
                    occluder_mask=occ_mask,
                    occludee_full_mask=full_mask,
                    occludee_visible_mask=visible_mask,
                    edge_darkness=1.0,
                    patch_size=7,
                    iterations=5,
                    skip_color_filter=True,
                )

            # Ensure pixels outside the mask are unchanged
            output = image_rgb.copy()
            output[minfo.mask > 127] = result[minfo.mask > 127]
            return output

        except ImportError as e:
            print(f"  [WARN] V25 inpainting not available: {e}")
            return None
        except Exception as e:
            print(f"  [ERROR] V25 inpainting failed: {e}")
            return None

    @staticmethod
    def _get_object_bbox(
        object_mask: Optional[np.ndarray],
        padding: int = 10,
    ) -> Optional[tuple]:
        """Get bounding box (y1, y2, x1, x2) of an object mask with padding."""
        if object_mask is None:
            return None
        binary = object_mask > 127
        ys, xs = np.where(binary)
        if len(ys) == 0:
            return None
        h, w = object_mask.shape[:2]
        y1 = max(0, ys.min() - padding)
        y2 = min(h, ys.max() + 1 + padding)
        x1 = max(0, xs.min() - padding)
        x2 = min(w, xs.max() + 1 + padding)
        return (y1, y2, x1, x2)

    def _compute_metrics(
        self,
        gt_rgb: np.ndarray,
        result_rgb: np.ndarray,
        mask: np.ndarray,
        crop_bbox: Optional[tuple] = None,
    ) -> Dict[str, float]:
        """Compute all inpainting metrics within occludee object region.

        - PSNR: within occlusion mask only
        - SSIM/LPIPS: within occludee (Object2) bounding box
        - COR/BS/TC: GT-free, localized by mask
        """
        metrics = {}

        mask_binary = mask > 127
        if mask_binary.sum() > 0:
            gt_region = gt_rgb[mask_binary]
            res_region = result_rgb[mask_binary]

            # PSNR in masked region only
            mse = np.mean(
                (gt_region.astype(np.float64) - res_region.astype(np.float64)) ** 2
            )
            metrics["psnr"] = 10 * np.log10(255.0**2 / max(mse, 1e-10))

            # SSIM/LPIPS: crop to occludee object region
            if crop_bbox is not None:
                y1, y2, x1, x2 = crop_bbox
                gt_crop = gt_rgb[y1:y2, x1:x2]
                res_crop = result_rgb[y1:y2, x1:x2]
            else:
                gt_crop = gt_rgb
                res_crop = result_rgb

            metrics["ssim"] = self._compute_ssim(gt_crop, res_crop)
            metrics["lpips"] = self._compute_lpips(gt_crop, res_crop)

        # GT-free metrics (localized by mask)
        metrics["cor"] = color_outlier_rate(result_rgb, mask)
        metrics["bs"] = boundary_smoothness(result_rgb, mask)
        metrics["tc"] = texture_coherence(result_rgb, mask)

        return metrics

    def _compute_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """SSIM computation (numpy)."""
        from kp3d.evaluation.experiments.restoration_eval import compute_ssim_numpy

        return compute_ssim_numpy(
            cv2.cvtColor(img1, cv2.COLOR_RGB2BGR),
            cv2.cvtColor(img2, cv2.COLOR_RGB2BGR),
        )

    def _compute_lpips(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """LPIPS computation (requires torch + lpips package)."""
        try:
            import torch

            if self._lpips_calc is None:
                from kp3d.metrics import MetricsCalculator

                self._lpips_calc = MetricsCalculator(use_lpips=True)

            t1 = (
                torch.from_numpy(img1.astype(np.float32) / 255.0)
                .permute(2, 0, 1)
                .unsqueeze(0)
            )
            t2 = (
                torch.from_numpy(img2.astype(np.float32) / 255.0)
                .permute(2, 0, 1)
                .unsqueeze(0)
            )
            return self._lpips_calc.compute_lpips(t1, t2)

        except (ImportError, RuntimeError):
            return 0.0

    def aggregate(self) -> Dict[str, Dict[str, float]]:
        """Aggregate results by method (mean across images and occlusion types)."""
        from collections import defaultdict

        method_results = defaultdict(list)
        for r in self.results:
            method_results[r.method].append(r)

        aggregated = {}
        for method, results in method_results.items():
            aggregated[method] = {
                "psnr": np.mean([r.psnr for r in results]),
                "ssim": np.mean([r.ssim for r in results]),
                "lpips": np.mean([r.lpips for r in results]),
                "cor": np.mean([r.cor for r in results]),
                "bs": np.mean([r.bs for r in results]),
                "tc": np.mean([r.tc for r in results]),
            }

        return aggregated
