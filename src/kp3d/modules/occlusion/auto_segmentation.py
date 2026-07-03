"""Automatic segmentation using SAM Automatic Mask Generator.

No prompts needed - fully automatic object detection and separation.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
import cv2
import warnings


class AutoSegmentation:
    """Fully automatic segmentation without prompts.

    Uses SAM's Automatic Mask Generator to detect all objects,
    then filters and ranks them for occlusion processing.
    """

    def __init__(
        self,
        model_type: str = "vit_h",
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.86,
        stability_score_thresh: float = 0.92,
        min_mask_region_area: int = 500,
        device: Optional[torch.device] = None
    ):
        """Initialize automatic segmentation.

        Args:
            model_type: SAM model variant ("vit_b", "vit_l", "vit_h").
            points_per_side: Grid density for point sampling.
            pred_iou_thresh: IoU prediction threshold.
            stability_score_thresh: Mask stability threshold.
            min_mask_region_area: Minimum mask area in pixels.
            device: Computation device.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.points_per_side = points_per_side
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.min_mask_region_area = min_mask_region_area

        self._sam = None
        self._mask_generator = None
        self._initialized = False

    def _load_sam(self) -> None:
        """Load SAM model and mask generator."""
        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
            import os

            # Find checkpoint
            checkpoint_paths = [
                f"sam_{self.model_type}.pth",
                f"checkpoints/sam_{self.model_type}.pth",
                os.path.expanduser(f"~/.cache/sam/sam_{self.model_type}.pth"),
                f"C:/Users/admin/.cache/sam/sam_{self.model_type}.pth",
            ]

            checkpoint = None
            for path in checkpoint_paths:
                if os.path.exists(path):
                    checkpoint = path
                    break

            if checkpoint is None:
                raise FileNotFoundError(
                    f"SAM checkpoint not found. Download from: "
                    f"https://github.com/facebookresearch/segment-anything#model-checkpoints"
                )

            print(f"Loading SAM from: {checkpoint}")
            self._sam = sam_model_registry[self.model_type](checkpoint=checkpoint)
            self._sam.to(self.device)

            self._mask_generator = SamAutomaticMaskGenerator(
                self._sam,
                points_per_side=self.points_per_side,
                pred_iou_thresh=self.pred_iou_thresh,
                stability_score_thresh=self.stability_score_thresh,
                min_mask_region_area=self.min_mask_region_area,
            )

        except ImportError:
            raise ImportError(
                "segment_anything not found. Install with: pip install segment-anything"
            )

    def _ensure_loaded(self) -> None:
        """Ensure model is loaded."""
        if not self._initialized:
            self._load_sam()
            self._initialized = True

    def generate_all_masks(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Generate all masks from image.

        Args:
            image: RGB image (H, W, 3), uint8.

        Returns:
            List of mask dicts with keys:
            - segmentation: binary mask (H, W)
            - area: mask area in pixels
            - bbox: [x, y, w, h]
            - predicted_iou: IoU prediction score
            - stability_score: mask stability score
        """
        self._ensure_loaded()
        return self._mask_generator.generate(image)

    def filter_masks(
        self,
        masks: List[Dict[str, Any]],
        min_area: Optional[int] = None,
        max_area: Optional[int] = None,
        min_stability: Optional[float] = None,
        top_n: Optional[int] = None,
        exclude_background: bool = True,
        image_shape: Optional[Tuple[int, int]] = None
    ) -> List[Dict[str, Any]]:
        """Filter masks by various criteria.

        Args:
            masks: List of mask dicts from generate_all_masks.
            min_area: Minimum mask area.
            max_area: Maximum mask area.
            min_stability: Minimum stability score.
            top_n: Keep only top N by area.
            exclude_background: Exclude masks that cover >80% of image.
            image_shape: (H, W) for background detection.

        Returns:
            Filtered list of masks.
        """
        filtered = masks.copy()

        # Area filters
        if min_area is not None:
            filtered = [m for m in filtered if m['area'] >= min_area]

        if max_area is not None:
            filtered = [m for m in filtered if m['area'] <= max_area]

        # Stability filter
        if min_stability is not None:
            filtered = [m for m in filtered if m['stability_score'] >= min_stability]

        # Exclude background (masks covering most of image)
        if exclude_background and image_shape is not None:
            total_pixels = image_shape[0] * image_shape[1]
            filtered = [m for m in filtered if m['area'] < total_pixels * 0.8]

        # Sort by area (descending) and take top N
        filtered = sorted(filtered, key=lambda x: x['area'], reverse=True)

        if top_n is not None:
            filtered = filtered[:top_n]

        return filtered

    def rank_by_depth(
        self,
        masks: List[Dict[str, Any]],
        depth_map: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Rank masks by mean depth value.

        Args:
            masks: List of mask dicts.
            depth_map: Depth map (H, W), higher = farther.

        Returns:
            Masks sorted by depth (foreground first).
        """
        for mask in masks:
            seg = mask['segmentation']
            if np.sum(seg) > 0:
                mask['mean_depth'] = np.mean(depth_map[seg > 0])
            else:
                mask['mean_depth'] = float('inf')

        # Sort by depth (lower = foreground = first)
        return sorted(masks, key=lambda x: x['mean_depth'])

    def find_best_pair(
        self,
        masks: List[Dict[str, Any]],
        min_overlap: float = 0.0,  # Changed: allow zero overlap
        max_overlap: float = 0.5
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Find best foreground/background pair from masks.

        Looks for two masks that:
        1. Are distinct objects (not one containing the other)
        2. Are adjacient or overlapping (related objects)
        3. Neither is the full background

        Args:
            masks: List of mask dicts.
            min_overlap: Minimum overlap ratio to consider related.
            max_overlap: Maximum overlap to consider distinct.

        Returns:
            Best (foreground, background) pair.
        """
        sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)

        # Exclude the largest mask if it's likely background (>50% of image)
        # and there are enough other masks
        if len(sorted_masks) > 2:
            largest = sorted_masks[0]
            if len(masks) > 0 and largest['area'] > sum(m['area'] for m in sorted_masks[1:]):
                # Largest mask is bigger than all others combined - likely background
                sorted_masks = sorted_masks[1:]

        best_pair = None
        best_score = -1

        for i, mask_a in enumerate(sorted_masks):
            for j, mask_b in enumerate(sorted_masks):
                if i >= j:
                    continue

                seg_a = mask_a['segmentation']
                seg_b = mask_b['segmentation']

                intersection = np.logical_and(seg_a, seg_b).sum()
                union = np.logical_or(seg_a, seg_b).sum()

                if union == 0:
                    continue

                iou = intersection / union
                overlap_ratio_a = intersection / seg_a.sum() if seg_a.sum() > 0 else 0
                overlap_ratio_b = intersection / seg_b.sum() if seg_b.sum() > 0 else 0

                # Skip if one completely contains the other
                if overlap_ratio_a > 0.8 or overlap_ratio_b > 0.8:
                    continue

                # Check adjacency: dilate masks and check overlap
                kernel = np.ones((15, 15), np.uint8)
                dilated_a = cv2.dilate(seg_a.astype(np.uint8), kernel, iterations=1)
                dilated_b = cv2.dilate(seg_b.astype(np.uint8), kernel, iterations=1)
                adjacent = np.logical_and(dilated_a, dilated_b).sum() > 0

                # Score: prefer similar-sized objects that are adjacent
                area_a, area_b = mask_a['area'], mask_b['area']
                size_ratio = min(area_a, area_b) / max(area_a, area_b)

                # Prefer pairs with similar sizes (size_ratio close to 1)
                # and that are adjacent or overlapping
                if adjacent or intersection > 0:
                    score = (area_a + area_b) * size_ratio * 2
                else:
                    score = (area_a + area_b) * size_ratio * 0.5

                if score > best_score:
                    best_score = score
                    # Determine which is foreground based on vertical position
                    center_y_a = np.mean(np.where(seg_a)[0]) if seg_a.sum() > 0 else 0
                    center_y_b = np.mean(np.where(seg_b)[0]) if seg_b.sum() > 0 else 0

                    if center_y_a < center_y_b:  # A is higher (foreground)
                        best_pair = (mask_a, mask_b)
                    else:
                        best_pair = (mask_b, mask_a)

        return best_pair if best_pair else (sorted_masks[0], sorted_masks[1])

    def select_foreground_background(
        self,
        masks: List[Dict[str, Any]],
        depth_map: Optional[np.ndarray] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Select foreground and background from masks.

        Uses smart pairing to find two distinct objects that overlap.

        Args:
            masks: List of mask dicts (at least 2).
            depth_map: Optional depth map for ranking.

        Returns:
            Tuple of (foreground_mask, background_mask).
        """
        if len(masks) < 2:
            raise ValueError("Need at least 2 masks")

        # Find best pair of distinct, overlapping objects
        fg, bg = self.find_best_pair(masks)

        print(f"  Selected pair - FG: {fg['area']} px, BG: {bg['area']} px")

        # If depth map available, verify/correct ordering
        # Only swap if depth difference is significant (>5%)
        if depth_map is not None:
            fg_depth = np.mean(depth_map[fg['segmentation'] > 0])
            bg_depth = np.mean(depth_map[bg['segmentation'] > 0])
            depth_diff = abs(fg_depth - bg_depth)

            print(f"  Depth: FG={fg_depth:.3f}, BG={bg_depth:.3f}, diff={depth_diff:.3f}")

            # Only use depth if difference is significant
            if depth_diff > 0.05:
                if fg_depth > bg_depth:  # Higher depth = farther = background
                    fg, bg = bg, fg
                    print("  Swapped based on depth (significant difference)")
            else:
                print("  Depth difference too small, keeping position-based order")

        return fg, bg

    def segment_auto(
        self,
        image: np.ndarray,
        depth_map: Optional[np.ndarray] = None,
        min_area_ratio: float = 0.01,
        max_area_ratio: float = 0.8,
        top_n: int = 5
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Full automatic segmentation pipeline.

        Args:
            image: RGB image (H, W, 3).
            depth_map: Optional depth map for ordering.
            min_area_ratio: Minimum mask area as ratio of image.
            max_area_ratio: Maximum mask area as ratio of image.
            top_n: Number of top masks to consider.

        Returns:
            Tuple of (foreground_mask, background_mask, all_filtered_masks).
        """
        h, w = image.shape[:2]
        total_pixels = h * w

        print("Generating all masks...")
        all_masks = self.generate_all_masks(image)
        print(f"  Generated {len(all_masks)} masks")

        # Filter
        print("Filtering masks...")
        filtered = self.filter_masks(
            all_masks,
            min_area=int(total_pixels * min_area_ratio),
            max_area=int(total_pixels * max_area_ratio),
            min_stability=0.9,
            top_n=top_n,
            exclude_background=True,
            image_shape=(h, w)
        )
        print(f"  Filtered to {len(filtered)} masks")

        if len(filtered) < 2:
            warnings.warn("Less than 2 significant masks found")
            if len(filtered) == 1:
                return filtered[0]['segmentation'], np.zeros((h, w), dtype=np.uint8), filtered
            return np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8), filtered

        # Select foreground/background
        print("Selecting foreground/background...")
        fg, bg = self.select_foreground_background(filtered, depth_map)

        print(f"  Foreground: {fg['area']} pixels")
        print(f"  Background: {bg['area']} pixels")

        return (
            fg['segmentation'].astype(np.uint8) * 255,
            bg['segmentation'].astype(np.uint8) * 255,
            filtered
        )

    def _ensure_predictor(self) -> None:
        """Ensure SamPredictor is initialized."""
        if not hasattr(self, '_predictor') or self._predictor is None:
            from segment_anything import SamPredictor
            self._ensure_loaded()
            self._predictor = SamPredictor(self._sam)

    def segment_with_points(
        self,
        image: np.ndarray,
        foreground_points: List[Tuple[int, int]],
        background_points: Optional[List[Tuple[int, int]]] = None,
        multimask_output: bool = False
    ) -> np.ndarray:
        """Segment using point prompts.

        Args:
            image: RGB image (H, W, 3).
            foreground_points: List of (x, y) points on the object.
            background_points: Optional list of (x, y) points on background.
            multimask_output: Return multiple mask options.

        Returns:
            Binary mask (H, W), uint8, 0 or 255.
        """
        self._ensure_predictor()

        # Set image
        self._predictor.set_image(image)

        # Prepare point coordinates and labels
        points = []
        labels = []

        # Add foreground points (label=1)
        for x, y in foreground_points:
            points.append([x, y])
            labels.append(1)

        # Add background points (label=0)
        if background_points:
            for x, y in background_points:
                points.append([x, y])
                labels.append(0)

        point_coords = np.array(points, dtype=np.float32)
        point_labels = np.array(labels, dtype=np.int32)

        # Predict
        masks, scores, logits = self._predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=multimask_output
        )

        # Select best mask
        if multimask_output:
            # Return mask with highest score
            best_idx = np.argmax(scores)
            mask = masks[best_idx]
        else:
            mask = masks[0]

        return (mask.astype(np.uint8) * 255)

    def segment_with_box(
        self,
        image: np.ndarray,
        box: Tuple[int, int, int, int],
        multimask_output: bool = False
    ) -> np.ndarray:
        """Segment using bounding box prompt.

        Args:
            image: RGB image (H, W, 3).
            box: Bounding box as (x1, y1, x2, y2).
            multimask_output: Return multiple mask options.

        Returns:
            Binary mask (H, W), uint8, 0 or 255.
        """
        self._ensure_predictor()

        # Set image
        self._predictor.set_image(image)

        # Prepare box
        box_array = np.array(box, dtype=np.float32)

        # Predict
        masks, scores, logits = self._predictor.predict(
            box=box_array,
            multimask_output=multimask_output
        )

        # Select best mask
        if multimask_output:
            best_idx = np.argmax(scores)
            mask = masks[best_idx]
        else:
            mask = masks[0]

        return (mask.astype(np.uint8) * 255)

    def segment_with_prompts(
        self,
        image: np.ndarray,
        foreground_points: Optional[List[Tuple[int, int]]] = None,
        background_points: Optional[List[Tuple[int, int]]] = None,
        box: Optional[Tuple[int, int, int, int]] = None,
        multimask_output: bool = False
    ) -> np.ndarray:
        """Segment using combined point and box prompts.

        Args:
            image: RGB image (H, W, 3).
            foreground_points: Points on the object.
            background_points: Points on background.
            box: Optional bounding box.
            multimask_output: Return multiple mask options.

        Returns:
            Binary mask (H, W), uint8, 0 or 255.
        """
        self._ensure_predictor()

        # Set image
        self._predictor.set_image(image)

        # Prepare points if provided
        point_coords = None
        point_labels = None

        if foreground_points or background_points:
            points = []
            labels = []

            if foreground_points:
                for x, y in foreground_points:
                    points.append([x, y])
                    labels.append(1)

            if background_points:
                for x, y in background_points:
                    points.append([x, y])
                    labels.append(0)

            point_coords = np.array(points, dtype=np.float32)
            point_labels = np.array(labels, dtype=np.int32)

        # Prepare box if provided
        box_array = None
        if box is not None:
            box_array = np.array(box, dtype=np.float32)

        # Predict
        masks, scores, logits = self._predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box_array,
            multimask_output=multimask_output
        )

        # Select best mask
        if multimask_output:
            best_idx = np.argmax(scores)
            mask = masks[best_idx]
        else:
            mask = masks[0]

        return (mask.astype(np.uint8) * 255)


def auto_segment_and_refine(
    image: np.ndarray,
    depth_map: Optional[np.ndarray] = None,
    use_refiner: bool = True,
    n_colors: int = 3,
    color_tolerance: int = 30
) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience function for full auto pipeline.

    Args:
        image: RGB image.
        depth_map: Optional depth map.
        use_refiner: Apply AutoMaskRefiner post-processing.
        n_colors: Number of colors for refiner.
        color_tolerance: Color tolerance for refiner.

    Returns:
        Tuple of (foreground_mask, background_mask).
    """
    # Auto segmentation
    segmenter = AutoSegmentation()
    fg_mask, bg_mask, _ = segmenter.segment_auto(image, depth_map)

    # Optional refinement
    if use_refiner and np.sum(fg_mask) > 0 and np.sum(bg_mask) > 0:
        from kp3d.modules.occlusion.auto_mask_refinement import AutoMaskRefiner

        refiner = AutoMaskRefiner(
            n_colors=n_colors,
            color_tolerance=color_tolerance
        )
        fg_mask, bg_mask = refiner.refine_both_masks(image, fg_mask, bg_mask)

    return fg_mask, bg_mask
