"""SAM-based mask refinement for LabelMe polygon annotations.

Refines coarse polygon masks from LabelMe into pixel-precise masks using SAM,
specifically designed for Korean painting objects where auto segmentation fails
but manual polygon annotations provide good approximate boundaries.

Strategy:
- Positive points: sampled from eroded interior (confident object region)
- Negative points: sampled from dilated exterior ring (confident background)
- Box prompt: bounding box of rough mask with padding
- 2-pass refinement: first predict -> logit -> second predict (SAM best practice)
- Margin constraint: SAM result clipped to dilated rough mask to prevent over-expansion
"""

from typing import List, Optional, Tuple
import numpy as np
import cv2


class SAMMaskRefiner:
    """Refine LabelMe polygon masks using SAM point/box prompts.

    Takes coarse polygon masks and produces pixel-accurate boundaries
    by leveraging SAM's ability to snap to object edges when given
    appropriate positive/negative point prompts.

    All pixel parameters (margin_px, erode_px, dilate_px) are reference
    values calibrated for 512px images.  When *adaptive=True* (default),
    they are scaled proportionally to each object's bounding-box size so
    that low-resolution images don't suffer from excessive expansion.

    Args:
        sam_predictor: Pre-initialized SAM predictor (SAM2ImagePredictor or SamPredictor).
        margin_px: Maximum expansion beyond original mask (pixels, ref 512px).
        min_area_ratio: Safety threshold - if refined area < rough_area * ratio, keep original.
        erode_px: Erosion radius for positive point sampling region (ref 512px).
        dilate_px: Dilation radius for negative point sampling ring (ref 512px).
        num_positive: Number of positive (foreground) prompt points.
        num_negative: Number of negative (background) prompt points.
        adaptive: Scale px params by object bbox size (reference = 512px).
    """

    _REFERENCE_SIZE: int = 512  # px params calibrated at this resolution

    def __init__(
        self,
        sam_predictor,
        margin_px: int = 15,
        min_area_ratio: float = 0.3,
        erode_px: int = 10,
        dilate_px: int = 15,
        num_positive: int = 5,
        num_negative: int = 8,
        adaptive: bool = True,
    ):
        self._predictor = sam_predictor
        self.margin_px = margin_px
        self.min_area_ratio = min_area_ratio
        self.erode_px = erode_px
        self.dilate_px = dilate_px
        self.num_positive = num_positive
        self.num_negative = num_negative
        self.adaptive = adaptive

    # ------------------------------------------------------------------
    # Adaptive scaling
    # ------------------------------------------------------------------

    def _adaptive_px(self, base_px: int, mask: np.ndarray, min_px: int = 1) -> int:
        """Scale a reference pixel value by the object's bbox size.

        The reference values (margin_px, erode_px, dilate_px) are designed
        for ~512px images.  For smaller objects / images the pixel values
        are scaled down proportionally so that they represent the same
        *fraction* of the object rather than a fixed absolute distance.

        Args:
            base_px: Reference pixel value (at 512px).
            mask: Binary mask of the object (H, W), uint8.
            min_px: Floor value to avoid zero-pixel kernels.

        Returns:
            Scaled pixel value, clamped to [min_px, base_px].
        """
        if not self.adaptive:
            return base_px

        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return base_px

        bbox_w = int(xs.max()) - int(xs.min()) + 1
        bbox_h = int(ys.max()) - int(ys.min()) + 1
        obj_size = max(bbox_w, bbox_h)

        scale = obj_size / self._REFERENCE_SIZE
        scaled = max(min_px, round(base_px * scale))
        # Never exceed the original reference value
        return min(scaled, base_px)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refine_mask(
        self,
        image: np.ndarray,
        rough_mask: np.ndarray,
        label: str,
        *,
        set_image: bool = False,
    ) -> np.ndarray:
        """Refine a single polygon mask using SAM.

        Args:
            image: RGB image (H, W, 3), uint8.  Only used if *set_image* is True.
            rough_mask: Binary mask (H, W), uint8, values 0 or 255.
            label: Shape label.  Masks whose label starts with "background"
                   are returned unchanged.
            set_image: If True, call ``predictor.set_image(image)`` before
                       prediction.  When processing multiple masks for the
                       same image, set this to True only for the first call.

        Returns:
            Refined binary mask (H, W), uint8, values 0 or 255.
        """
        # Skip background labels
        if label.lower().startswith("background"):
            return rough_mask

        # Skip tiny masks
        rough_area = np.sum(rough_mask > 0)
        if rough_area < 100:
            return rough_mask

        if set_image:
            self._predictor.set_image(image)

        # --- Compute adaptive pixel values for this object ---
        a_margin = self._adaptive_px(self.margin_px, rough_mask, min_px=2)
        a_erode = self._adaptive_px(self.erode_px, rough_mask, min_px=1)
        a_dilate = self._adaptive_px(self.dilate_px, rough_mask, min_px=2)

        # --- Prompt generation ---
        box = self._bbox_with_padding(rough_mask, pad_px=a_margin // 2)
        pos_points = self._sample_positive_points(rough_mask, erode_px=a_erode)
        neg_points = self._sample_negative_points(rough_mask, dilate_px=a_dilate)

        if pos_points is None or len(pos_points) == 0:
            return rough_mask

        point_coords = np.concatenate([pos_points, neg_points], axis=0) if neg_points is not None and len(neg_points) > 0 else pos_points
        point_labels = np.array(
            [1] * len(pos_points) + [0] * (len(neg_points) if neg_points is not None else 0),
            dtype=np.int32,
        )

        # --- 1st pass ---
        masks_1, scores_1, logits_1 = self._predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=True,
        )
        best_idx = int(np.argmax(scores_1))
        best_logit = logits_1[best_idx][None, :, :]  # (1, H_low, W_low)

        # --- 2nd pass (refine with logit) ---
        masks_2, scores_2, _ = self._predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            mask_input=best_logit,
            multimask_output=False,
        )
        refined = masks_2[0]  # bool array (H, W)

        # Convert to uint8 mask
        refined_mask = (refined > 0).astype(np.uint8) * 255

        # --- Constrain to original mask (shrink-only: never add background) ---
        refined_mask = cv2.bitwise_and(refined_mask, rough_mask)

        # --- Preserve eroded interior (prevent SAM from creating holes) ---
        interior_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * a_erode + 1, 2 * a_erode + 1),
        )
        eroded_interior = cv2.erode(rough_mask, interior_kernel, iterations=1)
        if np.sum(eroded_interior > 0) >= 10:
            refined_mask = cv2.bitwise_or(refined_mask, eroded_interior)

        # --- Safety check ---
        refined_area = np.sum(refined_mask > 0)
        if refined_area < rough_area * self.min_area_ratio:
            print(f"    [SAM Refine] {label}: area too small ({refined_area}/{rough_area}={refined_area/rough_area:.1%}), keeping original")
            return rough_mask

        area_change = (refined_area - rough_area) / rough_area * 100
        adaptive_tag = f" [margin={a_margin},erode={a_erode},dilate={a_dilate}]" if self.adaptive else ""
        print(f"    [SAM Refine] {label}: {rough_area} -> {refined_area} px ({area_change:+.1f}%){adaptive_tag}")
        return refined_mask

    def refine_all_shapes(
        self,
        image: np.ndarray,
        shapes: list,
        occlusion_labels: dict = None,
    ) -> list:
        """Refine all shapes in a LabelMe annotation list.

        Uses a 2-pass approach when occlusion info is provided:
        1. SAM-refine all shapes independently
        2. For occluded objects, preserve the occluded region using the
           **refined** occluder masks (not original) to avoid boundary mismatch

        Args:
            image: RGB image (H, W, 3), uint8.
            shapes: List of LabelMe shape dicts, each with at least
                    ``label``, ``points``, ``shape_type``.
            occlusion_labels: Optional dict mapping occludee_label to list of
                              occluder labels.  Used in pass 2 to preserve
                              occluded regions with refined occluder masks.

        Returns:
            New list of shape dicts with refined polygon points.
        """
        h, w = image.shape[:2]

        # Set image once for all masks
        self._predictor.set_image(image)

        # --- Pass 1: SAM-refine every shape independently ---
        shape_data = []  # (label, points, shape_type, rough_mask, refined_mask)
        for shape in shapes:
            label = shape["label"]
            points = shape["points"]
            shape_type = shape.get("shape_type", "polygon")

            rough_mask = np.zeros((h, w), dtype=np.uint8)
            pts = np.array(points, dtype=np.int32)
            cv2.fillPoly(rough_mask, [pts], 255)

            refined_mask = self.refine_mask(image, rough_mask, label, set_image=False)
            shape_data.append((label, points, shape_type, rough_mask, refined_mask))

        # --- Pass 2: preserve occluded regions using REFINED occluder masks ---
        if occlusion_labels:
            # Build label -> combined refined mask lookup
            refined_by_label: dict = {}
            for label, _, _, _, refined_mask in shape_data:
                if label not in refined_by_label:
                    refined_by_label[label] = np.zeros((h, w), dtype=np.uint8)
                refined_by_label[label] = np.maximum(refined_by_label[label], refined_mask)

            updated = []
            for label, points, shape_type, rough_mask, refined_mask in shape_data:
                if label in occlusion_labels:
                    # Build combined REFINED occluder mask
                    refined_occluder = np.zeros((h, w), dtype=np.uint8)
                    for occluder_label in occlusion_labels[label]:
                        if occluder_label in refined_by_label:
                            refined_occluder = np.maximum(
                                refined_occluder, refined_by_label[occluder_label]
                            )

                    # Occluded region = original rough polygon AND refined occluder
                    occluded_region = cv2.bitwise_and(rough_mask, refined_occluder)
                    refined_mask = cv2.bitwise_or(refined_mask, occluded_region)
                    preserved_px = int(np.sum(occluded_region > 0))
                    if preserved_px > 0:
                        print(f"    [SAM Refine] {label}: preserved {preserved_px} occluded px")

                updated.append((label, points, shape_type, rough_mask, refined_mask))
            shape_data = updated

        # --- Convert masks to polygon shapes ---
        refined_shapes = []
        for label, points, shape_type, rough_mask, refined_mask in shape_data:
            if np.array_equal(refined_mask, rough_mask):
                refined_shapes.append({
                    "label": label,
                    "points": points,
                    "shape_type": shape_type,
                })
            else:
                new_points = self._mask_to_polygon(refined_mask)
                if new_points is not None and len(new_points) >= 3:
                    refined_shapes.append({
                        "label": label,
                        "points": new_points,
                        "shape_type": shape_type,
                    })
                else:
                    refined_shapes.append({
                        "label": label,
                        "points": points,
                        "shape_type": shape_type,
                    })

        return refined_shapes

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bbox_with_padding(self, mask: np.ndarray, *, pad_px: int = None) -> np.ndarray:
        """Get bounding box of mask with padding.

        Args:
            mask: Binary mask (H, W), uint8.
            pad_px: Padding in pixels.  Falls back to ``margin_px // 2``.

        Returns:
            np.ndarray of shape (4,): [x1, y1, x2, y2].
        """
        h, w = mask.shape[:2]
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return np.array([0, 0, w, h], dtype=np.float32)

        pad = pad_px if pad_px is not None else self.margin_px // 2
        x1 = max(0, int(xs.min()) - pad)
        y1 = max(0, int(ys.min()) - pad)
        x2 = min(w, int(xs.max()) + pad)
        y2 = min(h, int(ys.max()) + pad)
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def _sample_positive_points(self, mask: np.ndarray, *, erode_px: int = None) -> Optional[np.ndarray]:
        """Sample positive points from eroded mask interior.

        Args:
            mask: Binary mask (H, W), uint8.
            erode_px: Erosion radius.  Falls back to ``self.erode_px``.

        Returns:
            np.ndarray of shape (N, 2) with (x, y) coordinates, or None.
        """
        epx = erode_px if erode_px is not None else self.erode_px
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * epx + 1, 2 * epx + 1),
        )
        eroded = cv2.erode(mask, kernel, iterations=1)

        # Fallback: if erosion eliminates everything, use original mask
        if np.sum(eroded > 0) < 10:
            eroded = mask

        ys, xs = np.where(eroded > 0)
        if len(xs) == 0:
            return None

        n = min(self.num_positive, len(xs))
        indices = np.linspace(0, len(xs) - 1, n, dtype=int)
        return np.stack([xs[indices], ys[indices]], axis=1).astype(np.float32)

    def _sample_negative_points(self, mask: np.ndarray, *, dilate_px: int = None) -> Optional[np.ndarray]:
        """Sample negative points from the ring outside the mask.

        The ring is: dilate(mask) - mask.

        Args:
            mask: Binary mask (H, W), uint8.
            dilate_px: Dilation radius.  Falls back to ``self.dilate_px``.

        Returns:
            np.ndarray of shape (N, 2) with (x, y) coordinates, or None.
        """
        dpx = dilate_px if dilate_px is not None else self.dilate_px
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * dpx + 1, 2 * dpx + 1),
        )
        dilated = cv2.dilate(mask, kernel, iterations=1)
        ring = cv2.bitwise_and(dilated, cv2.bitwise_not(mask))

        ys, xs = np.where(ring > 0)
        if len(xs) == 0:
            return None

        n = min(self.num_negative, len(xs))
        indices = np.linspace(0, len(xs) - 1, n, dtype=int)
        return np.stack([xs[indices], ys[indices]], axis=1).astype(np.float32)

    @staticmethod
    def _mask_to_polygon(mask: np.ndarray) -> Optional[list]:
        """Convert binary mask to polygon points (largest contour).

        Returns:
            List of [x, y] pairs, or None if no contour found.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
        if not contours:
            return None

        # Pick largest contour
        largest = max(contours, key=cv2.contourArea)

        # Simplify slightly to reduce point count
        epsilon = 0.5  # very mild approximation
        approx = cv2.approxPolyDP(largest, epsilon, True)

        return [[float(pt[0][0]), float(pt[0][1])] for pt in approx]
