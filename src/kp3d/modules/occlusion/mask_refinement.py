"""Mask refinement module for color-based post-processing.

Refines SAM segmentation masks using color information to:
1. Remove soban (red) pixels from ceramic mask
2. Remove ceramic (white) pixels from soban mask
3. Fill internal holes
4. Remove noise components
"""

from typing import Tuple, Optional
import numpy as np
import cv2


class MaskRefiner:
    """Refines segmentation masks using color-based filtering.

    Designed for Korean traditional painting separation where:
    - Ceramic (foreground): white/light colored
    - Soban (background): red/brown colored
    """

    def __init__(
        self,
        # Red (soban) color range in HSV
        red_h_low1: int = 0,
        red_h_high1: int = 12,
        red_h_low2: int = 168,
        red_h_high2: int = 180,
        red_s_min: int = 60,
        red_v_min: int = 40,
        red_v_max: int = 200,
        # Morphology parameters
        open_kernel_size: int = 3,
        close_kernel_size: int = 5,
        # Convex hull for top region
        top_hull_ratio: float = 0.35,
    ):
        """Initialize mask refiner.

        Args:
            red_h_low1: Lower hue for red detection (0-180).
            red_h_high1: Upper hue for red detection (first range).
            red_h_low2: Lower hue for red detection (second range).
            red_h_high2: Upper hue for red detection (180).
            red_s_min: Minimum saturation for red.
            red_v_min: Minimum value for red.
            red_v_max: Maximum value for red.
            open_kernel_size: Kernel size for morphological opening.
            close_kernel_size: Kernel size for morphological closing.
            top_hull_ratio: Ratio of top region for convex hull fill.
        """
        self.red_lower1 = np.array([red_h_low1, red_s_min, red_v_min])
        self.red_upper1 = np.array([red_h_high1, 255, red_v_max])
        self.red_lower2 = np.array([red_h_low2, red_s_min, red_v_min])
        self.red_upper2 = np.array([red_h_high2, 255, red_v_max])

        self.open_kernel_size = open_kernel_size
        self.close_kernel_size = close_kernel_size
        self.top_hull_ratio = top_hull_ratio

    def detect_red_mask(self, image_rgb: np.ndarray) -> np.ndarray:
        """Detect red (soban) pixels in image.

        Args:
            image_rgb: RGB image (H, W, 3), uint8.

        Returns:
            Binary mask of red pixels (H, W), uint8.
        """
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

        mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)

        return cv2.bitwise_or(mask1, mask2)

    def fill_holes(self, mask: np.ndarray) -> np.ndarray:
        """Fill internal holes in mask using flood fill.

        Args:
            mask: Binary mask (H, W), uint8.

        Returns:
            Mask with holes filled.
        """
        h, w = mask.shape

        # Invert mask
        inverted = cv2.bitwise_not(mask)

        # Flood fill from corner (marks background)
        flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        cv2.floodFill(inverted, flood_mask, (0, 0), 0)

        # Remaining white pixels are holes - add to original mask
        return cv2.bitwise_or(mask, inverted)

    def keep_largest_component(self, mask: np.ndarray) -> np.ndarray:
        """Keep only the largest connected component.

        Args:
            mask: Binary mask (H, W), uint8.

        Returns:
            Mask with only largest component.
        """
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

        if num_labels <= 2:
            return mask

        # Find largest component (excluding background label 0)
        areas = stats[1:, cv2.CC_STAT_AREA]
        main_label = 1 + np.argmax(areas)

        return (labels == main_label).astype(np.uint8) * 255

    def apply_top_convex_hull(
        self,
        mask: np.ndarray,
        exclude_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Apply convex hull fill to top region of mask.

        Args:
            mask: Binary mask (H, W), uint8.
            exclude_mask: Pixels to exclude from hull fill (e.g., red mask).

        Returns:
            Mask with top region filled by convex hull.
        """
        h, w = mask.shape

        # Find mask bounds
        rows = np.where(np.any(mask > 0, axis=1))[0]
        if len(rows) == 0:
            return mask

        y1, y2 = rows[0], rows[-1]
        top_cutoff = y1 + int((y2 - y1) * self.top_hull_ratio)

        # Extract top region
        top_region = np.zeros_like(mask)
        top_region[:top_cutoff] = mask[:top_cutoff]

        # Find contours and compute convex hull
        contours, _ = cv2.findContours(
            top_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return mask

        all_points = np.vstack(contours)
        hull = cv2.convexHull(all_points)

        # Create hull mask
        hull_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(hull_mask, [hull], 255)
        hull_mask[top_cutoff:] = 0

        # Exclude specified pixels (e.g., red)
        if exclude_mask is not None:
            hull_mask = cv2.bitwise_and(hull_mask, cv2.bitwise_not(exclude_mask))

        # Combine with original mask
        result = mask.copy()
        result[:top_cutoff] = cv2.bitwise_or(mask[:top_cutoff], hull_mask[:top_cutoff])

        return result

    def refine_ceramic_mask(
        self,
        image_rgb: np.ndarray,
        sam_mask: np.ndarray,
        notch_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Refine ceramic mask by removing red (soban) pixels.

        Args:
            image_rgb: RGB image (H, W, 3), uint8.
            sam_mask: SAM segmentation mask for ceramic (H, W), uint8.
            notch_mask: Optional mask of edge notches to restore.

        Returns:
            Refined ceramic mask.
        """
        # Detect red pixels
        red_mask = self.detect_red_mask(image_rgb)

        # Remove red from ceramic mask
        ceramic = cv2.bitwise_and(sam_mask, cv2.bitwise_not(red_mask))

        # Morphological cleanup
        kernel = np.ones((self.open_kernel_size, self.open_kernel_size), np.uint8)
        ceramic = cv2.morphologyEx(ceramic, cv2.MORPH_OPEN, kernel)

        # Fill holes
        ceramic = self.fill_holes(ceramic)

        # Keep largest component
        ceramic = self.keep_largest_component(ceramic)

        # Restore edge notches if provided
        if notch_mask is not None:
            # Only add notches that touch ceramic and are not red
            kernel_dilate = np.ones((3, 3), np.uint8)
            ceramic_dilated = cv2.dilate(ceramic, kernel_dilate)
            valid_notches = cv2.bitwise_and(notch_mask, ceramic_dilated)
            valid_notches = cv2.bitwise_and(valid_notches, cv2.bitwise_not(red_mask))
            ceramic = cv2.bitwise_or(ceramic, valid_notches)

        # Apply convex hull to top region
        ceramic = self.apply_top_convex_hull(ceramic, exclude_mask=red_mask)

        return ceramic

    def refine_soban_mask(
        self,
        image_rgb: np.ndarray,
        inpainted_rgb: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Extract soban mask using color-based detection.

        Args:
            image_rgb: RGB image (H, W, 3), uint8.
            inpainted_rgb: Optional inpainted image to extract from.

        Returns:
            Refined soban mask.
        """
        source = inpainted_rgb if inpainted_rgb is not None else image_rgb

        # Detect red pixels
        red_mask = self.detect_red_mask(source)

        # Morphological cleanup
        kernel = np.ones((self.close_kernel_size, self.close_kernel_size), np.uint8)
        soban = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        soban = cv2.morphologyEx(soban, cv2.MORPH_OPEN, kernel)

        # Keep largest component
        soban = self.keep_largest_component(soban)

        return soban

    def extract_object(
        self,
        image_rgb: np.ndarray,
        mask: np.ndarray,
        background_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> np.ndarray:
        """Extract object from image using mask.

        Args:
            image_rgb: RGB image (H, W, 3), uint8.
            mask: Binary mask (H, W), uint8.
            background_color: Background color for non-masked pixels.

        Returns:
            Extracted object image (H, W, 3), uint8.
        """
        h, w = mask.shape
        result = np.ones((h, w, 3), dtype=np.uint8) * np.array(background_color, dtype=np.uint8)
        result[mask > 0] = image_rgb[mask > 0]
        return result


def refine_masks_for_pipeline(
    image_rgb: np.ndarray,
    ceramic_sam_mask: np.ndarray,
    soban_sam_mask: np.ndarray,
    ceramic_notch_mask: Optional[np.ndarray] = None,
    inpainted_rgb: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience function to refine both masks.

    Args:
        image_rgb: RGB image (H, W, 3), uint8.
        ceramic_sam_mask: SAM mask for ceramic.
        soban_sam_mask: SAM mask for soban.
        ceramic_notch_mask: Optional notch mask for ceramic edge restoration.
        inpainted_rgb: Optional inpainted image for soban extraction.

    Returns:
        Tuple of (refined_ceramic_mask, refined_soban_mask).
    """
    refiner = MaskRefiner()

    ceramic_refined = refiner.refine_ceramic_mask(
        image_rgb, ceramic_sam_mask, ceramic_notch_mask
    )

    soban_refined = refiner.refine_soban_mask(
        image_rgb, inpainted_rgb
    )

    return ceramic_refined, soban_refined
