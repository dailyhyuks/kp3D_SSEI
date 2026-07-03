"""Auto mask refinement using automatic color analysis.

Automatically extracts dominant colors from each object mask
and removes cross-contamination between objects.
"""

from typing import List, Tuple, Optional
import numpy as np
import cv2
from sklearn.cluster import KMeans


class AutoMaskRefiner:
    """Automatically refines masks using color analysis.

    Unlike MaskRefiner which requires predefined color ranges,
    this class automatically extracts dominant colors from each
    object's mask and removes pixels belonging to other objects.
    """

    def __init__(
        self,
        n_colors: int = 3,
        color_tolerance: int = 30,
        min_cluster_ratio: float = 0.05,
        use_hsv: bool = True,
    ):
        """Initialize auto mask refiner.

        Args:
            n_colors: Number of dominant colors to extract per object.
            color_tolerance: Tolerance for color matching (in HSV or RGB).
            min_cluster_ratio: Minimum ratio of pixels for a color cluster.
            use_hsv: Use HSV color space for better color separation.
        """
        self.n_colors = n_colors
        self.color_tolerance = color_tolerance
        self.min_cluster_ratio = min_cluster_ratio
        self.use_hsv = use_hsv

    def extract_dominant_colors(
        self,
        image_rgb: np.ndarray,
        mask: np.ndarray,
        n_colors: Optional[int] = None
    ) -> List[np.ndarray]:
        """Extract dominant colors from masked region.

        Args:
            image_rgb: RGB image (H, W, 3), uint8.
            mask: Binary mask (H, W), uint8.
            n_colors: Override number of colors to extract.

        Returns:
            List of dominant colors in HSV or RGB.
        """
        n = n_colors or self.n_colors

        # Extract pixels within mask
        pixels = image_rgb[mask > 0]

        if len(pixels) < n:
            return []

        # Convert to HSV if needed
        if self.use_hsv:
            # Reshape for cv2
            pixels_reshaped = pixels.reshape(-1, 1, 3)
            pixels_hsv = cv2.cvtColor(pixels_reshaped, cv2.COLOR_RGB2HSV)
            pixels_for_clustering = pixels_hsv.reshape(-1, 3).astype(np.float32)
        else:
            pixels_for_clustering = pixels.astype(np.float32)

        # K-means clustering
        kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels_for_clustering)

        # Get cluster centers and their sizes
        centers = kmeans.cluster_centers_
        unique, counts = np.unique(labels, return_counts=True)

        # Filter by minimum ratio
        min_count = len(pixels) * self.min_cluster_ratio
        valid_colors = []

        for i, (label, count) in enumerate(zip(unique, counts)):
            if count >= min_count:
                valid_colors.append(centers[label].astype(np.uint8))

        return valid_colors

    def create_color_mask(
        self,
        image_rgb: np.ndarray,
        colors: List[np.ndarray],
        tolerance: Optional[int] = None
    ) -> np.ndarray:
        """Create mask for pixels matching given colors.

        Args:
            image_rgb: RGB image (H, W, 3), uint8.
            colors: List of colors to match (HSV or RGB).
            tolerance: Override color tolerance.

        Returns:
            Binary mask of matching pixels.
        """
        tol = tolerance or self.color_tolerance
        h, w = image_rgb.shape[:2]

        # Convert image to HSV if needed
        if self.use_hsv:
            image_color = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        else:
            image_color = image_rgb

        # Create combined mask for all colors
        combined_mask = np.zeros((h, w), dtype=np.uint8)

        for color in colors:
            # Create range around color
            if self.use_hsv:
                # Special handling for hue (circular)
                h_low = max(0, int(color[0]) - tol // 2)
                h_high = min(180, int(color[0]) + tol // 2)
                s_low = max(0, int(color[1]) - tol)
                s_high = min(255, int(color[1]) + tol)
                v_low = max(0, int(color[2]) - tol)
                v_high = min(255, int(color[2]) + tol)

                lower = np.array([h_low, s_low, v_low])
                upper = np.array([h_high, s_high, v_high])
            else:
                lower = np.maximum(0, color.astype(int) - tol).astype(np.uint8)
                upper = np.minimum(255, color.astype(int) + tol).astype(np.uint8)

            mask = cv2.inRange(image_color, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        return combined_mask

    def refine_mask(
        self,
        image_rgb: np.ndarray,
        target_mask: np.ndarray,
        other_mask: np.ndarray,
        inpainted_image: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Refine target mask by removing other object's colors.

        Args:
            image_rgb: Original RGB image.
            target_mask: Mask to refine.
            other_mask: Mask of other object (colors to remove).
            inpainted_image: Optional inpainted image for color extraction.

        Returns:
            Refined mask.
        """
        # Extract dominant colors from other object
        source_image = inpainted_image if inpainted_image is not None else image_rgb
        other_colors = self.extract_dominant_colors(source_image, other_mask)

        if not other_colors:
            return target_mask

        print(f"  Extracted {len(other_colors)} dominant colors from other object")

        # Create mask of other object's colors in the entire image
        other_color_mask = self.create_color_mask(image_rgb, other_colors)

        # Remove other object's colors from target mask
        refined = cv2.bitwise_and(target_mask, cv2.bitwise_not(other_color_mask))

        # Morphological cleanup
        kernel = np.ones((3, 3), np.uint8)
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel)

        # Fill holes
        refined = self._fill_holes(refined)

        # Keep largest component
        refined = self._keep_largest_component(refined)

        return refined

    def refine_both_masks(
        self,
        image_rgb: np.ndarray,
        mask_a: np.ndarray,
        mask_b: np.ndarray,
        inpainted_for_b: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Refine both masks by removing each other's colors.

        Args:
            image_rgb: Original RGB image.
            mask_a: First object mask (e.g., ceramic).
            mask_b: Second object mask (e.g., soban).
            inpainted_for_b: Inpainted image for extracting B's colors.

        Returns:
            Tuple of (refined_mask_a, refined_mask_b).
        """
        print("=== Auto Mask Refinement ===")

        # Extract colors from each object
        print("\nExtracting colors from object A...")
        colors_a = self.extract_dominant_colors(image_rgb, mask_a)
        print(f"  Found {len(colors_a)} dominant colors")

        source_b = inpainted_for_b if inpainted_for_b is not None else image_rgb
        print("\nExtracting colors from object B...")
        colors_b = self.extract_dominant_colors(source_b, mask_b)
        print(f"  Found {len(colors_b)} dominant colors")

        # Refine mask A by removing B's colors
        print("\nRefining mask A (removing B's colors)...")
        color_mask_b = self.create_color_mask(image_rgb, colors_b)
        refined_a = cv2.bitwise_and(mask_a, cv2.bitwise_not(color_mask_b))
        refined_a = self._cleanup_mask(refined_a)
        print(f"  {np.sum(mask_a > 0)} -> {np.sum(refined_a > 0)} pixels")

        # Refine mask B by removing A's colors (from inpainted if available)
        print("\nRefining mask B (removing A's colors)...")
        color_mask_a = self.create_color_mask(source_b, colors_a)
        refined_b = cv2.bitwise_and(mask_b, cv2.bitwise_not(color_mask_a))

        # For B, extract from inpainted image by removing A's colors
        if inpainted_for_b is not None:
            # Method: remove foreground colors from inpainted image
            # This preserves inpainted regions that may have slightly different colors
            color_mask_a_in_inpainted = self.create_color_mask(inpainted_for_b, colors_a)

            # Get non-white pixels from inpainted image (objects, not background)
            gray = cv2.cvtColor(inpainted_for_b, cv2.COLOR_RGB2GRAY)
            non_white = (gray < 250).astype(np.uint8) * 255

            # Soban = non-white pixels minus foreground colors
            refined_b = cv2.bitwise_and(non_white, cv2.bitwise_not(color_mask_a_in_inpainted))

        refined_b = self._cleanup_mask(refined_b)
        print(f"  {np.sum(mask_b > 0)} -> {np.sum(refined_b > 0)} pixels")

        return refined_a, refined_b

    def _fill_holes(self, mask: np.ndarray) -> np.ndarray:
        """Fill internal holes in mask."""
        h, w = mask.shape
        inverted = cv2.bitwise_not(mask)
        flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        cv2.floodFill(inverted, flood_mask, (0, 0), 0)
        return cv2.bitwise_or(mask, inverted)

    def _keep_largest_component(self, mask: np.ndarray) -> np.ndarray:
        """Keep only largest connected component."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if num_labels <= 2:
            return mask
        areas = stats[1:, cv2.CC_STAT_AREA]
        main_label = 1 + np.argmax(areas)
        return (labels == main_label).astype(np.uint8) * 255

    def _cleanup_mask(self, mask: np.ndarray) -> np.ndarray:
        """Apply standard cleanup operations."""
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = self._fill_holes(mask)
        mask = self._keep_largest_component(mask)
        return mask

    def extract_object(
        self,
        image_rgb: np.ndarray,
        mask: np.ndarray,
        background_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> np.ndarray:
        """Extract object from image using mask."""
        h, w = mask.shape
        result = np.ones((h, w, 3), dtype=np.uint8) * np.array(background_color, dtype=np.uint8)
        result[mask > 0] = image_rgb[mask > 0]
        return result


def auto_refine_masks(
    image_rgb: np.ndarray,
    foreground_mask: np.ndarray,
    background_mask: np.ndarray,
    inpainted_image: Optional[np.ndarray] = None,
    n_colors: int = 3,
    tolerance: int = 30
) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience function for automatic mask refinement.

    Args:
        image_rgb: Original RGB image.
        foreground_mask: Foreground object mask.
        background_mask: Background object mask.
        inpainted_image: Optional inpainted image for background.
        n_colors: Number of dominant colors to extract.
        tolerance: Color matching tolerance.

    Returns:
        Tuple of (refined_foreground, refined_background).
    """
    refiner = AutoMaskRefiner(n_colors=n_colors, color_tolerance=tolerance)
    return refiner.refine_both_masks(
        image_rgb, foreground_mask, background_mask, inpainted_image
    )
