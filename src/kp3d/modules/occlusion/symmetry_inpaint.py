"""
PatchMatch + Symmetry-based inpainting for objects with vertical symmetry.

This module provides inpainting techniques specifically designed for symmetric objects
like ceramics, using a combination of:
- Symmetry detection and mirroring
- PatchMatch algorithm for non-symmetric regions
- Edge blending for smooth transitions
"""

import numpy as np
import cv2
from typing import Optional, Tuple
from scipy import ndimage


class SymmetryDetector:
    """Detect symmetry axis in objects (primarily vertical for ceramics)."""

    def detect_vertical_symmetry(self, mask: np.ndarray, image: Optional[np.ndarray] = None) -> Optional[int]:
        """
        Find the vertical symmetry axis x-coordinate.

        Uses mask centroid and contour analysis.

        Args:
            mask: Binary mask (255 = object)
            image: Optional RGB image for texture-based refinement

        Returns:
            x-coordinate of symmetry axis or None if not found.
        """
        if mask.max() == 0:
            return None

        # Ensure binary mask
        binary_mask = (mask > 127).astype(np.uint8) * 255

        # Calculate moments for centroid
        moments = cv2.moments(binary_mask)
        if moments['m00'] == 0:
            return None

        # Initial estimate: centroid x-coordinate
        centroid_x = int(moments['m10'] / moments['m00'])

        # Refine by testing symmetry scores around centroid
        h, w = mask.shape
        search_range = min(w // 10, 50)  # Search within 10% of width or 50px

        best_score = -1
        best_axis = centroid_x

        for axis_x in range(max(0, centroid_x - search_range),
                          min(w, centroid_x + search_range + 1)):
            score = self.compute_symmetry_score(binary_mask, axis_x)
            if score > best_score:
                best_score = score
                best_axis = axis_x

        # Return axis if symmetry is reasonably good (>60%)
        if best_score > 0.6:
            return best_axis

        return None

    def compute_symmetry_score(self, mask: np.ndarray, axis_x: int) -> float:
        """
        Calculate how symmetric the mask is about the given axis.

        Args:
            mask: Binary mask (255 = object)
            axis_x: Vertical axis x-coordinate

        Returns:
            Symmetry score between 0 and 1 (1 = perfect symmetry)
        """
        h, w = mask.shape

        # Extract left and right halves
        left_width = axis_x
        right_width = w - axis_x

        if left_width == 0 or right_width == 0:
            return 0.0

        # Use the smaller width for comparison
        compare_width = min(left_width, right_width)

        # Extract regions to compare
        left_region = mask[:, axis_x - compare_width:axis_x]
        right_region = mask[:, axis_x:axis_x + compare_width]

        # Flip right region horizontally
        right_flipped = np.fliplr(right_region)

        # Calculate overlap
        intersection = np.logical_and(left_region > 127, right_flipped > 127).sum()
        union = np.logical_or(left_region > 127, right_flipped > 127).sum()

        if union == 0:
            return 0.0

        # IoU as symmetry score
        return intersection / union


class PatchMatchInpainter:
    """PatchMatch-based inpainting using exemplar regions."""

    def __init__(self, patch_size: int = 7, iterations: int = 5):
        """
        Initialize PatchMatch inpainter.

        Args:
            patch_size: Size of patches (should be odd)
            iterations: Number of PatchMatch iterations
        """
        self.patch_size = patch_size if patch_size % 2 == 1 else patch_size + 1
        self.iterations = iterations
        self.half_patch = self.patch_size // 2

    def inpaint(self, image: np.ndarray, mask: np.ndarray,
                exemplar_mask: np.ndarray) -> np.ndarray:
        """
        Inpaint using patches from exemplar region.

        Args:
            image: RGB image (H, W, 3)
            mask: Region to inpaint (255 = inpaint, 0 = keep)
            exemplar_mask: Region to sample patches from (255 = valid source)

        Returns:
            Inpainted image
        """
        result = image.copy()
        h, w = mask.shape[:2]

        # Convert masks to binary
        inpaint_binary = mask > 127
        exemplar_binary = exemplar_mask > 127

        # Get coordinates of pixels to inpaint (prioritize border pixels first)
        inpaint_coords = self._get_fill_order(inpaint_binary)

        # For each pixel to inpaint
        for y, x in inpaint_coords:
            # Skip if already filled
            if not inpaint_binary[y, x]:
                continue

            # Find best matching patch from exemplar region
            best_patch_center = self._find_best_patch(
                result, (y, x), exemplar_binary, inpaint_binary
            )

            if best_patch_center is not None:
                # Copy the best patch center pixel
                by, bx = best_patch_center
                result[y, x] = image[by, bx]
                inpaint_binary[y, x] = False

        return result

    def _get_fill_order(self, mask: np.ndarray) -> np.ndarray:
        """
        Get coordinates to fill, prioritizing boundary pixels.

        Returns:
            Array of (y, x) coordinates
        """
        # Compute distance transform (distance to nearest non-mask pixel)
        dist = ndimage.distance_transform_edt(mask)

        # Get coordinates of mask pixels
        coords = np.argwhere(mask)

        # Sort by distance (fill from outside in)
        distances = dist[coords[:, 0], coords[:, 1]]
        sorted_indices = np.argsort(distances)

        return coords[sorted_indices]

    def _find_best_patch(self, image: np.ndarray, target_pos: Tuple[int, int],
                        exemplar_mask: np.ndarray, inpaint_mask: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Find the best matching patch center in exemplar region.

        Args:
            image: Current image state
            target_pos: (y, x) of pixel to fill
            exemplar_mask: Valid source regions
            inpaint_mask: Current unfilled regions

        Returns:
            (y, x) of best patch center or None
        """
        h, w = image.shape[:2]
        ty, tx = target_pos

        # Get valid patch around target (only known pixels)
        target_patch, target_mask = self._extract_patch(
            image, ty, tx, ~inpaint_mask
        )

        if target_mask.sum() == 0:
            return None

        # Find exemplar patch centers with valid neighborhoods
        exemplar_coords = self._get_valid_exemplar_centers(exemplar_mask)

        if len(exemplar_coords) == 0:
            return None

        # Sample subset for efficiency (PatchMatch random search)
        sample_size = min(200, len(exemplar_coords))
        sampled_indices = np.random.choice(
            len(exemplar_coords), sample_size, replace=False
        )
        sampled_coords = exemplar_coords[sampled_indices]

        # Find best match
        best_ssd = float('inf')
        best_center = None

        for ey, ex in sampled_coords:
            exemplar_patch, _ = self._extract_patch(image, ey, ex, exemplar_mask)

            # Compute SSD only on valid (known) pixels in target
            diff = (target_patch - exemplar_patch) * target_mask[..., np.newaxis]
            ssd = np.sum(diff ** 2)

            if ssd < best_ssd:
                best_ssd = ssd
                best_center = (ey, ex)

        return best_center

    def _extract_patch(self, image: np.ndarray, cy: int, cx: int,
                      valid_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract patch centered at (cy, cx).

        Returns:
            (patch, valid_mask) where patch is (patch_size, patch_size, 3)
            and valid_mask is (patch_size, patch_size) boolean array
        """
        h, w = image.shape[:2]

        # Calculate patch bounds
        y1 = max(0, cy - self.half_patch)
        y2 = min(h, cy + self.half_patch + 1)
        x1 = max(0, cx - self.half_patch)
        x2 = min(w, cx + self.half_patch + 1)

        # Extract patch
        patch = np.zeros((self.patch_size, self.patch_size, 3), dtype=image.dtype)
        mask = np.zeros((self.patch_size, self.patch_size), dtype=bool)

        # Calculate offsets in patch array
        py1 = self.half_patch - (cy - y1)
        py2 = py1 + (y2 - y1)
        px1 = self.half_patch - (cx - x1)
        px2 = px1 + (x2 - x1)

        patch[py1:py2, px1:px2] = image[y1:y2, x1:x2]
        mask[py1:py2, px1:px2] = valid_mask[y1:y2, x1:x2]

        return patch, mask

    def _get_valid_exemplar_centers(self, exemplar_mask: np.ndarray) -> np.ndarray:
        """
        Get coordinates of valid exemplar patch centers.

        A center is valid if the entire patch around it is in the exemplar region.
        """
        # Erode mask by half_patch to ensure full patches
        kernel = np.ones((self.patch_size, self.patch_size), dtype=np.uint8)
        eroded = cv2.erode(exemplar_mask.astype(np.uint8) * 255, kernel) > 127

        return np.argwhere(eroded)


class SymmetryGuidedInpainter:
    """Inpaint using symmetry of the object."""

    def __init__(self, use_patchmatch_fallback: bool = True):
        """
        Initialize symmetry-guided inpainter.

        Args:
            use_patchmatch_fallback: Use PatchMatch for non-symmetric regions
        """
        self.symmetry_detector = SymmetryDetector()
        self.patchmatch = PatchMatchInpainter() if use_patchmatch_fallback else None

    def inpaint(self, image: np.ndarray, mask: np.ndarray,
                object_mask: np.ndarray) -> np.ndarray:
        """
        Inpaint occluded regions using object's symmetry.

        Algorithm:
        1. Detect symmetry axis of object
        2. For each pixel in mask to inpaint:
           a. Calculate mirrored position across symmetry axis
           b. If mirrored position is visible, copy pixel
           c. Else, use PatchMatch to find best patch
        3. Blend edges for smooth transition

        Args:
            image: RGB image (H, W, 3)
            mask: Region to inpaint (255 = inpaint, 0 = keep)
            object_mask: Full object mask (255 = object)

        Returns:
            Inpainted image
        """
        result = image.copy()

        # Detect symmetry axis
        axis_x = self.symmetry_detector.detect_vertical_symmetry(object_mask, image)

        if axis_x is None:
            # No symmetry detected, fallback to PatchMatch only
            if self.patchmatch is not None:
                # Use visible parts of object as exemplar
                exemplar_mask = cv2.bitwise_and(
                    object_mask,
                    cv2.bitwise_not(mask)
                )
                return self.patchmatch.inpaint(result, mask, exemplar_mask)
            return result

        # Convert masks to binary
        inpaint_binary = mask > 127
        object_binary = object_mask > 127

        # Get coordinates to inpaint
        inpaint_coords = np.argwhere(inpaint_binary)

        # Track which pixels were successfully filled by symmetry
        filled_by_symmetry = np.zeros_like(inpaint_binary)

        # Fill using symmetry
        for y, x in inpaint_coords:
            # Calculate mirrored position
            mirror_x = self.mirror_coordinates(np.array([[x]]), axis_x)[0, 0]

            # Check if mirrored position is valid and visible
            if (0 <= mirror_x < image.shape[1] and
                object_binary[y, mirror_x] and
                not inpaint_binary[y, mirror_x]):

                # Copy pixel from mirrored position
                result[y, x] = image[y, mirror_x]
                filled_by_symmetry[y, x] = True

        # Use PatchMatch for remaining unfilled pixels
        if self.patchmatch is not None:
            remaining_mask = (inpaint_binary & ~filled_by_symmetry).astype(np.uint8) * 255

            if remaining_mask.max() > 0:
                # Use visible parts of object (including symmetry-filled) as exemplar
                visible_mask = object_binary & ~inpaint_binary
                exemplar_mask = (visible_mask | filled_by_symmetry).astype(np.uint8) * 255

                result = self.patchmatch.inpaint(result, remaining_mask, exemplar_mask)

        # Blend edges for smooth transition
        result = self._blend_edges(result, image, inpaint_binary)

        return result

    def mirror_coordinates(self, coords: np.ndarray, axis_x: int) -> np.ndarray:
        """
        Mirror coordinates across vertical axis.

        Args:
            coords: Array of x-coordinates (N, 1) or (N,)
            axis_x: Vertical axis x-coordinate

        Returns:
            Mirrored x-coordinates with same shape
        """
        coords = np.asarray(coords)
        offset = coords - axis_x
        mirrored = axis_x - offset
        return mirrored.astype(np.int32)

    def _blend_edges(self, inpainted: np.ndarray, original: np.ndarray,
                    inpaint_mask: np.ndarray, blend_width: int = 5) -> np.ndarray:
        """
        Blend edges between inpainted and original regions.

        Args:
            inpainted: Inpainted image
            original: Original image
            inpaint_mask: Boolean mask of inpainted region
            blend_width: Width of blending region in pixels

        Returns:
            Blended image
        """
        # Create distance transform from inpaint boundary
        dist = ndimage.distance_transform_edt(~inpaint_mask)

        # Create blend weights (0 = inpainted, 1 = original)
        blend_weights = np.clip(dist / blend_width, 0, 1)

        # Expand to 3 channels
        blend_weights = blend_weights[..., np.newaxis]

        # Blend
        result = (inpainted * (1 - blend_weights) +
                 original * blend_weights).astype(np.uint8)

        return result


# Convenience functions

def symmetry_guided_inpaint(image: np.ndarray, mask: np.ndarray,
                           object_mask: np.ndarray) -> np.ndarray:
    """
    Quick symmetry-based inpainting.

    Args:
        image: RGB image
        mask: Region to inpaint (255 = inpaint)
        object_mask: Full object mask (255 = object)

    Returns:
        Inpainted image
    """
    inpainter = SymmetryGuidedInpainter()
    return inpainter.inpaint(image, mask, object_mask)


def detect_symmetry_axis(mask: np.ndarray) -> Optional[int]:
    """
    Quick symmetry axis detection.

    Args:
        mask: Binary mask (255 = object)

    Returns:
        x-coordinate of symmetry axis or None
    """
    detector = SymmetryDetector()
    return detector.detect_vertical_symmetry(mask)
