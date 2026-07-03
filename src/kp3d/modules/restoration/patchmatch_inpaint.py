"""Restoration-optimized PatchMatch inpainting for v7.

A specialized PatchMatch implementation for noise restoration that uses:
- Onion-peel filling order (boundary-first, inward)
- LAB color space for perceptually accurate patch matching
- Hanji (Korean paper) texture preservation mode
- Configurable search parameters

Reference: Barnes et al., "PatchMatch: A Randomized Correspondence Algorithm
for Structural Image Editing", SIGGRAPH 2009.
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class RestorationPatchMatch:
    """PatchMatch inpainter optimized for restoration tasks.

    Unlike generic PatchMatch, this implementation:
    1. Uses onion-peel (outside-in) filling for better coherence
    2. Matches in LAB space for perceptual accuracy on paintings
    3. Supports texture preservation for hanji paper
    4. Has configurable random search budget
    """

    def __init__(
        self,
        patch_size: int = 7,
        iterations: int = 5,
        search_samples: int = 100,
        preserve_texture: bool = False,
        texture_sigma: float = 2.0,
    ):
        """Initialize restoration PatchMatch.

        Args:
            patch_size: Patch size (will be made odd).
            iterations: Number of propagation iterations.
            search_samples: Random samples per pixel search.
            preserve_texture: Enable hanji texture preservation.
            texture_sigma: Sigma for texture extraction blur.
        """
        self.patch_size = patch_size if patch_size % 2 == 1 else patch_size + 1
        self.half_patch = self.patch_size // 2
        self.iterations = iterations
        self.search_samples = search_samples
        self.preserve_texture = preserve_texture
        self.texture_sigma = texture_sigma

    def inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Inpaint masked regions using restoration PatchMatch.

        Args:
            image: RGB image (uint8, H x W x 3).
            mask: Binary mask (uint8, 255 = inpaint region).

        Returns:
            Inpainted image (uint8).
        """
        if mask.max() == 0:
            return image.copy()

        result = image.copy()
        h, w = image.shape[:2]

        # Convert to LAB for matching
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        result_lab = lab.copy()

        # Extract texture before inpainting if preservation enabled
        texture = None
        if self.preserve_texture:
            texture = self._extract_texture(image)

        # Build valid source mask (non-masked, eroded for full patches)
        source_mask = (mask == 0).astype(np.uint8) * 255
        kernel = np.ones((self.patch_size, self.patch_size), np.uint8)
        source_eroded = cv2.erode(source_mask, kernel) > 127

        if not np.any(source_eroded):
            # Fallback: use OpenCV inpainting if no valid source patches
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            restored_bgr = cv2.inpaint(bgr, mask, 3, cv2.INPAINT_TELEA)
            return cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)

        # Get source patch centers
        source_coords = np.argwhere(source_eroded)

        # Onion-peel: get fill order from boundary inward
        fill_order = self._get_onion_peel_order(mask)

        if len(fill_order) == 0:
            return image.copy()

        # Track which pixels have been filled
        inpaint_remaining = (mask > 127).copy()

        # Fill in layers
        for y, x in fill_order:
            if not inpaint_remaining[y, x]:
                continue

            # Find best matching patch
            best_center = self._find_best_patch_lab(
                result_lab, (y, x), source_coords, inpaint_remaining
            )

            if best_center is not None:
                by, bx = best_center
                # Copy patch center pixel
                result_lab[y, x] = lab[by, bx]
                result[y, x] = image[by, bx]
                inpaint_remaining[y, x] = False

        # Apply texture preservation
        if self.preserve_texture and texture is not None:
            result = self._apply_texture(result, texture, mask)

        # Boundary blending
        result = self._blend_boundary(result, image, mask)

        return result

    def _get_onion_peel_order(self, mask: np.ndarray) -> np.ndarray:
        """Get fill coordinates in onion-peel order (boundary first).

        Args:
            mask: Binary mask (uint8).

        Returns:
            Array of (y, x) coordinates sorted by distance from boundary.
        """
        binary = (mask > 127).astype(np.uint8)

        # Distance from non-masked region
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

        # Get mask coordinates
        coords = np.argwhere(binary > 0)

        if len(coords) == 0:
            return np.array([], dtype=np.int32).reshape(0, 2)

        # Sort by distance (closest to boundary first)
        distances = dist[coords[:, 0], coords[:, 1]]
        sorted_idx = np.argsort(distances)

        return coords[sorted_idx]

    def _find_best_patch_lab(
        self,
        lab_image: np.ndarray,
        target_pos: Tuple[int, int],
        source_coords: np.ndarray,
        inpaint_mask: np.ndarray,
    ) -> Optional[Tuple[int, int]]:
        """Find best matching patch center in LAB space.

        Args:
            lab_image: Current LAB image state (float32).
            target_pos: (y, x) pixel to fill.
            source_coords: Valid source patch centers.
            inpaint_mask: Current unfilled mask.

        Returns:
            (y, x) of best matching source center, or None.
        """
        h, w = lab_image.shape[:2]
        ty, tx = target_pos

        # Extract target patch (only known pixels)
        target_patch, valid_mask = self._extract_patch(
            lab_image, ty, tx, ~inpaint_mask
        )

        if valid_mask.sum() == 0:
            return None

        # Random subset for search
        n_samples = min(self.search_samples, len(source_coords))
        indices = np.random.choice(len(source_coords), n_samples, replace=False)
        sampled = source_coords[indices]

        best_ssd = float("inf")
        best_center = None

        for sy, sx in sampled:
            source_patch, _ = self._extract_patch(
                lab_image, sy, sx, np.ones((h, w), dtype=bool)
            )

            # SSD on valid pixels only (LAB space)
            diff = (target_patch - source_patch) * valid_mask[..., np.newaxis]
            ssd = np.sum(diff ** 2)

            if ssd < best_ssd:
                best_ssd = ssd
                best_center = (sy, sx)

        return best_center

    def _extract_patch(
        self,
        image: np.ndarray,
        cy: int,
        cx: int,
        valid_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract patch with bounds checking.

        Args:
            image: Source image.
            cy, cx: Patch center.
            valid_mask: Boolean mask of valid pixels.

        Returns:
            (patch, valid_mask_patch) arrays.
        """
        h, w = image.shape[:2]
        ps = self.patch_size
        hp = self.half_patch

        y1 = max(0, cy - hp)
        y2 = min(h, cy + hp + 1)
        x1 = max(0, cx - hp)
        x2 = min(w, cx + hp + 1)

        channels = image.shape[2] if image.ndim == 3 else 1
        patch = np.zeros((ps, ps, channels), dtype=image.dtype)
        mask_patch = np.zeros((ps, ps), dtype=bool)

        py1 = hp - (cy - y1)
        py2 = py1 + (y2 - y1)
        px1 = hp - (cx - x1)
        px2 = px1 + (x2 - x1)

        if image.ndim == 3:
            patch[py1:py2, px1:px2] = image[y1:y2, x1:x2]
        else:
            patch[py1:py2, px1:px2, 0] = image[y1:y2, x1:x2]

        mask_patch[py1:py2, px1:px2] = valid_mask[y1:y2, x1:x2]

        return patch, mask_patch

    def _extract_texture(self, image: np.ndarray) -> np.ndarray:
        """Extract high-frequency texture component.

        Args:
            image: RGB image (uint8).

        Returns:
            Texture component (float32, centered at 0).
        """
        img_float = image.astype(np.float32)
        sigma = self.texture_sigma
        ksize = int(np.ceil(sigma * 6)) | 1
        low_freq = cv2.GaussianBlur(img_float, (ksize, ksize), sigma)
        return img_float - low_freq

    def _apply_texture(
        self,
        restored: np.ndarray,
        texture: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Re-apply hanji texture to restored regions.

        Args:
            restored: Inpainted image (uint8).
            texture: Texture from original (float32).
            mask: Inpaint mask.

        Returns:
            Image with texture restored (uint8).
        """
        result = restored.astype(np.float32)

        # Only apply in masked regions with smooth falloff
        dist = cv2.distanceTransform((mask > 127).astype(np.uint8), cv2.DIST_L2, 5)
        max_dist = dist.max() + 1e-6
        blend = np.clip(dist / max(max_dist * 0.5, 1), 0, 1)
        blend_3d = blend[:, :, np.newaxis]

        # Add texture with reduced strength in center of filled area
        texture_strength = 0.5 * (1 - blend_3d * 0.5)
        mask_3d = (mask > 127)[:, :, np.newaxis]

        result = np.where(
            mask_3d,
            result + texture * texture_strength,
            result,
        )

        return np.clip(result, 0, 255).astype(np.uint8)

    def _blend_boundary(
        self,
        inpainted: np.ndarray,
        original: np.ndarray,
        mask: np.ndarray,
        blend_width: int = 3,
    ) -> np.ndarray:
        """Smooth boundary between inpainted and original regions.

        Args:
            inpainted: Inpainted result.
            original: Original image.
            mask: Inpaint mask.
            blend_width: Blending width in pixels.

        Returns:
            Blended image (uint8).
        """
        # Distance from mask boundary (inside mask)
        dist_inside = cv2.distanceTransform(
            (mask > 127).astype(np.uint8), cv2.DIST_L2, 5
        )

        # Blend weight: 0 at boundary, 1 deep inside
        weight = np.clip(dist_inside / max(blend_width, 1), 0, 1)
        weight_3d = weight[:, :, np.newaxis]

        mask_bool = (mask > 127)[:, :, np.newaxis]

        result = np.where(
            mask_bool,
            inpainted.astype(np.float32) * weight_3d + original.astype(np.float32) * (1 - weight_3d),
            original.astype(np.float32),
        )

        return np.clip(result, 0, 255).astype(np.uint8)
