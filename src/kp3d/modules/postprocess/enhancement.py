"""Post-Upscale Enhancement Module.

Applies adaptive denoising and edge enhancement after super-resolution,
using CV hints like edge maps, gradient direction, and texture density.

Pipeline:
1. Extract hints (edge, gradient, texture, flat regions)
2. Adaptive denoising (region-aware)
3. Edge enhancement (gradient-aware sharpening + ΔE injection)
"""

import cv2
import numpy as np
import torch
from torch import Tensor
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import time

from kp3d.core.base import BasePreprocessModule, ModuleOutput


@dataclass
class EnhancementConfig:
    """Configuration for post-upscale enhancement.

    Attributes:
        # Denoising
        denoise_strength: Overall denoising strength (0-1)
        flat_region_strength: Denoising strength for flat regions
        texture_region_strength: Denoising strength for textured regions
        edge_region_strength: Denoising strength near edges

        # Edge Enhancement
        edge_enhance_strength: Overall edge enhancement strength (0-1)
        unsharp_amount: Unsharp mask amount
        unsharp_radius: Unsharp mask radius
        delta_e_injection: ΔE boundary injection strength
        gradient_aware: Use gradient direction for sharpening

        # Hint extraction
        edge_sigma: Sigma for edge proximity calculation
        texture_window: Window size for texture analysis
        flat_threshold: Variance threshold for flat region detection
    """
    # Denoising
    denoise_strength: float = 0.5
    flat_region_strength: float = 0.8
    texture_region_strength: float = 0.3
    edge_region_strength: float = 0.1

    # Edge Enhancement
    edge_enhance_strength: float = 0.5
    unsharp_amount: float = 1.5
    unsharp_radius: float = 1.0
    delta_e_injection: float = 0.3
    gradient_aware: bool = True

    # Hint extraction
    edge_sigma: float = 8.0
    texture_window: int = 7
    flat_threshold: float = 100.0


class PostUpscaleEnhancer(BasePreprocessModule):
    """Post-Upscale Enhancement Module.

    Applies adaptive denoising and edge enhancement using CV hints.
    """

    def __init__(
        self,
        config: Optional[EnhancementConfig] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        **kwargs: Any,
    ) -> None:
        super().__init__(device=device, dtype=dtype)
        self.config = config or EnhancementConfig()
        self._initialized = True

    @property
    def name(self) -> str:
        return "post_upscale_enhancer"

    def load_weights(self, checkpoint_path: str) -> None:
        """No pretrained weights needed for CV-based enhancement."""
        pass

    def _tensor_to_numpy(self, tensor: Tensor) -> np.ndarray:
        """Convert tensor to numpy RGB array."""
        if tensor.dim() == 4:
            tensor = tensor[0]
        arr = tensor.cpu().numpy()
        if arr.shape[0] == 3:
            arr = np.transpose(arr, (1, 2, 0))
        return (np.clip(arr, 0, 1) * 255).astype(np.uint8)

    def _numpy_to_tensor(self, arr: np.ndarray) -> Tensor:
        """Convert numpy RGB array to tensor."""
        arr = arr.astype(np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr).to(device=self.device, dtype=self.dtype)

    # =========================================================================
    # HINT EXTRACTION
    # =========================================================================

    def extract_edge_map(self, image: np.ndarray) -> np.ndarray:
        """Extract edge map using multi-scale Canny + ΔE.

        Returns:
            Edge map (float32, 0-1)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Multi-scale Canny
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        edges1 = cv2.Canny(filtered, 30, 80)
        edges2 = cv2.Canny(filtered, 50, 150)
        edges3 = cv2.Canny(filtered, 100, 200)
        canny = np.maximum(np.maximum(edges1, edges2), edges3)

        # ΔE (LAB color gradient)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        L_grad = np.sqrt(
            cv2.Sobel(lab[:,:,0], cv2.CV_32F, 1, 0, ksize=3)**2 +
            cv2.Sobel(lab[:,:,0], cv2.CV_32F, 0, 1, ksize=3)**2
        )
        A_grad = np.sqrt(
            cv2.Sobel(lab[:,:,1], cv2.CV_32F, 1, 0, ksize=3)**2 +
            cv2.Sobel(lab[:,:,1], cv2.CV_32F, 0, 1, ksize=3)**2
        )
        B_grad = np.sqrt(
            cv2.Sobel(lab[:,:,2], cv2.CV_32F, 1, 0, ksize=3)**2 +
            cv2.Sobel(lab[:,:,2], cv2.CV_32F, 0, 1, ksize=3)**2
        )
        delta_e = np.sqrt(L_grad**2 + A_grad**2 + B_grad**2)

        # Normalize ΔE
        if delta_e.max() > 0:
            delta_e = delta_e / delta_e.max()

        # Combine
        canny_norm = canny.astype(np.float32) / 255.0
        edge_map = np.maximum(canny_norm, delta_e)

        return edge_map.astype(np.float32)

    def extract_gradient_direction(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract gradient magnitude and direction.

        Returns:
            (magnitude, direction) - both float32
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)

        # Sobel gradients
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

        magnitude = np.sqrt(gx**2 + gy**2)
        direction = np.arctan2(gy, gx)  # -pi to pi

        # Normalize magnitude
        if magnitude.max() > 0:
            magnitude = magnitude / magnitude.max()

        return magnitude, direction

    def extract_texture_map(self, image: np.ndarray) -> np.ndarray:
        """Extract texture density map using local variance.

        High variance = textured region
        Low variance = flat region

        Returns:
            Texture map (float32, 0-1, 1=textured)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)

        window = self.config.texture_window

        # Local mean
        kernel = np.ones((window, window), np.float32) / (window * window)
        local_mean = cv2.filter2D(gray, -1, kernel)

        # Local variance
        local_sq_mean = cv2.filter2D(gray**2, -1, kernel)
        local_var = local_sq_mean - local_mean**2
        local_var = np.maximum(local_var, 0)  # Numerical stability

        # Normalize with soft threshold
        texture_map = 1 - np.exp(-local_var / self.config.flat_threshold)

        return texture_map.astype(np.float32)

    def extract_edge_proximity(self, edge_map: np.ndarray) -> np.ndarray:
        """Compute edge proximity map.

        Returns:
            Proximity map (float32, 0-1, 1=on edge)
        """
        # Binary edge
        edge_binary = (edge_map > 0.1).astype(np.uint8)

        # Distance transform
        non_edge = 1 - edge_binary
        dist = cv2.distanceTransform(non_edge, cv2.DIST_L2, 5)

        # Gaussian falloff
        sigma = self.config.edge_sigma
        proximity = np.exp(-(dist**2) / (2 * sigma**2))

        return proximity.astype(np.float32)

    def compute_region_masks(
        self,
        edge_proximity: np.ndarray,
        texture_map: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Compute soft masks for different regions.

        Returns:
            Dict with 'edge', 'texture', 'flat' masks (each 0-1)
        """
        # Edge region: high proximity
        edge_mask = edge_proximity

        # Flat region: low texture AND not edge
        flat_mask = (1 - texture_map) * (1 - edge_proximity)

        # Texture region: high texture AND not edge
        texture_mask = texture_map * (1 - edge_proximity * 0.5)

        # Normalize so they sum to ~1
        total = edge_mask + flat_mask + texture_mask + 1e-8

        return {
            'edge': edge_mask / total,
            'texture': texture_mask / total,
            'flat': flat_mask / total,
        }

    # =========================================================================
    # ADAPTIVE DENOISING
    # =========================================================================

    def adaptive_denoise(
        self,
        image: np.ndarray,
        region_masks: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Apply region-aware denoising.

        - Flat regions: Strong bilateral filter
        - Texture regions: Light guided filter
        - Edge regions: Minimal/no filtering
        """
        h, w = image.shape[:2]
        img_float = image.astype(np.float32)

        cfg = self.config

        # Strong denoising for flat regions
        if cfg.flat_region_strength > 0:
            d = int(5 + cfg.flat_region_strength * 10)
            sigma_color = 50 + cfg.flat_region_strength * 50
            sigma_space = 50 + cfg.flat_region_strength * 50
            flat_denoised = cv2.bilateralFilter(
                image, d, sigma_color, sigma_space
            ).astype(np.float32)
        else:
            flat_denoised = img_float

        # Light denoising for texture regions (preserve texture)
        if cfg.texture_region_strength > 0:
            # Guided filter - preserves edges and texture better
            radius = int(3 + cfg.texture_region_strength * 5)
            eps = 0.01 + (1 - cfg.texture_region_strength) * 0.1

            gray_guide = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            texture_denoised = np.zeros_like(img_float)
            for c in range(3):
                texture_denoised[:,:,c] = cv2.ximgproc.guidedFilter(
                    gray_guide, image[:,:,c], radius, eps
                ) if hasattr(cv2, 'ximgproc') else cv2.bilateralFilter(
                    image[:,:,c], 5, 30, 30
                )
            texture_denoised = texture_denoised.astype(np.float32)
        else:
            texture_denoised = img_float

        # Minimal denoising for edge regions (mostly preserve)
        if cfg.edge_region_strength > 0:
            edge_denoised = cv2.GaussianBlur(
                img_float, (3, 3), cfg.edge_region_strength * 0.5
            )
        else:
            edge_denoised = img_float

        # Blend based on region masks
        flat_mask = region_masks['flat'][:, :, np.newaxis]
        texture_mask = region_masks['texture'][:, :, np.newaxis]
        edge_mask = region_masks['edge'][:, :, np.newaxis]

        # Apply overall denoising strength
        strength = cfg.denoise_strength

        blended = (
            flat_denoised * flat_mask +
            texture_denoised * texture_mask +
            edge_denoised * edge_mask
        )

        # Mix with original based on overall strength
        result = img_float * (1 - strength) + blended * strength

        return np.clip(result, 0, 255).astype(np.uint8)

    # =========================================================================
    # EDGE ENHANCEMENT
    # =========================================================================

    def gradient_aware_sharpen(
        self,
        image: np.ndarray,
        magnitude: np.ndarray,
        direction: np.ndarray
    ) -> np.ndarray:
        """Apply gradient-direction-aware sharpening.

        Sharpens perpendicular to edge direction for cleaner edges.
        """
        cfg = self.config
        img_float = image.astype(np.float32)

        # Standard unsharp mask
        radius = int(cfg.unsharp_radius * 2) * 2 + 1  # Ensure odd
        blurred = cv2.GaussianBlur(img_float, (radius, radius), cfg.unsharp_radius)
        unsharp_mask = img_float - blurred

        # Weight by gradient magnitude (sharpen more at edges)
        edge_weight = magnitude[:, :, np.newaxis]

        # Apply sharpening
        amount = cfg.unsharp_amount * cfg.edge_enhance_strength
        sharpened = img_float + unsharp_mask * edge_weight * amount

        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def inject_delta_e_edges(
        self,
        image: np.ndarray,
        edge_map: np.ndarray
    ) -> np.ndarray:
        """Inject ΔE-based color boundaries.

        Darkens edges slightly to increase contrast.
        """
        cfg = self.config
        if cfg.delta_e_injection <= 0:
            return image

        img_float = image.astype(np.float32)

        # Smooth edge map for natural injection
        edge_smooth = cv2.GaussianBlur(edge_map, (5, 5), 1.0)

        # Compute edge gradient for directional darkening
        edge_dx = cv2.Sobel(edge_smooth, cv2.CV_32F, 1, 0, ksize=3)
        edge_dy = cv2.Sobel(edge_smooth, cv2.CV_32F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(edge_dx**2 + edge_dy**2)

        if edge_magnitude.max() > 0:
            edge_magnitude = edge_magnitude / edge_magnitude.max()

        # Injection (darkening at edges)
        edge_3d = edge_magnitude[:, :, np.newaxis]
        injection = edge_3d * cfg.delta_e_injection * cfg.edge_enhance_strength * 50

        result = img_float - injection

        return np.clip(result, 0, 255).astype(np.uint8)

    def enhance_edges(
        self,
        image: np.ndarray,
        edge_map: np.ndarray,
        magnitude: np.ndarray,
        direction: np.ndarray
    ) -> np.ndarray:
        """Apply full edge enhancement pipeline."""
        cfg = self.config

        # Step 1: Gradient-aware sharpening
        if cfg.gradient_aware:
            result = self.gradient_aware_sharpen(image, magnitude, direction)
        else:
            # Simple unsharp mask
            radius = int(cfg.unsharp_radius * 2) * 2 + 1
            blurred = cv2.GaussianBlur(image.astype(np.float32), (radius, radius), cfg.unsharp_radius)
            unsharp = image.astype(np.float32) - blurred
            result = image.astype(np.float32) + unsharp * cfg.unsharp_amount * cfg.edge_enhance_strength
            result = np.clip(result, 0, 255).astype(np.uint8)

        # Step 2: ΔE boundary injection
        result = self.inject_delta_e_edges(result, edge_map)

        return result

    # =========================================================================
    # MAIN FORWARD
    # =========================================================================

    def forward(self, image: Tensor, **kwargs: Any) -> ModuleOutput:
        """Apply post-upscale enhancement.

        Args:
            image: Input tensor (B, C, H, W)

        Returns:
            ModuleOutput with enhanced image
        """
        start_time = time.time()

        if image.dim() == 3:
            image = image.unsqueeze(0)

        # Convert to numpy
        img_np = self._tensor_to_numpy(image[0])

        # =====================================================================
        # STEP 1: Extract hints
        # =====================================================================
        edge_map = self.extract_edge_map(img_np)
        magnitude, direction = self.extract_gradient_direction(img_np)
        texture_map = self.extract_texture_map(img_np)
        edge_proximity = self.extract_edge_proximity(edge_map)
        region_masks = self.compute_region_masks(edge_proximity, texture_map)

        # =====================================================================
        # STEP 2: Adaptive denoising
        # =====================================================================
        denoised = self.adaptive_denoise(img_np, region_masks)

        # =====================================================================
        # STEP 3: Edge enhancement
        # =====================================================================
        # Re-extract hints from denoised image for better edge enhancement
        edge_map_denoised = self.extract_edge_map(denoised)
        magnitude_denoised, direction_denoised = self.extract_gradient_direction(denoised)

        enhanced = self.enhance_edges(
            denoised,
            edge_map_denoised,
            magnitude_denoised,
            direction_denoised
        )

        # =====================================================================
        # Build output
        # =====================================================================
        processing_time = time.time() - start_time

        result_tensor = self._numpy_to_tensor(enhanced).unsqueeze(0)

        # Intermediates for debugging
        intermediates = {
            'edge_map': self._numpy_to_tensor(
                np.stack([(edge_map * 255).astype(np.uint8)] * 3, axis=-1)
            ),
            'texture_map': self._numpy_to_tensor(
                np.stack([(texture_map * 255).astype(np.uint8)] * 3, axis=-1)
            ),
            'edge_proximity': self._numpy_to_tensor(
                np.stack([(edge_proximity * 255).astype(np.uint8)] * 3, axis=-1)
            ),
            'gradient_magnitude': self._numpy_to_tensor(
                np.stack([(magnitude * 255).astype(np.uint8)] * 3, axis=-1)
            ),
            'flat_mask': self._numpy_to_tensor(
                np.stack([(region_masks['flat'] * 255).astype(np.uint8)] * 3, axis=-1)
            ),
            'denoised': self._numpy_to_tensor(denoised),
        }

        metadata = {
            'method': 'post_upscale_enhancement',
            'processing_time': processing_time,
            'denoise_strength': self.config.denoise_strength,
            'edge_enhance_strength': self.config.edge_enhance_strength,
        }

        return ModuleOutput(
            result=result_tensor,
            intermediate=intermediates,
            metadata=metadata,
        )


__all__ = ["PostUpscaleEnhancer", "EnhancementConfig"]
