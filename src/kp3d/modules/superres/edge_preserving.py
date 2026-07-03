"""Edge-Preserving Super Resolution for Korean paintings.

Preserves edge information during upscaling by:
1. Extracting edges before upscaling
2. Upscaling edges separately (bicubic)
3. Injecting upscaled edges into high-frequency component after Real-ESRGAN

This addresses the ~50% edge loss observed in standard Real-ESRGAN upscaling.
"""

import cv2
import numpy as np
import torch
from torch import Tensor
from typing import Any, Optional, Dict, Tuple
import time

from loguru import logger

from kp3d.core.base import ModuleOutput
from kp3d.modules.superres.base import BaseSuperResolution, SuperResConfig, ScaleFactor


class EdgePreservingSuperRes(BaseSuperResolution):
    """Edge-preserving super-resolution module.

    Wraps Real-ESRGAN with edge extraction and re-injection to prevent
    edge loss during upscaling.

    Pipeline:
    1. Extract edges from input (Canny + color-based)
    2. Upscale image with Real-ESRGAN
    3. Upscale edge map (4x)
    4. Separate upscaled image into low/high frequency
    5. Inject upscaled edges into high frequency
    6. Reconstruct final image
    """

    def __init__(
        self,
        config: Optional[SuperResConfig] = None,
        device: Optional[torch.device] = None,
        edge_inject_strength: float = 0.25,
        edge_sharpen_strength: float = 0.15,
        **kwargs: Any,
    ) -> None:
        """Initialize Edge-Preserving Super Resolution.

        Args:
            config: Super-resolution configuration.
            device: Computation device.
            edge_inject_strength: Strength of edge injection (0-1).
            edge_sharpen_strength: Strength of edge sharpening (0-1).
        """
        super().__init__(config=config, device=device, **kwargs)

        self.edge_inject_strength = edge_inject_strength
        self.edge_sharpen_strength = edge_sharpen_strength

        # Initialize Real-ESRGAN
        self.real_esrgan = None
        self._init_real_esrgan()

    def _init_real_esrgan(self) -> None:
        """Initialize the underlying Real-ESRGAN module."""
        try:
            from kp3d.modules.superres.real_esrgan import RealESRGANModule
            self.real_esrgan = RealESRGANModule(
                config=self.config,
                device=self.device,
            )
            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize Real-ESRGAN: {e}")
            self._initialized = False

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

    def extract_edges_enhanced(self, image: np.ndarray) -> np.ndarray:
        """Extract edges using multi-method approach.

        Combines:
        1. Canny edge detection (multi-scale)
        2. LAB color gradient (ΔE)

        Args:
            image: RGB image (uint8)

        Returns:
            Edge map (uint8, 0-255)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Multi-scale Canny
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        edges1 = cv2.Canny(filtered, 30, 80)
        edges2 = cv2.Canny(filtered, 50, 150)
        edges3 = cv2.Canny(filtered, 100, 200)
        canny_edges = np.maximum(np.maximum(edges1, edges2), edges3)

        # LAB color gradient (ΔE)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

        # Sobel gradients for each channel
        L_grad = np.sqrt(cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)**2 +
                         cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)**2)
        A_grad = np.sqrt(cv2.Sobel(A, cv2.CV_32F, 1, 0, ksize=3)**2 +
                         cv2.Sobel(A, cv2.CV_32F, 0, 1, ksize=3)**2)
        B_grad = np.sqrt(cv2.Sobel(B, cv2.CV_32F, 1, 0, ksize=3)**2 +
                         cv2.Sobel(B, cv2.CV_32F, 0, 1, ksize=3)**2)

        delta_e = np.sqrt(L_grad**2 + A_grad**2 + B_grad**2)

        # Normalize ΔE to 0-255
        if delta_e.max() > 0:
            delta_e = (delta_e / delta_e.max() * 255).astype(np.uint8)
        else:
            delta_e = np.zeros_like(gray)

        # Combine: max of Canny and ΔE
        combined = np.maximum(canny_edges, delta_e)

        return combined

    def upscale_edges(
        self,
        edges: np.ndarray,
        scale: int = 4
    ) -> np.ndarray:
        """Upscale edge map preserving sharpness.

        Uses nearest-neighbor for binary edges to avoid blurring.

        Args:
            edges: Edge map (uint8)
            scale: Upscaling factor

        Returns:
            Upscaled edge map (uint8)
        """
        h, w = edges.shape[:2]
        new_h, new_w = h * scale, w * scale

        # Use INTER_NEAREST to preserve edge sharpness
        upscaled = cv2.resize(edges, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Optional: slight Gaussian blur to smooth jaggies
        upscaled = cv2.GaussianBlur(upscaled, (3, 3), 0.5)

        return upscaled

    def inject_edges_to_image(
        self,
        image: np.ndarray,
        edges: np.ndarray,
        strength: float = 0.25
    ) -> np.ndarray:
        """Inject edge information into image.

        Uses frequency separation to add edges to high-frequency component.

        Args:
            image: RGB image (uint8)
            edges: Edge map (uint8)
            strength: Injection strength (0-1)

        Returns:
            Edge-enhanced image (uint8)
        """
        if strength <= 0:
            return image

        img_float = image.astype(np.float32)

        # Extract low frequency (blur)
        sigma = 2.0
        ksize = int(sigma * 6) | 1
        low_freq = cv2.GaussianBlur(img_float, (ksize, ksize), sigma)

        # High frequency = original - low
        high_freq = img_float - low_freq

        # Normalize edges to 0-1
        edge_norm = edges.astype(np.float32) / 255.0

        # Smooth edges for natural blending
        edge_smooth = cv2.GaussianBlur(edge_norm, (5, 5), 1.0)

        # Create edge gradient for directional enhancement
        edge_dx = cv2.Sobel(edge_smooth, cv2.CV_32F, 1, 0, ksize=3)
        edge_dy = cv2.Sobel(edge_smooth, cv2.CV_32F, 0, 1, ksize=3)
        edge_mag = np.sqrt(edge_dx**2 + edge_dy**2)

        if edge_mag.max() > 0:
            edge_mag = edge_mag / edge_mag.max()

        # Expand to 3 channels
        edge_3d = edge_mag[:, :, np.newaxis]

        # Inject edges: darken at edge locations (creates contrast)
        edge_injection = edge_3d * strength * 40  # Scale factor

        # Subtract to create darkening effect at edges
        enhanced_high = high_freq - edge_injection

        # Reconstruct
        result = low_freq + enhanced_high
        result = np.clip(result, 0, 255).astype(np.uint8)

        return result

    def apply_edge_sharpening(
        self,
        image: np.ndarray,
        edges: np.ndarray,
        strength: float = 0.15
    ) -> np.ndarray:
        """Apply edge-guided sharpening.

        Sharpens only at edge locations to avoid amplifying noise.

        Args:
            image: RGB image (uint8)
            edges: Edge map (uint8)
            strength: Sharpening strength (0-1)

        Returns:
            Sharpened image (uint8)
        """
        if strength <= 0:
            return image

        # Unsharp mask
        img_float = image.astype(np.float32)
        blurred = cv2.GaussianBlur(img_float, (5, 5), 1.5)
        unsharp_mask = img_float - blurred

        # Edge-guided: apply sharpening only at edges
        edge_mask = edges.astype(np.float32) / 255.0
        edge_mask = cv2.GaussianBlur(edge_mask, (7, 7), 2.0)
        edge_mask = edge_mask[:, :, np.newaxis]

        # Apply masked sharpening
        sharpened = img_float + unsharp_mask * edge_mask * strength * 2
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

        return sharpened

    def combine_edge_maps(
        self,
        edges_upscaled: np.ndarray,
        edges_detected: np.ndarray,
        original_weight: float = 0.7
    ) -> np.ndarray:
        """Combine upscaled original edges with newly detected edges.

        Args:
            edges_upscaled: Upscaled original edge map
            edges_detected: Edges detected from upscaled image
            original_weight: Weight for original edges (0-1)

        Returns:
            Combined edge map
        """
        # Weighted combination favoring original edges
        combined = (edges_upscaled.astype(np.float32) * original_weight +
                    edges_detected.astype(np.float32) * (1 - original_weight))

        # Also take max to ensure we don't lose strong edges
        combined = np.maximum(combined, edges_upscaled.astype(np.float32) * 0.8)

        return np.clip(combined, 0, 255).astype(np.uint8)

    def compute_edge_proximity_mask(
        self,
        edges: np.ndarray,
        sigma: float = 8.0
    ) -> np.ndarray:
        """Compute smooth edge proximity mask for blending.

        Args:
            edges: Edge map (uint8)
            sigma: Gaussian sigma for proximity falloff

        Returns:
            Proximity mask (float32, 0-1, 1 = on edge)
        """
        # Distance transform from edges
        non_edges = (edges < 50).astype(np.uint8)
        dist = cv2.distanceTransform(non_edges, cv2.DIST_L2, 5)

        # Gaussian falloff
        proximity = np.exp(-(dist ** 2) / (2 * sigma ** 2))

        return proximity.astype(np.float32)

    def blend_esrgan_bicubic(
        self,
        esrgan_result: np.ndarray,
        original_image: np.ndarray,
        edge_proximity: np.ndarray,
        scale: int = 4,
        edge_blend_strength: float = 0.6
    ) -> np.ndarray:
        """Blend ESRGAN result with bicubic upscale based on edge proximity.

        Near edges: Use more of the sharper bicubic result
        Far from edges: Use ESRGAN (smoother but higher quality)

        Args:
            esrgan_result: Real-ESRGAN upscaled image
            original_image: Original low-res image
            edge_proximity: Edge proximity mask (0-1)
            scale: Upscaling factor
            edge_blend_strength: How much bicubic to blend at edges (0-1)

        Returns:
            Blended image
        """
        h, w = original_image.shape[:2]
        new_h, new_w = h * scale, w * scale

        # Bicubic upscale of original (preserves edges better)
        bicubic_result = cv2.resize(
            original_image,
            (new_w, new_h),
            interpolation=cv2.INTER_CUBIC
        )

        # Apply unsharp mask to bicubic for extra sharpness
        blurred = cv2.GaussianBlur(bicubic_result.astype(np.float32), (5, 5), 1.0)
        sharpened = bicubic_result.astype(np.float32) + 0.5 * (bicubic_result.astype(np.float32) - blurred)
        bicubic_sharp = np.clip(sharpened, 0, 255).astype(np.uint8)

        # Ensure size match
        if edge_proximity.shape[:2] != esrgan_result.shape[:2]:
            edge_proximity = cv2.resize(
                edge_proximity,
                (esrgan_result.shape[1], esrgan_result.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

        # Blend: at edges use bicubic, elsewhere use ESRGAN
        # blend_weight = edge_proximity * edge_blend_strength
        blend_weight = (edge_proximity * edge_blend_strength)[:, :, np.newaxis]

        blended = (esrgan_result.astype(np.float32) * (1 - blend_weight) +
                   bicubic_sharp.astype(np.float32) * blend_weight)

        return np.clip(blended, 0, 255).astype(np.uint8)

    def forward(
        self,
        image: Tensor,
        scale: Optional[ScaleFactor] = None,
        denoise: bool = True,
        **kwargs: Any,
    ) -> ModuleOutput:
        """Edge-preserving super resolution.

        Args:
            image: Input tensor (B, C, H, W)
            scale: Upscaling factor
            denoise: Apply denoising

        Returns:
            ModuleOutput with upscaled image
        """
        if not self._initialized or self.real_esrgan is None:
            raise RuntimeError("Real-ESRGAN not initialized")

        start_time = time.time()

        if image.dim() == 3:
            image = image.unsqueeze(0)

        # Convert to numpy for edge extraction
        img_np = self._tensor_to_numpy(image[0])

        # Step 1: Extract edges before upscaling
        edges_original = self.extract_edges_enhanced(img_np)

        # Step 2: Upscale with Real-ESRGAN
        esrgan_output = self.real_esrgan.forward(image, scale=scale, denoise=denoise)
        upscaled_tensor = esrgan_output.result

        # Convert upscaled to numpy
        upscaled_np = self._tensor_to_numpy(upscaled_tensor[0])

        # Step 3: Upscale edge map
        scale_factor = scale.value if scale else self.scale
        edges_upscaled = self.upscale_edges(edges_original, scale=scale_factor)

        # Ensure size match
        if edges_upscaled.shape[:2] != upscaled_np.shape[:2]:
            edges_upscaled = cv2.resize(
                edges_upscaled,
                (upscaled_np.shape[1], upscaled_np.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        # Step 4: Compute edge proximity mask
        edge_proximity = self.compute_edge_proximity_mask(edges_upscaled, sigma=12.0)

        # Step 5: Blend ESRGAN with sharpened bicubic at edge locations
        blended = self.blend_esrgan_bicubic(
            upscaled_np,
            img_np,
            edge_proximity,
            scale=scale_factor,
            edge_blend_strength=0.5  # 50% bicubic at edges
        )

        # Step 6: Extract edges from blended result
        edges_post_upscale = self.extract_edges_enhanced(blended)

        # Step 7: Combine original upscaled edges with newly detected edges
        edges_combined = self.combine_edge_maps(
            edges_upscaled,
            edges_post_upscale,
            original_weight=0.6
        )

        # Step 8: Inject combined edges into blended image
        edge_enhanced = self.inject_edges_to_image(
            blended,
            edges_combined,
            strength=self.edge_inject_strength
        )

        # Step 9: Apply edge-guided sharpening
        final_result = self.apply_edge_sharpening(
            edge_enhanced,
            edges_combined,
            strength=self.edge_sharpen_strength
        )

        # Convert back to tensor
        result_tensor = self._numpy_to_tensor(final_result).unsqueeze(0)

        processing_time = time.time() - start_time

        # Build intermediates for debugging
        intermediates = {
            'edges_original': self._numpy_to_tensor(
                np.stack([edges_original] * 3, axis=-1)
            ),
            'edges_upscaled': self._numpy_to_tensor(
                np.stack([edges_upscaled] * 3, axis=-1)
            ),
            'edge_proximity': self._numpy_to_tensor(
                np.stack([(edge_proximity * 255).astype(np.uint8)] * 3, axis=-1)
            ),
            'esrgan_output': esrgan_output.result[0],
            'blended': self._numpy_to_tensor(blended),
            'edges_combined': self._numpy_to_tensor(
                np.stack([edges_combined] * 3, axis=-1)
            ),
            'edge_enhanced': self._numpy_to_tensor(edge_enhanced),
        }

        metadata = {
            'method': 'edge_preserving_superres',
            'scale': scale_factor,
            'edge_inject_strength': self.edge_inject_strength,
            'edge_sharpen_strength': self.edge_sharpen_strength,
            'processing_time': processing_time,
            'esrgan_metadata': esrgan_output.metadata,
        }

        return ModuleOutput(
            result=result_tensor,
            intermediate=intermediates,
            metadata=metadata,
        )

    def load_weights(self, checkpoint_path: str) -> None:
        """Load weights (delegates to Real-ESRGAN)."""
        if self.real_esrgan:
            self.real_esrgan.load_weights(checkpoint_path)

    @property
    def name(self) -> str:
        return "edge_preserving_superres"


__all__ = ["EdgePreservingSuperRes"]
