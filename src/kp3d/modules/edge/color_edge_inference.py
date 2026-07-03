"""Color-based Edge Inference for Korean paintings.

Infers edges from color information when original edges are
missing or faded. Combines:
1. Color gradient (ΔE in LAB space)
2. Superpixel segmentation boundaries

This is crucial for paintings where:
- No ink outlines exist (몰골법 style)
- Original edges have faded away
- Pigment boundaries are the only edge source
"""

import cv2
import numpy as np
import torch
from torch import Tensor
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

from kp3d.core.base import ModuleOutput
from kp3d.modules.edge.base import BaseEdgeDetection, EdgeConfig


@dataclass
class ColorEdgeConfig:
    """Configuration for color-based edge inference.

    Attributes:
        delta_e_threshold: Minimum ΔE to consider as edge (0-100)
        delta_e_sigma: Gaussian sigma for ΔE smoothing
        superpixel_segments: Number of SLIC superpixel segments
        superpixel_compactness: SLIC compactness (higher = more square)
        boundary_thickness: Thickness of superpixel boundaries
        combine_mode: How to combine ΔE and superpixel edges
        weak_edge_boost: Amplification factor for weak edges
        gradient_method: 'sobel', 'scharr', or 'delta_e'
    """
    delta_e_threshold: float = 5.0
    delta_e_sigma: float = 1.0
    superpixel_segments: int = 500
    superpixel_compactness: float = 10.0
    boundary_thickness: int = 1
    combine_mode: str = "max"  # "max", "weighted", "multiply"
    weak_edge_boost: float = 2.0
    gradient_method: str = "delta_e"


class ColorEdgeInference(BaseEdgeDetection):
    """색상 기반 엣지 추론기

    원본에 엣지가 없거나 희미한 경우, 색상 정보에서 엣지를 추론합니다.

    Methods:
    1. ΔE (색차): LAB 공간에서 인접 픽셀 간 색 차이 계산
    2. Superpixel: SLIC으로 의미있는 영역 분할 후 경계 추출
    """

    def __init__(
        self,
        config: Optional[EdgeConfig] = None,
        color_config: Optional[ColorEdgeConfig] = None,
        **kwargs
    ):
        super().__init__(config=config, **kwargs)
        self.color_config = color_config or ColorEdgeConfig()
        self._initialized = True

    @property
    def name(self) -> str:
        return "color_edge_inference"

    def load_weights(self, checkpoint_path: str) -> None:
        self._initialized = True

    def _tensor_to_numpy(self, tensor: Tensor) -> np.ndarray:
        """Convert tensor to numpy array."""
        if tensor.dim() == 4:
            tensor = tensor[0]
        arr = tensor.cpu().numpy()
        if arr.shape[0] in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr.squeeze(2)
        return (np.clip(arr, 0, 1) * 255).astype(np.uint8)

    def _numpy_to_tensor(self, arr: np.ndarray, channels: int = 1) -> Tensor:
        """Convert numpy array to tensor."""
        arr = arr.astype(np.float32) / 255.0
        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]
        elif arr.ndim == 3 and arr.shape[2] in (1, 3):
            arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr).to(device=self.device, dtype=self.dtype)

    def compute_delta_e(self, image: np.ndarray) -> np.ndarray:
        """Compute color difference (ΔE) gradient map.

        Calculates the perceptual color difference between adjacent pixels
        in LAB color space. High ΔE = potential edge.

        Args:
            image: RGB image (uint8)

        Returns:
            ΔE map (float32, 0-100+ range)
        """
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)

        # Compute gradients in each LAB channel
        # Using Sobel for smooth gradients
        L = lab[:, :, 0]
        A = lab[:, :, 1]
        B = lab[:, :, 2]

        # Sobel gradients
        L_dx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
        L_dy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
        A_dx = cv2.Sobel(A, cv2.CV_32F, 1, 0, ksize=3)
        A_dy = cv2.Sobel(A, cv2.CV_32F, 0, 1, ksize=3)
        B_dx = cv2.Sobel(B, cv2.CV_32F, 1, 0, ksize=3)
        B_dy = cv2.Sobel(B, cv2.CV_32F, 0, 1, ksize=3)

        # ΔE = sqrt(ΔL² + Δa² + Δb²) for each direction
        delta_e_x = np.sqrt(L_dx**2 + A_dx**2 + B_dx**2)
        delta_e_y = np.sqrt(L_dy**2 + A_dy**2 + B_dy**2)

        # Combined magnitude
        delta_e = np.sqrt(delta_e_x**2 + delta_e_y**2)

        # Optional smoothing
        if self.color_config.delta_e_sigma > 0:
            ksize = int(self.color_config.delta_e_sigma * 4) | 1
            delta_e = cv2.GaussianBlur(delta_e, (ksize, ksize),
                                        self.color_config.delta_e_sigma)

        return delta_e

    def compute_superpixel_boundaries(self, image: np.ndarray) -> np.ndarray:
        """Extract boundaries from superpixel segmentation.

        Uses SLIC if available, otherwise falls back to K-means clustering.

        Args:
            image: RGB image (uint8)

        Returns:
            Binary boundary map (uint8, 0 or 255)
        """
        try:
            # Try SLIC if opencv-contrib is available
            slic = cv2.ximgproc.createSuperpixelSLIC(
                image,
                algorithm=cv2.ximgproc.SLIC,
                region_size=int(np.sqrt(image.shape[0] * image.shape[1] /
                                         self.color_config.superpixel_segments)),
                ruler=self.color_config.superpixel_compactness
            )
            slic.iterate(10)
            boundary_mask = slic.getLabelContourMask(thick_line=True)

        except AttributeError:
            # Fallback: K-means based region segmentation
            boundary_mask = self._compute_kmeans_boundaries(image)

        # Optionally dilate boundaries
        if self.color_config.boundary_thickness > 1:
            kernel = np.ones((self.color_config.boundary_thickness,
                            self.color_config.boundary_thickness), np.uint8)
            boundary_mask = cv2.dilate(boundary_mask, kernel, iterations=1)

        return boundary_mask

    def _compute_kmeans_boundaries(self, image: np.ndarray) -> np.ndarray:
        """Fallback boundary detection using K-means clustering.

        Args:
            image: RGB image (uint8)

        Returns:
            Binary boundary map (uint8, 0 or 255)
        """
        h, w = image.shape[:2]

        # Convert to LAB for perceptually uniform clustering
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

        # Reshape for K-means
        pixels = lab.reshape(-1, 3).astype(np.float32)

        # K-means clustering
        n_clusters = min(self.color_config.superpixel_segments // 50, 20)
        n_clusters = max(n_clusters, 5)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, _ = cv2.kmeans(pixels, n_clusters, None, criteria, 3,
                                   cv2.KMEANS_PP_CENTERS)

        # Reshape labels back to image
        label_map = labels.reshape(h, w)

        # Find boundaries using gradient
        # Compute gradient of label map
        gx = cv2.Sobel(label_map.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(label_map.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        gradient = np.sqrt(gx**2 + gy**2)

        # Threshold to get boundaries
        boundary_mask = (gradient > 0).astype(np.uint8) * 255

        return boundary_mask

    def boost_weak_edges(self, edge_map: np.ndarray) -> np.ndarray:
        """Amplify weak edges while preserving strong ones.

        Uses a non-linear transformation to boost weak signals
        without saturating strong ones.

        Args:
            edge_map: Edge map (float32)

        Returns:
            Boosted edge map (float32)
        """
        boost = self.color_config.weak_edge_boost
        if boost <= 1.0:
            return edge_map

        # Normalize to 0-1
        max_val = edge_map.max()
        if max_val < 1e-6:
            return edge_map

        normalized = edge_map / max_val

        # Non-linear boost: sqrt or power function
        # sqrt boosts weak signals more than strong ones
        boosted = np.power(normalized, 1.0 / boost)

        return boosted * max_val

    def combine_edge_maps(
        self,
        delta_e_map: np.ndarray,
        superpixel_map: np.ndarray
    ) -> np.ndarray:
        """Combine ΔE and superpixel edge maps.

        Args:
            delta_e_map: ΔE-based edge map (float32)
            superpixel_map: Superpixel boundary map (uint8)

        Returns:
            Combined edge map (uint8, 0-255)
        """
        # Normalize ΔE map to 0-255
        delta_e_norm = delta_e_map.copy()
        if delta_e_norm.max() > 0:
            # Apply threshold
            threshold = self.color_config.delta_e_threshold
            delta_e_norm = np.clip(delta_e_norm - threshold, 0, None)
            if delta_e_norm.max() > 0:
                delta_e_norm = (delta_e_norm / delta_e_norm.max() * 255)
        delta_e_norm = delta_e_norm.astype(np.float32)

        # Superpixel to float
        superpixel_float = superpixel_map.astype(np.float32)

        # Combine based on mode
        mode = self.color_config.combine_mode

        if mode == "max":
            combined = np.maximum(delta_e_norm, superpixel_float)
        elif mode == "weighted":
            # ΔE weighted more (0.6) as it's gradient-based
            combined = 0.6 * delta_e_norm + 0.4 * superpixel_float
        elif mode == "multiply":
            # Both must agree - more conservative
            combined = (delta_e_norm / 255) * superpixel_float
        else:
            combined = np.maximum(delta_e_norm, superpixel_float)

        return np.clip(combined, 0, 255).astype(np.uint8)

    def infer_edges(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Infer edges from color information.

        Args:
            image: RGB image (uint8)

        Returns:
            Tuple of (inferred_edges, intermediate_maps)
        """
        # Step 1: Compute ΔE gradient
        delta_e_map = self.compute_delta_e(image)

        # Step 2: Compute superpixel boundaries
        try:
            superpixel_map = self.compute_superpixel_boundaries(image)
        except Exception as e:
            # Fallback if SLIC fails (e.g., opencv-contrib not installed)
            print(f"Superpixel failed: {e}, using ΔE only")
            superpixel_map = np.zeros(image.shape[:2], dtype=np.uint8)

        # Step 3: Boost weak edges in ΔE map
        delta_e_boosted = self.boost_weak_edges(delta_e_map)

        # Step 4: Combine
        combined = self.combine_edge_maps(delta_e_boosted, superpixel_map)

        intermediates = {
            'delta_e_raw': delta_e_map,
            'delta_e_boosted': delta_e_boosted,
            'superpixel': superpixel_map,
            'combined': combined,
        }

        return combined, intermediates

    def forward(
        self,
        image: Tensor,
        return_all: bool = False,
        **kwargs: Any
    ) -> ModuleOutput:
        """Infer edges from image color information.

        Args:
            image: Input image tensor (B, C, H, W)
            return_all: If True, return all intermediate maps
            **kwargs: Additional parameters

        Returns:
            ModuleOutput with inferred edges
        """
        import time
        start = time.time()

        if image.dim() == 3:
            image = image.unsqueeze(0)

        # Convert to numpy
        img_np = self._tensor_to_numpy(image[0])

        # Infer edges
        inferred_edges, intermediates_np = self.infer_edges(img_np)

        elapsed = time.time() - start

        # Convert to tensors
        result_tensor = self._numpy_to_tensor(inferred_edges).unsqueeze(0)

        intermediates = {}
        if return_all:
            for key, arr in intermediates_np.items():
                if arr.dtype == np.float32:
                    arr_norm = (arr / (arr.max() + 1e-8) * 255).astype(np.uint8)
                else:
                    arr_norm = arr
                intermediates[key] = self._numpy_to_tensor(arr_norm)

        return ModuleOutput(
            result=result_tensor,
            intermediate=intermediates,
            metadata={
                'method': 'color_edge_inference',
                'processing_time': elapsed,
                'delta_e_threshold': self.color_config.delta_e_threshold,
                'superpixel_segments': self.color_config.superpixel_segments,
                'weak_edge_boost': self.color_config.weak_edge_boost,
                'combine_mode': self.color_config.combine_mode,
            }
        )


__all__ = ["ColorEdgeInference", "ColorEdgeConfig"]
