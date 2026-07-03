"""Edge Resynthesizer for Korean paintings.

Resynthesize preserved original edges into restored images
to recover edge information lost during restoration/denoising.
"""

import cv2
import numpy as np
import torch
from torch import Tensor
from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel, Field

from kp3d.core.base import ModuleOutput
from kp3d.modules.edge.base import BaseEdgeDetection, EdgeConfig


class EdgeResynthesizerConfig(BaseModel):
    """Configuration for edge resynthesis.

    Attributes:
        edge_weight: Weight for original edges (0-1).
        current_weight: Weight for current image edges (0-1).
        unsharp_strength: Strength of unsharp masking (0-3).
        unsharp_radius: Radius for unsharp masking (0.5-5).
        blend_mode: How to blend original and current edges.
        preserve_original_edges: Whether to preserve original edges.
        edge_enhance_sigma: Sigma for edge enhancement gaussian.
        min_edge_strength: Minimum edge strength to preserve.
        edge_detection_method: Edge detection method (sobel recommended).

    Research Notes (contour_enhancement experiments):
        - Sobel achieves F1=0.908 vs Canny F1=0.804 against pseudo GT
        - Sobel provides better precision/recall balance
        - See research/contour_enhancement/IDEA.md for details
    """
    edge_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    current_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    unsharp_strength: float = Field(default=1.5, ge=0.0, le=3.0)
    unsharp_radius: float = Field(default=1.0, ge=0.5, le=5.0)
    blend_mode: Literal["weighted", "max", "overlay", "softlight"] = "weighted"
    preserve_original_edges: bool = True
    edge_enhance_sigma: float = Field(default=1.0, ge=0.1, le=3.0)
    min_edge_strength: float = Field(default=0.1, ge=0.0, le=0.5)
    edge_detection_method: Literal["sobel", "canny", "scharr"] = "sobel"

    class Config:
        frozen = False


class EdgeResynthesizer(BaseEdgeDetection):
    """Edge Resynthesizer - 원본 엣지를 복원된 이미지에 재합성

    복원/디노이징 과정에서 손실된 엣지 정보를 원본에서
    추출한 엣지를 사용하여 복구합니다.

    Pipeline:
    1. 원본 이미지에서 엣지 추출 (사전 저장)
    2. 복원된 이미지에서 현재 엣지 추출
    3. 두 엣지 맵을 blend_mode에 따라 합성
    4. Unsharp masking으로 최종 선명화
    """

    def __init__(
        self,
        config: Optional[EdgeConfig] = None,
        resynth_config: Optional[EdgeResynthesizerConfig] = None,
        **kwargs
    ):
        """Initialize edge resynthesizer.

        Args:
            config: Base edge detection config.
            resynth_config: Resynthesis-specific configuration.
            **kwargs: Additional arguments for base class.
        """
        super().__init__(config=config, **kwargs)
        self.resynth_config = resynth_config or EdgeResynthesizerConfig()
        self._initialized = True

    @property
    def name(self) -> str:
        return "edge_resynth"

    def load_weights(self, checkpoint_path: str) -> None:
        """No weights needed for this module."""
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

    def extract_edges(self, image: np.ndarray) -> np.ndarray:
        """Extract edges from an image.

        Uses the configured edge detection method:
        - sobel: Best for Korean paintings (F1=0.908 in experiments)
        - canny: Traditional binary edge detection
        - scharr: More sensitive to small gradients

        Args:
            image: Input image (RGB or grayscale).

        Returns:
            Edge map (grayscale, 0-255).
        """
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        method = self.resynth_config.edge_detection_method

        if method == "sobel":
            # Sobel: Best for Korean paintings (F1=0.908)
            # See research/contour_enhancement/experiment_03_gt_comparison.py
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edges = np.sqrt(sobel_x**2 + sobel_y**2)
            edges = np.clip(edges, 0, 255).astype(np.uint8)

        elif method == "scharr":
            # Scharr: More sensitive to small gradients
            scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
            scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
            edges = np.sqrt(scharr_x**2 + scharr_y**2)
            edges = np.clip(edges, 0, 255).astype(np.uint8)

        else:  # canny (legacy)
            # Apply bilateral filter to reduce noise while preserving edges
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)
            # Multi-scale Canny for better edge detection
            edges1 = cv2.Canny(filtered, 30, 80)
            edges2 = cv2.Canny(filtered, 50, 150)
            edges = np.maximum(edges1, edges2)

        return edges

    def blend_edges_weighted(
        self,
        original_edges: np.ndarray,
        current_edges: np.ndarray
    ) -> np.ndarray:
        """Weighted blend of original and current edges."""
        cfg = self.resynth_config
        blended = (
            cfg.edge_weight * original_edges.astype(np.float32) +
            cfg.current_weight * current_edges.astype(np.float32)
        )
        return np.clip(blended, 0, 255).astype(np.uint8)

    def blend_edges_max(
        self,
        original_edges: np.ndarray,
        current_edges: np.ndarray
    ) -> np.ndarray:
        """Take maximum of original and current edges."""
        return np.maximum(original_edges, current_edges)

    def blend_edges_overlay(
        self,
        original_edges: np.ndarray,
        current_edges: np.ndarray
    ) -> np.ndarray:
        """Overlay blend mode for edges."""
        a = original_edges.astype(np.float32) / 255.0
        b = current_edges.astype(np.float32) / 255.0

        # Overlay formula: 2ab if a < 0.5 else 1 - 2(1-a)(1-b)
        result = np.where(
            a < 0.5,
            2 * a * b,
            1 - 2 * (1 - a) * (1 - b)
        )
        return (np.clip(result, 0, 1) * 255).astype(np.uint8)

    def blend_edges_softlight(
        self,
        original_edges: np.ndarray,
        current_edges: np.ndarray
    ) -> np.ndarray:
        """Soft light blend mode for edges."""
        a = original_edges.astype(np.float32) / 255.0
        b = current_edges.astype(np.float32) / 255.0

        # Soft light formula
        result = np.where(
            b < 0.5,
            a - (1 - 2 * b) * a * (1 - a),
            a + (2 * b - 1) * (np.sqrt(a) - a)
        )
        return (np.clip(result, 0, 1) * 255).astype(np.uint8)

    def blend_edges(
        self,
        original_edges: np.ndarray,
        current_edges: np.ndarray
    ) -> np.ndarray:
        """Blend edges using configured blend mode.

        Args:
            original_edges: Pre-extracted original edges.
            current_edges: Edges from current (restored) image.

        Returns:
            Blended edge map.
        """
        mode = self.resynth_config.blend_mode

        if mode == "weighted":
            return self.blend_edges_weighted(original_edges, current_edges)
        elif mode == "max":
            return self.blend_edges_max(original_edges, current_edges)
        elif mode == "overlay":
            return self.blend_edges_overlay(original_edges, current_edges)
        elif mode == "softlight":
            return self.blend_edges_softlight(original_edges, current_edges)
        else:
            raise ValueError(f"Unknown blend mode: {mode}")

    def apply_unsharp_masking(
        self,
        image: np.ndarray,
        edges: np.ndarray
    ) -> np.ndarray:
        """Apply unsharp masking guided by edge map.

        Args:
            image: Input image (RGB).
            edges: Edge map for guidance.

        Returns:
            Sharpened image.
        """
        cfg = self.resynth_config

        # Convert to float
        img_float = image.astype(np.float32) / 255.0

        # Gaussian blur for unsharp mask
        ksize = int(cfg.unsharp_radius * 4) | 1  # Ensure odd
        blurred = cv2.GaussianBlur(img_float, (ksize, ksize), cfg.unsharp_radius)

        # Unsharp mask: original + strength * (original - blurred)
        mask = img_float - blurred

        # Weight mask by edge map - enhance more where edges are strong
        edge_weight = edges.astype(np.float32) / 255.0
        if len(image.shape) == 3:
            edge_weight = edge_weight[:, :, np.newaxis]

        # Apply weighted unsharp masking
        sharpened = img_float + cfg.unsharp_strength * mask * (0.3 + 0.7 * edge_weight)

        return (np.clip(sharpened, 0, 1) * 255).astype(np.uint8)

    def resynthesize(
        self,
        current_image: np.ndarray,
        original_edges: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resynthesize edges into the current image.

        Args:
            current_image: Current (restored) image (RGB).
            original_edges: Pre-extracted original edges.

        Returns:
            Tuple of (enhanced_image, final_edge_map).
        """
        cfg = self.resynth_config

        # Extract edges from current image
        current_edges = self.extract_edges(current_image)

        # Blend original and current edges
        if cfg.preserve_original_edges and original_edges is not None:
            blended_edges = self.blend_edges(original_edges, current_edges)
        else:
            blended_edges = current_edges

        # Filter weak edges
        min_val = int(cfg.min_edge_strength * 255)
        blended_edges = np.where(blended_edges >= min_val, blended_edges, 0).astype(np.uint8)

        # Apply unsharp masking guided by edge map
        enhanced = self.apply_unsharp_masking(current_image, blended_edges)

        return enhanced, blended_edges

    def forward(
        self,
        image: Tensor,
        original_edges: Optional[Tensor] = None,
        **kwargs: Any
    ) -> ModuleOutput:
        """Resynthesize edges into a restored image.

        Args:
            image: Current (restored) image tensor (B, C, H, W).
            original_edges: Pre-extracted original edge tensor (B, 1, H, W).
                          If None, uses only current edges.
            **kwargs: Additional parameters.

        Returns:
            ModuleOutput with enhanced image and edge metadata.
        """
        import time
        start = time.time()

        # Handle batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)

        # Convert to numpy
        img_np = self._tensor_to_numpy(image[0])

        # Handle original edges
        if original_edges is not None:
            if original_edges.dim() == 3:
                original_edges = original_edges.unsqueeze(0)
            orig_edges_np = self._tensor_to_numpy(original_edges[0])
        else:
            # Extract from current if not provided (fallback)
            orig_edges_np = self.extract_edges(img_np)

        # Perform resynthesis
        enhanced, final_edges = self.resynthesize(img_np, orig_edges_np)

        elapsed = time.time() - start

        # Convert back to tensor
        if len(enhanced.shape) == 3 and enhanced.shape[2] == 3:
            result_tensor = self._numpy_to_tensor(enhanced)
        else:
            result_tensor = self._numpy_to_tensor(enhanced)

        result_tensor = result_tensor.unsqueeze(0)

        # Build intermediate outputs
        intermediates = {
            'original_edges': self._numpy_to_tensor(orig_edges_np) if orig_edges_np is not None else None,
            'current_edges': self._numpy_to_tensor(self.extract_edges(img_np)),
            'blended_edges': self._numpy_to_tensor(final_edges),
            'enhanced': result_tensor.squeeze(0),
        }

        return ModuleOutput(
            result=result_tensor,
            intermediate=intermediates,
            metadata={
                'method': 'edge_resynth',
                'processing_time': elapsed,
                'blend_mode': self.resynth_config.blend_mode,
                'edge_weight': self.resynth_config.edge_weight,
                'unsharp_strength': self.resynth_config.unsharp_strength,
            }
        )


__all__ = [
    "EdgeResynthesizer",
    "EdgeResynthesizerConfig",
]
