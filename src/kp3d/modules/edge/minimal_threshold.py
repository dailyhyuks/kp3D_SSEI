"""Minimal threshold edge detection for analysis.

Detects ALL possible edges with minimal thresholds to analyze
the difference between noise and actual contours.
"""

import cv2
import numpy as np
import torch
from torch import Tensor
from typing import Optional, Dict, Any

from kp3d.core.base import ModuleOutput
from kp3d.modules.edge.base import BaseEdgeDetection, EdgeConfig


class MinimalThresholdDetector(BaseEdgeDetection):
    """최소 임계치 엣지 검출기 - 분석용

    모든 가능한 엣지를 검출하여 노이즈와 실제 윤곽선을
    시각적으로 분석할 수 있도록 함.
    """

    def __init__(self, config: Optional[EdgeConfig] = None, **kwargs):
        super().__init__(config=config, **kwargs)
        self._initialized = True

    @property
    def name(self) -> str:
        return "minimal_threshold"

    def load_weights(self, checkpoint_path: str) -> None:
        self._initialized = True

    def _tensor_to_numpy_rgb(self, tensor: Tensor) -> np.ndarray:
        if tensor.dim() == 4:
            tensor = tensor[0]
        arr = tensor.cpu().numpy()
        if arr.shape[0] == 3:
            arr = np.transpose(arr, (1, 2, 0))
        return (np.clip(arr, 0, 1) * 255).astype(np.uint8)

    def _numpy_to_tensor(self, arr: np.ndarray) -> Tensor:
        arr = arr.astype(np.float32) / 255.0
        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]
        return torch.from_numpy(arr).to(device=self.device, dtype=self.dtype)

    def detect_all_edges(self, image_rgb: np.ndarray) -> Dict[str, np.ndarray]:
        """최소 임계치로 모든 엣지 검출"""
        results = {}

        # RGB to LAB
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        L, A, B = cv2.split(lab)

        # Grayscale
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        # 1. 다양한 임계치의 Canny
        L_blur = cv2.GaussianBlur(L, (3, 3), 0)
        gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

        results['canny_L_min'] = cv2.Canny(L_blur, 5, 15)
        results['canny_L_low'] = cv2.Canny(L_blur, 10, 30)
        results['canny_L_mid'] = cv2.Canny(L_blur, 30, 80)
        results['canny_gray_min'] = cv2.Canny(gray_blur, 5, 15)

        # 2. Sobel 그래디언트
        sobel_x = cv2.Sobel(L_blur, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(L_blur, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel_mag = (sobel_mag / (sobel_mag.max() + 1e-8) * 255).astype(np.uint8)
        results['sobel_magnitude'] = sobel_mag

        _, results['sobel_thresh_5'] = cv2.threshold(sobel_mag, 5, 255, cv2.THRESH_BINARY)
        _, results['sobel_thresh_10'] = cv2.threshold(sobel_mag, 10, 255, cv2.THRESH_BINARY)
        _, results['sobel_thresh_20'] = cv2.threshold(sobel_mag, 20, 255, cv2.THRESH_BINARY)

        # 3. Laplacian
        laplacian = cv2.Laplacian(L_blur, cv2.CV_64F, ksize=3)
        laplacian = np.abs(laplacian)
        laplacian = (laplacian / (laplacian.max() + 1e-8) * 255).astype(np.uint8)
        results['laplacian'] = laplacian
        _, results['laplacian_thresh_10'] = cv2.threshold(laplacian, 10, 255, cv2.THRESH_BINARY)

        # 4. 색상 채널 그래디언트
        A_blur = cv2.GaussianBlur(A, (3, 3), 0)
        B_blur = cv2.GaussianBlur(B, (3, 3), 0)

        sobel_A = np.sqrt(cv2.Sobel(A_blur, cv2.CV_64F, 1, 0)**2 + cv2.Sobel(A_blur, cv2.CV_64F, 0, 1)**2)
        sobel_B = np.sqrt(cv2.Sobel(B_blur, cv2.CV_64F, 1, 0)**2 + cv2.Sobel(B_blur, cv2.CV_64F, 0, 1)**2)

        results['color_A_gradient'] = (sobel_A / (sobel_A.max() + 1e-8) * 255).astype(np.uint8)
        results['color_B_gradient'] = (sobel_B / (sobel_B.max() + 1e-8) * 255).astype(np.uint8)

        # 5. 모든 채널 합성
        all_edges = np.maximum.reduce([
            results['canny_L_min'],
            results['canny_gray_min'],
            results['sobel_thresh_5'],
            results['laplacian_thresh_10']
        ])
        results['all_combined'] = all_edges

        # 6. 적응형 임계치
        adaptive = cv2.adaptiveThreshold(
            L_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        results['adaptive_threshold'] = 255 - adaptive

        return results

    def analyze_edge_density(self, edge_map: np.ndarray) -> Dict[str, float]:
        """엣지 밀도 분석"""
        total_pixels = edge_map.size
        edge_pixels = np.sum(edge_map > 0)

        return {
            'total_pixels': int(total_pixels),
            'edge_pixels': int(edge_pixels),
            'density_percent': float((edge_pixels / total_pixels) * 100)
        }

    def forward(self, image: Tensor, **kwargs: Any) -> ModuleOutput:
        """최소 임계치 엣지 검출"""
        import time
        start = time.time()

        if image.dim() == 3:
            image = image.unsqueeze(0)

        img_np = self._tensor_to_numpy_rgb(image[0])
        all_edges = self.detect_all_edges(img_np)

        intermediates = {}
        densities = {}

        for name, edge_map in all_edges.items():
            intermediates[name] = self._numpy_to_tensor(edge_map)
            densities[name] = self.analyze_edge_density(edge_map)

        result = intermediates['all_combined'].unsqueeze(0)
        elapsed = time.time() - start

        return ModuleOutput(
            result=result,
            intermediate=intermediates,
            metadata={
                'method': 'minimal_threshold',
                'processing_time': elapsed,
                'edge_densities': densities
            }
        )
