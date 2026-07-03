"""Smart edge fusion for Korean paintings.

Combines structure-based and color-based edge detection
with intelligent noise filtering.
"""

import cv2
import numpy as np
import torch
from torch import Tensor
from typing import Optional, Dict, Any, Tuple, List

from kp3d.core.base import ModuleOutput
from kp3d.modules.edge.base import BaseEdgeDetection, EdgeConfig


class SmartFusionDetector(BaseEdgeDetection):
    """스마트 엣지 융합 검출기

    구조 기반 + 색상 기반 엣지를 지능적으로 융합하고
    연결 성분 분석으로 노이즈를 제거합니다.
    """

    def __init__(
        self,
        config: Optional[EdgeConfig] = None,
        external_canny: Tuple[int, int] = (30, 80),
        internal_canny: Tuple[int, int] = (10, 30),
        color_threshold: int = 15,
        min_component_area: int = 10,
        min_edge_length: int = 30,
        **kwargs
    ):
        super().__init__(config=config, **kwargs)
        self.external_canny = external_canny
        self.internal_canny = internal_canny
        self.color_threshold = color_threshold
        self.min_component_area = min_component_area
        self.min_edge_length = min_edge_length
        self._initialized = True

    @property
    def name(self) -> str:
        return "smart_fusion"

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

    def detect_external_contours(self, L_blur: np.ndarray) -> np.ndarray:
        """외곽선 검출 - 높은 임계치로 깔끔하게"""
        return cv2.Canny(L_blur, self.external_canny[0], self.external_canny[1])

    def detect_internal_structure(self, L_blur: np.ndarray) -> np.ndarray:
        """내부 구조 검출 - 중간 임계치"""
        return cv2.Canny(L_blur, self.internal_canny[0], self.internal_canny[1])

    def detect_color_boundaries(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """색상 경계 검출 - LAB A/B 채널에 Canny 적용

        단순 threshold 대신 Canny를 사용하여 실제 엣지만 검출
        """
        A_blur = cv2.GaussianBlur(A, (3, 3), 0)
        B_blur = cv2.GaussianBlur(B, (3, 3), 0)

        # Apply Canny directly to color channels
        canny_A = cv2.Canny(A_blur, 20, 60)
        canny_B = cv2.Canny(B_blur, 20, 60)

        # Combine color edges
        color_edges = np.maximum(canny_A, canny_B)

        return color_edges

    def filter_by_connected_components(self, edge_map: np.ndarray) -> np.ndarray:
        """연결 성분 분석으로 노이즈 필터링

        - 너무 작은 성분 제거 (면적 < min_component_area)
        - 실제 엣지는 보존 (길이 > min_edge_length)
        """
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            edge_map, connectivity=8
        )

        result = np.zeros_like(edge_map)

        for i in range(1, num_labels):  # Skip background (0)
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]

            # Calculate approximate length (diagonal)
            length = np.sqrt(width**2 + height**2)

            # Keep if area is sufficient OR if it's a long edge
            if area >= self.min_component_area or length >= self.min_edge_length:
                result[labels == i] = 255

        return result

    def zhang_suen_thinning(self, binary: np.ndarray) -> np.ndarray:
        """Zhang-Suen 골격화로 1픽셀 선 생성"""
        skeleton = binary.copy()

        def iteration(img, step):
            changed = []
            rows, cols = img.shape

            for i in range(1, rows - 1):
                for j in range(1, cols - 1):
                    if img[i, j] == 0:
                        continue

                    # 8-neighbors
                    P = [img[i-1, j], img[i-1, j+1], img[i, j+1], img[i+1, j+1],
                         img[i+1, j], img[i+1, j-1], img[i, j-1], img[i-1, j-1]]

                    # B(P1) = number of non-zero neighbors
                    B = sum(P)

                    # A(P1) = number of 0->1 transitions
                    A = sum((P[k] == 0 and P[(k+1) % 8] == 1) for k in range(8))

                    if step == 0:
                        c1 = P[0] * P[2] * P[4]
                        c2 = P[2] * P[4] * P[6]
                    else:
                        c1 = P[0] * P[2] * P[6]
                        c2 = P[0] * P[4] * P[6]

                    if 2 <= B <= 6 and A == 1 and c1 == 0 and c2 == 0:
                        changed.append((i, j))

            for i, j in changed:
                img[i, j] = 0

            return len(changed) > 0

        # Normalize to binary
        skeleton = (skeleton > 0).astype(np.uint8)

        changed = True
        while changed:
            changed = False
            if iteration(skeleton, 0):
                changed = True
            if iteration(skeleton, 1):
                changed = True

        return skeleton * 255

    def forward(self, image: Tensor, **kwargs: Any) -> ModuleOutput:
        """스마트 융합 엣지 검출"""
        import time
        start = time.time()

        if image.dim() == 3:
            image = image.unsqueeze(0)

        # Convert to numpy RGB
        img_np = self._tensor_to_numpy_rgb(image[0])

        # Convert to LAB
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        L, A, B = cv2.split(lab)
        L_blur = cv2.GaussianBlur(L, (3, 3), 0)

        # Step 1: External contours (clean, high threshold)
        external = self.detect_external_contours(L_blur)

        # Step 2: Internal structure (medium threshold)
        internal = self.detect_internal_structure(L_blur)

        # Step 3: Color boundaries
        color_edges = self.detect_color_boundaries(A, B)

        # Step 4: Combine all edges
        raw_combined = np.maximum.reduce([external, internal, color_edges])

        # Step 5: Filter noise with connected component analysis
        filtered = self.filter_by_connected_components(raw_combined)

        # Step 6: Skip heavy morphological ops - CC filtering is sufficient
        # Only do very light cleanup if edges are thick
        cleaned = filtered  # Keep filtered result as-is

        # Step 7: Thinning for 1-pixel edges
        try:
            thinned = cv2.ximgproc.thinning(cleaned, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        except AttributeError:
            thinned = self.zhang_suen_thinning(cleaned)

        elapsed = time.time() - start

        # Store intermediates for visualization
        intermediates = {
            'external': self._numpy_to_tensor(external),
            'internal': self._numpy_to_tensor(internal),
            'color_edges': self._numpy_to_tensor(color_edges),
            'raw_combined': self._numpy_to_tensor(raw_combined),
            'filtered': self._numpy_to_tensor(filtered),
            'cleaned': self._numpy_to_tensor(cleaned),
            'thinned': self._numpy_to_tensor(thinned),
        }

        result = self._numpy_to_tensor(thinned).unsqueeze(0)

        return ModuleOutput(
            result=result,
            intermediate=intermediates,
            metadata={
                'method': 'smart_fusion',
                'processing_time': elapsed,
                'external_canny': self.external_canny,
                'internal_canny': self.internal_canny,
                'color_threshold': self.color_threshold,
            }
        )
