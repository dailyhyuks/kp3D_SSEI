"""Korean traditional painting specialized edge detector."""

import time
from typing import Any, Optional

import cv2
import numpy as np
import torch
from torch import Tensor

from kp3d.core.base import ModuleOutput
from kp3d.modules.edge.base import BaseEdgeDetection, EdgeConfig
from kp3d.modules.edge.canny import CannyEdgeDetector
from kp3d.modules.edge.hed import HEDEdgeDetector


class KoreanInkEdgeDetector(BaseEdgeDetection):
    """먹선/담채 경계 특화 검출기.

    Specialized edge detector for Korean traditional painting characteristics:
    - 먹선 (Ink lines): Sharp, high-contrast brush strokes
    - 담채 (Light color): Soft, gradual color transitions

    Combines Canny and HED detectors with adaptive weighting.
    """

    def __init__(
        self,
        config: Optional[EdgeConfig] = None,
        use_hed: Optional[bool] = None,
        **kwargs
    ) -> None:
        """Initialize Korean ink edge detector.

        Args:
            config: Edge detection configuration.
            use_hed: Whether to use HED. Overrides config if provided.
            **kwargs: Additional arguments.
        """
        super().__init__(config=config, **kwargs)

        # Override use_hed if explicitly provided
        if use_hed is not None:
            self.config.use_hed = use_hed

        # Initialize sub-detectors
        self.canny_detector = CannyEdgeDetector(
            config=self.config,
            device=self.device,
            dtype=self.dtype
        )

        if self.config.use_hed:
            try:
                self.hed_detector = HEDEdgeDetector(
                    config=self.config,
                    device=self.device,
                    dtype=self.dtype
                )
                self._has_hed = not self.hed_detector._fallback_to_canny
            except Exception as e:
                print(f"HED initialization failed: {e}")
                self.hed_detector = None
                self._has_hed = False
        else:
            self.hed_detector = None
            self._has_hed = False

        self._initialized = True

    @property
    def name(self) -> str:
        """Module name."""
        return "korean_ink_edge"

    def load_weights(self, checkpoint_path: str) -> None:
        """Load weights for HED component.

        Args:
            checkpoint_path: Path to HED weights.
        """
        if self.hed_detector is not None and not self.hed_detector._fallback_to_canny:
            self.hed_detector.load_weights(checkpoint_path)
        self._initialized = True

    def _tensor_to_numpy(self, tensor: Tensor) -> np.ndarray:
        """Convert tensor to numpy array.

        Args:
            tensor: Input tensor (B, C, H, W) or (C, H, W).

        Returns:
            Numpy array in HWC format.
        """
        if tensor.dim() == 4:
            tensor = tensor[0]

        array = tensor.cpu().numpy()

        if array.shape[0] in [1, 3]:
            array = np.transpose(array, (1, 2, 0))

        if array.max() <= 1.0:
            array = (array * 255).astype(np.uint8)
        else:
            array = array.astype(np.uint8)

        return array

    def _numpy_to_tensor(self, array: np.ndarray) -> Tensor:
        """Convert numpy array to tensor.

        Args:
            array: Input array (H, W) or (H, W, C).

        Returns:
            Tensor in CHW format.
        """
        array = array.astype(np.float32) / 255.0

        if array.ndim == 2:
            array = array[:, :, np.newaxis]

        tensor = torch.from_numpy(np.transpose(array, (2, 0, 1)))
        return tensor.to(device=self.device, dtype=self.dtype)

    def detect_ink_lines(self, image: Tensor) -> Tensor:
        """먹선(墨線) 전용 검출 - 높은 대비, 선명한 경계.

        Detects sharp, high-contrast ink brush strokes typical of
        Korean traditional painting outlines.

        Args:
            image: Input image tensor (B, C, H, W).

        Returns:
            Ink line edge map (B, 1, H, W).
        """
        # Use high thresholds for sharp edges
        ink_config = EdgeConfig(
            low_threshold=self.config.high_threshold,  # Use higher thresholds
            high_threshold=self.config.high_threshold * 1.5,
            multi_scale=False  # Single scale for sharp detection
        )

        # Temporarily update canny detector config
        original_config = self.canny_detector.config
        self.canny_detector.config = ink_config

        # Detect with Canny
        result = self.canny_detector.forward(image)
        ink_edges = result.result

        # Restore original config
        self.canny_detector.config = original_config

        # Apply morphological operations to connect ink strokes
        np_edges = self._tensor_to_numpy(ink_edges)
        if np_edges.ndim == 3:
            np_edges = np_edges.squeeze(-1)

        # Dilate slightly to connect nearby ink strokes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        np_edges = cv2.dilate(np_edges, kernel, iterations=1)

        # Convert back to tensor
        ink_edges = self._numpy_to_tensor(np_edges).unsqueeze(0)

        return ink_edges

    def detect_damchae_boundary(self, image: Tensor) -> Tensor:
        """담채(淡彩) 경계 검출 - 부드러운 색상 전이.

        Detects soft, gradual color transitions typical of light
        color washes in Korean traditional painting.

        Args:
            image: Input image tensor (B, C, H, W).

        Returns:
            Damchae boundary map (B, 1, H, W).
        """
        # Use lower thresholds for subtle transitions
        damchae_config = EdgeConfig(
            low_threshold=self.config.low_threshold * 0.5,
            high_threshold=self.config.low_threshold,
            multi_scale=True,  # Multi-scale for soft boundaries
            scales=[1.0, 0.5]
        )

        # Temporarily update canny detector config
        original_config = self.canny_detector.config
        self.canny_detector.config = damchae_config

        # Detect with Canny
        result = self.canny_detector.forward(image)
        damchae_edges = result.result

        # Restore original config
        self.canny_detector.config = original_config

        # Apply Gaussian blur for soft boundaries
        np_edges = self._tensor_to_numpy(damchae_edges)
        if np_edges.ndim == 3:
            np_edges = np_edges.squeeze(-1)

        # Blur to create soft boundaries
        np_edges = cv2.GaussianBlur(np_edges, (5, 5), 2.0)

        # Convert back to tensor
        damchae_edges = self._numpy_to_tensor(np_edges).unsqueeze(0)

        return damchae_edges

    def detect_internal_details(self, image: Tensor) -> Tensor:
        """내부 디테일 검출 - 적응형 임계값 + 텍스처 분석.

        Detects internal details like dragon patterns, plank textures, etc.
        Uses adaptive thresholding and texture analysis to capture fine details
        within object boundaries.

        Args:
            image: Input image tensor (B, C, H, W).

        Returns:
            Internal detail edge map (B, 1, H, W).
        """
        np_image = self._tensor_to_numpy(image)

        # Convert to grayscale for processing
        if np_image.ndim == 3 and np_image.shape[2] == 3:
            gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = np_image.squeeze()

        # 1. Adaptive Canny with local contrast
        if self.config.adaptive_threshold:
            # Calculate local mean and std
            mean = cv2.GaussianBlur(gray, (15, 15), 0)
            std = cv2.GaussianBlur((gray - mean) ** 2, (15, 15), 0) ** 0.5

            # Adaptive thresholds based on local statistics
            low_adaptive = np.clip(mean - 0.5 * std, 10, 100)
            high_adaptive = np.clip(mean + 0.5 * std, 20, 150)

            # Per-pixel adaptive Canny (approximation via multi-pass)
            edges_adaptive = np.zeros_like(gray)
            for thresh_level in [0.5, 1.0, 1.5]:
                low = np.percentile(low_adaptive, thresh_level * 33)
                high = np.percentile(high_adaptive, thresh_level * 33)
                edges_level = cv2.Canny(gray, low, high)
                edges_adaptive = cv2.bitwise_or(edges_adaptive, edges_level)
        else:
            edges_adaptive = cv2.Canny(
                gray,
                self.config.low_threshold * 0.7,
                self.config.high_threshold * 0.7
            )

        # 2. Laplacian of Gaussian for texture details
        # Apply Gaussian blur first to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 1.0)
        # Laplacian to detect texture variations
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
        laplacian = np.abs(laplacian).astype(np.uint8)
        # Threshold to get texture edges
        _, texture_edges = cv2.threshold(
            laplacian,
            np.percentile(laplacian, 85),  # Top 15% of textures
            255,
            cv2.THRESH_BINARY
        )

        # 3. Multi-channel edge detection for color images
        if np_image.ndim == 3 and np_image.shape[2] == 3:
            channel_edges = []
            for i in range(3):
                channel = np_image[:, :, i]
                edges_ch = cv2.Canny(
                    channel,
                    self.config.low_threshold * 0.8,
                    self.config.high_threshold * 0.8
                )
                channel_edges.append(edges_ch)

            # Combine channel edges
            color_edges = cv2.bitwise_or(
                cv2.bitwise_or(channel_edges[0], channel_edges[1]),
                channel_edges[2]
            )
        else:
            color_edges = np.zeros_like(gray)

        # 4. Combine all internal detail edges
        internal_edges = cv2.bitwise_or(edges_adaptive, texture_edges)
        internal_edges = cv2.bitwise_or(internal_edges, color_edges)

        # 5. Morphological refinement
        # Remove very small isolated pixels (noise)
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        internal_edges = cv2.morphologyEx(
            internal_edges,
            cv2.MORPH_OPEN,
            kernel_erode
        )

        # Slightly dilate to connect nearby details
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        internal_edges = cv2.dilate(internal_edges, kernel_dilate, iterations=1)

        # Convert back to tensor
        return self._numpy_to_tensor(internal_edges).unsqueeze(0)

    def detect_color_boundaries(self, image: Tensor) -> Tensor:
        """색상 전이 경계 검출.

        Detects color transition boundaries using LAB color space.
        Emphasizes traditional Korean five-color (오방색) boundaries.

        Args:
            image: Input image tensor (B, C, H, W).

        Returns:
            Color boundary edge map (B, 1, H, W).
        """
        np_image = self._tensor_to_numpy(image)

        # Skip if grayscale
        if np_image.ndim != 3 or np_image.shape[2] != 3:
            return torch.zeros_like(image[:, :1, :, :])

        # Convert RGB to LAB color space
        # LAB is better for perceptual color differences
        lab = cv2.cvtColor(np_image, cv2.COLOR_RGB2LAB)

        # Split channels
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Compute gradients for each channel
        # L channel (lightness)
        grad_l_x = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3)
        grad_l_y = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=3)
        grad_l = np.sqrt(grad_l_x**2 + grad_l_y**2)

        # A channel (green-red)
        grad_a_x = cv2.Sobel(a_channel, cv2.CV_64F, 1, 0, ksize=3)
        grad_a_y = cv2.Sobel(a_channel, cv2.CV_64F, 0, 1, ksize=3)
        grad_a = np.sqrt(grad_a_x**2 + grad_a_y**2)

        # B channel (blue-yellow)
        grad_b_x = cv2.Sobel(b_channel, cv2.CV_64F, 1, 0, ksize=3)
        grad_b_y = cv2.Sobel(b_channel, cv2.CV_64F, 0, 1, ksize=3)
        grad_b = np.sqrt(grad_b_x**2 + grad_b_y**2)

        # Combine gradients with emphasis on color channels (a, b)
        # Korean traditional colors have strong chromaticity
        color_gradient = (
            0.3 * grad_l +  # Lightness contributes less
            0.4 * grad_a +  # Red-green boundaries
            0.3 * grad_b    # Blue-yellow boundaries
        )

        # Normalize to 0-255
        color_gradient = (
            255 * (color_gradient - color_gradient.min()) /
            (color_gradient.max() - color_gradient.min() + 1e-8)
        ).astype(np.uint8)

        # Apply threshold to get color boundaries
        # Use adaptive threshold for varying color intensities
        color_edges = cv2.adaptiveThreshold(
            color_gradient,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=-2
        )

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        color_edges = cv2.morphologyEx(color_edges, cv2.MORPH_OPEN, kernel)

        # Convert back to tensor
        return self._numpy_to_tensor(color_edges).unsqueeze(0)

    def forward(self, image: Tensor, **kwargs: Any) -> ModuleOutput:
        """Detect edges optimized for Korean painting characteristics.

        Combines:
        1. Ink line detection (sharp, high-contrast)
        2. Internal detail detection (textures, patterns)
        3. Color boundary detection (color transitions)
        4. Damchae boundary detection (soft, gradual)
        5. Optional HED deep learning detection
        6. Weighted fusion based on configuration

        Args:
            image: Input image tensor (B, C, H, W).
            **kwargs: Additional parameters.

        Returns:
            ModuleOutput with combined edge map and all intermediate results.
        """
        start_time = time.time()

        intermediate = {"input": image}

        # 1. Detect ink lines (먹선) - outlines
        ink_edges = self.detect_ink_lines(image)
        intermediate["ink_lines"] = ink_edges

        # 2. Detect internal details - textures and patterns
        internal_edges = self.detect_internal_details(image)
        intermediate["internal_details"] = internal_edges

        # 3. Detect color boundaries - color transitions
        color_edges = self.detect_color_boundaries(image)
        intermediate["color_boundaries"] = color_edges

        # 4. Detect damchae boundaries (담채) - soft transitions
        damchae_edges = self.detect_damchae_boundary(image)
        intermediate["damchae_boundaries"] = damchae_edges

        # 5. Optional HED detection
        if self._has_hed and self.hed_detector is not None:
            hed_result = self.hed_detector.forward(image, **kwargs)
            hed_edges = hed_result.result
            intermediate["hed_edges"] = hed_edges
        else:
            hed_edges = None

        # 6. Weighted combination
        # Get weights from config
        ink_weight = self.config.ink_line_weight
        internal_weight = self.config.internal_detail_weight
        color_weight = self.config.color_boundary_weight
        damchae_weight = self.config.damchae_sensitivity

        # Normalize weights
        total_weight = ink_weight + internal_weight + color_weight + damchae_weight
        if self._has_hed and hed_edges is not None:
            total_weight += 1.0  # HED has weight 1.0

        ink_w = ink_weight / total_weight
        internal_w = internal_weight / total_weight
        color_w = color_weight / total_weight
        damchae_w = damchae_weight / total_weight
        hed_w = 1.0 / total_weight if self._has_hed and hed_edges is not None else 0.0

        # Combine edge maps
        combined = (
            ink_w * ink_edges +
            internal_w * internal_edges +
            color_w * color_edges +
            damchae_w * damchae_edges
        )

        if self._has_hed and hed_edges is not None:
            combined = combined + hed_w * hed_edges

        # Normalize to [0, 1]
        combined = torch.clamp(combined, 0.0, 1.0)

        # Apply threshold for binary edges
        threshold = kwargs.get('threshold', 0.5)
        edges_binary = (combined > threshold).float()
        intermediate["edges_binary"] = edges_binary

        elapsed = time.time() - start_time

        return ModuleOutput(
            result=combined,
            intermediate=intermediate,
            metadata={
                "method": "korean_ink",
                "ink_line_weight": ink_weight,
                "internal_detail_weight": internal_weight,
                "color_boundary_weight": color_weight,
                "damchae_sensitivity": damchae_weight,
                "use_hed": self._has_hed,
                "weights": {
                    "ink": ink_w,
                    "internal": internal_w,
                    "color": color_w,
                    "damchae": damchae_w,
                    "hed": hed_w
                },
                "threshold": threshold,
                "processing_time": elapsed
            }
        )
