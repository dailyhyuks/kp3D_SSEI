"""OpenCV-based enhancement/restoration baselines.

Traditional filtering methods for comparison with Spectral Interpolation.
These are general-purpose denoising filters, not specifically designed for
periodic grid removal — demonstrating the advantage of our frequency-domain approach.
"""

import cv2
import numpy as np

from kp3d.evaluation.baselines import register_enhancement_baseline
from kp3d.evaluation.baselines.base import BaseEnhancementBaseline


@register_enhancement_baseline
class BilateralBaseline(BaseEnhancementBaseline):
    """OpenCV bilateral filter baseline.

    Edge-preserving smoothing. Good for general denoising but not
    designed for periodic artifact removal.
    """

    name = "bilateral"

    def __init__(self, d: int = 9, sigma_color: float = 75, sigma_space: float = 75):
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def process(self, image_bgr: np.ndarray) -> np.ndarray:
        return cv2.bilateralFilter(
            image_bgr, self.d, self.sigma_color, self.sigma_space
        )


@register_enhancement_baseline
class NLMeansBaseline(BaseEnhancementBaseline):
    """OpenCV Non-Local Means denoising baseline.

    Patch-based denoising exploiting image self-similarity.
    """

    name = "nlmeans"

    def __init__(self, h: float = 10, h_color: float = 10,
                 template_window: int = 7, search_window: int = 21):
        self.h = h
        self.h_color = h_color
        self.template_window = template_window
        self.search_window = search_window

    def process(self, image_bgr: np.ndarray) -> np.ndarray:
        return cv2.fastNlMeansDenoisingColored(
            image_bgr, None,
            self.h, self.h_color,
            self.template_window, self.search_window,
        )


@register_enhancement_baseline
class MedianBaseline(BaseEnhancementBaseline):
    """Median filter baseline.

    Simple nonlinear filter. Removes impulse noise but blurs edges.
    """

    name = "median"

    def __init__(self, ksize: int = 5):
        self.ksize = ksize

    def process(self, image_bgr: np.ndarray) -> np.ndarray:
        return cv2.medianBlur(image_bgr, self.ksize)


@register_enhancement_baseline
class GuidedBaseline(BaseEnhancementBaseline):
    """Guided filter baseline.

    Edge-aware filtering using guided image filtering.
    Requires opencv-contrib (ximgproc).
    """

    name = "guided"

    def __init__(self, radius: int = 8, eps: float = 0.01):
        self.radius = radius
        self.eps = eps

    def process(self, image_bgr: np.ndarray) -> np.ndarray:
        guide = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        try:
            return cv2.ximgproc.guidedFilter(guide, image_bgr, self.radius, self.eps)
        except AttributeError:
            # ximgproc not available, fall back to bilateral
            return cv2.bilateralFilter(image_bgr, 9, 75, 75)


@register_enhancement_baseline
class ButterworthBaseline(BaseEnhancementBaseline):
    """Butterworth band-reject filter in frequency domain.

    Designed specifically for periodic grid removal. Places notch
    filters at grid harmonic frequencies.
    """

    name = "butterworth"

    def __init__(
        self,
        period_x: int = 9,
        period_y: int = 7,
        order: int = 4,
        bandwidth: float = 3.0,
        n_harmonics: int = 5,
    ):
        self.period_x = period_x
        self.period_y = period_y
        self.order = order
        self.bandwidth = bandwidth
        self.n_harmonics = n_harmonics

    def process(self, image_bgr: np.ndarray) -> np.ndarray:
        result = np.zeros_like(image_bgr, dtype=np.float32)

        for c in range(3):
            channel = image_bgr[:, :, c].astype(np.float64)
            f = np.fft.fft2(channel)
            f_shift = np.fft.fftshift(f)

            h, w = channel.shape
            cy, cx = h // 2, w // 2

            # Create Butterworth notch reject filter
            H = np.ones((h, w), dtype=np.float64)

            for harmonic in range(1, self.n_harmonics + 1):
                # Horizontal harmonics
                fx = harmonic * w / self.period_x
                # Vertical harmonics
                fy = harmonic * h / self.period_y

                for (dx, dy) in [(fx, 0), (-fx, 0), (0, fy), (0, -fy)]:
                    center_x = cx + dx
                    center_y = cy + dy
                    Y, X = np.ogrid[:h, :w]
                    dist = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
                    # Butterworth notch
                    notch = 1.0 / (
                        1.0 + (self.bandwidth / (dist + 1e-10)) ** (2 * self.order)
                    )
                    H *= notch

            filtered = f_shift * H
            result[:, :, c] = np.real(
                np.fft.ifft2(np.fft.ifftshift(filtered))
            ).astype(np.float32)

        return np.clip(result, 0, 255).astype(np.uint8)
