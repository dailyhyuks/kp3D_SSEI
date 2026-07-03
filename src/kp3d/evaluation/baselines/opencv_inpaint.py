"""OpenCV-based inpainting baselines (Telea, Navier-Stokes)."""

import cv2
import numpy as np

from kp3d.evaluation.baselines import register_inpainting_baseline
from kp3d.evaluation.baselines.base import BaseInpaintingBaseline


@register_inpainting_baseline
class OpenCVTeleaBaseline(BaseInpaintingBaseline):
    """OpenCV Telea (Fast Marching Method) inpainting.

    Reference: A. Telea, "An Image Inpainting Technique Based on the
    Fast Marching Method", JGTOOLS 2004.
    """

    name = "opencv_telea"

    def __init__(self, radius: int = 3):
        self.radius = radius

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        # OpenCV inpaint expects single-channel mask
        mask_gray = mask if mask.ndim == 2 else cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask_binary = (mask_gray > 127).astype(np.uint8)

        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        result_bgr = cv2.inpaint(img_bgr, mask_binary, self.radius, cv2.INPAINT_TELEA)
        return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)


@register_inpainting_baseline
class OpenCVNSBaseline(BaseInpaintingBaseline):
    """OpenCV Navier-Stokes inpainting.

    Reference: M. Bertalmio et al., "Navier-Stokes, Fluid Dynamics,
    and Image and Video Inpainting", CVPR 2001.
    """

    name = "opencv_ns"

    def __init__(self, radius: int = 3):
        self.radius = radius

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        mask_gray = mask if mask.ndim == 2 else cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask_binary = (mask_gray > 127).astype(np.uint8)

        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        result_bgr = cv2.inpaint(img_bgr, mask_binary, self.radius, cv2.INPAINT_NS)
        return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
