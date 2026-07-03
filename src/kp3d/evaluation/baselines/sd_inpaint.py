"""Stable Diffusion Inpainting baseline.

Uses SDInpainter from kp3d.modules.occlusion.inpainting.
Requires diffusers package.
"""

import cv2
import numpy as np

from kp3d.evaluation.baselines import register_inpainting_baseline
from kp3d.evaluation.baselines.base import BaseInpaintingBaseline


@register_inpainting_baseline
class SDInpaintBaseline(BaseInpaintingBaseline):
    """Stable Diffusion Inpainting baseline.

    Reference: Rombach et al., "High-Resolution Image Synthesis with
    Latent Diffusion Models", CVPR 2022.
    Model: runwayml/stable-diffusion-inpainting
    """

    name = "sd_inpaint"

    def __init__(self):
        self._inpainter = None
        self._available = None

    @property
    def available(self) -> bool:
        if self._available is None:
            try:
                import diffusers  # noqa: F401
                self._available = True
            except ImportError:
                self._available = False
        return self._available

    def _get_inpainter(self):
        if self._inpainter is None:
            from kp3d.modules.occlusion.inpainting import SDInpainter
            self._inpainter = SDInpainter(device="cuda")
        return self._inpainter

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if not self.available:
            raise RuntimeError(
                "SD Inpaint requires diffusers. "
                "Install with: pip install diffusers transformers accelerate"
            )

        h, w = image.shape[:2]
        mask_binary = (mask > 127).astype(np.uint8) * 255

        # Resize to 512x512 for SD pipeline
        image_resized = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LANCZOS4)
        mask_resized = cv2.resize(mask_binary, (512, 512), interpolation=cv2.INTER_NEAREST)

        inpainter = self._get_inpainter()
        result_512 = inpainter.inpaint(
            image=image_resized,
            mask=mask_resized,
            prompt="traditional Korean painting, ink wash style, clean background",
            num_inference_steps=30,
            seed=42,
        )

        # Resize back to original size
        result = cv2.resize(result_512, (w, h), interpolation=cv2.INTER_LANCZOS4)

        # Paste original outside mask for fair comparison
        output = image.copy()
        output[mask > 127] = result[mask > 127]

        return output
