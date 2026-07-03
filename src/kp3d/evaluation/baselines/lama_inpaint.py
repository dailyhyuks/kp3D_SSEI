"""LaMa (Large Mask Inpainting) baseline.

Uses the LaMa ONNX model via the existing LamaInpainter in
kp3d.modules.occlusion.inpainting. Gracefully skips if
dependencies (onnxruntime, huggingface_hub) are not installed.
"""

import numpy as np

from kp3d.evaluation.baselines import register_inpainting_baseline
from kp3d.evaluation.baselines.base import BaseInpaintingBaseline


@register_inpainting_baseline
class LaMaBaseline(BaseInpaintingBaseline):
    """LaMa deep learning inpainting baseline.

    Reference: Suvorov et al., "Resolution-robust Large Mask
    Inpainting with Fourier Convolutions", WACV 2022.

    Uses the existing LamaInpainter from kp3d.modules.occlusion.inpainting.
    """

    name = "lama"

    def __init__(self):
        self._inpainter = None
        self._available = None

    @property
    def available(self) -> bool:
        """Check if LaMa dependencies are installed."""
        if self._available is None:
            try:
                import onnxruntime  # noqa: F401
                self._available = True
            except ImportError:
                self._available = False
        return self._available

    def _get_inpainter(self):
        if self._inpainter is None:
            from kp3d.modules.occlusion.inpainting import LamaInpainter
            self._inpainter = LamaInpainter()
        return self._inpainter

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Inpaint using LaMa model.

        Args:
            image: Input RGB image (H, W, 3), uint8.
            mask: Binary mask (H, W), uint8, 255=inpaint region.

        Returns:
            Inpainted RGB image (H, W, 3), uint8.

        Raises:
            RuntimeError: If LaMa dependencies are not available.
        """
        if not self.available:
            raise RuntimeError(
                "LaMa baseline requires onnxruntime. "
                "Install with: pip install onnxruntime-gpu"
            )

        inpainter = self._get_inpainter()

        # LamaInpainter expects RGB input
        mask_binary = (mask > 127).astype(np.uint8) * 255

        # Use LamaInpainter.inpaint() which handles pre/post processing
        result = inpainter.inpaint(image, mask_binary)

        # Paste only mask region for fair comparison
        output = image.copy()
        output[mask > 127] = result[mask > 127]

        return output
