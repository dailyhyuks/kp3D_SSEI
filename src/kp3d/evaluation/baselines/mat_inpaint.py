"""MAT (Mask-Aware Transformer) inpainting baseline.

Uses MATInpainter from kp3d.modules.occlusion.inpainting.
Requires IOPaint or direct MAT model file.
"""

import numpy as np

from kp3d.evaluation.baselines import register_inpainting_baseline
from kp3d.evaluation.baselines.base import BaseInpaintingBaseline


@register_inpainting_baseline
class MATBaseline(BaseInpaintingBaseline):
    """MAT (CVPR 2022) inpainting baseline.

    Reference: Li et al., "MAT: Mask-Aware Transformer for Large Hole
    Image Inpainting", CVPR 2022.
    """

    name = "mat"

    def __init__(self):
        self._inpainter = None
        self._available = None

    @property
    def available(self) -> bool:
        if self._available is None:
            try:
                from kp3d.modules.occlusion.inpainting import MATInpainter
                # Check if IOPaint is available or model can be loaded
                try:
                    from iopaint.model.mat import MAT  # noqa: F401
                    self._available = True
                except ImportError:
                    # Check for local model file
                    from pathlib import Path
                    cache_dir = Path.home() / ".cache" / "mat"
                    model_file = cache_dir / "MAT_Places512_G_fp16.pkl"
                    self._available = model_file.exists()
            except ImportError:
                self._available = False
        return self._available

    def _get_inpainter(self):
        if self._inpainter is None:
            from kp3d.modules.occlusion.inpainting import MATInpainter
            self._inpainter = MATInpainter(device="cuda")
        return self._inpainter

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if not self.available:
            raise RuntimeError("MAT baseline requires IOPaint or MAT model.")

        inpainter = self._get_inpainter()
        mask_binary = (mask > 127).astype(np.uint8) * 255

        # MATInpainter.inpaint() accepts RGB numpy + mask
        result = inpainter.inpaint(image, mask_binary)
        return result
