"""BrushNet (ECCV 2024) inpainting baseline.

Uses BrushNet from external/BrushNet with segmentation mask checkpoint.
Requires diffusers and BrushNet's custom pipeline classes.
"""

import sys
from pathlib import Path

import numpy as np

from kp3d.evaluation.baselines import register_inpainting_baseline
from kp3d.evaluation.baselines.base import BaseInpaintingBaseline

# Add BrushNet's src to path for its custom diffusers extensions
_BRUSHNET_ROOT = Path(__file__).resolve().parents[4] / "external" / "BrushNet"
_BRUSHNET_SRC = _BRUSHNET_ROOT / "src"


@register_inpainting_baseline
class BrushNetBaseline(BaseInpaintingBaseline):
    """BrushNet (ECCV 2024) inpainting baseline.

    Reference: Xie et al., "BrushNet: A Plug-and-Play Image Inpainting
    Model with Decomposed Dual-Branch Diffusion", ECCV 2024.
    """

    name = "brushnet"

    def __init__(self):
        self._pipe = None
        self._available = None

    @property
    def available(self) -> bool:
        if self._available is None:
            try:
                # Check that BrushNet directory and checkpoint exist
                ckpt = _BRUSHNET_ROOT / "data" / "ckpt" / "segmentation_mask_brushnet_ckpt"
                if not ckpt.exists():
                    self._available = False
                else:
                    import diffusers  # noqa: F401
                    self._available = True
            except ImportError:
                self._available = False
        return self._available

    def _get_pipe(self):
        if self._pipe is not None:
            return self._pipe

        import torch

        # Add BrushNet src to path for custom pipeline/model classes
        if str(_BRUSHNET_SRC) not in sys.path:
            sys.path.insert(0, str(_BRUSHNET_SRC))

        from diffusers import StableDiffusionBrushNetPipeline, BrushNetModel, UniPCMultistepScheduler

        brushnet_path = str(
            _BRUSHNET_ROOT / "data" / "ckpt" / "segmentation_mask_brushnet_ckpt"
        )

        # Check for base SD model; fall back to HF hub
        base_model_path = _BRUSHNET_ROOT / "data" / "ckpt" / "realisticVisionV60B1_v51VAE"
        if not base_model_path.exists():
            base_model_path = "runwayml/stable-diffusion-v1-5"
        else:
            base_model_path = str(base_model_path)

        brushnet = BrushNetModel.from_pretrained(
            brushnet_path, torch_dtype=torch.float16
        )
        pipe = StableDiffusionBrushNetPipeline.from_pretrained(
            base_model_path,
            brushnet=brushnet,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=False,
            safety_checker=None,
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()

        self._pipe = pipe
        return self._pipe

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if not self.available:
            raise RuntimeError("BrushNet not available. Check external/BrushNet.")

        import cv2
        import torch
        from PIL import Image

        pipe = self._get_pipe()

        h, w = image.shape[:2]

        # BrushNet expects: masked image (zeros in mask area) + mask (3ch)
        mask_binary = (mask > 127).astype(np.float64)
        masked_image = image * (1 - mask_binary[:, :, np.newaxis])

        pil_image = Image.fromarray(masked_image.astype(np.uint8)).convert("RGB")
        pil_mask = Image.fromarray(
            (mask_binary[:, :, np.newaxis].repeat(3, axis=2) * 255).astype(np.uint8)
        ).convert("RGB")

        # Resize to 512x512 for SD pipeline
        pil_image_resized = pil_image.resize((512, 512), Image.LANCZOS)
        pil_mask_resized = pil_mask.resize((512, 512), Image.NEAREST)

        generator = torch.Generator("cuda").manual_seed(42)

        result = pipe(
            "traditional Korean painting, ink wash style, clean background",
            pil_image_resized,
            pil_mask_resized,
            num_inference_steps=30,
            generator=generator,
            brushnet_conditioning_scale=1.0,
        ).images[0]

        # Resize back and blend
        result_np = np.array(result.resize((w, h), Image.LANCZOS))

        # Paste original outside mask
        output = image.copy()
        output[mask > 127] = result_np[mask > 127]

        return output
