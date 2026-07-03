"""PowerPaint v2.1 (ECCV 2024) inpainting baseline.

Uses PowerPaint with BrushNet architecture from external/PowerPaint.
Follows the exact loading procedure from PowerPaint's app.py.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

from kp3d.evaluation.baselines import register_inpainting_baseline
from kp3d.evaluation.baselines.base import BaseInpaintingBaseline

_PP_ROOT = Path(__file__).resolve().parents[4] / "external" / "PowerPaint"


@register_inpainting_baseline
class PowerPaintBaseline(BaseInpaintingBaseline):
    """PowerPaint v2.1 (ECCV 2024) inpainting baseline.

    Reference: Zhuang et al., "PowerPaint: High-Quality Versatile
    Image Inpainting", ECCV 2024.
    """

    name = "powerpaint"

    def __init__(self):
        self._pipe = None
        self._available = None

    @property
    def available(self) -> bool:
        if self._available is None:
            try:
                ckpt = _PP_ROOT / "checkpoints" / "ppt-v2"
                base_model = ckpt / "realisticVisionV60B1_v51VAE"
                brushnet_ckpt = ckpt / "PowerPaint_Brushnet"
                self._available = (
                    base_model.exists()
                    and brushnet_ckpt.exists()
                    and (brushnet_ckpt / "diffusion_pytorch_model.safetensors").exists()
                )
                if self._available:
                    import diffusers  # noqa: F401
            except ImportError:
                self._available = False
        return self._available

    def _get_pipe(self):
        if self._pipe is not None:
            return self._pipe

        import torch
        from safetensors.torch import load_model
        from transformers import CLIPTextModel

        # Add PowerPaint to path for its custom models
        if str(_PP_ROOT) not in sys.path:
            sys.path.insert(0, str(_PP_ROOT))

        from powerpaint.models.BrushNet_CA import BrushNetModel
        from powerpaint.models.unet_2d_condition import UNet2DConditionModel
        from powerpaint.pipelines.pipeline_PowerPaint_Brushnet_CA import (
            StableDiffusionPowerPaintBrushNetPipeline,
        )
        from powerpaint.utils.utils import TokenizerWrapper, add_tokens
        from diffusers import UniPCMultistepScheduler

        ckpt = _PP_ROOT / "checkpoints" / "ppt-v2"
        base_model_path = str(ckpt / "realisticVisionV60B1_v51VAE")
        brushnet_ckpt_dir = str(ckpt / "PowerPaint_Brushnet")

        weight_dtype = torch.float16

        # Step 1: Create BrushNet from UNet structure
        unet = UNet2DConditionModel.from_pretrained(
            base_model_path,
            subfolder="unet",
            revision=None,
            torch_dtype=weight_dtype,
        )
        text_encoder_brushnet = CLIPTextModel.from_pretrained(
            base_model_path,
            subfolder="text_encoder",
            revision=None,
            torch_dtype=weight_dtype,
        )
        brushnet = BrushNetModel.from_unet(unet)

        # Step 2: Create pipeline
        pipe = StableDiffusionPowerPaintBrushNetPipeline.from_pretrained(
            base_model_path,
            brushnet=brushnet,
            text_encoder_brushnet=text_encoder_brushnet,
            torch_dtype=weight_dtype,
            low_cpu_mem_usage=False,
            safety_checker=None,
        )

        # Step 3: Load UNet from base model
        pipe.unet = UNet2DConditionModel.from_pretrained(
            base_model_path,
            subfolder="unet",
            revision=None,
            torch_dtype=weight_dtype,
        )

        # Step 4: Setup tokenizer with learned task tokens
        pipe.tokenizer = TokenizerWrapper(
            from_pretrained=base_model_path,
            subfolder="tokenizer",
            revision=None,
            torch_type=weight_dtype,
        )
        add_tokens(
            tokenizer=pipe.tokenizer,
            text_encoder=pipe.text_encoder_brushnet,
            placeholder_tokens=["P_ctxt", "P_shape", "P_obj"],
            initialize_tokens=["a", "a", "a"],
            num_vectors_per_token=10,
        )

        # Step 5: Load PowerPaint BrushNet weights
        load_model(
            pipe.brushnet,
            str(Path(brushnet_ckpt_dir) / "diffusion_pytorch_model.safetensors"),
        )

        # Step 6: Load text encoder brushnet weights
        pipe.text_encoder_brushnet.load_state_dict(
            torch.load(
                str(Path(brushnet_ckpt_dir) / "pytorch_model.bin"),
                map_location="cpu",
            ),
            strict=False,
        )

        # Step 7: Setup scheduler and offloading
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()

        self._pipe = pipe
        return self._pipe

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if not self.available:
            raise RuntimeError("PowerPaint not available. Check external/PowerPaint.")

        import torch
        from PIL import Image

        pipe = self._get_pipe()

        h, w = image.shape[:2]
        mask_float = (mask > 127).astype(np.float64)

        # Prepare masked image (zero out mask area)
        masked_image = image * (1 - mask_float[:, :, np.newaxis])

        # Resize to 512x512
        masked_resized = cv2.resize(
            masked_image.astype(np.uint8), (512, 512), interpolation=cv2.INTER_LANCZOS4
        )
        mask_resized = cv2.resize(
            mask_float, (512, 512), interpolation=cv2.INTER_NEAREST
        )

        pil_image = Image.fromarray(masked_resized).convert("RGB")
        pil_mask = Image.fromarray(
            (mask_resized[:, :, np.newaxis].repeat(3, axis=2) * 255).astype(np.uint8)
        ).convert("RGB")

        generator = torch.Generator("cuda").manual_seed(42)

        # PowerPaint uses object-removal control type:
        # promptA = " P_ctxt", promptB = " P_ctxt"
        # negative_promptA = " P_obj", negative_promptB = " P_obj"
        result = pipe(
            promptA="P_ctxt",
            promptB="P_ctxt",
            promptU="",
            negative_promptA="P_obj",
            negative_promptB="P_obj",
            negative_promptU="worst quality, low quality",
            image=pil_image,
            mask=pil_mask,
            num_inference_steps=30,
            generator=generator,
            brushnet_conditioning_scale=1.0,
            width=512,
            height=512,
        ).images[0]

        # Resize back and blend
        result_np = np.array(result.resize((w, h), Image.LANCZOS))

        output = image.copy()
        output[mask > 127] = result_np[mask > 127]

        return output
