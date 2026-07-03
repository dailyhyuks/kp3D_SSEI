"""Real-ESRGAN implementation for super-resolution."""

import os
from pathlib import Path
from typing import Any, Optional
import time

import numpy as np
import torch
from loguru import logger
from torch import Tensor

try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    REALESRGAN_AVAILABLE = True
except ImportError:
    REALESRGAN_AVAILABLE = False
    logger.warning(
        "Real-ESRGAN not available. Install with: pip install realesrgan basicsr"
    )

from kp3d.core.base import ModuleOutput
from kp3d.core.registry import register_module
from kp3d.modules.superres.base import BaseSuperResolution, ScaleFactor, SuperResConfig


@register_module("real_esrgan")
class RealESRGANModule(BaseSuperResolution):
    """Real-ESRGAN based super-resolution module.

    Provides high-quality image upscaling using the Real-ESRGAN model,
    with optimizations for processing traditional Korean paintings.

    Attributes:
        upsampler: The Real-ESRGAN upsampler instance.
        MODEL_PATHS: Dictionary mapping model names to download URLs.
    """

    MODEL_PATHS = {
        "RealESRGAN_x2plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "RealESRGAN_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "RealESRGAN_x4plus_anime_6B": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    }

    MODEL_CONFIGS = {
        "RealESRGAN_x2plus": {"num_block": 23, "num_feat": 64, "scale": 2},
        "RealESRGAN_x4plus": {"num_block": 23, "num_feat": 64, "scale": 4},
        "RealESRGAN_x4plus_anime_6B": {"num_block": 6, "num_feat": 64, "scale": 4},
    }

    def __init__(
        self,
        config: Optional[SuperResConfig] = None,
        device: Optional[torch.device] = None,
        half_precision: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize Real-ESRGAN module.

        Args:
            config: Super-resolution configuration.
            device: Computation device.
            half_precision: Use FP16 for faster inference.
            **kwargs: Additional configuration parameters.

        Raises:
            ImportError: If Real-ESRGAN is not installed.
        """
        if not REALESRGAN_AVAILABLE:
            raise ImportError(
                "Real-ESRGAN is not installed. "
                "Install with: pip install realesrgan basicsr"
            )

        super().__init__(config=config, device=device, **kwargs)

        self.half_precision = half_precision
        self.upsampler: Optional[RealESRGANer] = None

        # Initialize model
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the Real-ESRGAN model.

        Downloads weights if necessary and creates the upsampler.
        """
        model_name = self.config.model_name

        if model_name not in self.MODEL_CONFIGS:
            logger.warning(
                f"Unknown model '{model_name}', falling back to RealESRGAN_x4plus"
            )
            model_name = "RealESRGAN_x4plus"
            self.config.model_name = model_name

        # Get model configuration
        model_cfg = self.MODEL_CONFIGS[model_name]

        # Create model architecture
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=model_cfg["num_feat"],
            num_block=model_cfg["num_block"],
            num_grow_ch=32,
            scale=model_cfg["scale"],
        )

        # Set up model path
        model_dir = Path.home() / ".cache" / "kp3d" / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{model_name}.pth"

        # Download model if not exists
        if not model_path.exists():
            logger.info(f"Downloading {model_name} weights...")
            self._download_model(model_name, model_path)

        # Create upsampler
        try:
            # Determine device string for RealESRGANer
            device_str = "cuda" if self.device.type == "cuda" else "cpu"

            self.upsampler = RealESRGANer(
                scale=model_cfg["scale"],
                model_path=str(model_path),
                model=model,
                tile=self.config.tile_size,
                tile_pad=self.config.tile_overlap,
                pre_pad=0,
                half=self.half_precision and device_str == "cuda",
                device=device_str,
            )

            self._initialized = True
            logger.info(f"Real-ESRGAN model '{model_name}' initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Real-ESRGAN: {e}")
            raise RuntimeError(f"Model initialization failed: {e}") from e

    def _download_model(self, model_name: str, save_path: Path) -> None:
        """Download model weights from GitHub releases.

        Args:
            model_name: Name of the model to download.
            save_path: Path to save the downloaded weights.

        Raises:
            RuntimeError: If download fails.
        """
        try:
            import urllib.request
            from tqdm import tqdm

            url = self.MODEL_PATHS[model_name]

            # Download with progress bar
            class DownloadProgressBar(tqdm):
                def update_to(self, b=1, bsize=1, tsize=None):
                    if tsize is not None:
                        self.total = tsize
                    self.update(b * bsize - self.n)

            with DownloadProgressBar(
                unit='B',
                unit_scale=True,
                miniters=1,
                desc=model_name
            ) as t:
                urllib.request.urlretrieve(
                    url,
                    filename=str(save_path),
                    reporthook=t.update_to
                )

            logger.info(f"Downloaded {model_name} to {save_path}")

        except Exception as e:
            if save_path.exists():
                save_path.unlink()
            raise RuntimeError(f"Failed to download model: {e}") from e

    def load_weights(self, checkpoint_path: str) -> None:
        """Load pretrained weights from a checkpoint file.

        Args:
            checkpoint_path: Path to the checkpoint file.

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist.
            RuntimeError: If checkpoint is incompatible.
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        try:
            # Re-initialize with custom checkpoint
            model_name = self.config.model_name
            model_cfg = self.MODEL_CONFIGS.get(
                model_name,
                self.MODEL_CONFIGS["RealESRGAN_x4plus"]
            )

            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=model_cfg["num_feat"],
                num_block=model_cfg["num_block"],
                num_grow_ch=32,
                scale=model_cfg["scale"],
            )

            device_str = "cuda" if self.device.type == "cuda" else "cpu"

            self.upsampler = RealESRGANer(
                scale=model_cfg["scale"],
                model_path=str(checkpoint_path),
                model=model,
                tile=self.config.tile_size,
                tile_pad=self.config.tile_overlap,
                pre_pad=0,
                half=self.half_precision and device_str == "cuda",
                device=device_str,
            )

            self._initialized = True
            logger.info(f"Loaded weights from {checkpoint_path}")

        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}") from e

    def forward(
        self,
        image: Tensor,
        scale: Optional[ScaleFactor] = None,
        denoise: bool = True,
        **kwargs: Any,
    ) -> ModuleOutput:
        """Upscale an image using Real-ESRGAN.

        Args:
            image: Input image tensor (B, C, H, W) with values in [0, 1].
            scale: Optional override for upscaling factor.
            denoise: Whether to apply denoising.
            **kwargs: Additional parameters.

        Returns:
            ModuleOutput containing the upscaled image and metadata.

        Raises:
            RuntimeError: If model is not initialized.
        """
        if not self._initialized or self.upsampler is None:
            raise RuntimeError("Model not initialized. Call load_weights() first.")

        start_time = time.time()

        # Store intermediate results
        intermediate = {}

        # Convert tensor to numpy (B, C, H, W) -> (H, W, C)
        # Real-ESRGAN expects BGR uint8
        batch_size = image.shape[0]
        results = []

        for i in range(batch_size):
            img = image[i]  # (C, H, W)

            # Convert to numpy HWC format
            img_np = img.permute(1, 2, 0).cpu().numpy()  # (H, W, C)

            # Clip to [0, 1] and convert to uint8 [0, 255]
            img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)

            # Convert RGB to BGR for OpenCV compatibility
            img_bgr = img_np[:, :, ::-1].copy()

            # Store original for comparison
            if i == 0:
                intermediate["input"] = img

            # Apply denoising if requested
            if denoise and self.config.denoise_strength > 0:
                denoised_img = self._denoise(
                    img.unsqueeze(0),
                    self.config.denoise_strength
                ).squeeze(0)
                if i == 0:
                    intermediate["denoised"] = denoised_img
            else:
                denoised_img = img

            # Upscale using Real-ESRGAN
            try:
                output_bgr, _ = self.upsampler.enhance(
                    img_bgr,
                    outscale=scale.value if scale else self.scale
                )
            except Exception as e:
                logger.error(f"Real-ESRGAN enhancement failed: {e}")
                # Fallback to bicubic upsampling
                logger.warning("Falling back to bicubic interpolation")
                scale_factor = scale.value if scale else self.scale
                output_tensor = torch.nn.functional.interpolate(
                    img.unsqueeze(0),
                    scale_factor=scale_factor,
                    mode="bicubic",
                    align_corners=False,
                ).squeeze(0)
                results.append(output_tensor)
                continue

            # Convert back to RGB tensor
            output_rgb = output_bgr[:, :, ::-1].copy()
            output_tensor = torch.from_numpy(output_rgb).float() / 255.0
            output_tensor = output_tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
            output_tensor = output_tensor.to(device=self.device, dtype=self.dtype)

            if i == 0:
                intermediate["upscaled_raw"] = output_tensor

            results.append(output_tensor)

        # Stack results back into batch
        result = torch.stack(results, dim=0)

        # Calculate processing time
        processing_time = time.time() - start_time

        # Build metadata
        metadata = {
            "scale": scale.value if scale else self.scale,
            "model": self.config.model_name,
            "denoise_strength": self.config.denoise_strength if denoise else 0.0,
            "original_size": (image.shape[2], image.shape[3]),
            "output_size": (result.shape[2], result.shape[3]),
            "processing_time": processing_time,
            "tile_size": self.config.tile_size,
            "device": str(self.device),
        }

        return ModuleOutput(
            result=result,
            intermediate=intermediate,
            metadata=metadata,
        )

    @property
    def name(self) -> str:
        """Return the module's unique identifier name."""
        return "real_esrgan"
