"""Neural residual refinement for grid restoration (v13).

Provides SCUNet-based artifact cleanup after multiplicative deconvolution.
Falls back to bilateral filter if SCUNet is unavailable.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Model download URLs
MODEL_URLS = {
    "scunet": "https://github.com/cszn/SCUNet/releases/download/v0.0/scunet_color_real_psnr.pth",
}

# Cache directory for model weights
CACHE_DIR = Path.home() / ".cache" / "kp3d" / "models"


class NeuralResidualRefiner:
    """SCUNet-based residual artifact cleanup for grid restoration.

    Applies neural network refinement to clean up residual artifacts
    after multiplicative grid deconvolution. Falls back to bilateral
    filtering if the neural model is unavailable.

    Attributes:
        model_name: Name of the model to use ("scunet").
        device: Torch device for inference.
    """

    def __init__(
        self,
        model_name: str = "scunet",
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize the neural residual refiner.

        Args:
            model_name: Model to use. Currently only "scunet" is supported.
            device: Torch device for inference. Auto-detects if None.
        """
        self.model_name = model_name
        self._model: Optional[nn.Module] = None
        self._use_bilateral = False

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        logger.debug(
            f"NeuralResidualRefiner initialized with model={model_name}, device={self.device}"
        )

    def _download_model(self, url: str, save_path: Path) -> None:
        """Download model weights from URL with progress bar.

        Args:
            url: URL to download from.
            save_path: Local path to save the weights.

        Raises:
            RuntimeError: If download fails.
        """
        import urllib.request

        logger.info(f"Downloading {save_path.name} from {url}")

        try:
            from tqdm import tqdm

            class DownloadProgressBar(tqdm):
                def update_to(self, b=1, bsize=1, tsize=None):
                    if tsize is not None:
                        self.total = tsize
                    self.update(b * bsize - self.n)

            with DownloadProgressBar(
                unit="B", unit_scale=True, miniters=1, desc=save_path.name
            ) as t:
                urllib.request.urlretrieve(
                    url, filename=str(save_path), reporthook=t.update_to
                )
        except ImportError:
            # No tqdm available, download without progress bar
            logger.info("tqdm not available, downloading without progress bar...")
            urllib.request.urlretrieve(url, filename=str(save_path))

        logger.info(f"Downloaded {save_path.name} successfully")

    def _build_scunet_model(self) -> nn.Module:
        """Build or load SCUNet model architecture.

        Attempts to load SCUNet from the scunet package. If unavailable,
        sets the bilateral fallback flag.

        Returns:
            The SCUNet model instance.

        Raises:
            ImportError: If SCUNet is not available.
        """
        # Try importing from scunet package
        try:
            from scunet.models.network_scunet import SCUNet as RealSCUNet

            logger.debug("Using SCUNet from scunet package")
            # SCUNet for color images: in_nc=3, config=[4,4,4,4,4,4,4], dim=64
            model = RealSCUNet(in_nc=3, config=[4, 4, 4, 4, 4, 4, 4], dim=64)
            return model
        except ImportError:
            logger.debug("scunet package not available, trying torch hub")

        # Try loading from torch hub
        try:
            model = torch.hub.load(
                "cszn/SCUNet",
                "scunet_color_real_psnr",
                pretrained=False,
                trust_repo=True,
            )
            logger.debug("Loaded SCUNet architecture from torch hub")
            return model
        except Exception as e:
            logger.debug(f"torch hub load failed: {e}")

        # All attempts failed, use bilateral fallback
        logger.warning(
            "SCUNet architecture not available. "
            "Install with: pip install git+https://github.com/cszn/SCUNet.git "
            "Falling back to bilateral filter."
        )
        raise ImportError("SCUNet not available")

    def _ensure_model(self) -> None:
        """Ensure the model is loaded and ready for inference.

        Downloads weights if needed and loads the model into memory.
        Sets _use_bilateral flag if model loading fails.
        """
        if self._model is not None:
            return

        if self._use_bilateral:
            # Already determined that bilateral fallback is needed
            return

        model_path = CACHE_DIR / "scunet_color_real_psnr.pth"

        try:
            # Download model weights if not cached
            if not model_path.exists():
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                self._download_model(MODEL_URLS["scunet"], model_path)

            # Build model architecture
            self._model = self._build_scunet_model()

            # Load weights
            state_dict = torch.load(
                model_path, map_location=self.device, weights_only=True
            )
            self._model.load_state_dict(state_dict)
            self._model.eval()
            self._model.to(self.device)

            logger.info(f"SCUNet model loaded successfully on {self.device}")

        except Exception as e:
            logger.warning(f"Failed to load SCUNet model: {e}. Using bilateral fallback.")
            self._use_bilateral = True
            self._model = None

    def _refine_with_scunet(self, image_bgr: np.ndarray) -> np.ndarray:
        """Run SCUNet inference on BGR image.

        Args:
            image_bgr: Input image in BGR format (H, W, 3), uint8.

        Returns:
            Refined image in BGR format (H, W, 3), uint8.
        """
        # BGR -> RGB -> float32 -> [0,1] -> NCHW tensor
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_t = torch.from_numpy(rgb.astype(np.float32) / 255.0)
        img_t = img_t.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        img_t = img_t.to(self.device)

        with torch.no_grad():
            output = self._model(img_t)

        # Back to numpy BGR
        out_np = output[0].cpu().clamp(0, 1).numpy()
        out_np = np.transpose(out_np, (1, 2, 0))  # CHW -> HWC
        out_bgr = cv2.cvtColor((out_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        return out_bgr

    def _refine_with_bilateral(
        self, image_bgr: np.ndarray, iterations: int = 3
    ) -> np.ndarray:
        """Apply iterative bilateral filtering as fallback refinement.

        Args:
            image_bgr: Input image in BGR format (H, W, 3), uint8.
            iterations: Number of bilateral filter iterations.

        Returns:
            Refined image in BGR format (H, W, 3), uint8.
        """
        result = image_bgr.copy()
        for _ in range(iterations):
            result = cv2.bilateralFilter(result, d=5, sigmaColor=30, sigmaSpace=30)
        return result

    def refine(self, image_bgr: np.ndarray, strength: float = 0.3) -> np.ndarray:
        """Apply neural refinement with strength-based blending.

        The result is blended with the original image based on strength:
        result = original * (1 - strength) + refined * strength

        Args:
            image_bgr: Input image in BGR format (H, W, 3), uint8.
            strength: Blending strength (0.0 = original, 1.0 = fully refined).
                      Default is 0.3 for subtle refinement.

        Returns:
            Refined and blended image in BGR format (H, W, 3), uint8.
        """
        # Clamp strength to valid range
        strength = max(0.0, min(1.0, strength))

        if strength == 0.0:
            return image_bgr.copy()

        try:
            self._ensure_model()

            if self._use_bilateral or self._model is None:
                # Use bilateral fallback
                refined = self._refine_with_bilateral(image_bgr)
                logger.debug("Refined using bilateral filter fallback")
            else:
                # Use SCUNet
                refined = self._refine_with_scunet(image_bgr)
                logger.debug("Refined using SCUNet neural model")

        except Exception as e:
            logger.warning(f"Neural refinement failed: {e}. Using bilateral fallback.")
            refined = self._refine_with_bilateral(image_bgr)

        # Blend with original based on strength
        if strength == 1.0:
            return refined

        result = (
            image_bgr.astype(np.float64) * (1 - strength)
            + refined.astype(np.float64) * strength
        )
        return np.clip(result, 0, 255).astype(np.uint8)


__all__ = ["NeuralResidualRefiner", "MODEL_URLS", "CACHE_DIR"]
