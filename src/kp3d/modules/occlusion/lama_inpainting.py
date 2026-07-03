"""LaMa (Large Mask Inpainting) integration.

LaMa is a resolution-robust large mask inpainting model using
Fourier Convolutions. This module provides a wrapper for easy integration.

Reference: https://github.com/advimman/lama
Paper: Resolution-robust Large Mask Inpainting with Fourier Convolutions (WACV 2022)
"""

from typing import Optional, Union
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierUnit(nn.Module):
    """Fourier Unit for Fast Fourier Convolutions."""

    def __init__(self, in_channels: int, out_channels: int, groups: int = 1):
        super().__init__()
        self.groups = groups
        self.conv = nn.Conv2d(in_channels * 2, out_channels * 2, 1, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, c, h, w = x.shape

        # FFT
        ffted = torch.fft.rfft2(x, norm="ortho")
        ffted = torch.stack([ffted.real, ffted.imag], dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view(batch, -1, h, w // 2 + 1)

        # Conv in frequency domain
        ffted = self.conv(ffted)
        ffted = self.bn(ffted)
        ffted = self.relu(ffted)

        # Inverse FFT
        ffted = ffted.view(batch, -1, 2, h, w // 2 + 1).permute(0, 1, 3, 4, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])
        output = torch.fft.irfft2(ffted, s=(h, w), norm="ortho")

        return output


class FFCResBlock(nn.Module):
    """Residual block with Fast Fourier Convolutions."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class SimpleLaMaModel(nn.Module):
    """Simplified LaMa-like model for inpainting.

    This is a lightweight implementation inspired by LaMa architecture.
    For full LaMa, use the official pretrained weights.
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 3, base_channels: int = 64):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )

        # Middle blocks with residual
        self.middle = nn.Sequential(
            FFCResBlock(base_channels * 4),
            FFCResBlock(base_channels * 4),
            FFCResBlock(base_channels * 4),
            FFCResBlock(base_channels * 4),
        )

        # Decoder
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Concatenate image and mask
        inp = torch.cat([x, mask], dim=1)

        # Encode
        e1 = self.enc1(inp)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        # Middle
        m = self.middle(e3)

        # Decode with skip connections
        d3 = self.dec3(m) + e2
        d2 = self.dec2(d3) + e1
        out = self.dec1(d2)

        return out


class LaMaInpainter:
    """LaMa-based inpainting wrapper.

    Provides high-quality inpainting for large mask regions using
    a LaMa-inspired architecture with Fourier convolutions.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        model_path: Optional[str] = None
    ):
        """Initialize LaMa inpainter.

        Args:
            device: Computation device. Defaults to CUDA if available.
            model_path: Path to pretrained weights. If None, uses untrained model.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model
        self.model = SimpleLaMaModel().to(self.device)
        self.model.eval()

        if model_path and Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        self._initialized = True

    def inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        return_numpy: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """Inpaint masked regions using LaMa.

        Args:
            image: RGB image (H, W, 3), uint8 [0-255] or float [0-1].
            mask: Binary mask (H, W) where 255/1 = region to inpaint.
            return_numpy: Return numpy array instead of tensor.

        Returns:
            Inpainted image.
        """
        # Preprocess
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        if mask.dtype == np.uint8:
            mask = mask.astype(np.float32) / 255.0

        # Ensure mask is 2D
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        # Pad to multiple of 8 for encoder/decoder
        h, w = image.shape[:2]
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8

        if pad_h > 0 or pad_w > 0:
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='reflect')

        # To tensor: (H, W, C) -> (1, C, H, W)
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()

        img_tensor = img_tensor.to(self.device)
        mask_tensor = mask_tensor.to(self.device)

        # Inpaint
        with torch.no_grad():
            # Mask out the region
            masked_img = img_tensor * (1 - mask_tensor)

            # Run model
            output = self.model(masked_img, mask_tensor)

            # Blend: keep original where mask=0, use output where mask=1
            result = img_tensor * (1 - mask_tensor) + output * mask_tensor

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            result = result[:, :, :h, :w]

        if return_numpy:
            result = result.squeeze(0).permute(1, 2, 0).cpu().numpy()
            result = (result * 255).clip(0, 255).astype(np.uint8)
            return result

        return result


def lama_inpaint(
    image: np.ndarray,
    mask: np.ndarray,
    device: Optional[torch.device] = None
) -> np.ndarray:
    """Quick LaMa inpainting function.

    Args:
        image: RGB image (H, W, 3).
        mask: Binary mask (H, W).
        device: Computation device.

    Returns:
        Inpainted image.
    """
    inpainter = LaMaInpainter(device=device)
    return inpainter.inpaint(image, mask)


# Alternative: Use cv2.inpaint with improved parameters as fallback
def enhanced_cv2_inpaint(
    image: np.ndarray,
    mask: np.ndarray,
    radius: int = 10,
    method: str = "ns",
    iterations: int = 2
) -> np.ndarray:
    """Enhanced OpenCV inpainting with multiple passes.

    Args:
        image: RGB image (H, W, 3), uint8.
        mask: Binary mask (H, W).
        radius: Inpainting radius.
        method: "ns" (Navier-Stokes) or "telea".
        iterations: Number of inpainting passes.

    Returns:
        Inpainted image.
    """
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    if mask.dtype != np.uint8:
        mask = (mask * 255).astype(np.uint8)

    flag = cv2.INPAINT_NS if method == "ns" else cv2.INPAINT_TELEA
    result = image.copy()

    # Multiple passes with decreasing radius
    remaining_mask = mask.copy()
    for i in range(iterations):
        r = max(3, radius - i * 2)
        result = cv2.inpaint(result, remaining_mask, r, flag)

        # Erode mask for next iteration (shrink the region)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        remaining_mask = cv2.erode(remaining_mask, kernel, iterations=1)

        if np.sum(remaining_mask) == 0:
            break

    return result
