"""HED (Holistically-nested Edge Detection) implementation."""

import os
import time
from pathlib import Path
from typing import Any, Optional
from urllib.request import urlretrieve

import torch
import torch.nn as nn
from torch import Tensor

from kp3d.core.base import ModuleOutput
from kp3d.modules.edge.base import BaseEdgeDetection, EdgeConfig


class HEDNetwork(nn.Module):
    """HED Network Architecture.

    Based on VGG16 backbone with side outputs at multiple scales.
    Reference: https://github.com/sniklaus/pytorch-hed
    """

    def __init__(self):
        """Initialize HED network."""
        super().__init__()

        # VGG16 conv layers
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        # Side outputs
        self.side1 = nn.Conv2d(64, 1, kernel_size=1)
        self.side2 = nn.Conv2d(128, 1, kernel_size=1)
        self.side3 = nn.Conv2d(256, 1, kernel_size=1)
        self.side4 = nn.Conv2d(512, 1, kernel_size=1)
        self.side5 = nn.Conv2d(512, 1, kernel_size=1)

        # Fuse layer
        self.fuse = nn.Conv2d(5, 1, kernel_size=1)

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ReLU
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through HED network.

        Args:
            x: Input tensor (B, 3, H, W).

        Returns:
            Edge probability map (B, 1, H, W).
        """
        h, w = x.size()[2:]

        # Block 1
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        side1 = self.side1(x)
        x = self.pool(x)

        # Block 2
        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        side2 = self.side2(x)
        x = self.pool(x)

        # Block 3
        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.relu(self.conv3_3(x))
        side3 = self.side3(x)
        x = self.pool(x)

        # Block 4
        x = self.relu(self.conv4_1(x))
        x = self.relu(self.conv4_2(x))
        x = self.relu(self.conv4_3(x))
        side4 = self.side4(x)
        x = self.pool(x)

        # Block 5
        x = self.relu(self.conv5_1(x))
        x = self.relu(self.conv5_2(x))
        x = self.relu(self.conv5_3(x))
        side5 = self.side5(x)

        # Upsample side outputs to original size
        side1 = side1
        side2 = nn.functional.interpolate(side2, size=(h, w), mode='bilinear', align_corners=True)
        side3 = nn.functional.interpolate(side3, size=(h, w), mode='bilinear', align_corners=True)
        side4 = nn.functional.interpolate(side4, size=(h, w), mode='bilinear', align_corners=True)
        side5 = nn.functional.interpolate(side5, size=(h, w), mode='bilinear', align_corners=True)

        # Fuse all side outputs
        fuse = self.fuse(torch.cat([side1, side2, side3, side4, side5], dim=1))

        return torch.sigmoid(fuse)


class HEDEdgeDetector(BaseEdgeDetection):
    """HED-based edge detector.

    Uses deep learning for holistically-nested edge detection.
    Falls back to Canny if weights cannot be loaded.
    """

    HED_WEIGHTS_URL = "http://content.sniklaus.com/github/pytorch-hed/network-bsds500.pytorch"

    def __init__(
        self,
        config: Optional[EdgeConfig] = None,
        weights_path: Optional[str] = None,
        **kwargs
    ) -> None:
        """Initialize HED edge detector.

        Args:
            config: Edge detection configuration.
            weights_path: Path to HED weights file. If None, will try to download.
            **kwargs: Additional arguments.
        """
        super().__init__(config=config, **kwargs)
        self.weights_path = weights_path
        self.network = None
        self._fallback_to_canny = False

        # Try to initialize network
        self._init_network()

    @property
    def name(self) -> str:
        """Module name."""
        return "hed_edge" if not self._fallback_to_canny else "hed_edge_canny_fallback"

    def _get_weights_path(self) -> Path:
        """Get path to weights file, downloading if necessary.

        Returns:
            Path to weights file.
        """
        if self.weights_path:
            return Path(self.weights_path)

        # Default cache location
        cache_dir = Path.home() / ".cache" / "kp3d" / "hed"
        cache_dir.mkdir(parents=True, exist_ok=True)
        weights_file = cache_dir / "network-bsds500.pytorch"

        return weights_file

    def _download_weights(self, output_path: Path) -> bool:
        """Download HED weights from GitHub.

        Args:
            output_path: Where to save the weights.

        Returns:
            True if download successful, False otherwise.
        """
        try:
            print(f"Downloading HED weights to {output_path}...")
            urlretrieve(self.HED_WEIGHTS_URL, output_path)
            print("Download complete.")
            return True
        except Exception as e:
            print(f"Failed to download HED weights: {e}")
            return False

    def _init_network(self) -> None:
        """Initialize HED network and load weights."""
        try:
            # Create network
            self.network = HEDNetwork()
            self.network = self.network.to(self.device)
            self.network.eval()

            # Get weights path
            weights_file = self._get_weights_path()

            # Download if not exists
            if not weights_file.exists():
                if not self._download_weights(weights_file):
                    raise RuntimeError("Failed to download weights")

            # Load weights
            if weights_file.exists():
                self.load_weights(str(weights_file))
            else:
                raise FileNotFoundError(f"Weights not found: {weights_file}")

        except Exception as e:
            print(f"HED initialization failed: {e}")
            print("Falling back to Canny edge detection")
            self._fallback_to_canny = True
            self.network = None
            self._initialized = True  # Still mark as initialized for fallback

    def load_weights(self, checkpoint_path: str) -> None:
        """Load pretrained HED weights.

        Args:
            checkpoint_path: Path to checkpoint file.

        Raises:
            FileNotFoundError: If checkpoint doesn't exist.
            RuntimeError: If loading fails.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        try:
            state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

            # Map keys from the official format to our naming convention
            # Official format uses: moduleVggOne.0, moduleVggTwo.1, etc.
            # Our format uses: conv1_1, conv1_2, conv2_1, etc.
            key_mapping = {
                # VGG blocks
                'moduleVggOne.0': 'conv1_1',
                'moduleVggOne.2': 'conv1_2',
                'moduleVggTwo.1': 'conv2_1',
                'moduleVggTwo.3': 'conv2_2',
                'moduleVggThr.1': 'conv3_1',
                'moduleVggThr.3': 'conv3_2',
                'moduleVggThr.5': 'conv3_3',
                'moduleVggFou.1': 'conv4_1',
                'moduleVggFou.3': 'conv4_2',
                'moduleVggFou.5': 'conv4_3',
                'moduleVggFiv.1': 'conv5_1',
                'moduleVggFiv.3': 'conv5_2',
                'moduleVggFiv.5': 'conv5_3',
                # Side outputs
                'moduleScoreOne': 'side1',
                'moduleScoreTwo': 'side2',
                'moduleScoreThr': 'side3',
                'moduleScoreFou': 'side4',
                'moduleScoreFiv': 'side5',
                # Fuse layer
                'moduleCombine.0': 'fuse',
            }

            # Convert keys
            new_state_dict = {}
            for old_key, value in state_dict.items():
                # Extract the base key and parameter type (weight/bias)
                for old_prefix, new_prefix in key_mapping.items():
                    if old_key.startswith(old_prefix):
                        param_type = old_key.split('.')[-1]  # 'weight' or 'bias'
                        new_key = f"{new_prefix}.{param_type}"
                        new_state_dict[new_key] = value
                        break

            self.network.load_state_dict(new_state_dict)
            self._initialized = True
            print(f"Loaded HED weights from {checkpoint_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load weights: {e}")

    def _preprocess_input(self, image: Tensor) -> Tensor:
        """Preprocess image for HED network.

        Args:
            image: Input tensor (B, C, H, W), values in [0, 1].

        Returns:
            Preprocessed tensor.
        """
        # HED expects RGB input in [0, 1]
        # Convert grayscale to RGB if needed
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)

        return image

    def forward(self, image: Tensor, **kwargs: Any) -> ModuleOutput:
        """Detect edges using HED network.

        Args:
            image: Input image tensor (B, C, H, W).
            **kwargs: Additional parameters.

        Returns:
            ModuleOutput with edge map and intermediate results.
        """
        start_time = time.time()

        # Fallback to Canny if HED not available
        if self._fallback_to_canny:
            from kp3d.modules.edge.canny import CannyEdgeDetector
            canny = CannyEdgeDetector(config=self.config, device=self.device, dtype=self.dtype)
            return canny.forward(image, **kwargs)

        # Preprocess
        input_tensor = self._preprocess_input(image)

        # Inference
        with torch.no_grad():
            edge_map = self.network(input_tensor)

        # Post-processing
        threshold = kwargs.get('threshold', 0.5)
        edge_binary = (edge_map > threshold).float()

        elapsed = time.time() - start_time

        return ModuleOutput(
            result=edge_map,
            intermediate={
                "input": image,
                "edges_binary": edge_binary,
            },
            metadata={
                "method": "hed",
                "threshold": threshold,
                "processing_time": elapsed,
                "network_params": sum(p.numel() for p in self.network.parameters()),
            }
        )
