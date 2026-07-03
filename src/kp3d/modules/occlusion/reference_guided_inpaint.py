"""
Reference-Guided Diffusion Inpainting Module

Implements style-constrained inpainting using visible regions as style reference.
Uses Gram matrix style features and color statistics to guide diffusion models.
"""

import numpy as np
import cv2
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class StyleFeatureExtractor:
    """Extract Gram matrix style features from image regions."""

    def __init__(self, use_vgg: bool = True):
        """
        Initialize feature extractor.

        Args:
            use_vgg: Use VGG-19 for deep features. If False, uses color statistics only.
        """
        self.use_vgg = use_vgg
        self._vgg = None
        self._device = None

    @property
    def vgg(self):
        """Lazy load VGG model."""
        if self._vgg is None and self.use_vgg:
            try:
                from torchvision.models import vgg19, VGG19_Weights

                # Determine device
                if torch.cuda.is_available():
                    self._device = torch.device("cuda")
                else:
                    self._device = torch.device("cpu")
                    logger.warning("CUDA not available, using CPU for VGG features (slower)")

                # Load VGG-19 pretrained model
                vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).to(self._device)
                vgg.eval()

                # Extract feature layers (conv1_1, conv2_1, conv3_1, conv4_1, conv5_1)
                self._vgg = nn.ModuleDict({
                    'conv1_1': vgg.features[:2],
                    'conv2_1': vgg.features[2:7],
                    'conv3_1': vgg.features[7:12],
                    'conv4_1': vgg.features[12:21],
                    'conv5_1': vgg.features[21:30]
                })

                for param in self._vgg.parameters():
                    param.requires_grad = False

                logger.info(f"VGG-19 loaded on {self._device}")

            except ImportError:
                logger.warning("torchvision not available, falling back to color statistics only")
                self.use_vgg = False
                self._vgg = None
            except Exception as e:
                logger.warning(f"Failed to load VGG-19: {e}. Falling back to color statistics only")
                self.use_vgg = False
                self._vgg = None

        return self._vgg

    def extract_gram_features(self, image: np.ndarray, mask: np.ndarray) -> Dict:
        """
        Extract Gram matrix style features from masked region.

        Args:
            image: RGB image (H, W, 3), uint8 [0-255]
            mask: Region to extract features from (255 = valid)

        Returns:
            Dict with 'gram_matrices', 'color_histogram', 'mean_color', 'std_color'
        """
        features = {}

        # Ensure mask is binary
        mask_binary = (mask > 127).astype(np.uint8)

        # Extract color statistics
        masked_pixels = image[mask_binary > 0]

        if len(masked_pixels) > 0:
            features['mean_color'] = masked_pixels.mean(axis=0)
            features['std_color'] = masked_pixels.std(axis=0)

            # Compute color histogram per channel
            features['color_histogram'] = []
            for ch in range(3):
                hist, _ = np.histogram(masked_pixels[:, ch], bins=32, range=(0, 256))
                hist = hist.astype(np.float32) / (hist.sum() + 1e-8)
                features['color_histogram'].append(hist)
        else:
            features['mean_color'] = np.array([128, 128, 128], dtype=np.float32)
            features['std_color'] = np.array([50, 50, 50], dtype=np.float32)
            features['color_histogram'] = [np.ones(32) / 32.0 for _ in range(3)]

        # Extract deep style features if VGG is available
        if self.use_vgg and self.vgg is not None:
            try:
                gram_matrices = self._compute_gram_matrices(image, mask_binary)
                features['gram_matrices'] = gram_matrices
            except Exception as e:
                logger.warning(f"Failed to compute Gram matrices: {e}")
                features['gram_matrices'] = None
        else:
            features['gram_matrices'] = None

        return features

    def _compute_gram_matrices(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute Gram matrices from VGG feature maps."""
        # Normalize image for VGG (ImageNet stats)
        img_normalized = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        img_normalized = (img_normalized - mean) / std

        # Convert to torch tensor (B, C, H, W)
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.float().to(self._device)

        # Resize mask to image size if needed
        mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask_tensor = torch.from_numpy(mask_resized).float().to(self._device) / 255.0

        gram_matrices = {}

        with torch.no_grad():
            x = img_tensor
            for layer_name, layer in self.vgg.items():
                x = layer(x)

                # Compute Gram matrix for this layer
                b, c, h, w = x.shape
                features = x.view(b, c, h * w)

                # Apply mask if dimensions match (downsample mask)
                mask_downsampled = torch.nn.functional.interpolate(
                    mask_tensor.unsqueeze(0).unsqueeze(0),
                    size=(h, w),
                    mode='nearest'
                ).view(1, 1, h * w)

                # Mask features
                features_masked = features * mask_downsampled

                # Gram matrix: G = F * F^T / (C * H * W)
                gram = torch.bmm(features_masked, features_masked.transpose(1, 2))
                gram = gram / (c * h * w)

                gram_matrices[layer_name] = gram.cpu().numpy()[0]

        return gram_matrices

    def compute_style_distance(self, features1: Dict, features2: Dict) -> float:
        """
        Compute style distance between two feature sets.

        L_style = Σ ||Gram(generated) - Gram(reference)||²
        """
        distance = 0.0

        # Gram matrix distance (if available)
        if features1.get('gram_matrices') is not None and features2.get('gram_matrices') is not None:
            gram1 = features1['gram_matrices']
            gram2 = features2['gram_matrices']

            for layer_name in gram1.keys():
                if layer_name in gram2:
                    diff = gram1[layer_name] - gram2[layer_name]
                    distance += np.sum(diff ** 2)

        # Color histogram distance
        hist1 = features1.get('color_histogram', [])
        hist2 = features2.get('color_histogram', [])

        for h1, h2 in zip(hist1, hist2):
            distance += np.sum((h1 - h2) ** 2)

        # Color statistics distance
        mean_dist = np.sum((features1['mean_color'] - features2['mean_color']) ** 2)
        std_dist = np.sum((features1['std_color'] - features2['std_color']) ** 2)

        distance += (mean_dist + std_dist) / 255.0

        return float(distance)

    def extract_color_statistics(self, image: np.ndarray, mask: np.ndarray) -> Dict:
        """
        Extract only color statistics from masked region (fast, no VGG).

        Args:
            image: RGB image (H, W, 3), uint8 [0-255]
            mask: Region to extract features from (255 = valid)

        Returns:
            Dict with 'mean_color', 'std_color', 'color_histogram'
        """
        features = {}

        # Ensure mask is binary
        mask_binary = (mask > 127).astype(np.uint8)

        # Extract masked pixels
        masked_pixels = image[mask_binary > 0]

        if len(masked_pixels) > 0:
            features['mean_color'] = masked_pixels.mean(axis=0)
            features['std_color'] = masked_pixels.std(axis=0)

            # Compute color histogram per channel
            features['color_histogram'] = []
            for ch in range(3):
                hist, _ = np.histogram(masked_pixels[:, ch], bins=32, range=(0, 256))
                hist = hist.astype(np.float32) / (hist.sum() + 1e-8)
                features['color_histogram'].append(hist)
        else:
            features['mean_color'] = np.array([128, 128, 128], dtype=np.float32)
            features['std_color'] = np.array([50, 50, 50], dtype=np.float32)
            features['color_histogram'] = [np.ones(32) / 32.0 for _ in range(3)]

        return features


class ReferenceGuidedInpainter:
    """Reference-guided diffusion inpainting using visible region as style reference."""

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-inpainting",
        device: str = "cuda",
        style_weight: float = 0.3
    ):
        """
        Initialize reference-guided inpainter.

        Args:
            model_id: HuggingFace model ID for Stable Diffusion inpainting
            device: 'cuda' or 'cpu'
            style_weight: Weight for style guidance (0-1)
        """
        self.model_id = model_id
        self.device = device if torch.cuda.is_available() else "cpu"
        self.style_weight = style_weight
        self.style_extractor = StyleFeatureExtractor(use_vgg=(self.device == "cuda"))
        self._pipeline = None

        if self.device == "cpu":
            logger.warning("Running on CPU - inpainting will be slower and use simpler color matching")

    @property
    def pipeline(self):
        """Lazy load diffusion pipeline."""
        if self._pipeline is None:
            try:
                from diffusers import StableDiffusionInpaintPipeline

                logger.info(f"Loading {self.model_id} on {self.device}...")
                self._pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                ).to(self.device)

                # Enable memory optimizations if on CPU
                if self.device == "cpu":
                    self._pipeline.enable_attention_slicing()

                logger.info("Diffusion pipeline loaded successfully")

            except ImportError:
                logger.error("diffusers library not installed. Install with: pip install diffusers transformers accelerate")
                raise
            except Exception as e:
                logger.error(f"Failed to load diffusion pipeline: {e}")
                raise

        return self._pipeline

    def inpaint(
        self,
        image: np.ndarray,
        inpaint_mask: np.ndarray,
        reference_mask: np.ndarray,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Inpaint with style-constrained diffusion.

        Args:
            image: RGB image (H, W, 3), uint8 [0-255]
            inpaint_mask: Region to inpaint (255 = inpaint)
            reference_mask: Region to use as style reference (255 = valid)
            num_inference_steps: Diffusion steps
            guidance_scale: CFG scale
            seed: Random seed for reproducibility

        Returns:
            Inpainted image with style matching reference region

        Algorithm:
        1. Extract style features from reference region
        2. Generate texture description prompt from features
        3. Run SD inpainting with generated prompt
        4. Apply post-processing to match color statistics
        """
        # Extract reference style features
        logger.info("Extracting style features from reference region...")
        reference_features = self.style_extractor.extract_gram_features(image, reference_mask)

        # Generate style-aware prompt
        prompt = self._generate_style_prompt(reference_features)
        logger.info(f"Generated style prompt: {prompt}")

        # Prepare inputs for diffusion
        from PIL import Image

        # Convert to PIL
        pil_image = Image.fromarray(image)
        pil_mask = Image.fromarray(inpaint_mask)

        # Set seed if provided
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        # Run diffusion inpainting
        logger.info(f"Running diffusion inpainting ({num_inference_steps} steps)...")

        result = self.pipeline(
            prompt=prompt,
            image=pil_image,
            mask_image=pil_mask,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]

        result_np = np.array(result)

        # Post-process to match color statistics
        logger.info("Matching color statistics to reference...")
        result_matched = self._match_color_statistics(
            result_np,
            inpaint_mask,
            reference_features
        )

        return result_matched

    def _generate_style_prompt(self, features: Dict) -> str:
        """Generate text prompt describing the style features."""
        mean_color = features['mean_color']
        std_color = features['std_color']

        # Determine dominant color
        r, g, b = mean_color

        if r > g and r > b:
            dominant_color = "reddish"
        elif g > r and g > b:
            dominant_color = "greenish"
        elif b > r and b > g:
            dominant_color = "bluish"
        elif abs(r - g) < 20 and abs(r - b) < 20:
            if r > 200:
                dominant_color = "light"
            elif r < 50:
                dominant_color = "dark"
            else:
                dominant_color = "neutral"
        else:
            dominant_color = "colored"

        # Determine texture variation
        avg_std = std_color.mean()

        if avg_std < 20:
            texture = "smooth, uniform texture"
        elif avg_std < 40:
            texture = "subtle texture variation"
        else:
            texture = "detailed texture with variation"

        prompt = f"natural inpainting, {dominant_color} tones, {texture}, seamless blend, high quality"

        return prompt

    def _match_color_statistics(
        self,
        result: np.ndarray,
        inpaint_mask: np.ndarray,
        reference_features: Dict
    ) -> np.ndarray:
        """Post-process to match color mean/std of reference."""
        result_matched = result.copy()

        # Get inpainted region
        mask_binary = (inpaint_mask > 127).astype(np.uint8)

        if mask_binary.sum() == 0:
            return result_matched

        # Current statistics in inpainted region
        inpainted_pixels = result[mask_binary > 0]
        current_mean = inpainted_pixels.mean(axis=0)
        current_std = inpainted_pixels.std(axis=0) + 1e-8

        # Target statistics
        target_mean = reference_features['mean_color']
        target_std = reference_features['std_color']

        # Match statistics per channel
        for ch in range(3):
            # Normalize to zero mean, unit variance
            normalized = (result_matched[:, :, ch] - current_mean[ch]) / current_std[ch]

            # Scale to target statistics
            matched = normalized * target_std[ch] + target_mean[ch]

            # Apply only in inpainted region
            result_matched[:, :, ch] = np.where(
                mask_binary > 0,
                matched,
                result_matched[:, :, ch]
            )

        # Clip to valid range
        result_matched = np.clip(result_matched, 0, 255).astype(np.uint8)

        # Blend edges to avoid hard boundaries
        kernel_size = 15
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask_dilated = cv2.dilate(mask_binary, kernel, iterations=1)
        mask_eroded = cv2.erode(mask_binary, kernel, iterations=1)

        blend_region = (mask_dilated - mask_eroded).astype(np.float32) / 255.0
        blend_region = cv2.GaussianBlur(blend_region, (15, 15), 0)

        for ch in range(3):
            result_matched[:, :, ch] = (
                blend_region * result_matched[:, :, ch] +
                (1 - blend_region) * result[:, :, ch]
            ).astype(np.uint8)

        return result_matched


def match_histograms(
    source: np.ndarray,
    reference: np.ndarray,
    source_mask: Optional[np.ndarray] = None,
    reference_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Match histogram of source to reference.

    Args:
        source: Source image (H, W, 3)
        reference: Reference image (H, W, 3)
        source_mask: Optional mask for source (255 = valid)
        reference_mask: Optional mask for reference (255 = valid)

    Returns:
        Source image with histogram matched to reference
    """
    matched = source.copy()

    for ch in range(3):
        # Get source and reference pixels
        if source_mask is not None:
            src_pixels = source[:, :, ch][source_mask > 127].flatten()
        else:
            src_pixels = source[:, :, ch].flatten()

        if reference_mask is not None:
            ref_pixels = reference[:, :, ch][reference_mask > 127].flatten()
        else:
            ref_pixels = reference[:, :, ch].flatten()

        if len(src_pixels) == 0 or len(ref_pixels) == 0:
            continue

        # Compute CDFs
        src_values, src_counts = np.unique(src_pixels, return_counts=True)
        ref_values, ref_counts = np.unique(ref_pixels, return_counts=True)

        src_cdf = np.cumsum(src_counts).astype(np.float64)
        src_cdf /= src_cdf[-1]

        ref_cdf = np.cumsum(ref_counts).astype(np.float64)
        ref_cdf /= ref_cdf[-1]

        # Build lookup table
        lookup = np.interp(src_cdf, ref_cdf, ref_values)
        lookup_table = np.zeros(256, dtype=np.uint8)

        for i, val in enumerate(src_values):
            lookup_table[val] = lookup[i]

        # Apply lookup
        matched[:, :, ch] = lookup_table[source[:, :, ch]]

    return matched


def reference_guided_inpaint(
    image: np.ndarray,
    inpaint_mask: np.ndarray,
    reference_mask: np.ndarray,
    style_weight: float = 0.3,
    device: str = "cuda",
    num_inference_steps: int = 30,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Quick reference-guided inpainting.

    Args:
        image: RGB image (H, W, 3)
        inpaint_mask: Region to inpaint (255 = inpaint)
        reference_mask: Region to use as style reference (255 = valid)
        style_weight: Weight for style guidance (0-1)
        device: 'cuda' or 'cpu'
        num_inference_steps: Diffusion steps
        seed: Random seed

    Returns:
        Inpainted image with style matching reference
    """
    inpainter = ReferenceGuidedInpainter(
        device=device,
        style_weight=style_weight
    )

    return inpainter.inpaint(
        image=image,
        inpaint_mask=inpaint_mask,
        reference_mask=reference_mask,
        num_inference_steps=num_inference_steps,
        seed=seed
    )
