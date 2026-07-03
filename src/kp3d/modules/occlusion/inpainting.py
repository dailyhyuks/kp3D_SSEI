"""Inpainting module for filling occluded regions.

Provides OpenCV-based inpainting with Telea and Navier-Stokes methods,
LaMa (Large Mask Inpainting) for high-quality results,
plus utilities for object extraction.
"""

from typing import Literal, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import cv2
from scipy import ndimage

# Lazy ONNX session cache
_lama_session = None
_LAMA_MODEL_SIZE = 512  # LaMa ONNX model fixed input size


def _download_lama_onnx() -> str:
    """Download LaMa ONNX model from HuggingFace Hub.

    Returns:
        Path to downloaded model file.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "LaMa inpainting requires huggingface_hub. "
            "Install with: pip install huggingface-hub"
        )

    model_path = hf_hub_download(
        repo_id="Carve/LaMa-ONNX",
        filename="lama_fp32.onnx",
        cache_dir=Path.home() / ".cache" / "lama"
    )
    return model_path


def _get_lama_session():
    """Get or create ONNX Runtime session for LaMa."""
    global _lama_session
    if _lama_session is None:
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "LaMa inpainting requires onnxruntime. "
                "Install with: pip install onnxruntime-gpu"
            )

        model_path = _download_lama_onnx()

        # Configure ONNX Runtime
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        _lama_session = ort.InferenceSession(model_path, providers=providers)

    return _lama_session


class LamaInpainter:
    """LaMa (Large Mask Inpainting) based inpainter.

    Uses ONNX Runtime for high-quality deep learning inpainting.
    Significantly better than OpenCV methods for complex textures.

    Reference: https://github.com/advimman/lama
    Model: https://huggingface.co/Carve/LaMa-ONNX
    """

    def __init__(self):
        """Initialize LaMa inpainter (model loaded on first use)."""
        self._session = None

    @property
    def session(self):
        """Lazy load ONNX session."""
        if self._session is None:
            self._session = _get_lama_session()
        return self._session

    def _preprocess(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        """Preprocess image and mask for LaMa model.

        Args:
            image: RGB image (H, W, 3), uint8.
            mask: Binary mask (H, W).

        Returns:
            Tuple of (image_tensor, mask_tensor, original_size).
        """
        original_size = (image.shape[1], image.shape[0])  # (W, H)

        # Resize to model input size
        img_resized = cv2.resize(
            image, (_LAMA_MODEL_SIZE, _LAMA_MODEL_SIZE),
            interpolation=cv2.INTER_LANCZOS4
        )
        mask_resized = cv2.resize(
            mask, (_LAMA_MODEL_SIZE, _LAMA_MODEL_SIZE),
            interpolation=cv2.INTER_NEAREST
        )

        # Normalize image to [0, 1]
        img_norm = img_resized.astype(np.float32) / 255.0

        # Normalize mask to [0, 1]
        mask_norm = (mask_resized > 127).astype(np.float32)

        # Convert to NCHW format
        # Image: (H, W, 3) -> (1, 3, H, W)
        img_tensor = np.transpose(img_norm, (2, 0, 1))[np.newaxis, ...]

        # Mask: (H, W) -> (1, 1, H, W)
        mask_tensor = mask_norm[np.newaxis, np.newaxis, ...]

        return img_tensor, mask_tensor, original_size

    def _postprocess(
        self,
        output: np.ndarray,
        original_size: Tuple[int, int]
    ) -> np.ndarray:
        """Postprocess LaMa output.

        Args:
            output: Model output tensor (1, 3, H, W).
            original_size: Original image size (W, H).

        Returns:
            RGB image (H, W, 3), uint8.
        """
        # (1, 3, H, W) -> (H, W, 3)
        result = np.transpose(output[0], (1, 2, 0))

        # LaMa ONNX outputs values in [0, 255] range
        result = np.clip(result, 0, 255).astype(np.uint8)

        # Resize back to original size
        if original_size != (_LAMA_MODEL_SIZE, _LAMA_MODEL_SIZE):
            result = cv2.resize(
                result, original_size,
                interpolation=cv2.INTER_LANCZOS4
            )

        return result

    def inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """Inpaint masked regions using LaMa.

        Args:
            image: RGB image (H, W, 3), uint8.
            mask: Binary mask (H, W) where 255 = region to inpaint.

        Returns:
            Inpainted image (H, W, 3), uint8.
        """
        # Ensure uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        if mask.dtype != np.uint8:
            mask = (mask > 0).astype(np.uint8) * 255

        # Preprocess
        img_tensor, mask_tensor, original_size = self._preprocess(image, mask)

        # Run inference
        input_name_img = self.session.get_inputs()[0].name
        input_name_mask = self.session.get_inputs()[1].name
        output_name = self.session.get_outputs()[0].name

        output = self.session.run(
            [output_name],
            {input_name_img: img_tensor, input_name_mask: mask_tensor}
        )[0]

        # Postprocess
        result = self._postprocess(output, original_size)

        return result


class InpaintingModule:
    """Inpainting module for occluded region filling.

    Supports multiple algorithms:
    - Telea: Fast Marching Method based on Alexandru Telea's algorithm.
    - NS (Navier-Stokes): PDE-based approach producing smoother results.
    - LaMa: Deep learning based, best for large masks and complex textures.
    - SD: Stable Diffusion inpainting for generative fill.
    - ControlNet: SD + ControlNet for structure-preserving inpainting.
    """

    def __init__(
        self,
        method: Literal["telea", "ns", "lama", "sd", "controlnet"] = "telea",
        radius: int = 5,
        sd_prompt: Optional[str] = None,
        sd_steps: int = 30
    ):
        """Initialize inpainting module.

        Args:
            method: Inpainting algorithm.
            radius: Radius for OpenCV inpainting (ignored for others).
            sd_prompt: Default prompt for SD/ControlNet methods.
            sd_steps: Number of inference steps for SD/ControlNet.
        """
        self.method = method
        self.radius = radius
        self.sd_prompt = sd_prompt or "seamless texture, high quality"
        self.sd_steps = sd_steps

        # OpenCV inpainting flags
        self._flags = {
            "telea": cv2.INPAINT_TELEA,
            "ns": cv2.INPAINT_NS
        }

        # Lazy-loaded inpainters
        self._lama = None
        self._sd = None
        self._controlnet = None

    @property
    def lama(self) -> LamaInpainter:
        """Get LaMa inpainter (lazy loaded)."""
        if self._lama is None:
            self._lama = LamaInpainter()
        return self._lama

    @property
    def sd(self) -> "SDInpainter":
        """Get SD inpainter (lazy loaded)."""
        if self._sd is None:
            self._sd = SDInpainter(use_controlnet=False)
        return self._sd

    @property
    def controlnet(self) -> "SDInpainter":
        """Get ControlNet inpainter (lazy loaded)."""
        if self._controlnet is None:
            self._controlnet = SDInpainter(use_controlnet=True)
        return self._controlnet

    def inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        radius: Optional[int] = None,
        method: Optional[str] = None,
        prompt: Optional[str] = None
    ) -> np.ndarray:
        """Inpaint masked regions of an image.

        Args:
            image: RGB image (H, W, 3), uint8.
            mask: Binary mask (H, W) where 255 = region to inpaint.
            radius: Override default inpainting radius (OpenCV only).
            method: Override default method ("telea", "ns", "lama", "sd", "controlnet").
            prompt: Text prompt for SD/ControlNet methods.

        Returns:
            Inpainted image (H, W, 3), uint8.
        """
        method = method or self.method
        radius = radius or self.radius
        prompt = prompt or self.sd_prompt

        # Ensure mask is uint8
        if mask.dtype != np.uint8:
            mask = (mask > 0).astype(np.uint8) * 255

        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        # Use LaMa for deep learning inpainting
        if method == "lama":
            return self.lama.inpaint(image, mask)

        # Use Stable Diffusion inpainting
        if method == "sd":
            return self.sd.inpaint(
                image, mask,
                prompt=prompt,
                num_inference_steps=self.sd_steps
            )

        # Use ControlNet inpainting
        if method == "controlnet":
            return self.controlnet.inpaint(
                image, mask,
                prompt=prompt,
                num_inference_steps=self.sd_steps
            )

        # Use OpenCV for traditional inpainting
        flag = self._flags[method]
        result = cv2.inpaint(image, mask, radius, flag)
        return result

    def inpaint_with_dilated_mask(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        dilation_size: int = 3,
        dilation_iterations: int = 1
    ) -> np.ndarray:
        """Inpaint with dilated mask for better edge blending.

        Args:
            image: RGB image (H, W, 3).
            mask: Binary mask (H, W), can be 0-1 or 0-255.
            dilation_size: Kernel size for dilation.
            dilation_iterations: Number of dilation passes.

        Returns:
            Inpainted image with smooth boundaries.
        """
        # Normalize mask to binary 0-1 first (handles both 0-1 and 0-255 input)
        binary_mask = (mask > 0).astype(np.uint8)

        # Dilate mask to cover edges
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (dilation_size, dilation_size)
        )
        dilated_mask = cv2.dilate(
            binary_mask,
            kernel,
            iterations=dilation_iterations
        )

        # Convert to 0-255 for inpaint
        return self.inpaint(image, dilated_mask * 255)

    def extract_object(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        background_color: Tuple[int, int, int] = (255, 255, 255),
        return_rgba: bool = True
    ) -> np.ndarray:
        """Extract object from image using mask.

        Args:
            image: RGB image (H, W, 3).
            mask: Binary mask (H, W) where 255/True = object.
            background_color: Color for non-object regions (if RGB output).
            return_rgba: Return RGBA with alpha channel instead of RGB.

        Returns:
            Extracted object as RGB(A) image.
        """
        # Ensure proper types
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        if mask.dtype != np.uint8:
            mask = (mask > 0).astype(np.uint8) * 255

        if return_rgba:
            # Create RGBA output
            rgba = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
            rgba[:, :, :3] = image
            rgba[:, :, 3] = mask
            return rgba
        else:
            # Create RGB with background color
            result = np.full_like(image, background_color, dtype=np.uint8)
            result[mask > 0] = image[mask > 0]
            return result

    def remove_object(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        radius: Optional[int] = None
    ) -> np.ndarray:
        """Remove object from image by inpainting its region.

        Convenience method that inverts the typical inpainting logic.

        Args:
            image: RGB image (H, W, 3).
            mask: Binary mask of object to remove.
            radius: Inpainting radius.

        Returns:
            Image with object removed and region filled.
        """
        return self.inpaint(image, mask, radius)

    def blend_edges(
        self,
        original: np.ndarray,
        inpainted: np.ndarray,
        mask: np.ndarray,
        blend_width: int = 5
    ) -> np.ndarray:
        """Blend edges between original and inpainted regions.

        Creates a smooth transition at mask boundaries using
        alpha blending with a gradient.

        Args:
            original: Original image (H, W, 3).
            inpainted: Inpainted image (H, W, 3).
            mask: Binary mask of inpainted region.
            blend_width: Width of blending zone in pixels.

        Returns:
            Blended image with smooth transitions.
        """
        # Create distance transform for soft blending
        if mask.dtype != np.uint8:
            mask = (mask > 0).astype(np.uint8)

        # Compute distance to mask boundary
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        dist_inv = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 5)

        # Create blend weight (0 at boundary, 1 away from it)
        blend_weight = np.clip(dist / blend_width, 0, 1)
        blend_weight_inv = np.clip(dist_inv / blend_width, 0, 1)

        # Normalize weights
        total_weight = blend_weight + blend_weight_inv + 1e-8
        w_inpaint = blend_weight / total_weight
        w_original = blend_weight_inv / total_weight

        # Expand for RGB channels
        w_inpaint = w_inpaint[:, :, np.newaxis]
        w_original = w_original[:, :, np.newaxis]

        # Blend
        result = (inpainted * w_inpaint + original * w_original).astype(np.uint8)
        return result

    def inpaint_with_reference(
        self,
        image: np.ndarray,
        inpaint_mask: np.ndarray,
        reference_mask: np.ndarray,
        noise_factor: float = 0.7
    ) -> np.ndarray:
        """Inpaint using color statistics from a reference region.

        More effective than standard inpainting when the region to fill
        should match a specific part of the image (e.g., filling occluded
        table surface using visible table texture).

        Args:
            image: RGB image (H, W, 3), uint8.
            inpaint_mask: Binary mask of region to fill.
            reference_mask: Binary mask of region to sample colors from.
            noise_factor: Texture variation (0=flat color, 1=full variation).

        Returns:
            Inpainted image with texture matching reference region.
        """
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        result = image.copy()

        # Get reference region statistics
        ref_pixels = image[reference_mask > 0]
        if len(ref_pixels) == 0:
            return self.inpaint(image, inpaint_mask)

        mean_color = np.mean(ref_pixels, axis=0)
        std_color = np.std(ref_pixels, axis=0)

        # Fill inpaint region with textured color
        fill_region = inpaint_mask > 0
        num_fill = np.sum(fill_region)

        if num_fill > 0:
            # Generate texture noise
            noise = np.random.randn(num_fill, 3) * std_color * noise_factor
            fill_colors = (mean_color + noise).clip(0, 255).astype(np.uint8)
            result[fill_region] = fill_colors

            # Smooth boundaries
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            boundary = cv2.dilate(inpaint_mask.astype(np.uint8), kernel) - inpaint_mask.astype(np.uint8)

            # Apply Gaussian blur at boundary
            result_blurred = cv2.GaussianBlur(result, (7, 7), 0)
            alpha = cv2.GaussianBlur(boundary.astype(np.float32), (11, 11), 0)
            alpha = alpha / (alpha.max() + 1e-8)
            alpha_3ch = np.stack([alpha] * 3, axis=2)

            result = (result.astype(np.float32) * (1 - alpha_3ch * 0.5) +
                     result_blurred.astype(np.float32) * alpha_3ch * 0.5).astype(np.uint8)

            # Final inpaint for boundary cleanup
            result = cv2.inpaint(result, boundary, 3, cv2.INPAINT_NS)

        return result

    def inpaint_texture_clone(
        self,
        image: np.ndarray,
        inpaint_mask: np.ndarray,
        reference_mask: np.ndarray,
        patch_size: int = 5,
        use_position_weight: bool = True
    ) -> np.ndarray:
        """Inpaint using multi-scale NS with coherence enhancement.

        Uses multi-scale Navier-Stokes inpainting for better structure,
        then enhances local coherence with guided filtering.

        Args:
            image: RGB image (H, W, 3), uint8.
            inpaint_mask: Binary mask of region to fill (0-255).
            reference_mask: Binary mask of reference texture region (0-255).
            patch_size: Base inpainting radius for OpenCV.
            use_position_weight: Not used, kept for API compatibility.

        Returns:
            Inpainted image.
        """
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        h, w = image.shape[:2]

        # Normalize masks
        inpaint_binary = (inpaint_mask > 127).astype(np.uint8)

        if np.sum(inpaint_binary) == 0:
            return image.copy()

        # Multi-scale inpainting for better structure preservation
        # Scale 1: Coarse (downscaled) - captures overall structure
        scale_factor = 0.5
        small_h, small_w = int(h * scale_factor), int(w * scale_factor)

        img_small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
        mask_small = cv2.resize(inpaint_binary * 255, (small_w, small_h), interpolation=cv2.INTER_NEAREST)

        # Inpaint at coarse scale with larger radius
        coarse_inpainted = cv2.inpaint(img_small, mask_small, patch_size * 2, cv2.INPAINT_NS)

        # Upscale coarse result
        coarse_upscaled = cv2.resize(coarse_inpainted, (w, h), interpolation=cv2.INTER_CUBIC)

        # Scale 2: Fine (original) - captures details
        # Use coarse result as initialization
        initialized = image.copy()
        initialized[inpaint_binary > 0] = coarse_upscaled[inpaint_binary > 0]

        # Fine inpainting with smaller radius for detail
        fine_inpainted = cv2.inpaint(initialized, inpaint_binary * 255, patch_size, cv2.INPAINT_NS)

        # Blend coarse structure with fine details
        # Use distance from edge to weight: center uses coarse, edge uses fine
        dist_from_edge = cv2.distanceTransform(inpaint_binary, cv2.DIST_L2, 5)
        max_dist = np.max(dist_from_edge) + 1e-6

        # Normalize distance
        center_weight = np.clip(dist_from_edge / (max_dist * 0.5), 0, 1)
        center_weight = center_weight[:, :, np.newaxis]

        # Combine: more coarse in center, more fine at edges
        combined = (coarse_upscaled.astype(float) * center_weight * 0.3 +
                   fine_inpainted.astype(float) * (1 - center_weight * 0.3))
        combined = np.clip(combined, 0, 255).astype(np.uint8)

        # Apply only to inpaint region
        result = image.copy()
        result[inpaint_binary > 0] = combined[inpaint_binary > 0]

        # Edge coherence: ensure edges align with surrounding
        # Use guided filter to propagate edge information
        try:
            # ximgproc module for guided filter
            guide = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            for c in range(3):
                channel = result[:, :, c].astype(np.float32)
                filtered = cv2.ximgproc.guidedFilter(
                    guide.astype(np.float32),
                    channel,
                    radius=3,
                    eps=100
                )
                # Apply only at boundary
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                boundary = cv2.dilate(inpaint_binary, kernel) - cv2.erode(inpaint_binary, kernel)
                result[:, :, c] = np.where(boundary > 0, filtered.astype(np.uint8), result[:, :, c])
        except (cv2.error, AttributeError):
            # ximgproc not available, use bilateral filter as fallback
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            boundary = cv2.dilate(inpaint_binary, kernel) - cv2.erode(inpaint_binary, kernel)
            if np.sum(boundary) > 0:
                blurred = cv2.bilateralFilter(result, 5, 50, 50)
                result[boundary > 0] = blurred[boundary > 0]

        # Final boundary cleanup with small NS pass
        kernel_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edge_band = cv2.dilate(inpaint_binary, kernel_edge) - inpaint_binary
        if np.sum(edge_band) > 0:
            result = cv2.inpaint(result, edge_band * 255, 2, cv2.INPAINT_NS)

        return result

    def inpaint_lama_guided(
        self,
        image: np.ndarray,
        inpaint_mask: np.ndarray,
        reference_mask: np.ndarray,
        prefill_method: str = "ns",
        lama_weight: float = 0.3
    ) -> np.ndarray:
        """LaMa inpainting with context pre-filling (V2 improved).

        Addresses the "existing region reference problem" where LaMa alone
        produces white/generic fills because it sees no visual context.

        V2 Approach:
        1. Pre-fill masked region with NS inpainting (gives LaMa context)
        2. Run LaMa on pre-filled image (now sees surrounding texture)
        3. Weighted blend of LaMa structure with pre-filled texture
        4. Edge-aware boundary smoothing
        5. Final micro-boundary cleanup

        Args:
            image: RGB image (H, W, 3), uint8.
            inpaint_mask: Binary mask of region to fill (0-255).
            reference_mask: Binary mask of reference texture region (unused in V2).
            prefill_method: "ns" for Navier-Stokes prefill, "none" to skip.
            lama_weight: Weight for LaMa result in blending (0-1). Lower values
                        preserve more texture from prefill, higher values keep
                        more LaMa structure. Default 0.3 works well.

        Returns:
            Inpainted image with context-matched colors and textures.
        """
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        h, w = image.shape[:2]

        # Normalize mask
        inpaint_binary = (inpaint_mask > 127).astype(np.uint8)

        if np.sum(inpaint_binary) == 0:
            return image.copy()

        # Step 1: Pre-fill masked region with surrounding context
        # This gives LaMa visual context to work with instead of blank area
        if prefill_method == "ns":
            # NS inpainting provides fast, reasonable context
            prefilled = cv2.inpaint(image, inpaint_binary * 255, 7, cv2.INPAINT_NS)
        else:
            # No prefill - blank the region (original behavior, causes white fill)
            prefilled = image.copy()
            prefilled[inpaint_binary > 0] = [255, 255, 255]

        # Step 2: Run LaMa on pre-filled image (now has visual context!)
        lama_result = self.lama.inpaint(prefilled, inpaint_binary * 255)

        # Step 3: Weighted blend of LaMa structure with pre-filled texture
        # Lower lama_weight = more texture preservation
        # Higher lama_weight = more LaMa structure (but may have color drift)
        result = image.copy()
        lama_region = lama_result[inpaint_binary > 0].astype(np.float32)
        prefill_region = prefilled[inpaint_binary > 0].astype(np.float32)

        blended = lama_region * lama_weight + prefill_region * (1 - lama_weight)
        result[inpaint_binary > 0] = blended.astype(np.uint8)

        # Step 4: Edge-aware boundary smoothing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        boundary = cv2.dilate(inpaint_binary, kernel) - cv2.erode(inpaint_binary, kernel)

        if np.sum(boundary) > 0:
            # Gaussian blur at boundary for smooth transition
            blurred = cv2.GaussianBlur(result, (5, 5), 1.5)
            alpha = cv2.GaussianBlur(boundary.astype(np.float32), (7, 7), 2.0)
            alpha = alpha / (alpha.max() + 1e-8)
            alpha = alpha[:, :, np.newaxis]

            result = (result.astype(np.float32) * (1 - alpha * 0.5) +
                     blurred.astype(np.float32) * alpha * 0.5).astype(np.uint8)

        # Step 5: Final micro-boundary cleanup with NS
        micro_boundary = cv2.dilate(inpaint_binary, np.ones((3, 3), np.uint8)) - inpaint_binary
        if np.sum(micro_boundary) > 0:
            result = cv2.inpaint(result, micro_boundary * 255, 2, cv2.INPAINT_NS)

        # Preserve original outside inpaint region
        final = image.copy()
        final[inpaint_binary > 0] = result[inpaint_binary > 0]

        return final

    def inpaint_lama_guided_legacy(
        self,
        image: np.ndarray,
        inpaint_mask: np.ndarray,
        reference_mask: np.ndarray
    ) -> np.ndarray:
        """Legacy LaMa guided inpainting (V1 - kept for reference).

        Original implementation that uses post-hoc LAB color matching.
        Has issues with white fill when LaMa produces generic results.

        Args:
            image: RGB image (H, W, 3), uint8.
            inpaint_mask: Binary mask of region to fill (0-255).
            reference_mask: Binary mask of reference texture region (0-255).

        Returns:
            Inpainted image with color matching applied.
        """
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        h, w = image.shape[:2]

        # Normalize masks
        inpaint_binary = (inpaint_mask > 127).astype(np.uint8)

        if np.sum(inpaint_binary) == 0:
            return image.copy()

        # Run LaMa inpainting (may produce white fill without context)
        lama_result = self.lama.inpaint(image, inpaint_binary * 255)

        # Get surrounding context (multi-scale)
        kernel_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        dilated_inner = cv2.dilate(inpaint_binary, kernel_inner)
        surrounding_inner = dilated_inner - inpaint_binary

        kernel_outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        dilated_outer = cv2.dilate(inpaint_binary, kernel_outer)
        surrounding_outer = dilated_outer - dilated_inner

        inner_pixels = image[surrounding_inner > 0]
        outer_pixels = image[surrounding_outer > 0]

        if len(inner_pixels) == 0 and len(outer_pixels) == 0:
            return lama_result

        # LAB color space matching
        image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        lama_lab = cv2.cvtColor(lama_result, cv2.COLOR_RGB2LAB).astype(np.float32)

        if len(inner_pixels) > 0:
            inner_lab = cv2.cvtColor(inner_pixels.reshape(1, -1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
            inner_mean = np.mean(inner_lab, axis=0)
            inner_std = np.std(inner_lab, axis=0) + 1e-6
        else:
            inner_mean = None

        if len(outer_pixels) > 0:
            outer_lab = cv2.cvtColor(outer_pixels.reshape(1, -1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
            outer_mean = np.mean(outer_lab, axis=0)
            outer_std = np.std(outer_lab, axis=0) + 1e-6
        else:
            outer_mean = None

        if inner_mean is not None and outer_mean is not None:
            tgt_mean = inner_mean * 0.7 + outer_mean * 0.3
            tgt_std = inner_std * 0.7 + outer_std * 0.3
        elif inner_mean is not None:
            tgt_mean, tgt_std = inner_mean, inner_std
        else:
            tgt_mean, tgt_std = outer_mean, outer_std

        # Color transfer in LAB space
        lama_inpaint_lab = lama_lab[inpaint_binary > 0]
        if len(lama_inpaint_lab) > 0:
            src_mean = np.mean(lama_inpaint_lab, axis=0)
            src_std = np.std(lama_inpaint_lab, axis=0) + 1e-6

            result_lab = lama_lab.copy()
            for c in range(3):
                channel = result_lab[:, :, c]
                inpaint_vals = channel[inpaint_binary > 0]
                normalized = (inpaint_vals - src_mean[c]) / src_std[c]
                transferred = normalized * tgt_std[c] + tgt_mean[c]
                channel[inpaint_binary > 0] = inpaint_vals * 0.2 + transferred * 0.8

            result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
            result = cv2.cvtColor(result_lab, cv2.COLOR_LAB2RGB)
        else:
            result = lama_result.copy()

        # Denoise and edge smoothing
        denoised = cv2.fastNlMeansDenoisingColored(result, None, 6, 6, 7, 21)
        result[inpaint_binary > 0] = denoised[inpaint_binary > 0]

        # Final boundary cleanup
        kernel_micro = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        micro_edge = cv2.dilate(inpaint_binary, kernel_micro) - inpaint_binary
        if np.sum(micro_edge) > 0:
            result = cv2.inpaint(result, micro_edge * 255, 2, cv2.INPAINT_NS)

        final = image.copy()
        final[inpaint_binary > 0] = result[inpaint_binary > 0]

        return final

    def inpaint_seamless_clone(
        self,
        image: np.ndarray,
        inpaint_mask: np.ndarray,
        reference_mask: np.ndarray
    ) -> np.ndarray:
        """Inpaint using OpenCV seamlessClone for natural blending.

        Clones texture from reference region into inpaint region with
        automatic color/lighting adjustment.

        Args:
            image: RGB image (H, W, 3), uint8.
            inpaint_mask: Binary mask of region to fill.
            reference_mask: Binary mask of reference texture region.

        Returns:
            Inpainted image with seamlessly cloned texture.
        """
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        h, w = image.shape[:2]

        # Normalize masks
        inpaint_binary = (inpaint_mask > 127).astype(np.uint8)
        ref_binary = (reference_mask > 127).astype(np.uint8)

        # Get reference region
        ref_coords = np.where(ref_binary > 0)
        if len(ref_coords[0]) == 0:
            return cv2.inpaint(image, inpaint_binary * 255, 5, cv2.INPAINT_NS)

        ref_y_min, ref_y_max = ref_coords[0].min(), ref_coords[0].max()
        ref_x_min, ref_x_max = ref_coords[1].min(), ref_coords[1].max()

        # Extract source patch
        src_patch = image[ref_y_min:ref_y_max+1, ref_x_min:ref_x_max+1].copy()
        src_mask = ref_binary[ref_y_min:ref_y_max+1, ref_x_min:ref_x_max+1] * 255

        # Get inpaint region center
        inpaint_coords = np.where(inpaint_binary > 0)
        if len(inpaint_coords[0]) == 0:
            return image

        center_y = int(np.mean(inpaint_coords[0]))
        center_x = int(np.mean(inpaint_coords[1]))

        # Resize source to fit inpaint region if needed
        inp_y_min, inp_y_max = inpaint_coords[0].min(), inpaint_coords[0].max()
        inp_x_min, inp_x_max = inpaint_coords[1].min(), inpaint_coords[1].max()
        inp_h = inp_y_max - inp_y_min + 1
        inp_w = inp_x_max - inp_x_min + 1

        src_h, src_w = src_patch.shape[:2]

        # Scale source if inpaint region is larger
        if inp_h > src_h or inp_w > src_w:
            scale = max(inp_h / src_h, inp_w / src_w) * 1.2
            new_h, new_w = int(src_h * scale), int(src_w * scale)
            src_patch = cv2.resize(src_patch, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            src_mask = cv2.resize(src_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        try:
            # Apply seamless clone
            result = cv2.seamlessClone(
                src_patch, image, src_mask,
                (center_x, center_y),
                cv2.NORMAL_CLONE
            )
        except cv2.error:
            # Fallback to simple inpaint
            result = cv2.inpaint(image, inpaint_binary * 255, 5, cv2.INPAINT_NS)

        return result

    def inpaint_foreground_removal(
        self,
        image: np.ndarray,
        foreground_mask: np.ndarray,
        background_mask: np.ndarray,
        background_hull_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Remove foreground object and fill with background texture.

        Specialized method for occlusion handling: removes a foreground
        object and fills the region with texture matching the background.

        Args:
            image: RGB image (H, W, 3), uint8.
            foreground_mask: Binary mask of foreground object to remove.
            background_mask: Binary mask of visible background (for texture).
            background_hull_mask: Optional convex hull of background for
                                  estimating occluded extent.

        Returns:
            Image with foreground removed and background extended.
        """
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        result = image.copy()
        h, w = image.shape[:2]

        # Compute background hull if not provided
        if background_hull_mask is None:
            contours, _ = cv2.findContours(
                background_mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                all_pts = np.vstack(contours)
                hull = cv2.convexHull(all_pts)
                background_hull_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(background_hull_mask, [hull], 255)
            else:
                background_hull_mask = background_mask.copy()

        # Region to fill: foreground that overlaps with background hull
        # (this is where foreground occludes background)
        overlap_region = cv2.bitwise_and(
            foreground_mask.astype(np.uint8),
            background_hull_mask.astype(np.uint8)
        )

        # Region outside hull: fill with white/background
        outside_hull = cv2.bitwise_and(
            foreground_mask.astype(np.uint8),
            cv2.bitwise_not(background_hull_mask)
        )

        # Get pure background (not occluded by foreground)
        pure_background = cv2.bitwise_and(
            background_mask.astype(np.uint8),
            cv2.bitwise_not(foreground_mask.astype(np.uint8))
        )

        # Fill overlap region with background texture
        if np.sum(overlap_region) > 0 and np.sum(pure_background) > 0:
            result = self.inpaint_with_reference(
                result,
                overlap_region,
                pure_background,
                noise_factor=0.7
            )

        # Fill outside region with white
        result[outside_hull > 0] = [252, 252, 252]

        # Final boundary cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        outer_boundary = cv2.dilate(outside_hull, kernel) - outside_hull
        if np.sum(outer_boundary) > 0:
            result = cv2.inpaint(result, outer_boundary, 3, cv2.INPAINT_NS)

        return result


def sample_from_boundary_neighborhood(
    image_rgb: np.ndarray,
    occlusion_mask: np.ndarray,
    occludee_visible_mask: np.ndarray,
    max_distance: int = 10
) -> np.ndarray:
    """Sample colors from occludee region near occlusion boundary.

    Uses distance transform to find pixels closest to the occlusion boundary.
    This ensures we get the actual color that should fill the occluded area,
    rather than relying on K-means clustering which may select wrong colors
    for light-colored objects.

    Args:
        image_rgb: RGB image (H, W, 3), uint8.
        occlusion_mask: Binary mask of occlusion region.
        occludee_visible_mask: Binary mask of visible occludee region.
        max_distance: Maximum distance from occlusion boundary to sample.

    Returns:
        Array of sampled pixel colors (N, 3).
    """
    h, w = image_rgb.shape[:2]
    occ_binary = (occlusion_mask > 0).astype(np.uint8)
    visible_binary = (occludee_visible_mask > 0).astype(np.uint8)

    if np.sum(visible_binary) == 0:
        return np.array([])

    # Get occlusion boundary
    occ_contour = np.zeros((h, w), dtype=np.uint8)
    contours, _ = cv2.findContours(occ_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(occ_contour, contours, -1, 255, thickness=1)

    # Distance from each pixel to occlusion boundary
    dist_from_boundary = cv2.distanceTransform(255 - occ_contour, cv2.DIST_L2, 5)

    # Only consider visible occludee pixels
    dist_masked = dist_from_boundary.copy()
    dist_masked[visible_binary == 0] = 9999

    # Find pixels within max_distance of occlusion boundary
    near_boundary = (dist_masked <= max_distance) & (visible_binary > 0)

    if np.sum(near_boundary) < 10:
        # Fallback: expand search radius
        for radius in [20, 30, 50]:
            near_boundary = (dist_masked <= radius) & (visible_binary > 0)
            if np.sum(near_boundary) >= 10:
                break

    if np.sum(near_boundary) < 10:
        # Ultimate fallback: use all visible pixels
        near_boundary = visible_binary > 0

    sampled_pixels = image_rgb[near_boundary]
    return sampled_pixels


def get_intersection_edge(
    occludee_full_mask: np.ndarray,
    occlusion_mask: np.ndarray,
    thickness: int = 2,
    use_erosion_edge: bool = True,
) -> np.ndarray:
    """Get edge where object boundary meets occlusion boundary.

    Args:
        occludee_full_mask: Full mask of occludee object.
        occlusion_mask: Mask of occlusion region.
        thickness: Dilation thickness for intersection tolerance.
        use_erosion_edge: If True, use erosion-based boundary (no 1px offset).
            If False, use legacy contour-based boundary.

    Returns:
        Binary mask of intersection edge.
    """
    h, w = occludee_full_mask.shape
    occ_binary = (occlusion_mask > 0).astype(np.uint8)
    obj_binary = (occludee_full_mask > 0).astype(np.uint8)

    if use_erosion_edge:
        # Erosion-based: boundary sits exactly on the mask edge (no offset)
        erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        obj_edge = obj_binary - cv2.erode(obj_binary, erode_k)
        occ_edge = occ_binary - cv2.erode(occ_binary, erode_k)

        # Apply thickness via dilation
        if thickness > 1:
            thick_k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (thickness, thickness)
            )
            obj_edge = cv2.dilate(obj_edge, thick_k)
            occ_edge = cv2.dilate(occ_edge, thick_k)

        intersection = cv2.bitwise_and(obj_edge * 255, occ_edge * 255)
        return intersection

    # Legacy contour-based (has ~1px outward offset)
    obj_contour = np.zeros((h, w), dtype=np.uint8)
    contours, _ = cv2.findContours(obj_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(obj_contour, contours, -1, 255, thickness=1)

    occ_contour = np.zeros((h, w), dtype=np.uint8)
    contours, _ = cv2.findContours(occ_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(occ_contour, contours, -1, 255, thickness=1)

    kernel = np.ones((thickness, thickness), dtype=np.uint8)
    obj_dilated = cv2.dilate(obj_contour, kernel)
    occ_dilated = cv2.dilate(occ_contour, kernel)

    intersection = cv2.bitwise_and(obj_dilated, occ_dilated)
    return intersection


def sample_edge_colors_from_visible(
    image_rgb: np.ndarray,
    intersection_edge: np.ndarray,
    occludee_visible_mask: np.ndarray,
    max_search_distance: int = 30,
    edge_threshold_low: int = 50,
    edge_threshold_high: int = 150,
    smooth_edges: bool = True,
    smooth_kernel_size: int = 5,
    occluder_mask: np.ndarray = None,  # V24: for safe color sampling
    min_safe_distance: int = 3  # V24: min distance from occluder for color sampling
) -> np.ndarray:
    """Sample edge colors from nearby visible edges (V23/V24 algorithm).

    Instead of using a fixed darkness factor, this samples actual edge colors
    from the visible portion of the occludee object, making the intersection
    edge look more natural and consistent with the object's appearance.

    V24 improvement: When occluder_mask is provided, avoids sampling colors
    from edges too close to the occluder boundary (which may be contaminated
    by synthetic occlusion colors in test scenarios).

    Args:
        image_rgb: RGB image (H, W, 3), uint8.
        intersection_edge: Binary mask of intersection edge to fill.
        occludee_visible_mask: Visible portion of occludee object.
        max_search_distance: Max distance to search for visible edges.
        edge_threshold_low: Canny low threshold.
        edge_threshold_high: Canny high threshold.
        smooth_edges: Whether to apply smoothing to sampled colors.
        smooth_kernel_size: Kernel size for median smoothing.
        occluder_mask: Binary mask of occluder (for V24 safe sampling).
        min_safe_distance: Minimum distance from occluder for safe color sampling.

    Returns:
        Edge color image (H, W, 3) with sampled colors for intersection edge pixels.
    """
    h, w = image_rgb.shape[:2]
    visible_binary = (occludee_visible_mask > 0).astype(np.uint8)
    edge_binary = (intersection_edge > 0).astype(np.uint8)

    if np.sum(edge_binary) == 0:
        return image_rgb.copy()

    # Step 1: Detect edges in the visible region using Canny
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    edges_canny = cv2.Canny(gray, edge_threshold_low, edge_threshold_high)

    # Only keep edges within visible region
    visible_edges = cv2.bitwise_and(edges_canny, visible_binary * 255)

    # Step 2: Also detect edges using gradient magnitude (backup)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobelx**2 + sobely**2)
    gradient_edges = (gradient_mag > np.percentile(gradient_mag[visible_binary > 0], 80)).astype(np.uint8) * 255
    gradient_edges = cv2.bitwise_and(gradient_edges, visible_binary * 255)

    # Combine Canny and gradient edges
    combined_edges = cv2.bitwise_or(visible_edges, gradient_edges)

    # =========================================================================
    # V24: Compute distance from occluder for safe color sampling
    # =========================================================================
    dist_from_occluder = None
    if occluder_mask is not None:
        occluder_binary = (occluder_mask > 0).astype(np.uint8)
        # Distance transform: distance from each pixel to nearest occluder pixel
        dist_from_occluder = cv2.distanceTransform(
            1 - occluder_binary, cv2.DIST_L2, 3
        )

    # =========================================================================
    # BACKUP: Alternative approaches (commented out for reference)
    # =========================================================================
    # --- Alternative 1: Exclude occluder contour from edges ---
    # if occluder_mask is not None:
    #     occluder_binary = (occluder_mask > 0).astype(np.uint8)
    #     occluder_contour = cv2.Canny(occluder_binary * 255, 50, 150)
    #     # Dilate contour slightly to ensure coverage
    #     kernel = np.ones((3, 3), dtype=np.uint8)
    #     occluder_contour = cv2.dilate(occluder_contour, kernel, iterations=1)
    #     combined_edges = cv2.bitwise_and(combined_edges, cv2.bitwise_not(occluder_contour))
    #
    # --- Alternative 2: Blur occluder region before edge detection ---
    # if occluder_mask is not None:
    #     blurred = cv2.GaussianBlur(image_rgb, (31, 31), 0)
    #     image_for_edge = image_rgb.copy()
    #     image_for_edge[occluder_mask > 0] = blurred[occluder_mask > 0]
    #     gray = cv2.cvtColor(image_for_edge, cv2.COLOR_RGB2GRAY)
    #     edges_canny = cv2.Canny(gray, edge_threshold_low, edge_threshold_high)
    # =========================================================================

    # If no edges found, fall back to boundary of visible region
    if np.sum(combined_edges) < 10:
        kernel = np.ones((3, 3), dtype=np.uint8)
        dilated = cv2.dilate(visible_binary, kernel)
        combined_edges = ((dilated - visible_binary) * 255).astype(np.uint8)

    # Step 3: For each intersection edge pixel, find nearest visible edge
    edge_coords = np.where(edge_binary > 0)
    visible_edge_coords = np.where(combined_edges > 0)

    # Minimum visible edge pixels for reliable sampling
    min_visible_edges = 20
    if len(visible_edge_coords[0]) < min_visible_edges:
        # Not enough visible edges for reliable sampling
        # Return None to signal fallback to V22
        return None

    # Build KD-tree for fast nearest neighbor search
    from scipy.spatial import cKDTree
    visible_edge_points = np.column_stack((visible_edge_coords[0], visible_edge_coords[1]))
    tree = cKDTree(visible_edge_points)

    # Query nearest visible edge for each intersection edge pixel
    # Use k=10 for more robust color averaging (reduces outlier impact)
    query_points = np.column_stack((edge_coords[0], edge_coords[1]))
    k_neighbors = min(10, len(visible_edge_points))
    distances, indices = tree.query(query_points, k=k_neighbors)

    # Step 4: Sample colors from nearest visible edges with distance weighting
    result = np.zeros_like(image_rgb)

    for i, (y, x) in enumerate(zip(edge_coords[0], edge_coords[1])):
        if np.isscalar(indices[i]):
            idx_list = [indices[i]]
            dist_list = [distances[i]]
        else:
            idx_list = indices[i]
            dist_list = distances[i]

        # Filter by max distance
        valid_mask = np.array(dist_list) < max_search_distance
        if not np.any(valid_mask):
            # All too far, use closest anyway
            valid_mask[0] = True

        valid_indices = np.array(idx_list)[valid_mask]
        valid_distances = np.array(dist_list)[valid_mask]

        # V24: Filter out edges too close to occluder (contaminated colors)
        if dist_from_occluder is not None:
            safe_mask = []
            for idx in valid_indices:
                vy, vx = visible_edge_points[idx]
                safe_mask.append(dist_from_occluder[vy, vx] >= min_safe_distance)
            safe_mask = np.array(safe_mask)

            if np.any(safe_mask):
                valid_indices = valid_indices[safe_mask]
                valid_distances = valid_distances[safe_mask]

        # Distance-weighted color sampling
        weights = 1.0 / (valid_distances + 1.0)
        weights = weights / np.sum(weights)

        sampled_color = np.zeros(3, dtype=np.float32)
        for idx, weight in zip(valid_indices, weights):
            vy, vx = visible_edge_points[idx]
            sampled_color += image_rgb[vy, vx].astype(np.float32) * weight

        result[y, x] = np.clip(sampled_color, 0, 255).astype(np.uint8)

    # Step 5: Apply smoothing to reduce color variation (improves SSIM)
    if smooth_edges and smooth_kernel_size > 0:
        # Make kernel size odd
        ksize = smooth_kernel_size if smooth_kernel_size % 2 == 1 else smooth_kernel_size + 1

        # Use median filter instead of Gaussian for outlier robustness
        if ksize >= 3:
            smoothed = cv2.medianBlur(result, ksize)
            result[edge_binary > 0] = smoothed[edge_binary > 0]

    return result


def measure_edge_width_map(
    image_rgb: np.ndarray,
    visible_mask: np.ndarray,
    edge_threshold_low: int = 50,
    edge_threshold_high: int = 150,
    min_edge_width: int = 1,
    max_edge_width: int = 8,
    width_smoothing_sigma: float = 1.5,
) -> tuple:
    """Compute edge skeleton and width map from visible region.

    V25 Dynamic Edge Morphology: Extracts edge skeleton and measures
    local thickness at each skeleton pixel using distance transform.

    Args:
        image_rgb: RGB image (H, W, 3), uint8.
        visible_mask: Binary mask of visible region.
        edge_threshold_low: Canny low threshold.
        edge_threshold_high: Canny high threshold.
        min_edge_width: Minimum edge width in pixels.
        max_edge_width: Maximum edge width in pixels (clipping).
        width_smoothing_sigma: Gaussian sigma for width map smoothing.

    Returns:
        tuple: (skeleton, width_map, edge_mask)
            - skeleton: (H, W) uint8 - edge centerline binary mask
            - width_map: (H, W) float32 - local thickness at skeleton pixels
            - edge_mask: (H, W) uint8 - original edge binary mask
    """
    # Step 1: Edge detection (Canny + Gradient)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    canny = cv2.Canny(gray, edge_threshold_low, edge_threshold_high)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)

    visible_pixels = visible_mask > 0
    if np.sum(visible_pixels) == 0:
        h, w = image_rgb.shape[:2]
        return (np.zeros((h, w), dtype=np.uint8),
                np.zeros((h, w), dtype=np.float32),
                np.zeros((h, w), dtype=np.uint8))

    grad_edges = (grad_mag > np.percentile(
        grad_mag[visible_pixels], 80
    )).astype(np.uint8) * 255

    edge_mask = cv2.bitwise_or(canny, grad_edges)
    edge_mask = cv2.bitwise_and(edge_mask, (visible_mask > 0).astype(np.uint8) * 255)

    # Step 2: Morphological closing to connect fragmented edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edge_mask_closed = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, kernel)

    # Step 3: Distance transform - distance from each edge pixel to edge boundary
    edge_binary = (edge_mask_closed > 0).astype(np.uint8)
    dist_inside = cv2.distanceTransform(edge_binary, cv2.DIST_L2, 3)
    thickness_field = dist_inside * 2.0  # both sides = full thickness

    # Step 4: Skeleton extraction
    try:
        skeleton = cv2.ximgproc.thinning(
            edge_mask_closed, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN
        )
    except AttributeError:
        # Fallback: skimage.morphology.skeletonize
        from skimage.morphology import skeletonize
        skeleton = (skeletonize(edge_binary.astype(bool)) * 255).astype(np.uint8)

    # Step 5: Width map - store thickness only at skeleton pixels
    width_map = np.zeros_like(thickness_field, dtype=np.float32)
    skel_coords = np.where(skeleton > 0)
    if len(skel_coords[0]) > 0:
        width_map[skel_coords] = thickness_field[skel_coords]

        # Step 6: Clipping
        width_map = np.clip(width_map, 0, max_edge_width)
        width_map[width_map > 0] = np.clip(
            width_map[width_map > 0], min_edge_width, max_edge_width
        )

        # Step 7: Smoothing for thickness continuity
        if width_smoothing_sigma > 0:
            width_smooth = cv2.GaussianBlur(width_map, (0, 0), width_smoothing_sigma)
            width_map = np.where(skeleton > 0, width_smooth, 0).astype(np.float32)
            # Re-clip after smoothing (blur can push values below min)
            width_map[width_map > 0] = np.clip(
                width_map[width_map > 0], min_edge_width, max_edge_width
            )

    return skeleton, width_map, edge_mask_closed


def extract_edge_color_profile(
    image_rgb: np.ndarray,
    skeleton: np.ndarray,
    width_map: np.ndarray,
    n_profile_samples: int = 5,
    bilateral_sampling: bool = True,
) -> dict:
    """Extract per-skeleton-pixel color profile along normal direction.

    V25: Samples center and boundary colors along the edge normal,
    capturing the ink gradient from dark center to lighter boundary.

    Args:
        image_rgb: RGB image (H, W, 3), uint8.
        skeleton: (H, W) uint8 - edge skeleton binary mask.
        width_map: (H, W) float32 - local thickness at skeleton pixels.
        n_profile_samples: Number of samples along normal (unused currently, reserved).

    Returns:
        dict with keys: 'coords', 'widths', 'center_colors', 'edge_colors', 'orientations'
        or None if no skeleton pixels found.
    """
    skel_coords = np.column_stack(np.where(skeleton > 0))
    if len(skel_coords) == 0:
        return None

    h, w = image_rgb.shape[:2]

    # Edge orientation via Sobel gradient direction
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)

    widths = width_map[skel_coords[:, 0], skel_coords[:, 1]]
    center_colors = image_rgb[skel_coords[:, 0], skel_coords[:, 1]].astype(np.float32)

    # Normal direction = gradient direction (perpendicular to edge tangent)
    normals_y = gy[skel_coords[:, 0], skel_coords[:, 1]]
    normals_x = gx[skel_coords[:, 0], skel_coords[:, 1]]
    norms = np.sqrt(normals_y**2 + normals_x**2) + 1e-8
    normals_y = normals_y / norms
    normals_x = normals_x / norms

    # Boundary colors sampling
    edge_colors = np.zeros_like(center_colors)
    half_w = widths / 2.0

    if bilateral_sampling:
        # V25.1: Sample both sides, pick lighter (boundary) color
        # Positive direction
        ey_pos = np.clip(skel_coords[:, 0] + (normals_y * half_w), 0, h - 1).astype(int)
        ex_pos = np.clip(skel_coords[:, 1] + (normals_x * half_w), 0, w - 1).astype(int)
        color_pos = image_rgb[ey_pos, ex_pos].astype(np.float32)
        # Negative direction
        ey_neg = np.clip(skel_coords[:, 0] - (normals_y * half_w), 0, h - 1).astype(int)
        ex_neg = np.clip(skel_coords[:, 1] - (normals_x * half_w), 0, w - 1).astype(int)
        color_neg = image_rgb[ey_neg, ex_neg].astype(np.float32)
        # Pick lighter color (boundary is typically lighter than center)
        brightness_pos = np.mean(color_pos, axis=1)
        brightness_neg = np.mean(color_neg, axis=1)
        use_pos = brightness_pos >= brightness_neg
        edge_colors[use_pos] = color_pos[use_pos]
        edge_colors[~use_pos] = color_neg[~use_pos]
    else:
        # Original: sample positive direction only
        ey = np.clip(skel_coords[:, 0] + (normals_y * half_w), 0, h - 1).astype(int)
        ex = np.clip(skel_coords[:, 1] + (normals_x * half_w), 0, w - 1).astype(int)
        edge_colors = image_rgb[ey, ex].astype(np.float32)

    return {
        'coords': skel_coords,
        'widths': widths,
        'center_colors': center_colors.astype(np.uint8),
        'edge_colors': edge_colors.astype(np.uint8),
        'orientations': np.column_stack((normals_y, normals_x)),
    }


def render_dynamic_intersection_edge(
    image_rgb: np.ndarray,
    intersection_centerline: np.ndarray,
    edge_profile: dict,
    occluder_mask: np.ndarray = None,
    min_safe_distance: int = 3,
    k_neighbors: int = 10,
    max_search_distance: int = 30,
    # V26: Adaptive Kernel Smoothing
    _v26_smoothstep_gradient: bool = False,
) -> tuple:
    """Render variable-thickness intersection edge with color gradient.

    V25: For each intersection centerline pixel, finds nearest visible
    skeleton matches via KD-tree, computes weighted average thickness
    and center/edge colors, then renders with radial gradient.

    Args:
        image_rgb: RGB image (H, W, 3), uint8.
        intersection_centerline: (H, W) binary mask of intersection centerline.
        edge_profile: Dict from extract_edge_color_profile().
        occluder_mask: Binary mask of occluder (for V24 safe filtering).
        min_safe_distance: Min distance from occluder for safe sampling.
        k_neighbors: Number of nearest skeleton neighbors to consider.
        max_search_distance: Max distance for neighbor search.

    Returns:
        tuple: (rendered_edge, edge_mask)
            - rendered_edge: (H, W, 3) rendered edge colors
            - edge_mask: (H, W) uint8 variable-thickness edge binary mask
    """
    from scipy.spatial import cKDTree

    h, w = image_rgb.shape[:2]
    rendered_edge = np.zeros_like(image_rgb)
    edge_mask = np.zeros((h, w), dtype=np.uint8)

    cl_coords = np.column_stack(np.where(intersection_centerline > 0))
    if len(cl_coords) == 0 or edge_profile is None:
        return rendered_edge, edge_mask

    # V24: Distance from occluder for safe filtering
    dist_from_occluder = None
    if occluder_mask is not None:
        occ_binary = (occluder_mask > 0).astype(np.uint8)
        dist_from_occluder = cv2.distanceTransform(1 - occ_binary, cv2.DIST_L2, 3)

    # KD-tree for visible skeleton matching
    tree = cKDTree(edge_profile['coords'])
    k = min(k_neighbors, len(edge_profile['coords']))
    distances, indices = tree.query(cl_coords, k=k)

    # Ensure 2D arrays even when k=1
    if k == 1:
        distances = distances.reshape(-1, 1)
        indices = indices.reshape(-1, 1)

    for i, (y, x) in enumerate(cl_coords):
        idxs = indices[i]
        dists = distances[i]

        # Distance filtering
        valid = dists < max_search_distance
        if not valid.any():
            valid[0] = True

        idxs_v = idxs[valid]
        dists_v = dists[valid]

        # V24 Safe filter
        if dist_from_occluder is not None:
            safe = np.array([
                dist_from_occluder[
                    edge_profile['coords'][idx][0],
                    edge_profile['coords'][idx][1]
                ] >= min_safe_distance
                for idx in idxs_v
            ])
            if safe.any():
                idxs_v = idxs_v[safe]
                dists_v = dists_v[safe]

        # Distance weights
        weights = 1.0 / (dists_v + 1.0)
        weights = weights / weights.sum()

        # Weighted average thickness & colors
        local_width = np.sum(edge_profile['widths'][idxs_v] * weights)
        local_center = np.sum(
            edge_profile['center_colors'][idxs_v].astype(np.float32) *
            weights[:, None], axis=0
        )
        local_edge = np.sum(
            edge_profile['edge_colors'][idxs_v].astype(np.float32) *
            weights[:, None], axis=0
        )

        # Render with radial gradient (center → edge color transition)
        radius = max(1, int(round(local_width / 2.0)))

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    d = np.sqrt(dy * dy + dx * dx)
                    if d <= radius:
                        t = d / radius  # 0=center, 1=edge
                        # V26 Imp1: Hermite smoothstep (smoother center/edge transitions)
                        # Note: radius >= 2 needed (radius=1 is mathematically identical to linear)
                        if _v26_smoothstep_gradient and radius >= 2:
                            t = t * t * (3.0 - 2.0 * t)
                        color = local_center * (1 - t) + local_edge * t
                        rendered_edge[ny, nx] = np.clip(color, 0, 255).astype(np.uint8)
                        edge_mask[ny, nx] = 255

    return rendered_edge, edge_mask


def inpaint_occlusion_boundary_guided(
    image_rgb: np.ndarray,
    occlusion_mask: np.ndarray,
    occluder_mask: np.ndarray,
    occludee_full_mask: np.ndarray,
    occludee_visible_mask: np.ndarray,
    edge_darkness: float = 0.3,
    max_sample_distance: int = 10,
    # V25: Dynamic Edge Morphology parameters
    use_dynamic_edge: bool = True,
    min_edge_width: int = 1,
    max_edge_width: int = 8,
    width_smoothing_sigma: float = 1.5,
    min_safe_distance: int = 3,
    use_erosion_edge: bool = True,
    # V25.1 improvement flags (for ablation testing)
    _use_weighted_body: bool = False,    # was True, reverted (hurts -0.64 dB)
    _use_bilateral_color: bool = True,   # keep (helps +0.02 dB)
    _skip_edge_darkness: bool = False,   # was True, reverted (hurts -0.05 dB)
    _protect_edge_smooth: bool = False,  # was True, reverted (hurts -0.16 dB)
    # V26: Adaptive Kernel Smoothing flags (for ablation testing)
    _v26_smoothstep_gradient: bool = False,   # Imp1: Hermite smoothstep for edge gradient
    _v26_adaptive_smooth: bool = False,       # Imp2: width-based boundary smoothing divisor
    _v26_adaptive_aa: bool = False,           # Imp3: adaptive AA kernel (3x3 or 5x5)
    _v26_feathered_transition: bool = False,  # Imp4: edge-body feathered blending
) -> np.ndarray:
    """Inpaint occlusion region using boundary neighborhood sampling.

    V21 algorithm: Samples colors from the occludee region closest to the
    occlusion boundary, ensuring accurate color matching for both dark and
    light colored objects.

    V25 upgrade: When use_dynamic_edge=True, uses skeleton-based width
    measurement and color profile extraction for variable-thickness edge
    rendering with center-to-boundary color gradients.

    Args:
        image_rgb: RGB image (H, W, 3), uint8.
        occlusion_mask: Binary mask of occlusion region to fill.
        occluder_mask: Binary mask of occluder object.
        occludee_full_mask: Full mask of occludee object (including occluded).
        occludee_visible_mask: Visible portion of occludee.
        edge_darkness: Darkness factor for edge (0-1, lower = darker).
        max_sample_distance: Max distance from boundary to sample colors.
        use_dynamic_edge: Enable V25 dynamic edge morphology.
        min_edge_width: Minimum edge width in pixels (V25).
        max_edge_width: Maximum edge width in pixels (V25).
        width_smoothing_sigma: Gaussian sigma for width map smoothing (V25).
        min_safe_distance: Min distance from occluder for safe sampling (V24/V25).
        _v26_smoothstep_gradient: V26 Imp1 - Hermite smoothstep for wide edge gradient.
        _v26_adaptive_smooth: V26 Imp2 - width-adaptive boundary smoothing divisor.
        _v26_adaptive_aa: V26 Imp3 - adaptive AA kernel size (3x3 or 5x5).
        _v26_feathered_transition: V26 Imp4 - feathered edge-body transition zone.

    Returns:
        Inpainted image (H, W, 3), uint8.
    """
    h, w = image_rgb.shape[:2]
    occ_binary = (occlusion_mask > 0).astype(np.uint8)
    occluder_binary = (occluder_mask > 0).astype(np.uint8)

    if np.sum(occ_binary) == 0:
        return image_rgb.copy()

    # Sample colors from boundary neighborhood
    boundary_pixels = sample_from_boundary_neighborhood(
        image_rgb, occlusion_mask, occludee_visible_mask, max_sample_distance
    )

    if len(boundary_pixels) == 0:
        return image_rgb.copy()

    result = image_rgb.copy()
    avg = np.mean(boundary_pixels, axis=0).astype(np.uint8)

    # Blank occluder with average color
    result[occluder_binary > 0] = avg

    # Fill occlusion region
    occ_coords = np.where(occ_binary > 0)
    if len(occ_coords[0]) > 0:
        if _use_weighted_body:
            # V25.1: Distance-weighted nearest-neighbor sampling
            from scipy.spatial import cKDTree as _cKDTree

            # Get boundary pixel coordinates (visible occludee near occlusion)
            visible_binary = (occludee_visible_mask > 0).astype(np.uint8)
            occ_contour = np.zeros((h, w), dtype=np.uint8)
            _contours, _ = cv2.findContours(occ_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(occ_contour, _contours, -1, 255, thickness=1)
            dist_to_boundary = cv2.distanceTransform(255 - occ_contour, cv2.DIST_L2, 5)
            near_visible = (visible_binary > 0) & (dist_to_boundary <= max_sample_distance * 3)
            if np.sum(near_visible) < 10:
                near_visible = visible_binary > 0

            boundary_yx = np.column_stack(np.where(near_visible))
            boundary_colors = image_rgb[near_visible]

            if len(boundary_yx) > 0:
                body_tree = _cKDTree(boundary_yx)
                occ_yx = np.column_stack([occ_coords[0], occ_coords[1]])
                k_body = min(5, len(boundary_yx))
                body_dists, body_idxs = body_tree.query(occ_yx, k=k_body)
                if k_body == 1:
                    body_dists = body_dists.reshape(-1, 1)
                    body_idxs = body_idxs.reshape(-1, 1)
                body_weights = 1.0 / (body_dists + 1.0)
                body_weights = body_weights / body_weights.sum(axis=1, keepdims=True)
                # Weighted average of k-nearest colors
                filled_colors = np.zeros((len(occ_yx), 3), dtype=np.float32)
                for ki in range(k_body):
                    filled_colors += boundary_colors[body_idxs[:, ki]].astype(np.float32) * body_weights[:, ki:ki+1]
                result[occ_coords[0], occ_coords[1]] = np.clip(filled_colors, 0, 255).astype(np.uint8)
            else:
                # Fallback: random sampling
                indices = np.random.choice(len(boundary_pixels), len(occ_coords[0]), replace=True)
                result[occ_coords[0], occ_coords[1]] = boundary_pixels[indices]
        else:
            # Original: random sampling
            indices = np.random.choice(len(boundary_pixels), len(occ_coords[0]), replace=True)
            result[occ_coords[0], occ_coords[1]] = boundary_pixels[indices]

        # Blur body fill
        if _use_weighted_body:
            # V25.1: Adaptive blur kernel based on occlusion area
            occ_area = np.sum(occ_binary)
            occ_diameter = int(np.sqrt(occ_area) * 0.5)
            kernel_size = max(3, min(15, occ_diameter // 2))
            kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
            blurred = cv2.GaussianBlur(result, (kernel_size, kernel_size), 0)
        else:
            # Original: fixed 9x9 blur
            blurred = cv2.GaussianBlur(result, (9, 9), 0)
        result[occ_binary > 0] = blurred[occ_binary > 0]

        # Add texture noise
        std = np.std(boundary_pixels, axis=0)
        if _use_weighted_body:
            # V25.1: Adaptive noise scale
            texture_scale = np.clip(np.mean(std) / 50.0, 0.1, 0.5)
            noise = np.random.randn(len(occ_coords[0]), 3) * std * texture_scale
        else:
            # Original: fixed 0.25 noise scale
            noise = np.random.randn(len(occ_coords[0]), 3) * std * 0.25
        current = result[occ_coords[0], occ_coords[1]].astype(np.float32)
        result[occ_coords[0], occ_coords[1]] = np.clip(current + noise, 0, 255).astype(np.uint8)

    # =========================================================================
    # V25: Dynamic Edge Morphology
    # =========================================================================
    v25_applied = False
    edge_profile = None  # V26: initialized for adaptive smooth access
    if use_dynamic_edge:
        # 1. Extract intersection centerline (thickness=1)
        centerline = get_intersection_edge(
            occludee_full_mask, occlusion_mask, thickness=1, use_erosion_edge=use_erosion_edge
        )

        if np.sum(centerline > 0) > 0:
            # 2. Measure visible edge widths
            skeleton, width_map, _ = measure_edge_width_map(
                image_rgb, occludee_visible_mask,
                min_edge_width=min_edge_width,
                max_edge_width=max_edge_width,
                width_smoothing_sigma=width_smoothing_sigma,
            )

            # 3. Extract color profile
            edge_profile = extract_edge_color_profile(
                image_rgb, skeleton, width_map,
                bilateral_sampling=_use_bilateral_color,
            )

            # 4. Render variable-thickness edge with gradient colors
            if edge_profile is not None and len(edge_profile['coords']) > 0:
                rendered, edge_mask = render_dynamic_intersection_edge(
                    image_rgb, centerline, edge_profile,
                    occluder_mask=occluder_mask,
                    min_safe_distance=min_safe_distance,
                    _v26_smoothstep_gradient=_v26_smoothstep_gradient,
                )

                # Apply rendered edge to result
                if np.sum(edge_mask > 0) > 0:
                    if _skip_edge_darkness:
                        # V25.1: Use extracted color profile directly (no darkness adjustment)
                        result[edge_mask > 0] = rendered[edge_mask > 0]
                    else:
                        # Original: Apply darkness adjustment
                        if edge_darkness < 1.0:
                            edge_pixels = np.where(edge_mask > 0)
                            colors = rendered[edge_pixels[0], edge_pixels[1]].astype(np.float32)
                            factor = edge_darkness * (2 - edge_darkness)
                            colors = colors * factor
                            result[edge_pixels[0], edge_pixels[1]] = np.clip(colors, 0, 255).astype(np.uint8)
                        else:
                            result[edge_mask > 0] = rendered[edge_mask > 0]

                    # V26 Imp4: Feathered transition zone (edge → body)
                    if _v26_feathered_transition:
                        # Dilate edge_mask by 2px to find transition zone
                        feather_kernel = np.ones((5, 5), dtype=np.uint8)  # ~2px radius
                        feather_dilated = cv2.dilate(edge_mask, feather_kernel)
                        feather_zone = (feather_dilated > 0) & (edge_mask == 0) & (occ_binary > 0)
                        feather_coords = np.where(feather_zone)
                        if len(feather_coords[0]) > 0:
                            # Distance from edge_mask boundary (max 2px)
                            edge_inv = (255 - edge_mask).astype(np.uint8)
                            feather_dist = cv2.distanceTransform(edge_inv, cv2.DIST_L2, 3)
                            fd = feather_dist[feather_coords[0], feather_coords[1]]
                            # Blend weight: 0 at edge boundary, 1 at 2px away
                            blend_w = np.clip(fd / 2.0, 0, 1)
                            body_color = result[feather_coords[0], feather_coords[1]].astype(np.float32)
                            edge_border_color = rendered[feather_coords[0], feather_coords[1]].astype(np.float32)
                            feathered = body_color * blend_w[:, None] + edge_border_color * (1 - blend_w[:, None])
                            result[feather_coords[0], feather_coords[1]] = np.clip(
                                feathered, 0, 255
                            ).astype(np.uint8)

                    # Anti-alias edge boundary (vectorized)
                    # V26 Imp3: Adaptive AA kernel based on edge width
                    if _v26_adaptive_aa:
                        edge_dt = cv2.distanceTransform(edge_mask, cv2.DIST_L2, 3)
                        max_edge_dt = np.max(edge_dt) if np.max(edge_dt) > 0 else 0
                        aa_size = 5 if max_edge_dt >= 2.5 else 3
                    else:
                        aa_size = 3
                    aa_kernel = np.ones((aa_size, aa_size), dtype=np.uint8)
                    dilated = cv2.dilate(edge_mask, aa_kernel)
                    boundary = dilated - edge_mask
                    boundary_coords = np.where(boundary > 0)
                    if len(boundary_coords[0]) > 0:
                        curr = result[boundary_coords[0], boundary_coords[1]].astype(np.float32)
                        edge_color = rendered[boundary_coords[0], boundary_coords[1]].astype(np.float32)
                        if not _skip_edge_darkness and edge_darkness < 1.0:
                            edge_color = edge_color * (0.5 + edge_darkness * 0.5)
                        blended = curr * 0.5 + edge_color * 0.5
                        result[boundary_coords[0], boundary_coords[1]] = np.clip(
                            blended, 0, 255
                        ).astype(np.uint8)

                    v25_applied = True

    # =========================================================================
    # Fallback: V24 fixed-thickness edge (when V25 not applied)
    # =========================================================================
    if not v25_applied:
        edge_mask = get_intersection_edge(occludee_full_mask, occlusion_mask, thickness=2, use_erosion_edge=use_erosion_edge)

        if np.sum(edge_mask > 0) > 0:
            sampled_edge_colors = sample_edge_colors_from_visible(
                image_rgb, edge_mask, occludee_visible_mask,
                max_search_distance=30,
                occluder_mask=occluder_mask,
                min_safe_distance=min_safe_distance,
            )

            if sampled_edge_colors is not None:
                edge_coords = np.where(edge_mask > 0)

                # Vectorized edge application with corrected darkness
                sampled = sampled_edge_colors[edge_coords[0], edge_coords[1]].astype(np.float32)
                if edge_darkness < 1.0:
                    sampled = sampled * (1 - edge_darkness)  # Blend toward black
                result[edge_coords[0], edge_coords[1]] = np.clip(sampled, 0, 255).astype(np.uint8)

                # Anti-alias boundary (vectorized)
                kernel = np.ones((3, 3), dtype=np.uint8)
                dilated = cv2.dilate(edge_mask, kernel)
                boundary = dilated - edge_mask
                boundary_coords = np.where(boundary > 0)
                if len(boundary_coords[0]) > 0:
                    current = result[boundary_coords[0], boundary_coords[1]].astype(np.float32)
                    nearest = sampled_edge_colors[boundary_coords[0], boundary_coords[1]].astype(np.float32)
                    if edge_darkness < 1.0:
                        nearest = nearest * (0.5 + edge_darkness * 0.5)
                    blended = current * 0.5 + nearest * 0.5
                    result[boundary_coords[0], boundary_coords[1]] = np.clip(
                        blended, 0, 255
                    ).astype(np.uint8)

    # Boundary smoothing with distance transform
    dist = cv2.distanceTransform(occ_binary, cv2.DIST_L2, 5)
    # V26 Imp2: Adaptive divisor based on nearest edge width
    if _v26_adaptive_smooth and v25_applied and edge_profile is not None and len(edge_profile['coords']) > 0:
        from scipy.spatial import cKDTree as _cKDTree_v26
        edge_coords_yx = np.array(edge_profile['coords'])  # (N, 2) in (y, x)
        edge_widths = np.array(edge_profile['widths'])      # (N,)
        occ_yx = np.column_stack(np.where(occ_binary > 0))  # (M, 2)
        if len(occ_yx) > 0 and len(edge_coords_yx) > 0:
            tree_v26 = _cKDTree_v26(edge_coords_yx)
            _, nearest_idx = tree_v26.query(occ_yx, k=1)
            nearest_widths = edge_widths[nearest_idx]
            # divisor: thick edge → wider transition (up to 4.0), thin → narrow (down to 1.5)
            # Safe: clip ensures divisor >= 1.5, no division-by-zero risk
            divisor_vals = np.clip(nearest_widths / 2.0, 1.5, 4.0)
            divisor_map = np.full((h, w), 2.0, dtype=np.float32)  # default = 2 (V25 behavior)
            divisor_map[occ_yx[:, 0], occ_yx[:, 1]] = divisor_vals
            alpha = np.clip(dist / divisor_map, 0, 1)
        else:
            alpha = np.clip(dist / 2, 0, 1)
    else:
        alpha = np.clip(dist / 2, 0, 1)
    if v25_applied and _protect_edge_smooth:
        # V25.1: Exclude edge pixels from smoothing to preserve V25 rendering
        smooth_mask = (occ_binary > 0) & (edge_mask == 0)
    else:
        smooth_mask = occ_binary > 0
    smooth_coords = np.where(smooth_mask)
    if len(smooth_coords[0]) > 0:
        a = alpha[smooth_coords[0], smooth_coords[1]]
        current = result[smooth_coords[0], smooth_coords[1]].astype(np.float32)
        result[smooth_coords[0], smooth_coords[1]] = np.clip(
            avg * (1 - a)[:, None] + current * a[:, None], 0, 255
        ).astype(np.uint8)

    # Final: apply only to occlusion region
    final = image_rgb.copy()
    final[occ_binary > 0] = result[occ_binary > 0]

    return final


def inpaint_occlusion_patchmatch_guided(
    image_rgb: np.ndarray,
    occlusion_mask: np.ndarray,
    occluder_mask: np.ndarray,
    occludee_full_mask: np.ndarray,
    occludee_visible_mask: np.ndarray,
    edge_darkness: float = 0.3,
    patch_size: int = 7,
    iterations: int = 5,
    max_exemplar_distance: int = 0,  # 0 = auto
    use_edge_sampling: bool = True,  # V23: sample edge colors from visible edges
    edge_smooth_kernel: int = 5,  # V23: smoothing kernel size for SSIM improvement
    # Ablation toggle flags (all default False = production behavior unchanged)
    skip_color_filter: bool = False,  # Skip all color outlier filters (Filter 1/2/2.5/3)
    skip_outside_in: bool = False,  # Use random fill order instead of distance-based outside-in
    skip_post_processing: bool = False,  # Skip post-processing (median + bilateral + gaussian)
) -> np.ndarray:
    """Inpaint occlusion region using PatchMatch with boundary guidance.

    V22 algorithm: Uses PatchMatch to propagate texture patches from the
    visible occludee region near the occlusion boundary, maintaining spatial
    coherence and texture patterns while avoiding distant region contamination.
    Falls back to boundary_guided if visible region is too small.

    Args:
        image_rgb: RGB image (H, W, 3), uint8.
        occlusion_mask: Binary mask of occlusion region to fill.
        occluder_mask: Binary mask of occluder object.
        occludee_full_mask: Full mask of occludee object (including occluded).
        occludee_visible_mask: Visible portion of occludee.
        edge_darkness: Darkness factor for edge (0-1, lower = darker).
        patch_size: Size of patches for PatchMatch (odd number).
        iterations: Number of PatchMatch iterations.
        max_exemplar_distance: Max distance from occlusion boundary to sample
            exemplar patches (0 = auto based on image size).

    Returns:
        Inpainted image (H, W, 3), uint8.
    """
    h, w = image_rgb.shape[:2]
    occ_binary = (occlusion_mask > 0).astype(np.uint8)
    occluder_binary = (occluder_mask > 0).astype(np.uint8)
    visible_binary = (occludee_visible_mask > 0).astype(np.uint8)

    if np.sum(occ_binary) == 0:
        return image_rgb.copy()

    # Auto-adjust parameters based on image size
    img_size = max(h, w)
    if max_exemplar_distance == 0:
        # Scale exemplar distance with image size
        # Small images (<200px): use 15-20px
        # Large images (>500px): use 30-50px
        if img_size < 200:
            max_exemplar_distance = 15
        elif img_size < 400:
            max_exemplar_distance = 25
        else:
            max_exemplar_distance = 40

    # Also adjust patch size for small images
    if img_size < 150 and patch_size > 5:
        patch_size = 5

    # Check if we have enough visible region for PatchMatch
    min_visible_pixels = patch_size * patch_size * 10  # At least 10 patches worth
    if np.sum(visible_binary) < min_visible_pixels:
        # Fallback to original boundary_guided
        return inpaint_occlusion_boundary_guided(
            image_rgb, occlusion_mask, occluder_mask,
            occludee_full_mask, occludee_visible_mask, edge_darkness
        )

    # Get intersection edge
    edge_mask = get_intersection_edge(occludee_full_mask, occlusion_mask, thickness=2)

    # Sample colors for average (used for edge and fallback)
    boundary_pixels = sample_from_boundary_neighborhood(
        image_rgb, occlusion_mask, occludee_visible_mask, max_distance=15
    )
    avg = np.mean(boundary_pixels, axis=0).astype(np.uint8) if len(boundary_pixels) > 0 else np.array([128, 128, 128], dtype=np.uint8)

    result = image_rgb.copy()

    # Blank occluder with average color first
    result[occluder_binary > 0] = avg

    # PatchMatch inpainting
    # Prepare masks for PatchMatch
    inpaint_mask = (occ_binary * 255).astype(np.uint8)

    # CRITICAL FIX: Limit exemplar region to near occlusion boundary
    # This prevents sampling from distant regions that may have different textures
    occ_dist = cv2.distanceTransform(255 - occ_binary * 255, cv2.DIST_L2, 5)
    near_boundary = (occ_dist <= max_exemplar_distance) & (visible_binary > 0)
    exemplar_mask = (near_boundary.astype(np.uint8) * 255)

    # If near-boundary region is too small, expand gradually
    if np.sum(exemplar_mask > 0) < min_visible_pixels:
        for expand_dist in [max_exemplar_distance * 2, max_exemplar_distance * 3]:
            near_boundary = (occ_dist <= expand_dist) & (visible_binary > 0)
            exemplar_mask = (near_boundary.astype(np.uint8) * 255)
            if np.sum(exemplar_mask > 0) >= min_visible_pixels:
                break
        else:
            # Still not enough, use full visible region
            exemplar_mask = (visible_binary * 255).astype(np.uint8)

    # Ensure patch_size is odd
    patch_size = patch_size if patch_size % 2 == 1 else patch_size + 1
    half_patch = patch_size // 2

    # Get fill order
    inpaint_binary = occ_binary > 0
    coords = np.argwhere(inpaint_binary)
    if skip_outside_in:
        # Ablation: random fill order instead of outside-in
        np.random.shuffle(coords)
        fill_coords = coords
    else:
        # Production: outside-in for better results
        dist = ndimage.distance_transform_edt(inpaint_binary)
        distances = dist[coords[:, 0], coords[:, 1]]
        sorted_indices = np.argsort(distances)
        fill_coords = coords[sorted_indices]

    # Get valid exemplar centers (eroded to ensure full patches)
    kernel = np.ones((patch_size, patch_size), dtype=np.uint8)
    eroded_exemplar = cv2.erode(exemplar_mask, kernel) > 127
    exemplar_coords = np.argwhere(eroded_exemplar)

    if len(exemplar_coords) == 0:
        # Not enough exemplar region, fallback
        return inpaint_occlusion_boundary_guided(
            image_rgb, occlusion_mask, occluder_mask,
            occludee_full_mask, occludee_visible_mask, edge_darkness
        )

    # Track which pixels are filled
    filled_mask = ~inpaint_binary.copy()

    # PatchMatch filling loop
    for y, x in fill_coords:
        if filled_mask[y, x]:
            continue

        # Extract target patch (known pixels only)
        y1 = max(0, y - half_patch)
        y2 = min(h, y + half_patch + 1)
        x1 = max(0, x - half_patch)
        x2 = min(w, x + half_patch + 1)

        target_patch = result[y1:y2, x1:x2].copy()
        target_valid = filled_mask[y1:y2, x1:x2]

        if target_valid.sum() == 0:
            # No valid neighbors yet, use average color
            result[y, x] = avg
            filled_mask[y, x] = True
            continue

        # Random sample of exemplar centers for efficiency
        sample_size = min(100, len(exemplar_coords))
        sampled_indices = np.random.choice(len(exemplar_coords), sample_size, replace=False)
        sampled_coords = exemplar_coords[sampled_indices]

        # Find best matching patch
        best_ssd = float('inf')
        best_pixel = avg

        if skip_color_filter:
            # Ablation: no color filtering, just find best SSD match
            for ey, ex in sampled_coords:
                ey1 = max(0, ey - half_patch)
                ey2 = min(h, ey + half_patch + 1)
                ex1 = max(0, ex - half_patch)
                ex2 = min(w, ex + half_patch + 1)

                if (ey2 - ey1) != (y2 - y1) or (ex2 - ex1) != (x2 - x1):
                    continue

                exemplar_patch = image_rgb[ey1:ey2, ex1:ex2]
                diff = (target_patch.astype(float) - exemplar_patch.astype(float))
                diff_masked = diff * target_valid[..., np.newaxis]
                ssd = np.sum(diff_masked ** 2)

                if ssd < best_ssd:
                    best_ssd = ssd
                    best_pixel = image_rgb[ey, ex]
        else:
            # Production: multi-stage color outlier filtering
            # Compute color tolerance for outlier filtering (VERY STRICT)
            # Allow pixels within 1.5 standard deviations of boundary mean
            if len(boundary_pixels) > 10:
                color_std = np.std(boundary_pixels, axis=0)
                color_tolerance = np.maximum(color_std * 1.5, 20)  # Very strict: 1.5σ, min 20
                # Narrow percentile range for stricter filtering
                color_min = np.percentile(boundary_pixels, 10, axis=0)
                color_max = np.percentile(boundary_pixels, 90, axis=0)
            else:
                color_tolerance = np.array([40, 40, 40])
                color_min = avg - 40
                color_max = avg + 40

            # Try with strict filtering first, then relax if no match found
            for filter_level in range(3):  # 0=strict, 1=medium, 2=relaxed
                if filter_level == 0:
                    tol_mult, range_margin, imbalance_margin = 1.0, 10, 30
                elif filter_level == 1:
                    tol_mult, range_margin, imbalance_margin = 1.5, 20, 50
                else:
                    tol_mult, range_margin, imbalance_margin = 2.5, 40, 100

                for ey, ex in sampled_coords:
                    ey1 = max(0, ey - half_patch)
                    ey2 = min(h, ey + half_patch + 1)
                    ex1 = max(0, ex - half_patch)
                    ex2 = min(w, ex + half_patch + 1)

                    # Skip if patch sizes don't match
                    if (ey2 - ey1) != (y2 - y1) or (ex2 - ex1) != (x2 - x1):
                        continue

                    # COLOR OUTLIER FILTER 1: Skip patches with center pixel too different from average
                    center_pixel = image_rgb[ey, ex].astype(float)
                    color_diff = np.abs(center_pixel - avg.astype(float))
                    if np.any(color_diff > color_tolerance * tol_mult):
                        continue  # Skip this outlier patch

                    # COLOR OUTLIER FILTER 2: Check if center pixel is within boundary color range
                    if np.any(center_pixel < color_min - range_margin) or np.any(center_pixel > color_max + range_margin):
                        continue  # Skip pixels outside expected color range

                    # COLOR OUTLIER FILTER 2.5: Check for color channel imbalance (e.g., green/red spike)
                    channel_diff_rg = abs(center_pixel[0] - center_pixel[1])  # R vs G
                    channel_diff_rb = abs(center_pixel[0] - center_pixel[2])  # R vs B
                    channel_diff_gb = abs(center_pixel[1] - center_pixel[2])  # G vs B
                    avg_channel_diff_rg = abs(float(avg[0]) - float(avg[1]))
                    avg_channel_diff_rb = abs(float(avg[0]) - float(avg[2]))
                    avg_channel_diff_gb = abs(float(avg[1]) - float(avg[2]))
                    # If channel imbalance is much greater than average, skip
                    if (channel_diff_rg > avg_channel_diff_rg + imbalance_margin or
                        channel_diff_rb > avg_channel_diff_rb + imbalance_margin or
                        channel_diff_gb > avg_channel_diff_gb + imbalance_margin):
                        continue  # Skip pixels with abnormal color channel imbalance

                    exemplar_patch = image_rgb[ey1:ey2, ex1:ex2]

                    # COLOR OUTLIER FILTER 3: Check patch mean color
                    patch_mean = np.mean(exemplar_patch, axis=(0, 1))
                    if np.any(np.abs(patch_mean - avg) > color_tolerance * tol_mult * 1.2):
                        continue  # Skip patches with outlier average color

                    # Compute SSD only on valid pixels
                    diff = (target_patch.astype(float) - exemplar_patch.astype(float))
                    diff_masked = diff * target_valid[..., np.newaxis]
                    ssd = np.sum(diff_masked ** 2)

                    if ssd < best_ssd:
                        best_ssd = ssd
                        best_pixel = image_rgb[ey, ex]

                # If we found a match, stop trying relaxed filters
                if best_ssd < float('inf'):
                    break

        # Final fallback: if still no match, use nearest filled neighbor
        if best_ssd == float('inf'):
            # Find nearest filled pixel
            for dy in range(-half_patch, half_patch + 1):
                for dx in range(-half_patch, half_patch + 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and filled_mask[ny, nx]:
                        best_pixel = result[ny, nx]
                        break
                if best_ssd != float('inf') or not np.array_equal(best_pixel, avg):
                    break

        result[y, x] = best_pixel
        filled_mask[y, x] = True

    # POST-PROCESSING: Aggressive outlier removal and smoothing
    if not skip_post_processing:
        occ_coords = np.where(occ_binary > 0)
        if len(occ_coords[0]) > 0:
            # Step 0: Detect and fix dark outliers (pixels much darker than expected)
            # These often occur when no valid patch was found
            avg_brightness = np.mean(avg)
            pixel_brightness = np.mean(result, axis=2)
            dark_outlier = (pixel_brightness < avg_brightness * 0.5) & (occ_binary > 0)
            if np.sum(dark_outlier) > 0:
                # Replace dark outliers with avg color initially
                result[dark_outlier] = avg

            # Step 1: Stronger median filter to remove salt-and-pepper outliers
            median_filtered = cv2.medianBlur(result, 5)  # Larger kernel (5 vs 3)

            # Step 2: Detect outlier pixels (more sensitive threshold)
            diff_from_median = np.abs(result.astype(float) - median_filtered.astype(float))
            outlier_mask = np.max(diff_from_median, axis=2) > 25  # Stricter: 25 (was 40)
            outlier_in_occ = outlier_mask & (occ_binary > 0)

            # Replace outliers with median filtered values
            result[outlier_in_occ] = median_filtered[outlier_in_occ]

            # Step 2.5: Second pass - detect color channel outliers specifically
            # Check if any pixel has abnormal channel imbalance vs neighborhood
            for _ in range(2):  # Two passes for thorough cleaning
                median_pass2 = cv2.medianBlur(result, 3)
                # Color channel imbalance detection
                r, g, b = result[:,:,0].astype(float), result[:,:,1].astype(float), result[:,:,2].astype(float)
                mr, mg, mb = median_pass2[:,:,0].astype(float), median_pass2[:,:,1].astype(float), median_pass2[:,:,2].astype(float)
                # Detect if any channel is unusually different from median
                channel_outlier = ((np.abs(r - mr) > 20) | (np.abs(g - mg) > 20) | (np.abs(b - mb) > 20)) & (occ_binary > 0)
                result[channel_outlier] = median_pass2[channel_outlier]

            # Step 2.7: Re-check dark outliers after median filtering
            median_final = cv2.medianBlur(result, 5)
            pixel_brightness = np.mean(result, axis=2)
            dark_outlier = (pixel_brightness < avg_brightness * 0.6) & (occ_binary > 0)
            result[dark_outlier] = median_final[dark_outlier]

            # Step 3: Bilateral filter for edge-preserving smoothing
            bilateral_filtered = cv2.bilateralFilter(result, 5, 50, 50)

            # Step 4: Gaussian smoothing blend
            blurred = cv2.GaussianBlur(result, (5, 5), 1.0)
            # Blend: 60% bilateral, 40% gaussian for smooth texture
            for y, x in zip(occ_coords[0], occ_coords[1]):
                result[y, x] = (bilateral_filtered[y, x].astype(float) * 0.6 +
                               blurred[y, x].astype(float) * 0.4).astype(np.uint8)

    # Draw intersection edge
    if np.sum(edge_mask > 0) > 0:
        edge_coords = np.where(edge_mask > 0)

        use_v23 = use_edge_sampling
        sampled_edge_colors = None

        if use_edge_sampling:
            # V23: Try to sample edge colors from visible edges
            sampled_edge_colors = sample_edge_colors_from_visible(
                image_rgb, edge_mask, occludee_visible_mask,
                max_search_distance=30,
                smooth_edges=(edge_smooth_kernel > 0),
                smooth_kernel_size=edge_smooth_kernel,
                occluder_mask=occluder_mask  # V24: pass occluder for safe sampling
            )
            # If sampling failed (not enough visible edges), fallback to V22
            if sampled_edge_colors is None:
                use_v23 = False

        if use_v23 and sampled_edge_colors is not None:
            # V23: Apply sampled edge colors (with optional darkness adjustment)
            for i, (y, x) in enumerate(zip(edge_coords[0], edge_coords[1])):
                sampled = sampled_edge_colors[y, x].astype(np.float32)
                # Apply edge_darkness as a blend factor (1.0 = use sampled as-is)
                if edge_darkness < 1.0:
                    # Blend between sampled and darker version
                    darkened = sampled * edge_darkness
                    final_color = sampled * edge_darkness + darkened * (1 - edge_darkness)
                    result[y, x] = np.clip(final_color, 0, 255).astype(np.uint8)
                else:
                    result[y, x] = sampled.astype(np.uint8)

            # Anti-alias edge with sampled colors
            kernel = np.ones((3, 3), dtype=np.uint8)
            dilated = cv2.dilate(edge_mask, kernel)
            boundary = dilated - edge_mask
            boundary_coords = np.where(boundary > 0)
            if len(boundary_coords[0]) > 0:
                for y, x in zip(boundary_coords[0], boundary_coords[1]):
                    current = result[y, x].astype(np.float32)
                    # Find nearest edge pixel's color for anti-aliasing
                    nearest_edge_color = sampled_edge_colors[y, x].astype(np.float32)
                    if edge_darkness < 1.0:
                        nearest_edge_color = nearest_edge_color * (0.5 + edge_darkness * 0.5)
                    blended = current * 0.5 + nearest_edge_color * 0.5
                    result[y, x] = np.clip(blended, 0, 255).astype(np.uint8)
        else:
            # V22: Fixed darkness factor (original algorithm)
            edge_color = (avg * edge_darkness).astype(np.uint8)
            for y, x in zip(edge_coords[0], edge_coords[1]):
                result[y, x] = edge_color

            # Anti-alias edge with fixed color
            kernel = np.ones((3, 3), dtype=np.uint8)
            dilated = cv2.dilate(edge_mask, kernel)
            boundary = dilated - edge_mask
            boundary_coords = np.where(boundary > 0)
            if len(boundary_coords[0]) > 0:
                for y, x in zip(boundary_coords[0], boundary_coords[1]):
                    current = result[y, x].astype(np.float32)
                    blended = current * 0.5 + edge_color.astype(np.float32) * 0.5
                    result[y, x] = np.clip(blended, 0, 255).astype(np.uint8)

    # Final: apply only to occlusion region
    final = image_rgb.copy()
    final[occ_binary > 0] = result[occ_binary > 0]

    return final


def inpaint_occlusion_patchmatch_v25(
    image_rgb: np.ndarray,
    occlusion_mask: np.ndarray,
    occluder_mask: np.ndarray,
    occludee_full_mask: np.ndarray,
    occludee_visible_mask: np.ndarray,
    roi_edge_mask: Optional[np.ndarray] = None,
    edge_darkness: float = 0.3,
    patch_size: int = 7,
    iterations: int = 5,
    max_exemplar_distance: int = 0,
    # V25 Dynamic Edge parameters
    min_edge_width: int = 1,
    max_edge_width: int = 8,
    width_smoothing_sigma: float = 1.5,
    min_safe_distance: int = 3,
    # PatchMatch controls
    skip_color_filter: bool = False,
    skip_outside_in: bool = False,
    skip_post_processing: bool = False,
    # V26: Adaptive Kernel Smoothing flags (for ablation testing)
    _v26_smoothstep_gradient: bool = False,   # Imp1: Hermite smoothstep for edge gradient
    _v26_adaptive_aa: bool = False,           # Imp3: adaptive AA kernel (3x3 or 5x5)
) -> np.ndarray:
    """Inpaint using PatchMatch body fill + V25 dynamic edge rendering.

    Combines V22's PatchMatch texture propagation for the body region
    with V25's skeleton-based variable-thickness edge rendering for
    the intersection edge. This leverages texture coherence from
    PatchMatch while preserving the natural brush-stroke appearance
    from dynamic edge morphology.

    Args:
        image_rgb: RGB image (H, W, 3), uint8.
        occlusion_mask: Binary mask of occlusion region to fill.
        occluder_mask: Binary mask of occluder object.
        occludee_full_mask: Full mask of occludee object (including occluded).
        occludee_visible_mask: Visible portion of occludee.
        edge_darkness: Darkness factor for edge (0-1, lower = darker).
        patch_size: Size of patches for PatchMatch (odd number).
        iterations: Number of PatchMatch iterations.
        max_exemplar_distance: Max distance from boundary for exemplar (0=auto).
        min_edge_width: Minimum edge width in pixels (V25).
        max_edge_width: Maximum edge width in pixels (V25).
        width_smoothing_sigma: Gaussian sigma for width map smoothing (V25).
        min_safe_distance: Min distance from occluder for safe sampling (V25).
        skip_color_filter: Skip color outlier filters (ablation).
        skip_outside_in: Use random fill order (ablation).
        skip_post_processing: Skip post-processing (ablation).

    Returns:
        Inpainted image (H, W, 3), uint8.
    """
    h, w = image_rgb.shape[:2]
    occ_binary = (occlusion_mask > 0).astype(np.uint8)
    occluder_binary = (occluder_mask > 0).astype(np.uint8)
    visible_binary = (occludee_visible_mask > 0).astype(np.uint8)

    if np.sum(occ_binary) == 0:
        return image_rgb.copy()

    # Auto-adjust parameters based on image size
    img_size = max(h, w)
    if max_exemplar_distance == 0:
        if img_size < 200:
            max_exemplar_distance = 15
        elif img_size < 400:
            max_exemplar_distance = 25
        else:
            max_exemplar_distance = 40

    if img_size < 150 and patch_size > 5:
        patch_size = 5

    # Check if we have enough visible region for PatchMatch
    min_visible_pixels = patch_size * patch_size * 10
    if np.sum(visible_binary) < min_visible_pixels:
        # Fallback to V25 boundary_guided
        return inpaint_occlusion_boundary_guided(
            image_rgb, occlusion_mask, occluder_mask,
            occludee_full_mask, occludee_visible_mask,
            edge_darkness=edge_darkness,
            use_dynamic_edge=True,
            min_edge_width=min_edge_width,
            max_edge_width=max_edge_width,
            width_smoothing_sigma=width_smoothing_sigma,
            min_safe_distance=min_safe_distance,
            _v26_smoothstep_gradient=_v26_smoothstep_gradient,
            _v26_adaptive_aa=_v26_adaptive_aa,
        )

    # Sample colors for average
    boundary_pixels = sample_from_boundary_neighborhood(
        image_rgb, occlusion_mask, occludee_visible_mask, max_distance=15
    )
    avg = np.mean(boundary_pixels, axis=0).astype(np.uint8) if len(boundary_pixels) > 0 else np.array([128, 128, 128], dtype=np.uint8)

    result = image_rgb.copy()
    result[occluder_binary > 0] = avg

    # =========================================================================
    # PHASE 1: PatchMatch body fill (from V22)
    # =========================================================================
    inpaint_mask = (occ_binary * 255).astype(np.uint8)

    # Limit exemplar region to near occlusion boundary
    occ_dist = cv2.distanceTransform(255 - occ_binary * 255, cv2.DIST_L2, 5)
    near_boundary = (occ_dist <= max_exemplar_distance) & (visible_binary > 0)
    exemplar_mask = (near_boundary.astype(np.uint8) * 255)

    if np.sum(exemplar_mask > 0) < min_visible_pixels:
        for expand_dist in [max_exemplar_distance * 2, max_exemplar_distance * 3]:
            near_boundary = (occ_dist <= expand_dist) & (visible_binary > 0)
            exemplar_mask = (near_boundary.astype(np.uint8) * 255)
            if np.sum(exemplar_mask > 0) >= min_visible_pixels:
                break
        else:
            exemplar_mask = (visible_binary * 255).astype(np.uint8)

    patch_size = patch_size if patch_size % 2 == 1 else patch_size + 1
    half_patch = patch_size // 2

    # Get fill order
    inpaint_binary = occ_binary > 0
    coords = np.argwhere(inpaint_binary)
    if skip_outside_in:
        np.random.shuffle(coords)
        fill_coords = coords
    else:
        dist = ndimage.distance_transform_edt(inpaint_binary)
        distances = dist[coords[:, 0], coords[:, 1]]
        sorted_indices = np.argsort(distances)
        fill_coords = coords[sorted_indices]

    # Get valid exemplar centers
    kernel = np.ones((patch_size, patch_size), dtype=np.uint8)
    eroded_exemplar = cv2.erode(exemplar_mask, kernel) > 127
    exemplar_coords = np.argwhere(eroded_exemplar)

    if len(exemplar_coords) == 0:
        return inpaint_occlusion_boundary_guided(
            image_rgb, occlusion_mask, occluder_mask,
            occludee_full_mask, occludee_visible_mask,
            edge_darkness=edge_darkness,
            use_dynamic_edge=True,
            min_edge_width=min_edge_width,
            max_edge_width=max_edge_width,
            width_smoothing_sigma=width_smoothing_sigma,
            min_safe_distance=min_safe_distance,
            _v26_smoothstep_gradient=_v26_smoothstep_gradient,
            _v26_adaptive_aa=_v26_adaptive_aa,
        )

    filled_mask = ~inpaint_binary.copy()

    # PatchMatch filling loop
    for y, x in fill_coords:
        if filled_mask[y, x]:
            continue

        y1 = max(0, y - half_patch)
        y2 = min(h, y + half_patch + 1)
        x1 = max(0, x - half_patch)
        x2 = min(w, x + half_patch + 1)

        target_patch = result[y1:y2, x1:x2].copy()
        target_valid = filled_mask[y1:y2, x1:x2]

        if target_valid.sum() == 0:
            result[y, x] = avg
            filled_mask[y, x] = True
            continue

        sample_size = min(100, len(exemplar_coords))
        sampled_indices = np.random.choice(len(exemplar_coords), sample_size, replace=False)
        sampled_coords = exemplar_coords[sampled_indices]

        best_ssd = float('inf')
        best_pixel = avg

        if skip_color_filter:
            for ey, ex in sampled_coords:
                ey1 = max(0, ey - half_patch)
                ey2 = min(h, ey + half_patch + 1)
                ex1 = max(0, ex - half_patch)
                ex2 = min(w, ex + half_patch + 1)
                if (ey2 - ey1) != (y2 - y1) or (ex2 - ex1) != (x2 - x1):
                    continue
                exemplar_patch = image_rgb[ey1:ey2, ex1:ex2]
                diff = (target_patch.astype(float) - exemplar_patch.astype(float))
                diff_masked = diff * target_valid[..., np.newaxis]
                ssd = np.sum(diff_masked ** 2)
                if ssd < best_ssd:
                    best_ssd = ssd
                    best_pixel = image_rgb[ey, ex]
        else:
            if len(boundary_pixels) > 10:
                color_std = np.std(boundary_pixels, axis=0)
                color_tolerance = np.maximum(color_std * 1.5, 20)
                color_min = np.percentile(boundary_pixels, 10, axis=0)
                color_max = np.percentile(boundary_pixels, 90, axis=0)
            else:
                color_tolerance = np.array([40, 40, 40])
                color_min = avg - 40
                color_max = avg + 40

            for filter_level in range(3):
                if filter_level == 0:
                    tol_mult, range_margin, imbalance_margin = 1.0, 10, 30
                elif filter_level == 1:
                    tol_mult, range_margin, imbalance_margin = 1.5, 20, 50
                else:
                    tol_mult, range_margin, imbalance_margin = 2.5, 40, 100

                for ey, ex in sampled_coords:
                    ey1 = max(0, ey - half_patch)
                    ey2 = min(h, ey + half_patch + 1)
                    ex1 = max(0, ex - half_patch)
                    ex2 = min(w, ex + half_patch + 1)
                    if (ey2 - ey1) != (y2 - y1) or (ex2 - ex1) != (x2 - x1):
                        continue

                    center_pixel = image_rgb[ey, ex].astype(float)
                    color_diff = np.abs(center_pixel - avg.astype(float))
                    if np.any(color_diff > color_tolerance * tol_mult):
                        continue
                    if np.any(center_pixel < color_min - range_margin) or np.any(center_pixel > color_max + range_margin):
                        continue

                    channel_diff_rg = abs(center_pixel[0] - center_pixel[1])
                    channel_diff_rb = abs(center_pixel[0] - center_pixel[2])
                    channel_diff_gb = abs(center_pixel[1] - center_pixel[2])
                    avg_channel_diff_rg = abs(float(avg[0]) - float(avg[1]))
                    avg_channel_diff_rb = abs(float(avg[0]) - float(avg[2]))
                    avg_channel_diff_gb = abs(float(avg[1]) - float(avg[2]))
                    if (channel_diff_rg > avg_channel_diff_rg + imbalance_margin or
                        channel_diff_rb > avg_channel_diff_rb + imbalance_margin or
                        channel_diff_gb > avg_channel_diff_gb + imbalance_margin):
                        continue

                    exemplar_patch = image_rgb[ey1:ey2, ex1:ex2]
                    patch_mean = np.mean(exemplar_patch, axis=(0, 1))
                    if np.any(np.abs(patch_mean - avg) > color_tolerance * tol_mult * 1.2):
                        continue

                    diff = (target_patch.astype(float) - exemplar_patch.astype(float))
                    diff_masked = diff * target_valid[..., np.newaxis]
                    ssd = np.sum(diff_masked ** 2)
                    if ssd < best_ssd:
                        best_ssd = ssd
                        best_pixel = image_rgb[ey, ex]

                if best_ssd < float('inf'):
                    break

        if best_ssd == float('inf'):
            for dy in range(-half_patch, half_patch + 1):
                for dx in range(-half_patch, half_patch + 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and filled_mask[ny, nx]:
                        best_pixel = result[ny, nx]
                        break
                if best_ssd != float('inf') or not np.array_equal(best_pixel, avg):
                    break

        result[y, x] = best_pixel
        filled_mask[y, x] = True

    # Post-processing
    if not skip_post_processing:
        occ_coords = np.where(occ_binary > 0)
        if len(occ_coords[0]) > 0:
            avg_brightness = np.mean(avg)
            pixel_brightness = np.mean(result, axis=2)
            dark_outlier = (pixel_brightness < avg_brightness * 0.5) & (occ_binary > 0)
            if np.sum(dark_outlier) > 0:
                result[dark_outlier] = avg

            median_filtered = cv2.medianBlur(result, 5)
            diff_from_median = np.abs(result.astype(float) - median_filtered.astype(float))
            outlier_mask = np.max(diff_from_median, axis=2) > 25
            outlier_in_occ = outlier_mask & (occ_binary > 0)
            result[outlier_in_occ] = median_filtered[outlier_in_occ]

            for _ in range(2):
                median_pass2 = cv2.medianBlur(result, 3)
                r, g, b = result[:,:,0].astype(float), result[:,:,1].astype(float), result[:,:,2].astype(float)
                mr, mg, mb = median_pass2[:,:,0].astype(float), median_pass2[:,:,1].astype(float), median_pass2[:,:,2].astype(float)
                channel_outlier = ((np.abs(r - mr) > 20) | (np.abs(g - mg) > 20) | (np.abs(b - mb) > 20)) & (occ_binary > 0)
                result[channel_outlier] = median_pass2[channel_outlier]

            median_final = cv2.medianBlur(result, 5)
            pixel_brightness = np.mean(result, axis=2)
            dark_outlier = (pixel_brightness < avg_brightness * 0.6) & (occ_binary > 0)
            result[dark_outlier] = median_final[dark_outlier]

            bilateral_filtered = cv2.bilateralFilter(result, 5, 50, 50)
            blurred = cv2.GaussianBlur(result, (5, 5), 1.0)
            for y, x in zip(occ_coords[0], occ_coords[1]):
                result[y, x] = (bilateral_filtered[y, x].astype(float) * 0.6 +
                               blurred[y, x].astype(float) * 0.4).astype(np.uint8)

    # =========================================================================
    # PHASE 2: V25 Dynamic Edge rendering (replaces V22/V23/V24 edge)
    # =========================================================================
    v25_applied = False

    # Extract intersection centerline (thickness=1 for V25)
    if roi_edge_mask is not None and np.sum(roi_edge_mask > 0) > 0:
        # Use ROI polygon boundary directly for edge placement
        occ_binary = (occlusion_mask > 0).astype(np.uint8)
        erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        occ_edge = occ_binary - cv2.erode(occ_binary, erode_k)
        # Dilate both to ensure intersection tolerance
        thick_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        roi_dilated = cv2.dilate((roi_edge_mask > 0).astype(np.uint8), thick_k)
        occ_dilated = cv2.dilate(occ_edge, thick_k)
        centerline = cv2.bitwise_and(roi_dilated * 255, occ_dilated * 255)
    else:
        centerline = get_intersection_edge(
            occludee_full_mask, occlusion_mask, thickness=1
        )

    if np.sum(centerline > 0) > 0:
        # Measure visible edge widths
        skeleton, width_map, _ = measure_edge_width_map(
            image_rgb, occludee_visible_mask,
            min_edge_width=min_edge_width,
            max_edge_width=max_edge_width,
            width_smoothing_sigma=width_smoothing_sigma,
        )

        # Extract color profile
        edge_profile = extract_edge_color_profile(
            image_rgb, skeleton, width_map,
        )

        # Render variable-thickness edge
        if edge_profile is not None and len(edge_profile['coords']) > 0:
            rendered, edge_mask = render_dynamic_intersection_edge(
                image_rgb, centerline, edge_profile,
                occluder_mask=occluder_mask,
                min_safe_distance=min_safe_distance,
                _v26_smoothstep_gradient=_v26_smoothstep_gradient,
            )

            if np.sum(edge_mask > 0) > 0:
                # Apply with darkness adjustment
                if edge_darkness < 1.0:
                    edge_pixels = np.where(edge_mask > 0)
                    for y, x in zip(edge_pixels[0], edge_pixels[1]):
                        color = rendered[y, x].astype(np.float32)
                        darkened = color * edge_darkness
                        final_color = color * edge_darkness + darkened * (1 - edge_darkness)
                        result[y, x] = np.clip(final_color, 0, 255).astype(np.uint8)
                else:
                    result[edge_mask > 0] = rendered[edge_mask > 0]

                # Anti-alias
                # V26 Imp3: Adaptive AA kernel based on edge width
                if _v26_adaptive_aa:
                    edge_dt = cv2.distanceTransform(edge_mask, cv2.DIST_L2, 3)
                    max_edge_dt = np.max(edge_dt) if np.max(edge_dt) > 0 else 0
                    aa_size = 5 if max_edge_dt >= 2.5 else 3
                else:
                    aa_size = 3
                aa_kernel = np.ones((aa_size, aa_size), dtype=np.uint8)
                dilated = cv2.dilate(edge_mask, aa_kernel)
                boundary = dilated - edge_mask
                boundary_coords = np.where(boundary > 0)
                if len(boundary_coords[0]) > 0:
                    for y, x in zip(boundary_coords[0], boundary_coords[1]):
                        curr = result[y, x].astype(np.float32)
                        edge_color = rendered[y, x].astype(np.float32)
                        if edge_darkness < 1.0:
                            edge_color = edge_color * (0.5 + edge_darkness * 0.5)
                        blended = curr * 0.5 + edge_color * 0.5
                        result[y, x] = np.clip(blended, 0, 255).astype(np.uint8)

                v25_applied = True

    # Fallback: V24-style fixed edge if V25 failed
    if not v25_applied:
        if roi_edge_mask is not None and np.sum(roi_edge_mask > 0) > 0:
            occ_binary = (occlusion_mask > 0).astype(np.uint8)
            erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            occ_edge = occ_binary - cv2.erode(occ_binary, erode_k)
            thick_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            roi_dilated = cv2.dilate((roi_edge_mask > 0).astype(np.uint8), thick_k)
            occ_dilated = cv2.dilate(occ_edge, thick_k)
            edge_mask = cv2.bitwise_and(roi_dilated * 255, occ_dilated * 255)
        else:
            edge_mask = get_intersection_edge(occludee_full_mask, occlusion_mask, thickness=2)
        if np.sum(edge_mask > 0) > 0:
            sampled_edge_colors = sample_edge_colors_from_visible(
                image_rgb, edge_mask, occludee_visible_mask,
                max_search_distance=30,
                occluder_mask=occluder_mask,
                min_safe_distance=min_safe_distance,
            )
            if sampled_edge_colors is not None:
                edge_coords = np.where(edge_mask > 0)
                for i, (y, x) in enumerate(zip(edge_coords[0], edge_coords[1])):
                    sampled = sampled_edge_colors[y, x].astype(np.float32)
                    if edge_darkness < 1.0:
                        darkened = sampled * edge_darkness
                        final_color = sampled * edge_darkness + darkened * (1 - edge_darkness)
                        result[y, x] = np.clip(final_color, 0, 255).astype(np.uint8)
                    else:
                        result[y, x] = sampled.astype(np.uint8)
            else:
                edge_color = (avg * edge_darkness).astype(np.uint8)
                edge_coords = np.where(edge_mask > 0)
                for y, x in zip(edge_coords[0], edge_coords[1]):
                    result[y, x] = edge_color

    # Final: apply only to occlusion region
    final = image_rgb.copy()
    final[occ_binary > 0] = result[occ_binary > 0]

    return final


def quick_inpaint(
    image: np.ndarray,
    mask: np.ndarray,
    method: str = "telea",
    radius: int = 5
) -> np.ndarray:
    """Quick inpainting function without class instantiation.

    Args:
        image: RGB image (H, W, 3).
        mask: Binary mask (H, W).
        method: "telea", "ns", or "lama".
        radius: Inpainting radius (OpenCV only).

    Returns:
        Inpainted image.
    """
    module = InpaintingModule(method=method, radius=radius)
    return module.inpaint(image, mask)


def quick_lama_inpaint(
    image: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:
    """Quick LaMa inpainting (convenience function).

    Args:
        image: RGB image (H, W, 3).
        mask: Binary mask (H, W).

    Returns:
        Inpainted image using LaMa.
    """
    inpainter = LamaInpainter()
    return inpainter.inpaint(image, mask)


# Lazy SD pipeline cache
_sd_inpaint_pipeline = None
_controlnet_inpaint_pipeline = None


class SDInpainter:
    """Stable Diffusion based inpainter.

    Uses diffusers StableDiffusionInpaintPipeline for high-quality
    generative inpainting with optional ControlNet guidance.

    Reference: https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/inpaint
    """

    # Default models - Using Canny edge ControlNet for better structure preservation
    DEFAULT_SD_MODEL = "runwayml/stable-diffusion-inpainting"
    DEFAULT_CONTROLNET = "lllyasviel/control_v11p_sd15_canny"

    def __init__(
        self,
        model_id: Optional[str] = None,
        use_controlnet: bool = False,
        controlnet_id: Optional[str] = None,
        device: str = "cuda",
        dtype: str = "float16"
    ):
        """Initialize SD inpainter.

        Args:
            model_id: HuggingFace model ID for SD inpainting model.
            use_controlnet: Whether to use ControlNet for better structure.
            controlnet_id: HuggingFace model ID for ControlNet (Canny edge).
            device: Device to run on ("cuda" or "cpu").
            dtype: Model precision ("float16" or "float32").
        """
        self.model_id = model_id or self.DEFAULT_SD_MODEL
        self.use_controlnet = use_controlnet
        self.controlnet_id = controlnet_id or self.DEFAULT_CONTROLNET
        self.device = device
        self.dtype = dtype

        self._pipeline = None

    @property
    def pipeline(self):
        """Lazy load pipeline."""
        if self._pipeline is None:
            self._pipeline = self._load_pipeline()
        return self._pipeline

    def _load_pipeline(self):
        """Load the appropriate inpainting pipeline."""
        try:
            import torch
            from diffusers import (
                StableDiffusionInpaintPipeline,
                ControlNetModel,
                StableDiffusionControlNetInpaintPipeline,
            )
        except ImportError:
            raise ImportError(
                "SD inpainting requires diffusers. "
                "Install with: pip install diffusers transformers accelerate"
            )

        torch_dtype = torch.float16 if self.dtype == "float16" else torch.float32

        if self.use_controlnet:
            # Load ControlNet model
            print(f"Loading ControlNet: {self.controlnet_id}")
            controlnet = ControlNetModel.from_pretrained(
                self.controlnet_id,
                torch_dtype=torch_dtype
            )

            # Load ControlNet Inpaint pipeline
            print(f"Loading SD ControlNet Inpaint: {self.model_id}")
            pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                self.model_id,
                controlnet=controlnet,
                torch_dtype=torch_dtype,
                safety_checker=None,
            )
        else:
            # Load standard SD Inpaint pipeline
            print(f"Loading SD Inpaint: {self.model_id}")
            pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                safety_checker=None,
            )

        pipeline = pipeline.to(self.device)

        # Enable optimizations
        try:
            pipeline.enable_xformers_memory_efficient_attention()
        except Exception:
            pass  # xformers not available

        return pipeline

    def _prepare_control_image(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> "Image":
        """Prepare Canny edge control image for ControlNet inpainting.

        Extracts Canny edges from the visible (non-masked) region to guide
        structure-preserving inpainting.

        Args:
            image: RGB image (H, W, 3), uint8.
            mask: Binary mask (H, W) where 255 = region to inpaint.

        Returns:
            PIL Image with Canny edges (RGB).
        """
        from PIL import Image

        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Extract Canny edges
        edges = cv2.Canny(gray, 50, 150)

        # Remove edges in masked region (keep only visible area edges)
        edges[mask > 127] = 0

        # Convert to RGB (ControlNet expects 3 channel)
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        return Image.fromarray(edges_rgb)

    def inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        prompt: str = "red wooden traditional Korean furniture, carved wood texture, antique",
        negative_prompt: str = "white, ceramic, vase, jar, modern, blurry, low quality",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        controlnet_scale: float = 0.5,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """Inpaint masked regions using Stable Diffusion.

        Args:
            image: RGB image (H, W, 3), uint8.
            mask: Binary mask (H, W) where 255 = region to inpaint.
            prompt: Text prompt describing desired fill.
            negative_prompt: Text describing what to avoid.
            num_inference_steps: Number of denoising steps (more = better quality).
            guidance_scale: How strongly to follow the prompt.
            controlnet_scale: ControlNet conditioning scale (0.0-1.0).
            seed: Random seed for reproducibility.

        Returns:
            Inpainted image (H, W, 3), uint8.
        """
        from PIL import Image
        import torch

        # Ensure uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        if mask.dtype != np.uint8:
            mask = (mask > 0).astype(np.uint8) * 255

        # Convert to PIL
        img_pil = Image.fromarray(image)
        mask_pil = Image.fromarray(mask).convert('L')

        # Set seed if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Run inference
        if self.use_controlnet:
            control_image = self._prepare_control_image(image, mask)
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=img_pil,
                mask_image=mask_pil,
                control_image=control_image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_scale,
                generator=generator,
            ).images[0]
        else:
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=img_pil,
                mask_image=mask_pil,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]

        # Convert back to numpy
        result_np = np.array(result)
        return result_np

    def inpaint_texture(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        reference_region: Optional[np.ndarray] = None,
        num_inference_steps: int = 30,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """Inpaint with automatic texture-matching prompt.

        Analyzes the surrounding region to generate an appropriate prompt.

        Args:
            image: RGB image (H, W, 3), uint8.
            mask: Binary mask (H, W).
            reference_region: Optional mask of reference area for texture.
            num_inference_steps: Denoising steps.
            seed: Random seed.

        Returns:
            Inpainted image.
        """
        # Analyze image colors for prompt
        if reference_region is not None:
            ref_pixels = image[reference_region > 0]
        else:
            # Use area around mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
            dilated = cv2.dilate(mask, kernel)
            surrounding = dilated - mask
            ref_pixels = image[surrounding > 0]

        if len(ref_pixels) > 0:
            mean_color = np.mean(ref_pixels, axis=0)
            # Generate color-based prompt
            r, g, b = mean_color
            if r > 150 and g < 100 and b < 100:
                color_desc = "reddish brown wooden"
            elif r > 200 and g > 200 and b > 200:
                color_desc = "white ceramic"
            elif r < 50 and g < 50 and b < 50:
                color_desc = "dark"
            else:
                color_desc = "textured"

            prompt = f"{color_desc} surface, seamless texture, high quality, realistic"
        else:
            prompt = "seamless texture, high quality, realistic"

        return self.inpaint(
            image, mask,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            seed=seed
        )


def quick_sd_inpaint(
    image: np.ndarray,
    mask: np.ndarray,
    prompt: str = "high quality texture, seamless",
    use_controlnet: bool = False,
    num_steps: int = 30
) -> np.ndarray:
    """Quick SD inpainting (convenience function).

    Args:
        image: RGB image (H, W, 3).
        mask: Binary mask (H, W).
        prompt: Text prompt for inpainting.
        use_controlnet: Whether to use ControlNet.
        num_steps: Number of inference steps.

    Returns:
        Inpainted image using Stable Diffusion.
    """
    inpainter = SDInpainter(use_controlnet=use_controlnet)
    return inpainter.inpaint(image, mask, prompt=prompt, num_inference_steps=num_steps)


# MAT (Mask-Aware Transformer) Inpainting
# Reference: https://github.com/fenglinglwb/MAT
# Paper: MAT: Mask-Aware Transformer for Large Hole Image Inpainting (CVPR 2022)

_mat_model = None
_MAT_MODEL_SIZE = 512  # MAT model fixed input size

MAT_MODEL_URL = "https://huggingface.co/Sanster/IOPaint-models/resolve/main/MAT_Places512_G_fp16.pkl"
MAT_MODEL_MD5 = "8ca927f7c0c2e7c3f7c5e7c3f7c5e7c3"


def _download_mat_model() -> str:
    """Download MAT model from HuggingFace Hub.

    Returns:
        Path to downloaded model file.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "MAT inpainting requires huggingface_hub. "
            "Install with: pip install huggingface-hub"
        )

    model_path = hf_hub_download(
        repo_id="Sanster/IOPaint-models",
        filename="MAT_Places512_G_fp16.pkl",
        cache_dir=Path.home() / ".cache" / "mat"
    )
    return model_path


class MATInpainter:
    """MAT (Mask-Aware Transformer) based inpainter.

    Uses transformer architecture for high-quality large hole inpainting.
    Particularly effective for structured content and large masks.

    Reference: https://github.com/fenglinglwb/MAT
    Paper: CVPR 2022 - MAT: Mask-Aware Transformer for Large Hole Image Inpainting
    """

    def __init__(self, device: str = "cuda", fp16: bool = True):
        """Initialize MAT inpainter.

        Args:
            device: Device to run on ("cuda" or "cpu").
            fp16: Use half precision for faster inference (GPU only).
        """
        self.device = device
        self.fp16 = fp16 and "cuda" in device
        self._model = None
        self._z = None
        self._label = None

    def _load_model(self):
        """Load MAT model."""
        try:
            import torch
        except ImportError:
            raise ImportError("MAT requires PyTorch. Install with: pip install torch")

        # Try IOPaint's MAT implementation first
        try:
            from iopaint.model.mat import MAT as IOPaintMAT
            from iopaint.schema import InpaintRequest

            print("Loading MAT via IOPaint...")
            mat = IOPaintMAT(device=self.device)
            mat.init_model(self.device, no_half=not self.fp16)
            return mat, "iopaint"
        except ImportError:
            pass

        # Fallback: load directly
        print("Loading MAT model directly...")
        model_path = _download_mat_model()

        torch_dtype = torch.float16 if self.fp16 else torch.float32

        # Try to use pickle-based loading (MAT format)
        try:
            import pickle
            import sys

            # MAT uses custom dnnlib and legacy modules
            # We need to handle the pickle loading carefully
            with open(model_path, 'rb') as f:
                data = pickle.load(f)

            if 'G_ema' in data:
                model = data['G_ema']
            elif 'G' in data:
                model = data['G']
            else:
                raise ValueError("Could not find generator in MAT checkpoint")

            model = model.to(self.device).to(torch_dtype).eval()
            model.requires_grad_(False)

            # Initialize z and label
            z_dim = getattr(model, 'z_dim', 512)
            c_dim = getattr(model, 'c_dim', 0)

            self._z = torch.from_numpy(
                np.random.randn(1, z_dim)
            ).to(torch_dtype).to(self.device)
            self._label = torch.zeros([1, c_dim], device=self.device).to(torch_dtype)

            return model, "direct"

        except Exception as e:
            print(f"Warning: Could not load MAT model directly: {e}")
            print("Falling back to LaMa...")
            return None, "fallback"

    @property
    def model(self):
        """Lazy load model."""
        if self._model is None:
            self._model, self._mode = self._load_model()
        return self._model

    def inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """Inpaint masked regions using MAT.

        Args:
            image: RGB image (H, W, 3), uint8.
            mask: Binary mask (H, W) where 255 = region to inpaint.

        Returns:
            Inpainted image (H, W, 3), uint8.
        """
        import torch

        # Ensure uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        if mask.dtype != np.uint8:
            mask = (mask > 0).astype(np.uint8) * 255

        original_size = (image.shape[1], image.shape[0])  # (W, H)

        # Check mode
        model = self.model
        if model is None or self._mode == "fallback":
            # Fallback to LaMa
            print("Using LaMa fallback for MAT")
            lama = LamaInpainter()
            return lama.inpaint(image, mask)

        # IOPaint mode
        if self._mode == "iopaint":
            try:
                from iopaint.schema import InpaintRequest

                # Resize to 512x512
                img_resized = cv2.resize(
                    image, (_MAT_MODEL_SIZE, _MAT_MODEL_SIZE),
                    interpolation=cv2.INTER_LANCZOS4
                )
                mask_resized = cv2.resize(
                    mask, (_MAT_MODEL_SIZE, _MAT_MODEL_SIZE),
                    interpolation=cv2.INTER_NEAREST
                )

                # Convert RGB to BGR for IOPaint
                img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)

                # Run inference
                config = InpaintRequest()
                result_bgr = model.forward(img_bgr, mask_resized, config)

                # Convert back to RGB
                result = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

                # Resize back to original size
                if original_size != (_MAT_MODEL_SIZE, _MAT_MODEL_SIZE):
                    result = cv2.resize(
                        result, original_size,
                        interpolation=cv2.INTER_LANCZOS4
                    )

                return result

            except Exception as e:
                print(f"IOPaint MAT failed: {e}, falling back to LaMa")
                lama = LamaInpainter()
                return lama.inpaint(image, mask)

        # Direct mode
        torch_dtype = torch.float16 if self.fp16 else torch.float32

        # Resize to 512x512
        img_resized = cv2.resize(
            image, (_MAT_MODEL_SIZE, _MAT_MODEL_SIZE),
            interpolation=cv2.INTER_LANCZOS4
        )
        mask_resized = cv2.resize(
            mask, (_MAT_MODEL_SIZE, _MAT_MODEL_SIZE),
            interpolation=cv2.INTER_NEAREST
        )

        # Normalize image to [-1, 1]
        img_norm = img_resized.astype(np.float32) / 127.5 - 1.0

        # MAT mask convention: 0 = inpaint, 1 = keep
        mask_norm = 1.0 - (mask_resized.astype(np.float32) / 255.0)

        # Convert to tensors (NCHW format)
        img_tensor = torch.from_numpy(
            img_norm.transpose(2, 0, 1)
        ).unsqueeze(0).to(torch_dtype).to(self.device)

        mask_tensor = torch.from_numpy(
            mask_norm
        ).unsqueeze(0).unsqueeze(0).to(torch_dtype).to(self.device)

        # Run inference
        with torch.no_grad():
            output = model(
                img_tensor, mask_tensor,
                self._z, self._label,
                truncation_psi=1.0,
                noise_mode="none"
            )

        # Post-process output
        output = (output.permute(0, 2, 3, 1) * 127.5 + 127.5)
        output = output.round().clamp(0, 255).to(torch.uint8)
        output = output[0].cpu().numpy()

        # Resize back to original size
        if original_size != (_MAT_MODEL_SIZE, _MAT_MODEL_SIZE):
            output = cv2.resize(
                output, original_size,
                interpolation=cv2.INTER_LANCZOS4
            )

        return output


def quick_mat_inpaint(
    image: np.ndarray,
    mask: np.ndarray,
    device: str = "cuda"
) -> np.ndarray:
    """Quick MAT inpainting (convenience function).

    Args:
        image: RGB image (H, W, 3).
        mask: Binary mask (H, W).
        device: Device to use ("cuda" or "cpu").

    Returns:
        Inpainted image using MAT.
    """
    inpainter = MATInpainter(device=device)
    return inpainter.inpaint(image, mask)
