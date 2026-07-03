"""Hybrid Inpainting Selector for automatic strategy selection.

This module implements intelligent inpainting method selection based on region
complexity analysis. It automatically chooses between PatchMatch, Symmetry-guided,
and diffusion-based approaches for optimal results.

Strategy Selection:
| Condition | Method | Reason |
|-----------|--------|--------|
| occlusion_ratio < 0.3 & simple texture | PatchMatch | Fast, preserves style |
| symmetric object (ceramic) & ratio < 0.5 | Symmetry + PatchMatch | Uses object symmetry |
| ratio >= 0.3 or complex pattern | LaMa/Reference Diffusion | Generative capability |
"""

from enum import Enum
from typing import Dict, Optional, Tuple
import numpy as np
import cv2
from scipy import ndimage

from .symmetry_inpaint import SymmetryGuidedInpainter, PatchMatchInpainter
from .inpainting import LamaInpainter, InpaintingModule


class InpaintingStrategy(Enum):
    """Available inpainting strategies."""
    PATCHMATCH = "patchmatch"
    SYMMETRY_PATCHMATCH = "symmetry_patchmatch"
    REFERENCE_DIFFUSION = "reference_diffusion"
    LAMA = "lama"
    OPENCV = "opencv"


class RegionAnalyzer:
    """Analyze occluded region to determine optimal inpainting strategy."""

    def __init__(self):
        """Initialize region analyzer."""
        self._texture_complexity_threshold = 0.5
        self._symmetry_score_threshold = 0.6

    def analyze(
        self,
        image: np.ndarray,
        inpaint_mask: np.ndarray,
        object_mask: np.ndarray,
        object_type: Optional[str] = None
    ) -> Dict:
        """
        Analyze region characteristics to recommend best inpainting strategy.

        Args:
            image: RGB image (H, W, 3)
            inpaint_mask: Binary mask of region to inpaint (255 = inpaint)
            object_mask: Binary mask of full object (255 = object)
            object_type: Optional object type hint ('ceramic', 'furniture', etc.)

        Returns:
            Dict with:
            - occlusion_ratio: float (0-1)
            - texture_complexity: float (0-1, 0=simple, 1=complex)
            - has_symmetry: bool
            - edge_density: float (0-1)
            - recommended_strategy: InpaintingStrategy
        """
        # Ensure binary masks
        inpaint_binary = (inpaint_mask > 127).astype(np.uint8)
        object_binary = (object_mask > 127).astype(np.uint8)

        # Calculate occlusion ratio
        object_area = np.sum(object_binary)
        inpaint_area = np.sum(inpaint_binary)
        occlusion_ratio = inpaint_area / (object_area + 1e-8)

        # Calculate texture complexity
        texture_complexity = self._compute_texture_complexity(image, inpaint_binary)

        # Check for symmetry potential
        has_symmetry = self._check_symmetry_potential(object_binary, object_type)

        # Calculate edge density
        edge_density = self._compute_edge_density(image, inpaint_binary)

        # Recommend strategy
        recommended_strategy = self._select_strategy(
            occlusion_ratio=occlusion_ratio,
            texture_complexity=texture_complexity,
            has_symmetry=has_symmetry,
            edge_density=edge_density,
            object_type=object_type
        )

        return {
            'occlusion_ratio': occlusion_ratio,
            'texture_complexity': texture_complexity,
            'has_symmetry': has_symmetry,
            'edge_density': edge_density,
            'recommended_strategy': recommended_strategy
        }

    def _compute_texture_complexity(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> float:
        """
        Compute texture complexity using gradient magnitude.

        Args:
            image: RGB image
            mask: Binary mask of region to analyze

        Returns:
            Complexity score between 0 and 1 (0=simple, 1=complex)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Compute gradients around the inpaint region (border context)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dilated = cv2.dilate(mask, kernel)
        border_region = dilated - mask

        if np.sum(border_region) == 0:
            return 0.5  # Default to medium complexity

        # Calculate gradient magnitude using Sobel
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        # Get gradient statistics in border region
        border_gradients = grad_mag[border_region > 0]

        # Normalize gradient magnitude
        # High gradient = complex texture, Low gradient = simple texture
        avg_gradient = np.mean(border_gradients)
        std_gradient = np.std(border_gradients)

        # Normalize to 0-1 scale (empirical thresholds)
        complexity = np.clip(avg_gradient / 50.0 + std_gradient / 30.0, 0, 1)

        return complexity

    def _check_symmetry_potential(
        self,
        object_mask: np.ndarray,
        object_type: Optional[str] = None
    ) -> bool:
        """
        Check if object likely has exploitable symmetry.

        Args:
            object_mask: Binary mask of object
            object_type: Optional object type hint

        Returns:
            True if symmetry is likely to be useful
        """
        # Object type hints
        symmetric_types = {'ceramic', 'vase', 'jar', 'bowl', 'cup', 'furniture'}
        if object_type and object_type.lower() in symmetric_types:
            return True

        # Geometric symmetry check
        from .symmetry_inpaint import SymmetryDetector

        detector = SymmetryDetector()
        axis_x = detector.detect_vertical_symmetry(object_mask)

        if axis_x is None:
            return False

        # Check symmetry score
        score = detector.compute_symmetry_score(object_mask, axis_x)
        return score > self._symmetry_score_threshold

    def _compute_edge_density(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> float:
        """
        Compute edge density around the inpaint region.

        Args:
            image: RGB image
            mask: Binary mask of region

        Returns:
            Edge density score between 0 and 1
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Detect edges using Canny
        edges = cv2.Canny(gray, 50, 150)

        # Look at border region
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dilated = cv2.dilate(mask, kernel)
        border_region = dilated - mask

        if np.sum(border_region) == 0:
            return 0.0

        # Calculate edge density in border
        edge_pixels = np.sum(edges[border_region > 0] > 0)
        total_pixels = np.sum(border_region > 0)

        density = edge_pixels / (total_pixels + 1e-8)

        return np.clip(density, 0, 1)

    def _select_strategy(
        self,
        occlusion_ratio: float,
        texture_complexity: float,
        has_symmetry: bool,
        edge_density: float,
        object_type: Optional[str] = None
    ) -> InpaintingStrategy:
        """
        Select optimal inpainting strategy based on analysis.

        Args:
            occlusion_ratio: Ratio of occluded area to object area
            texture_complexity: Texture complexity score (0-1)
            has_symmetry: Whether object has useful symmetry
            edge_density: Edge density score (0-1)
            object_type: Optional object type hint

        Returns:
            Recommended InpaintingStrategy
        """
        # Strategy decision tree

        # Large occlusion or complex texture -> Use generative methods
        if occlusion_ratio >= 0.3 or texture_complexity > 0.7:
            return InpaintingStrategy.LAMA

        # Symmetric object with moderate occlusion -> Use symmetry
        if has_symmetry and occlusion_ratio < 0.5:
            return InpaintingStrategy.SYMMETRY_PATCHMATCH

        # Small occlusion with simple texture -> Use PatchMatch
        if occlusion_ratio < 0.3 and texture_complexity < 0.5:
            return InpaintingStrategy.PATCHMATCH

        # Default to LaMa for complex cases
        return InpaintingStrategy.LAMA


class HybridInpainter:
    """Hybrid inpainter that selects optimal method based on region analysis.

    Strategy selection:
    | Condition | Method | Reason |
    |-----------|--------|--------|
    | occlusion_ratio < 0.3 & simple texture | PatchMatch | Fast, preserves style |
    | symmetric object (ceramic) & ratio < 0.5 | Symmetry + PatchMatch | Uses object symmetry |
    | ratio >= 0.3 or complex pattern | LaMa | Generative capability |
    """

    def __init__(
        self,
        enable_symmetry: bool = True,
        enable_diffusion: bool = True,
        fallback_to_lama: bool = True
    ):
        """
        Initialize hybrid inpainter.

        Args:
            enable_symmetry: Enable symmetry-guided inpainting
            enable_diffusion: Enable diffusion-based inpainting
            fallback_to_lama: Use LaMa as fallback for complex cases
        """
        self.enable_symmetry = enable_symmetry
        self.enable_diffusion = enable_diffusion
        self.fallback_to_lama = fallback_to_lama

        # Region analyzer
        self.analyzer = RegionAnalyzer()

        # Lazy load inpainters
        self._patchmatch = None
        self._symmetry = None
        self._lama = None
        self._opencv = None

    @property
    def patchmatch(self) -> PatchMatchInpainter:
        """Get PatchMatch inpainter (lazy loaded)."""
        if self._patchmatch is None:
            self._patchmatch = PatchMatchInpainter()
        return self._patchmatch

    @property
    def symmetry(self) -> SymmetryGuidedInpainter:
        """Get symmetry-guided inpainter (lazy loaded)."""
        if self._symmetry is None:
            self._symmetry = SymmetryGuidedInpainter(use_patchmatch_fallback=True)
        return self._symmetry

    @property
    def lama(self) -> LamaInpainter:
        """Get LaMa inpainter (lazy loaded)."""
        if self._lama is None:
            self._lama = LamaInpainter()
        return self._lama

    @property
    def opencv(self) -> InpaintingModule:
        """Get OpenCV inpainter (lazy loaded)."""
        if self._opencv is None:
            self._opencv = InpaintingModule(method="ns", radius=7)
        return self._opencv

    def select_strategy(
        self,
        occlusion_info: Dict,
        object_type: Optional[str] = None
    ) -> InpaintingStrategy:
        """
        Select optimal inpainting strategy based on analysis.

        Args:
            occlusion_info: Dict from RegionAnalyzer.analyze()
            object_type: Optional object type hint

        Returns:
            Selected InpaintingStrategy
        """
        recommended = occlusion_info['recommended_strategy']

        # Check if recommended strategy is enabled
        if recommended == InpaintingStrategy.SYMMETRY_PATCHMATCH and not self.enable_symmetry:
            # Fallback to PatchMatch or LaMa
            if occlusion_info['occlusion_ratio'] < 0.3:
                return InpaintingStrategy.PATCHMATCH
            else:
                return InpaintingStrategy.LAMA if self.fallback_to_lama else InpaintingStrategy.OPENCV

        if recommended == InpaintingStrategy.LAMA and not self.enable_diffusion:
            # Fallback to OpenCV
            return InpaintingStrategy.OPENCV

        return recommended

    def inpaint(
        self,
        image: np.ndarray,
        inpaint_mask: np.ndarray,
        object_mask: np.ndarray,
        reference_mask: Optional[np.ndarray] = None,
        object_type: Optional[str] = None,
        strategy: Optional[InpaintingStrategy] = None
    ) -> Tuple[np.ndarray, InpaintingStrategy]:
        """
        Inpaint using optimal strategy.

        Args:
            image: RGB image (H, W, 3), uint8
            inpaint_mask: Region to inpaint (255 = inpaint, 0 = keep)
            object_mask: Full object mask (for symmetry analysis)
            reference_mask: Optional reference region mask (for reference-guided methods)
            object_type: Object type hint ('ceramic', 'furniture', etc.)
            strategy: Force specific strategy (overrides auto-selection)

        Returns:
            Tuple of (inpainted_image, strategy_used)
        """
        # Ensure uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        if inpaint_mask.dtype != np.uint8:
            inpaint_mask = (inpaint_mask > 0).astype(np.uint8) * 255
        if object_mask.dtype != np.uint8:
            object_mask = (object_mask > 0).astype(np.uint8) * 255

        # Analyze region if strategy not specified
        if strategy is None:
            occlusion_info = self.analyzer.analyze(
                image, inpaint_mask, object_mask, object_type
            )
            strategy = self.select_strategy(occlusion_info, object_type)

        # Execute inpainting based on strategy
        if strategy == InpaintingStrategy.PATCHMATCH:
            # Use PatchMatch with visible object parts as exemplar
            exemplar_mask = cv2.bitwise_and(
                object_mask,
                cv2.bitwise_not(inpaint_mask)
            )
            result = self.patchmatch.inpaint(image, inpaint_mask, exemplar_mask)

        elif strategy == InpaintingStrategy.SYMMETRY_PATCHMATCH:
            # Use symmetry-guided inpainting
            result = self.symmetry.inpaint(image, inpaint_mask, object_mask)

        elif strategy == InpaintingStrategy.LAMA:
            # Use LaMa deep learning inpainting
            result = self.lama.inpaint(image, inpaint_mask)

        elif strategy == InpaintingStrategy.REFERENCE_DIFFUSION:
            # Reference-guided inpainting (if reference mask provided)
            if reference_mask is not None:
                # Use reference-based inpainting
                opencv_module = InpaintingModule(method="ns")
                result = opencv_module.inpaint_with_reference(
                    image, inpaint_mask, reference_mask
                )
            else:
                # Fallback to LaMa
                result = self.lama.inpaint(image, inpaint_mask)

        else:  # OPENCV
            # Use OpenCV Navier-Stokes
            result = self.opencv.inpaint(image, inpaint_mask)

        return result, strategy

    def analyze_region(
        self,
        image: np.ndarray,
        inpaint_mask: np.ndarray,
        object_mask: np.ndarray,
        object_type: Optional[str] = None
    ) -> Dict:
        """
        Analyze region and return recommendations without inpainting.

        Args:
            image: RGB image
            inpaint_mask: Region to analyze
            object_mask: Full object mask
            object_type: Optional object type hint

        Returns:
            Analysis results dict
        """
        return self.analyzer.analyze(image, inpaint_mask, object_mask, object_type)


def hybrid_inpaint(
    image: np.ndarray,
    inpaint_mask: np.ndarray,
    object_mask: np.ndarray,
    reference_mask: Optional[np.ndarray] = None,
    object_type: Optional[str] = None
) -> np.ndarray:
    """
    Quick hybrid inpainting with automatic strategy selection.

    Convenience function for one-shot inpainting without managing the HybridInpainter instance.

    Args:
        image: RGB image (H, W, 3), uint8
        inpaint_mask: Region to inpaint (255 = inpaint, 0 = keep)
        object_mask: Full object mask (for symmetry and analysis)
        reference_mask: Optional reference region mask
        object_type: Object type hint ('ceramic', 'furniture', etc.)

    Returns:
        Inpainted image (H, W, 3), uint8

    Example:
        >>> result = hybrid_inpaint(image, mask, object_mask, object_type='ceramic')
    """
    inpainter = HybridInpainter()
    result, _ = inpainter.inpaint(
        image,
        inpaint_mask,
        object_mask,
        reference_mask=reference_mask,
        object_type=object_type
    )
    return result
