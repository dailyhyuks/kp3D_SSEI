"""
Morphological Grid Detector for Korean Paintings.

This module provides a MorphologicalGridDetector class that uses directional
and multi-scale morphological operations to detect grid patterns commonly
found in traditional Korean paintings. The detector employs rotated linear
structuring elements combined with top-hat transforms to extract both bright
and dark grid lines at multiple orientations and scales.

The algorithm aggregates responses using geometric mean across scales (to
reduce false positives) and arithmetic mean across angles (to capture grid
lines in all orientations).
"""

from typing import Dict, Tuple, List
import numpy as np
import cv2


class MorphologicalGridDetector:
    """
    Detector for grid patterns using directional morphological operations.

    This class detects grid lines in grayscale images by applying white and
    black top-hat transforms with rotated linear structuring elements at
    multiple scales and orientations. The responses are aggregated using
    geometric mean across scales and arithmetic mean across angles to produce
    a robust grid confidence map.

    Parameters
    ----------
    line_lengths : tuple of int
        Lengths of the linear structuring elements for multi-scale analysis.
        Default is (5, 9, 15).
    line_width : int
        Width of the linear structuring elements. Default is 1.
    angles : tuple of float
        Rotation angles in degrees for directional analysis.
        Default is (0, 45, 90, 135) covering horizontal, diagonal, and vertical.
    threshold : float
        Threshold for binary mask generation from confidence map.
        Default is 0.3.

    Attributes
    ----------
    line_lengths : tuple of int
        Configured line lengths for multi-scale detection.
    line_width : int
        Width of linear kernels.
    angles : tuple of float
        Rotation angles for directional kernels.
    threshold : float
        Binary mask threshold value.
    kernels : dict
        Dictionary mapping (angle, length) tuples to kernel arrays.
    """

    def __init__(
        self,
        line_lengths: Tuple[int, ...] = (5, 9, 15),
        line_width: int = 1,
        angles: Tuple[float, ...] = (0, 45, 90, 135),
        threshold: float = 0.3
    ) -> None:
        """
        Initialize the MorphologicalGridDetector.

        Parameters
        ----------
        line_lengths : tuple of int
            Lengths of linear structuring elements for multi-scale analysis.
        line_width : int
            Width of the linear structuring elements.
        angles : tuple of float
            Rotation angles in degrees for directional analysis.
        threshold : float
            Threshold for binary mask generation.
        """
        self.line_lengths = line_lengths
        self.line_width = line_width
        self.angles = angles
        self.threshold = threshold
        self.kernels: Dict[Tuple[float, int], np.ndarray] = {}

        # Pre-create all directional kernels
        self.kernels = self.create_directional_kernels()

    def create_directional_kernels(self) -> Dict[Tuple[float, int], np.ndarray]:
        """
        Create rotated linear structuring elements for each angle and scale.

        For each combination of angle and line length, creates a rotated
        linear kernel using cv2.getRotationMatrix2D and cv2.warpAffine.
        The base kernel is a horizontal line centered in a square matrix.

        Returns
        -------
        dict
            Dictionary mapping (angle, length) tuples to numpy arrays
            representing the rotated linear structuring elements.
        """
        kernels: Dict[Tuple[float, int], np.ndarray] = {}

        for length in self.line_lengths:
            # Create a square matrix large enough to contain rotated line
            # The diagonal of the line is the maximum extent
            max_dim = int(np.ceil(length * np.sqrt(2))) + 2
            # Ensure odd dimension for proper centering
            if max_dim % 2 == 0:
                max_dim += 1

            center = max_dim // 2

            for angle in self.angles:
                # Create base horizontal line kernel
                base_kernel = np.zeros((max_dim, max_dim), dtype=np.uint8)

                # Draw horizontal line through center
                half_length = length // 2
                start_col = center - half_length
                end_col = center + half_length + (1 if length % 2 == 1 else 0)

                # Draw the line with specified width
                for w in range(self.line_width):
                    row = center - self.line_width // 2 + w
                    if 0 <= row < max_dim:
                        base_kernel[row, start_col:end_col] = 1

                if angle == 0:
                    # No rotation needed for horizontal
                    rotated_kernel = base_kernel
                else:
                    # Get rotation matrix
                    rotation_matrix = cv2.getRotationMatrix2D(
                        center=(center, center),
                        angle=angle,
                        scale=1.0
                    )

                    # Apply rotation
                    rotated_kernel = cv2.warpAffine(
                        base_kernel,
                        rotation_matrix,
                        (max_dim, max_dim),
                        flags=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0
                    )

                # Ensure kernel has at least some non-zero elements
                if np.sum(rotated_kernel) == 0:
                    # Fallback: create a simple line kernel
                    rotated_kernel[center, center] = 1

                kernels[(angle, length)] = rotated_kernel

        return kernels

    def white_tophat_directional(
        self,
        image_gray: np.ndarray,
        kernel: np.ndarray
    ) -> np.ndarray:
        """
        Extract bright grid lines using white top-hat transform.

        The white top-hat transform highlights structures that are brighter
        than their surroundings and smaller than the structuring element.

        Parameters
        ----------
        image_gray : np.ndarray
            Grayscale input image.
        kernel : np.ndarray
            Structuring element for morphological operation.

        Returns
        -------
        np.ndarray
            White top-hat response showing bright line structures.
        """
        return cv2.morphologyEx(image_gray, cv2.MORPH_TOPHAT, kernel)

    def black_tophat_directional(
        self,
        image_gray: np.ndarray,
        kernel: np.ndarray
    ) -> np.ndarray:
        """
        Extract dark grid lines using black top-hat transform.

        The black top-hat transform (bottom-hat) highlights structures that
        are darker than their surroundings and smaller than the structuring
        element.

        Parameters
        ----------
        image_gray : np.ndarray
            Grayscale input image.
        kernel : np.ndarray
            Structuring element for morphological operation.

        Returns
        -------
        np.ndarray
            Black top-hat response showing dark line structures.
        """
        return cv2.morphologyEx(image_gray, cv2.MORPH_BLACKHAT, kernel)

    def extract_grid_lines(
        self,
        image_gray: np.ndarray
    ) -> Tuple[np.ndarray, Dict[float, np.ndarray]]:
        """
        Multi-scale and multi-direction grid line extraction.

        For each combination of angle and line length:
        1. Apply white top-hat and black top-hat transforms
        2. Combine responses: grid_response = white_tophat + black_tophat
        3. Aggregate across scales using geometric mean
        4. Aggregate across angles using arithmetic mean

        The geometric mean across scales reduces false positives by requiring
        consistent responses across multiple scales. The arithmetic mean
        across angles captures grid lines in all orientations equally.

        Parameters
        ----------
        image_gray : np.ndarray
            Grayscale input image (float32 recommended).

        Returns
        -------
        tuple
            - grid_mask_float : np.ndarray
                Combined grid response as float array.
            - per_direction_responses : dict
                Dictionary mapping each angle to its scale-aggregated response.
        """
        # Ensure float32 for precision
        if image_gray.dtype != np.float32:
            image_gray = image_gray.astype(np.float32)

        per_direction_responses: Dict[float, np.ndarray] = {}

        for angle in self.angles:
            # Collect responses for all scales at this angle
            scale_responses: List[np.ndarray] = []

            for length in self.line_lengths:
                kernel = self.kernels[(angle, length)]

                # Apply both top-hat transforms
                white_response = self.white_tophat_directional(image_gray, kernel)
                black_response = self.black_tophat_directional(image_gray, kernel)

                # Combine bright and dark line responses
                grid_response = white_response + black_response
                scale_responses.append(grid_response)

            # Aggregate across scales using geometric mean
            # This requires consistent response across scales, reducing noise
            if len(scale_responses) > 0:
                # Stack responses and compute geometric mean
                responses_stack = np.stack(scale_responses, axis=0)
                n_scales = len(scale_responses)

                # Add small epsilon to avoid issues with zero values
                epsilon = 1e-10
                responses_stack = np.maximum(responses_stack, epsilon)

                # Geometric mean: (prod of values)^(1/n)
                combined_scale = np.prod(responses_stack, axis=0) ** (1.0 / n_scales)

                per_direction_responses[angle] = combined_scale

        # Aggregate across angles using arithmetic mean
        if len(per_direction_responses) > 0:
            direction_responses_list = list(per_direction_responses.values())
            direction_stack = np.stack(direction_responses_list, axis=0)
            grid_mask_float = np.mean(direction_stack, axis=0)
        else:
            # Fallback for empty case
            grid_mask_float = np.zeros_like(image_gray)

        return grid_mask_float, per_direction_responses

    def compute_grid_confidence_map(
        self,
        image_bgr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full pipeline to compute grid confidence map from BGR image.

        Pipeline steps:
        1. Convert BGR to grayscale float32
        2. Extract grid lines using multi-scale/multi-direction analysis
        3. Normalize confidence to [0, 1] range
        4. Apply Gaussian smoothing
        5. Generate binary mask using threshold

        Parameters
        ----------
        image_bgr : np.ndarray
            Input image in BGR color format (as from cv2.imread).

        Returns
        -------
        tuple
            - confidence_map : np.ndarray
                Float32 array in [0, 1] range indicating grid line confidence.
            - binary_mask : np.ndarray
                Binary uint8 mask where 255 indicates detected grid lines.
        """
        # Step 1: Convert BGR to grayscale float32
        if len(image_bgr.shape) == 3:
            image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image_bgr.copy()

        image_gray = image_gray.astype(np.float32)

        # Step 2: Extract grid lines
        grid_response, per_direction = self.extract_grid_lines(image_gray)

        # Step 3: Normalize to [0, 1]
        min_val = np.min(grid_response)
        max_val = np.max(grid_response)

        if max_val - min_val > 1e-10:
            confidence_map = (grid_response - min_val) / (max_val - min_val)
        else:
            confidence_map = np.zeros_like(grid_response)

        # Step 4: Apply Gaussian blur for smoothing
        confidence_map = cv2.GaussianBlur(
            confidence_map,
            ksize=(5, 5),
            sigmaX=1.0,
            sigmaY=1.0
        )

        # Ensure still in [0, 1] after blur
        confidence_map = np.clip(confidence_map, 0.0, 1.0)

        # Step 5: Generate binary mask using threshold
        binary_mask = (confidence_map > self.threshold).astype(np.uint8) * 255

        return confidence_map, binary_mask


__all__ = ["MorphologicalGridDetector"]
