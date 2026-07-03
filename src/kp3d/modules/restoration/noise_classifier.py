"""Shape-aware noise type classifier for v7 restoration.

Classifies detected noise blobs into categories based on geometric
properties of their contours, enabling type-specific inpainting strategies.

Categories:
- dust: Small, circular isolated spots from pigment degradation
- crack: Elongated, branching patterns from surface damage
- stain: Irregular, large area discolorations
- mold: Fuzzy, semi-circular clusters from biological growth
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple


class NoiseShapeClassifier:
    """Classifies noise blobs by shape analysis.

    Uses contour features: circularity, aspect ratio, solidity,
    and fractal dimension estimate to determine noise type.
    """

    # Feature thresholds for classification
    PROFILES = {
        "dust": {
            "circularity_min": 0.6,
            "aspect_ratio_max": 2.0,
            "solidity_min": 0.7,
            "area_max": 150,
        },
        "crack": {
            "circularity_max": 0.3,
            "aspect_ratio_min": 3.0,
            "solidity_max": 0.6,
            "area_min": 20,
        },
        "stain": {
            "circularity_max": 0.7,
            "solidity_min": 0.5,
            "area_min": 100,
            "fractal_dim_max": 1.3,
        },
        "mold": {
            "circularity_min": 0.3,
            "circularity_max": 0.7,
            "solidity_max": 0.7,
            "area_min": 30,
            "fractal_dim_min": 1.2,
        },
    }

    # Default inpainting strategy per type
    DEFAULT_STRATEGY_MAP = {
        "dust": "patchmatch",
        "crack": "ns",
        "stain": "color_aware",
        "mold": "patchmatch",
    }

    def __init__(
        self,
        strategy_map: Optional[Dict[str, str]] = None,
    ):
        """Initialize classifier.

        Args:
            strategy_map: Override for noise_type -> inpaint_method mapping.
        """
        self.strategy_map = strategy_map or self.DEFAULT_STRATEGY_MAP.copy()

    def compute_features(self, contour: np.ndarray) -> Dict[str, float]:
        """Compute shape features for a single contour.

        Args:
            contour: OpenCV contour array.

        Returns:
            Feature dictionary with circularity, aspect_ratio, solidity,
            fractal_dim, and area.
        """
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        # Circularity: 4*pi*area / perimeter^2
        circularity = 0.0
        if perimeter > 1e-6:
            circularity = min(4 * np.pi * area / (perimeter ** 2), 1.0)

        # Aspect ratio from bounding rect
        rect = cv2.minAreaRect(contour)
        (_, (w_rect, h_rect), _) = rect
        if min(w_rect, h_rect) > 0:
            aspect_ratio = max(w_rect, h_rect) / min(w_rect, h_rect)
        else:
            aspect_ratio = 1.0

        # Solidity: area / convex hull area
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / (hull_area + 1e-6)

        # Fractal dimension estimate (box-counting approximation)
        fractal_dim = self._estimate_fractal_dimension(contour)

        return {
            "circularity": circularity,
            "aspect_ratio": aspect_ratio,
            "solidity": solidity,
            "fractal_dim": fractal_dim,
            "area": area,
        }

    def _estimate_fractal_dimension(self, contour: np.ndarray) -> float:
        """Estimate fractal dimension using perimeter-area relationship.

        D ≈ 2 * log(perimeter/4) / log(area) for simple shapes.
        More complex boundaries yield higher D values.

        Args:
            contour: OpenCV contour.

        Returns:
            Estimated fractal dimension (1.0 - 2.0).
        """
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if area < 4 or perimeter < 4:
            return 1.0

        # Normalized complexity: perimeter^2 / (4*pi*area)
        # For a circle this equals 1.0, for complex shapes > 1.0
        complexity = (perimeter ** 2) / (4 * np.pi * area)

        # Map to fractal dimension estimate [1.0, 2.0]
        fractal_dim = 1.0 + np.log(max(complexity, 1.0)) / np.log(100)
        fractal_dim = np.clip(fractal_dim, 1.0, 2.0)

        return float(fractal_dim)

    def classify_single(self, features: Dict[str, float]) -> str:
        """Classify a single blob based on its features.

        Uses scoring system: each matching criterion adds to a type's score.
        The type with highest score wins. Defaults to "dust" on ties.

        Args:
            features: Feature dictionary from compute_features().

        Returns:
            Noise type string: "dust", "crack", "stain", or "mold".
        """
        scores = {"dust": 0.0, "crack": 0.0, "stain": 0.0, "mold": 0.0}

        for noise_type, profile in self.PROFILES.items():
            for key, threshold in profile.items():
                feat_name, comp = key.rsplit("_", 1)
                feat_val = features.get(feat_name, 0.0)

                if comp == "min" and feat_val >= threshold:
                    scores[noise_type] += 1.0
                elif comp == "max" and feat_val <= threshold:
                    scores[noise_type] += 1.0

        # Return type with highest score
        best_type = max(scores, key=lambda k: (scores[k], k == "dust"))
        return best_type

    def classify_mask(
        self,
        mask: np.ndarray,
    ) -> Tuple[Dict[int, str], Dict[int, Dict[str, float]]]:
        """Classify all blobs in a binary mask.

        Args:
            mask: Binary mask (uint8, 255=noise).

        Returns:
            Tuple of:
                label_types: mapping from label id -> noise type
                label_features: mapping from label id -> feature dict
        """
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        # Find contours for shape analysis
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Build contour-to-label map using centroids
        contour_label_map = {}
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                for i in range(1, num_labels):
                    lx, ly = centroids[i]
                    if abs(cx - lx) < 2 and abs(cy - ly) < 2:
                        contour_label_map[i] = contour
                        break

        label_types = {}
        label_features = {}

        for label_id in range(1, num_labels):
            if label_id in contour_label_map:
                features = self.compute_features(contour_label_map[label_id])
            else:
                # Fallback: use basic stats
                area = float(stats[label_id, cv2.CC_STAT_AREA])
                features = {
                    "circularity": 0.5,
                    "aspect_ratio": 1.5,
                    "solidity": 0.7,
                    "fractal_dim": 1.1,
                    "area": area,
                }

            label_features[label_id] = features
            label_types[label_id] = self.classify_single(features)

        return label_types, label_features

    def get_strategy(self, noise_type: str) -> str:
        """Get inpainting strategy for a noise type.

        Args:
            noise_type: One of "dust", "crack", "stain", "mold".

        Returns:
            Inpainting method string.
        """
        return self.strategy_map.get(noise_type, "patchmatch")

    def create_strategy_masks(
        self,
        mask: np.ndarray,
        label_types: Dict[int, str],
    ) -> Dict[str, np.ndarray]:
        """Create per-strategy binary masks.

        Args:
            mask: Original binary mask.
            label_types: Label -> type mapping from classify_mask().

        Returns:
            Dict mapping strategy name -> binary mask for that strategy.
        """
        num_labels, labels = cv2.connectedComponents(mask, connectivity=8)

        strategy_masks: Dict[str, np.ndarray] = {}

        for label_id, noise_type in label_types.items():
            strategy = self.get_strategy(noise_type)
            if strategy not in strategy_masks:
                strategy_masks[strategy] = np.zeros_like(mask)
            strategy_masks[strategy][labels == label_id] = 255

        return strategy_masks

    def get_classification_summary(
        self,
        label_types: Dict[int, str],
    ) -> Dict[str, int]:
        """Get count summary of classifications.

        Args:
            label_types: Label -> type mapping.

        Returns:
            Dict mapping noise_type -> count.
        """
        summary = {"dust": 0, "crack": 0, "stain": 0, "mold": 0}
        for noise_type in label_types.values():
            summary[noise_type] = summary.get(noise_type, 0) + 1
        return summary
