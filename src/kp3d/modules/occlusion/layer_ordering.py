"""Layer ordering module based on depth estimation.

Determines foreground/background relationships between segmented
objects using monocular depth information.
"""

from typing import List, Tuple, Optional
import numpy as np

from kp3d.modules.occlusion.base import LayerInfo
from kp3d.modules.occlusion.depth import DepthEstimatorWrapper


class LayerOrderingModule:
    """Determine layer order (front-to-back) from depth maps.

    Uses mean depth values to establish which objects are in front
    of others. Lower depth values indicate closer (foreground) objects.
    """

    def __init__(
        self,
        depth_estimator: Optional[DepthEstimatorWrapper] = None,
        device=None
    ):
        """Initialize layer ordering module.

        Args:
            depth_estimator: Pre-initialized depth estimator.
                            Created automatically if not provided.
            device: Computation device for depth estimation.
        """
        self.depth_estimator = depth_estimator or DepthEstimatorWrapper(device=device)

    def compute_layer_depths(
        self,
        depth_map: np.ndarray,
        layers: List[LayerInfo]
    ) -> List[LayerInfo]:
        """Compute mean depth for each layer and update LayerInfo.

        Args:
            depth_map: Full depth map (H, W), values in [0, 1].
            layers: List of LayerInfo with masks.

        Returns:
            Updated layers with mean_depth filled in.
        """
        for layer in layers:
            layer.mean_depth = self.depth_estimator.get_mean_depth(
                depth_map, layer.mask
            )

        return layers

    def order_layers(
        self,
        layers: List[LayerInfo],
        depth_map: Optional[np.ndarray] = None
    ) -> List[LayerInfo]:
        """Order layers from foreground (front) to background (back).

        Args:
            layers: List of LayerInfo objects.
            depth_map: Optional depth map to compute depths if not already set.

        Returns:
            Layers sorted front-to-back (index 0 = foreground).
        """
        if depth_map is not None:
            layers = self.compute_layer_depths(depth_map, layers)

        # Sort by mean_depth (ascending = closer first)
        sorted_layers = sorted(layers, key=lambda x: x.mean_depth)

        # Mark foreground/background
        if len(sorted_layers) > 0:
            sorted_layers[0].is_foreground = True
            for layer in sorted_layers[1:]:
                layer.is_foreground = False

        return sorted_layers

    def identify_foreground(
        self,
        layers: List[LayerInfo],
        depth_map: Optional[np.ndarray] = None
    ) -> Tuple[LayerInfo, List[LayerInfo]]:
        """Identify the foreground layer and separate from background layers.

        Args:
            layers: List of LayerInfo objects.
            depth_map: Optional depth map for depth computation.

        Returns:
            Tuple of (foreground_layer, background_layers).

        Raises:
            ValueError: If no layers provided.
        """
        if not layers:
            raise ValueError("No layers provided for ordering")

        ordered = self.order_layers(layers, depth_map)

        foreground = ordered[0]
        background = ordered[1:] if len(ordered) > 1 else []

        return foreground, background

    def get_occlusion_order(
        self,
        layers: List[LayerInfo]
    ) -> List[Tuple[LayerInfo, LayerInfo]]:
        """Get pairs of (occluding, occluded) layers.

        Returns all pairs where the first layer occludes (is in front of)
        the second layer.

        Args:
            layers: Pre-ordered layers (front to back).

        Returns:
            List of (front_layer, back_layer) tuples.
        """
        pairs = []
        for i, front_layer in enumerate(layers):
            for back_layer in layers[i+1:]:
                # Check if masks overlap
                overlap = np.logical_and(front_layer.mask, back_layer.mask)
                if np.any(overlap):
                    pairs.append((front_layer, back_layer))

        return pairs


class SimpleLayerOrdering:
    """Simplified layer ordering for Phase 1 (2-layer case).

    Designed for the ceramic-on-soban case where:
    - Foreground: ceramic vase (closer, smaller)
    - Background: wooden table/soban (farther, larger)
    """

    def __init__(self):
        """Initialize simple ordering module."""
        pass

    def order_by_label(
        self,
        layers: List[LayerInfo],
        foreground_keywords: List[str] = ["ceramic", "vase", "jar", "pot"],
        background_keywords: List[str] = ["table", "soban", "wooden", "stand"]
    ) -> Tuple[Optional[LayerInfo], Optional[LayerInfo]]:
        """Order layers by matching label keywords.

        Fallback method when depth is ambiguous.

        Args:
            layers: List of LayerInfo objects.
            foreground_keywords: Keywords indicating foreground objects.
            background_keywords: Keywords indicating background objects.

        Returns:
            Tuple of (foreground, background) or (None, None) if not found.
        """
        foreground = None
        background = None

        for layer in layers:
            label_lower = layer.label.lower()

            if any(kw in label_lower for kw in foreground_keywords):
                foreground = layer
            elif any(kw in label_lower for kw in background_keywords):
                background = layer

        return foreground, background

    def order_by_area(
        self,
        layers: List[LayerInfo]
    ) -> Tuple[LayerInfo, LayerInfo]:
        """Order layers by mask area (smaller = foreground).

        Heuristic: In many paintings, foreground objects are smaller
        and background surfaces are larger.

        Args:
            layers: List of exactly 2 LayerInfo objects.

        Returns:
            Tuple of (smaller_layer, larger_layer).

        Raises:
            ValueError: If not exactly 2 layers.
        """
        if len(layers) != 2:
            raise ValueError(f"Expected 2 layers, got {len(layers)}")

        areas = [np.sum(layer.mask > 0) for layer in layers]

        if areas[0] < areas[1]:
            return layers[0], layers[1]
        else:
            return layers[1], layers[0]

    def order_combined(
        self,
        layers: List[LayerInfo],
        depth_map: Optional[np.ndarray] = None
    ) -> Tuple[LayerInfo, LayerInfo]:
        """Combined ordering using depth, labels, and area.

        Priority:
        1. Depth (if available and difference > threshold)
        2. Label keywords
        3. Area (fallback)

        Args:
            layers: List of exactly 2 LayerInfo objects.
            depth_map: Optional depth map.

        Returns:
            Tuple of (foreground, background).
        """
        if len(layers) != 2:
            raise ValueError(f"Expected 2 layers, got {len(layers)}")

        # Try depth first
        if depth_map is not None:
            from kp3d.modules.occlusion.depth import DepthEstimatorWrapper
            estimator = DepthEstimatorWrapper()

            d0 = estimator.get_mean_depth(depth_map, layers[0].mask)
            d1 = estimator.get_mean_depth(depth_map, layers[1].mask)

            # If depth difference is significant (> 0.05)
            if abs(d0 - d1) > 0.05:
                if d0 < d1:
                    layers[0].is_foreground = True
                    return layers[0], layers[1]
                else:
                    layers[1].is_foreground = True
                    return layers[1], layers[0]

        # Try label keywords
        fg, bg = self.order_by_label(layers)
        if fg is not None and bg is not None:
            fg.is_foreground = True
            return fg, bg

        # Fallback to area
        fg, bg = self.order_by_area(layers)
        fg.is_foreground = True
        return fg, bg
