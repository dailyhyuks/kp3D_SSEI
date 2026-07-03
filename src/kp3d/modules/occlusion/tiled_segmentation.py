"""Tiled segmentation for high-resolution images.

Splits large images into overlapping tiles, processes each tile with SAM,
and merges results using Non-Maximum Suppression (NMS).
"""

from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
import numpy as np
import cv2
import warnings


@dataclass
class TileConfig:
    """Configuration for tiled segmentation.

    Attributes:
        tile_size: Size of each tile (width and height).
        overlap: Overlap between adjacent tiles in pixels.
        min_area_ratio: Minimum mask area as ratio of tile size.
        max_area_ratio: Maximum mask area as ratio of tile size.
        iou_threshold: IoU threshold for NMS merging.
        skip_empty_tiles: Skip tiles with low content (mostly background).
        empty_threshold: Threshold for considering a tile empty (0-1).
        multiscale: Also process at full resolution for large objects.
        multiscale_factor: Downscale factor for full-image pass.
    """
    tile_size: int = 1024
    overlap: int = 256
    min_area_ratio: float = 0.005
    max_area_ratio: float = 0.9
    iou_threshold: float = 0.5
    skip_empty_tiles: bool = True
    empty_threshold: float = 0.95
    multiscale: bool = True
    multiscale_factor: float = 0.25


@dataclass
class DetectedObject:
    """Represents a detected and segmented object.

    Attributes:
        mask: Binary mask (H, W), uint8, 0 or 255.
        bbox: Bounding box [x1, y1, x2, y2].
        area: Mask area in pixels.
        score: Detection confidence score.
        stability_score: SAM stability score.
        label: Optional text label (from Grounded-SAM).
        source_tile: Which tile this detection came from.
    """
    mask: np.ndarray
    bbox: List[int]
    area: int
    score: float = 1.0
    stability_score: float = 1.0
    label: str = ""
    source_tile: int = -1
    mean_depth: float = 0.0


class TiledSegmentation:
    """High-resolution image segmentation using tiled processing.

    Splits large images into overlapping tiles, runs segmentation on each,
    then merges results with NMS to remove duplicates.

    Example:
        >>> tiled = TiledSegmentation(tile_size=1024, overlap=256)
        >>> objects = tiled.segment(large_image)
        >>> for obj in objects:
        ...     print(f"Object: {obj.area} pixels, bbox: {obj.bbox}")
    """

    def __init__(
        self,
        config: Optional[TileConfig] = None,
        segmenter: Optional[Any] = None,
        use_grounded_sam: bool = False,
        text_prompts: Optional[List[str]] = None,
        device: Optional[str] = None,
        verbose: bool = True
    ):
        """Initialize tiled segmentation.

        Args:
            config: Tile configuration. Uses defaults if None.
            segmenter: Pre-initialized segmenter (AutoSegmentation or GroundedSAM).
            use_grounded_sam: Use Grounded-SAM instead of AutoSegmentation.
            text_prompts: Text prompts for Grounded-SAM.
            device: Computation device ("cuda" or "cpu").
            verbose: Print progress information.
        """
        self.config = config or TileConfig()
        self.verbose = verbose
        self.use_grounded_sam = use_grounded_sam
        self.text_prompts = text_prompts or []
        self.device = device

        # Lazy load segmenter
        self._segmenter = segmenter
        self._initialized = segmenter is not None

    def _init_segmenter(self) -> None:
        """Initialize segmentation model."""
        if self._initialized:
            return

        if self.use_grounded_sam:
            from kp3d.modules.occlusion.grounded_sam import GroundedSAM
            self._segmenter = GroundedSAM(
                sam_model_type="vit_b",
                device=self.device
            )
        else:
            from kp3d.modules.occlusion.auto_segmentation import AutoSegmentation
            self._segmenter = AutoSegmentation(
                model_type="vit_h",
                points_per_side=32,
                min_mask_region_area=500,
                device=self.device
            )

        self._initialized = True

    def _log(self, msg: str) -> None:
        """Print message if verbose."""
        if self.verbose:
            print(msg)

    def _generate_tiles(
        self,
        image: np.ndarray
    ) -> List[Tuple[np.ndarray, Tuple[int, int, int, int], int]]:
        """Generate overlapping tiles from image.

        Args:
            image: Input image (H, W, 3).

        Returns:
            List of (tile_image, (y1, x1, y2, x2), tile_index).
        """
        h, w = image.shape[:2]
        tile_size = self.config.tile_size
        overlap = self.config.overlap
        stride = tile_size - overlap

        tiles = []
        tile_idx = 0

        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y1 = y
                x1 = x
                y2 = min(y + tile_size, h)
                x2 = min(x + tile_size, w)

                tile = image[y1:y2, x1:x2].copy()

                # Pad if tile is smaller than tile_size
                if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                    padded = np.zeros((tile_size, tile_size, 3), dtype=tile.dtype)
                    padded[:tile.shape[0], :tile.shape[1]] = tile
                    tile = padded

                tiles.append((tile, (y1, x1, y2, x2), tile_idx))
                tile_idx += 1

        return tiles

    def _is_empty_tile(self, tile: np.ndarray) -> bool:
        """Check if tile is mostly empty (background).

        Uses variance to detect low-content tiles.
        """
        if not self.config.skip_empty_tiles:
            return False

        # Convert to grayscale
        gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)

        # Check if mostly uniform (low variance)
        variance = np.var(gray) / 255.0

        # Also check if mostly white/light (typical painting background)
        mean_val = np.mean(gray) / 255.0

        # Empty if low variance AND light background
        is_empty = variance < 0.01 and mean_val > self.config.empty_threshold

        return is_empty

    def _segment_tile(
        self,
        tile: np.ndarray,
        tile_bounds: Tuple[int, int, int, int],
        tile_idx: int
    ) -> List[DetectedObject]:
        """Segment a single tile.

        Args:
            tile: Tile image (tile_size, tile_size, 3).
            tile_bounds: (y1, x1, y2, x2) in original image coordinates.
            tile_idx: Tile index for tracking.

        Returns:
            List of DetectedObject with global coordinates.
        """
        y1, x1, y2, x2 = tile_bounds
        tile_h, tile_w = y2 - y1, x2 - x1

        objects = []

        if self.use_grounded_sam and self.text_prompts:
            # Grounded-SAM: text-based detection
            for prompt in self.text_prompts:
                try:
                    detections = self._segmenter.detect_objects(tile, prompt)

                    for det in detections:
                        # Get mask for this detection
                        box = det["box"]
                        mask = self._segmenter.segment_with_boxes(tile, [box])[0]

                        # Crop mask to actual tile size (remove padding)
                        mask = mask[:tile_h, :tile_w]

                        # Convert to global coordinates
                        global_mask, global_bbox = self._to_global_coords(
                            mask, box, (y1, x1)
                        )

                        obj = DetectedObject(
                            mask=global_mask,
                            bbox=global_bbox,
                            area=int(np.sum(global_mask > 0)),
                            score=det["score"],
                            label=prompt,
                            source_tile=tile_idx
                        )
                        objects.append(obj)

                except Exception as e:
                    self._log(f"    Warning: Detection failed for '{prompt}': {e}")
        else:
            # AutoSegmentation: automatic detection
            try:
                masks = self._segmenter.generate_all_masks(tile)

                # Filter by area
                tile_pixels = tile_h * tile_w
                min_area = int(tile_pixels * self.config.min_area_ratio)
                max_area = int(tile_pixels * self.config.max_area_ratio)

                for mask_dict in masks:
                    area = mask_dict["area"]
                    if area < min_area or area > max_area:
                        continue

                    seg = mask_dict["segmentation"].astype(np.uint8) * 255
                    bbox = mask_dict["bbox"]  # [x, y, w, h] format

                    # Crop to actual tile size
                    seg = seg[:tile_h, :tile_w]

                    # Convert bbox from [x, y, w, h] to [x1, y1, x2, y2]
                    box_xyxy = [
                        bbox[0],
                        bbox[1],
                        bbox[0] + bbox[2],
                        bbox[1] + bbox[3]
                    ]

                    # Convert to global coordinates
                    global_mask, global_bbox = self._to_global_coords(
                        seg, box_xyxy, (y1, x1)
                    )

                    obj = DetectedObject(
                        mask=global_mask,
                        bbox=global_bbox,
                        area=int(np.sum(global_mask > 0)),
                        score=mask_dict.get("predicted_iou", 1.0),
                        stability_score=mask_dict.get("stability_score", 1.0),
                        source_tile=tile_idx
                    )
                    objects.append(obj)

            except Exception as e:
                self._log(f"    Warning: Segmentation failed: {e}")

        return objects

    def _to_global_coords(
        self,
        local_mask: np.ndarray,
        local_bbox: List[float],
        offset: Tuple[int, int]
    ) -> Tuple[np.ndarray, List[int]]:
        """Convert local tile coordinates to global image coordinates.

        Args:
            local_mask: Mask in tile coordinates.
            local_bbox: Bounding box [x1, y1, x2, y2] in tile coordinates.
            offset: (y_offset, x_offset) for this tile.

        Returns:
            (global_mask_info, global_bbox) - mask stays local, bbox is global.
        """
        y_off, x_off = offset

        global_bbox = [
            int(local_bbox[0] + x_off),
            int(local_bbox[1] + y_off),
            int(local_bbox[2] + x_off),
            int(local_bbox[3] + y_off)
        ]

        return local_mask, global_bbox

    def _compute_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute IoU between two masks."""
        intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
        union = np.logical_or(mask1 > 0, mask2 > 0).sum()

        if union == 0:
            return 0.0

        return intersection / union

    def _compute_bbox_iou(
        self,
        box1: List[int],
        box2: List[int]
    ) -> float:
        """Compute IoU between two bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _embed_mask_in_full_image(
        self,
        obj: DetectedObject,
        image_shape: Tuple[int, int],
        tile_bounds: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """Embed a tile-local mask into full image coordinates.

        Args:
            obj: DetectedObject with local mask.
            image_shape: (H, W) of full image.
            tile_bounds: (y1, x1, y2, x2) of the tile.

        Returns:
            Full-size mask with object in correct position.
        """
        h, w = image_shape
        full_mask = np.zeros((h, w), dtype=np.uint8)

        y1, x1, y2, x2 = tile_bounds
        tile_h, tile_w = y2 - y1, x2 - x1

        # Get the local mask portion that fits in the tile
        local_mask = obj.mask[:tile_h, :tile_w]

        # Place in full image
        full_mask[y1:y2, x1:x2] = local_mask

        return full_mask

    def _nms_merge(
        self,
        objects: List[DetectedObject],
        image_shape: Tuple[int, int],
        tile_info: Dict[int, Tuple[int, int, int, int]]
    ) -> List[DetectedObject]:
        """Merge overlapping detections using NMS.

        Args:
            objects: List of detected objects from all tiles.
            image_shape: (H, W) of original image.
            tile_info: Mapping of tile_idx to (y1, x1, y2, x2).

        Returns:
            Deduplicated list of objects.
        """
        if not objects:
            return []

        # Sort by score (descending)
        objects = sorted(objects, key=lambda x: x.score, reverse=True)

        # First pass: quick bbox-based filtering
        keep_indices = []
        suppressed = set()

        for i, obj_i in enumerate(objects):
            if i in suppressed:
                continue

            keep_indices.append(i)

            for j in range(i + 1, len(objects)):
                if j in suppressed:
                    continue

                obj_j = objects[j]

                # Quick bbox IoU check
                bbox_iou = self._compute_bbox_iou(obj_i.bbox, obj_j.bbox)

                if bbox_iou > self.config.iou_threshold:
                    # Compute actual mask IoU for confirmation
                    # Embed masks in full image coordinates
                    mask_i = self._embed_mask_in_full_image(
                        obj_i, image_shape, tile_info[obj_i.source_tile]
                    )
                    mask_j = self._embed_mask_in_full_image(
                        obj_j, image_shape, tile_info[obj_j.source_tile]
                    )

                    mask_iou = self._compute_iou(mask_i, mask_j)

                    if mask_iou > self.config.iou_threshold:
                        # Suppress the lower-scored detection
                        suppressed.add(j)

        # Create final list with full-image masks
        merged_objects = []
        for idx in keep_indices:
            obj = objects[idx]

            # Create full-size mask
            full_mask = self._embed_mask_in_full_image(
                obj, image_shape, tile_info[obj.source_tile]
            )

            # Update object with full mask
            merged_obj = DetectedObject(
                mask=full_mask,
                bbox=obj.bbox,
                area=int(np.sum(full_mask > 0)),
                score=obj.score,
                stability_score=obj.stability_score,
                label=obj.label,
                source_tile=obj.source_tile
            )
            merged_objects.append(merged_obj)

        return merged_objects

    def _merge_same_label_objects(
        self,
        objects: List[DetectedObject]
    ) -> List[DetectedObject]:
        """Merge objects with same label that might be split across tiles.

        For Grounded-SAM results, merge nearby objects with same label.
        """
        if not self.use_grounded_sam:
            return objects

        # Group by label
        label_groups: Dict[str, List[DetectedObject]] = {}
        for obj in objects:
            if obj.label not in label_groups:
                label_groups[obj.label] = []
            label_groups[obj.label].append(obj)

        merged = []
        for label, group in label_groups.items():
            if len(group) == 1:
                merged.extend(group)
                continue

            # Try to merge nearby objects
            # For now, keep all - could add spatial clustering
            merged.extend(group)

        return merged

    def segment(
        self,
        image: np.ndarray,
        depth_map: Optional[np.ndarray] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[DetectedObject]:
        """Segment high-resolution image using tiled processing.

        Args:
            image: RGB image (H, W, 3), uint8.
            depth_map: Optional depth map for ordering.
            progress_callback: Optional callback(current, total) for progress.

        Returns:
            List of DetectedObject with full-image masks.
        """
        self._init_segmenter()

        h, w = image.shape[:2]
        self._log(f"Processing image: {w}x{h}")
        self._log(f"Tile size: {self.config.tile_size}, overlap: {self.config.overlap}")

        # Generate tiles
        tiles = self._generate_tiles(image)
        total_tiles = len(tiles)
        self._log(f"Generated {total_tiles} tiles")

        # Track tile bounds for coordinate conversion
        tile_info: Dict[int, Tuple[int, int, int, int]] = {}

        # Process each tile
        all_objects: List[DetectedObject] = []
        processed = 0
        skipped = 0

        for tile, bounds, tile_idx in tiles:
            tile_info[tile_idx] = bounds

            if self._is_empty_tile(tile):
                skipped += 1
                if progress_callback:
                    progress_callback(processed + skipped, total_tiles)
                continue

            self._log(f"  Processing tile {tile_idx + 1}/{total_tiles} at {bounds}")

            objects = self._segment_tile(tile, bounds, tile_idx)
            all_objects.extend(objects)

            self._log(f"    Found {len(objects)} objects")

            processed += 1
            if progress_callback:
                progress_callback(processed + skipped, total_tiles)

        self._log(f"Processed {processed} tiles, skipped {skipped} empty tiles")
        self._log(f"Total detections before NMS: {len(all_objects)}")

        # Optional: Multi-scale pass for large objects
        if self.config.multiscale and len(all_objects) > 0:
            self._log("Running multi-scale pass...")
            multiscale_objects = self._multiscale_pass(image)
            all_objects.extend(multiscale_objects)
            self._log(f"  Added {len(multiscale_objects)} from multi-scale pass")

        # Merge overlapping detections
        self._log("Merging overlapping detections (NMS)...")
        merged_objects = self._nms_merge(all_objects, (h, w), tile_info)
        self._log(f"After NMS: {len(merged_objects)} objects")

        # Merge same-label objects (for Grounded-SAM)
        if self.use_grounded_sam:
            merged_objects = self._merge_same_label_objects(merged_objects)

        # Add depth information if available
        if depth_map is not None:
            for obj in merged_objects:
                mask_pixels = obj.mask > 0
                if np.sum(mask_pixels) > 0:
                    obj.mean_depth = float(np.mean(depth_map[mask_pixels]))

        # Sort by area (largest first)
        merged_objects = sorted(merged_objects, key=lambda x: x.area, reverse=True)

        self._log(f"Final: {len(merged_objects)} objects detected")

        return merged_objects

    def _multiscale_pass(self, image: np.ndarray) -> List[DetectedObject]:
        """Process downscaled full image for large objects.

        Catches objects that span multiple tiles.
        """
        h, w = image.shape[:2]
        factor = self.config.multiscale_factor

        # Downscale
        new_h, new_w = int(h * factor), int(w * factor)
        small_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        objects = []

        if self.use_grounded_sam and self.text_prompts:
            for prompt in self.text_prompts:
                try:
                    mask = self._segmenter.segment_with_text(
                        small_image, prompt, return_all=True
                    )

                    if np.sum(mask) > 0:
                        # Upscale mask back to original size
                        full_mask = cv2.resize(
                            mask, (w, h), interpolation=cv2.INTER_NEAREST
                        )

                        # Get bbox
                        ys, xs = np.where(full_mask > 0)
                        if len(xs) > 0:
                            bbox = [int(xs.min()), int(ys.min()),
                                   int(xs.max()), int(ys.max())]

                            obj = DetectedObject(
                                mask=full_mask,
                                bbox=bbox,
                                area=int(np.sum(full_mask > 0)),
                                score=0.8,  # Slightly lower score for multiscale
                                label=prompt,
                                source_tile=-1  # Mark as from multiscale pass
                            )
                            objects.append(obj)

                except Exception as e:
                    self._log(f"    Multiscale detection failed for '{prompt}': {e}")
        else:
            try:
                masks = self._segmenter.generate_all_masks(small_image)

                # Only keep large masks (>5% of image)
                min_area = new_h * new_w * 0.05

                for mask_dict in masks:
                    if mask_dict["area"] < min_area:
                        continue

                    seg = mask_dict["segmentation"].astype(np.uint8) * 255

                    # Upscale mask
                    full_mask = cv2.resize(
                        seg, (w, h), interpolation=cv2.INTER_NEAREST
                    )

                    # Get bbox
                    ys, xs = np.where(full_mask > 0)
                    if len(xs) > 0:
                        bbox = [int(xs.min()), int(ys.min()),
                               int(xs.max()), int(ys.max())]

                        obj = DetectedObject(
                            mask=full_mask,
                            bbox=bbox,
                            area=int(np.sum(full_mask > 0)),
                            score=mask_dict.get("predicted_iou", 1.0) * 0.9,
                            stability_score=mask_dict.get("stability_score", 1.0),
                            source_tile=-1
                        )
                        objects.append(obj)

            except Exception as e:
                self._log(f"    Multiscale segmentation failed: {e}")

        return objects

    def segment_with_prompts(
        self,
        image: np.ndarray,
        text_prompts: List[str],
        depth_map: Optional[np.ndarray] = None
    ) -> Dict[str, List[DetectedObject]]:
        """Segment specific objects by text prompts.

        Args:
            image: RGB image (H, W, 3), uint8.
            text_prompts: List of object descriptions.
            depth_map: Optional depth map.

        Returns:
            Dict mapping each prompt to list of matching objects.
        """
        # Temporarily enable Grounded-SAM mode
        original_mode = self.use_grounded_sam
        original_prompts = self.text_prompts

        self.use_grounded_sam = True
        self.text_prompts = text_prompts
        self._initialized = False  # Force re-init

        try:
            objects = self.segment(image, depth_map)

            # Group by label
            results: Dict[str, List[DetectedObject]] = {p: [] for p in text_prompts}
            for obj in objects:
                if obj.label in results:
                    results[obj.label].append(obj)

            return results

        finally:
            # Restore original mode
            self.use_grounded_sam = original_mode
            self.text_prompts = original_prompts
            self._initialized = False


def tiled_segment(
    image: np.ndarray,
    tile_size: int = 1024,
    overlap: int = 256,
    text_prompts: Optional[List[str]] = None,
    verbose: bool = True
) -> List[DetectedObject]:
    """Convenience function for tiled segmentation.

    Args:
        image: RGB image (H, W, 3), uint8.
        tile_size: Size of each tile.
        overlap: Overlap between tiles.
        text_prompts: Optional text prompts for Grounded-SAM.
        verbose: Print progress.

    Returns:
        List of DetectedObject.
    """
    config = TileConfig(tile_size=tile_size, overlap=overlap)

    tiled = TiledSegmentation(
        config=config,
        use_grounded_sam=text_prompts is not None,
        text_prompts=text_prompts,
        verbose=verbose
    )

    return tiled.segment(image)


def visualize_detections(
    image: np.ndarray,
    objects: List[DetectedObject],
    alpha: float = 0.5,
    show_labels: bool = True,
    show_scores: bool = True
) -> np.ndarray:
    """Visualize detected objects on image.

    Args:
        image: Original RGB image.
        objects: List of DetectedObject.
        alpha: Mask transparency (0-1).
        show_labels: Show object labels.
        show_scores: Show confidence scores.

    Returns:
        Visualization image (RGB).
    """
    vis = image.copy()

    # Generate colors for each object
    np.random.seed(42)
    colors = np.random.randint(0, 255, (len(objects), 3))

    for i, obj in enumerate(objects):
        color = colors[i].tolist()

        # Overlay mask
        mask = obj.mask > 0
        vis[mask] = (
            vis[mask] * (1 - alpha) +
            np.array(color) * alpha
        ).astype(np.uint8)

        # Draw bbox
        x1, y1, x2, y2 = obj.bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # Draw label and score
        label_parts = []
        if show_labels and obj.label:
            label_parts.append(obj.label)
        if show_scores:
            label_parts.append(f"{obj.score:.2f}")

        if label_parts:
            label = " | ".join(label_parts)
            cv2.putText(
                vis, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

    return vis


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python tiled_segmentation.py <image_path> [text_prompt]")
        print("Example: python tiled_segmentation.py painting.jpg")
        print("Example: python tiled_segmentation.py painting.jpg 'boat'")
        sys.exit(1)

    image_path = sys.argv[1]
    text_prompt = sys.argv[2] if len(sys.argv) > 2 else None

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load: {image_path}")
        sys.exit(1)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print(f"\nProcessing: {image_path}")
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")

    # Segment
    prompts = [text_prompt] if text_prompt else None
    objects = tiled_segment(image_rgb, text_prompts=prompts)

    print(f"\nDetected {len(objects)} objects:")
    for i, obj in enumerate(objects):
        print(f"  {i+1}. Area: {obj.area} px, BBox: {obj.bbox}, Label: {obj.label or 'N/A'}")

    # Visualize
    vis = visualize_detections(image_rgb, objects)
    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

    output_path = image_path.replace(".", "_tiled_seg.")
    cv2.imwrite(output_path, vis_bgr)
    print(f"\nSaved visualization: {output_path}")
