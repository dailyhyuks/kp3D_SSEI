"""Segmentation module using Grounding DINO + SAM2.

Provides text-prompted object detection and precise mask generation
for separating overlapping objects in paintings.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import warnings

from kp3d.modules.occlusion.base import LayerInfo


class SegmentationModule:
    """Text-prompted segmentation using Grounding DINO and SAM2.

    Pipeline:
    1. Grounding DINO: Detect objects from text prompts → bounding boxes
    2. SAM2: Generate precise masks from bounding boxes

    This enables semantic object separation based on natural language
    descriptions (e.g., "white ceramic vase", "red wooden table").
    """

    def __init__(
        self,
        sam_model_type: str = "vit_h",
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        device: Optional[torch.device] = None
    ):
        """Initialize segmentation module.

        Args:
            sam_model_type: SAM2 model variant ("vit_b", "vit_l", "vit_h").
            box_threshold: Confidence threshold for bounding box detection.
            text_threshold: Confidence threshold for text matching.
            device: Computation device.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sam_model_type = sam_model_type
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        # Lazy-loaded models
        self._grounding_dino = None
        self._grounding_processor = None
        self._sam_model = None
        self._sam_predictor = None

        self._initialized = False

    def _load_grounding_dino(self) -> None:
        """Load Grounding DINO model for object detection."""
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

            model_id = "IDEA-Research/grounding-dino-tiny"

            self._grounding_processor = AutoProcessor.from_pretrained(model_id)
            self._grounding_dino = AutoModelForZeroShotObjectDetection.from_pretrained(
                model_id
            ).to(self.device)

        except ImportError:
            raise ImportError(
                "Grounding DINO requires transformers>=4.36.0. "
                "Install with: pip install transformers>=4.36.0"
            )

    def _load_sam2(self) -> None:
        """Load SAM2 model for mask generation."""
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            # Model checkpoint paths (adjust as needed)
            checkpoint_map = {
                "vit_b": "sam2_hiera_base_plus.pt",
                "vit_l": "sam2_hiera_large.pt",
                "vit_h": "sam2_hiera_huge.pt"
            }

            # Try to build SAM2 from pretrained
            # Note: SAM2 may require downloading checkpoints separately
            self._sam_predictor = SAM2ImagePredictor.from_pretrained(
                f"facebook/sam2-hiera-{self.sam_model_type.replace('vit_', '')}"
            )
            self._sam_predictor.model.to(self.device)

        except ImportError:
            raise ImportError(
                "SAM2 not found. Install with: pip install git+https://github.com/facebookresearch/sam2.git"
            )
        except Exception as e:
            warnings.warn(f"SAM2 loading failed: {e}. Falling back to SAM1.")
            self._load_sam1_fallback()

    def _load_sam1_fallback(self) -> None:
        """Fallback to SAM1 if SAM2 is not available."""
        try:
            from segment_anything import sam_model_registry, SamPredictor

            # Try common checkpoint locations
            import os
            checkpoint_paths = [
                f"sam_{self.sam_model_type}.pth",
                f"checkpoints/sam_{self.sam_model_type}.pth",
                os.path.expanduser(f"~/.cache/sam/sam_{self.sam_model_type}.pth")
            ]

            checkpoint = None
            for path in checkpoint_paths:
                if os.path.exists(path):
                    checkpoint = path
                    break

            if checkpoint is None:
                raise FileNotFoundError(
                    f"SAM checkpoint not found. Please download from: "
                    f"https://github.com/facebookresearch/segment-anything#model-checkpoints"
                )

            sam = sam_model_registry[self.sam_model_type](checkpoint=checkpoint)
            sam.to(self.device)
            self._sam_predictor = SamPredictor(sam)

        except ImportError:
            raise ImportError(
                "segment_anything not found. Install with: pip install segment-anything"
            )

    def _ensure_models_loaded(self) -> None:
        """Ensure all required models are loaded."""
        if self._initialized:
            return

        self._load_grounding_dino()
        self._load_sam2()
        self._initialized = True

    def detect_objects(
        self,
        image: np.ndarray,
        text_prompts: List[str]
    ) -> List[Dict[str, Any]]:
        """Detect objects using Grounding DINO.

        Args:
            image: RGB image (H, W, 3), uint8.
            text_prompts: List of text descriptions for objects to detect.

        Returns:
            List of detections with 'label', 'bbox', 'score'.
        """
        self._ensure_models_loaded()

        from PIL import Image

        # Convert to PIL
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        # Join prompts with ". " as required by Grounding DINO
        text = ". ".join(text_prompts) + "."

        # Process
        inputs = self._grounding_processor(
            images=pil_image,
            text=text,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self._grounding_dino(**inputs)

        # Post-process
        results = self._grounding_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[pil_image.size[::-1]]  # (H, W)
        )[0]

        detections = []
        for box, score, label in zip(
            results["boxes"],
            results["scores"],
            results["labels"]
        ):
            detections.append({
                "label": label,
                "bbox": box.cpu().numpy().astype(int).tolist(),  # [x1, y1, x2, y2]
                "score": float(score)
            })

        return detections

    def generate_masks(
        self,
        image: np.ndarray,
        boxes: List[List[int]]
    ) -> List[np.ndarray]:
        """Generate precise masks using SAM from bounding boxes.

        Args:
            image: RGB image (H, W, 3), uint8.
            boxes: List of bounding boxes as [x1, y1, x2, y2].

        Returns:
            List of binary masks (H, W) as numpy arrays.
        """
        self._ensure_models_loaded()

        # Set image for SAM
        self._sam_predictor.set_image(image)

        masks = []
        for box in boxes:
            # SAM expects box as np array
            box_np = np.array(box)

            # Predict mask
            mask_output, scores, _ = self._sam_predictor.predict(
                box=box_np,
                multimask_output=True
            )

            # Take best mask
            best_idx = np.argmax(scores)
            masks.append(mask_output[best_idx])

        return masks

    def segment(
        self,
        image: np.ndarray,
        text_prompts: List[str]
    ) -> List[LayerInfo]:
        """Full segmentation pipeline: detect + mask.

        Args:
            image: RGB image (H, W, 3), uint8.
            text_prompts: Text descriptions for objects to segment.

        Returns:
            List of LayerInfo with masks and metadata.
        """
        # Detect objects
        detections = self.detect_objects(image, text_prompts)

        if not detections:
            warnings.warn(f"No objects detected for prompts: {text_prompts}")
            return []

        # Generate masks for each detection
        boxes = [d["bbox"] for d in detections]
        masks = self.generate_masks(image, boxes)

        # Create LayerInfo objects
        layers = []
        for det, mask in zip(detections, masks):
            layer = LayerInfo(
                label=det["label"],
                mask=mask.astype(np.uint8),
                bbox=tuple(det["bbox"]),
                mean_depth=0.0,  # Will be filled by depth module
                is_foreground=False  # Will be determined by layer ordering
            )
            layers.append(layer)

        return layers

    def segment_interactive(
        self,
        image: np.ndarray,
        points: Optional[List[Tuple[int, int]]] = None,
        point_labels: Optional[List[int]] = None,
        box: Optional[List[int]] = None
    ) -> np.ndarray:
        """Interactive segmentation with point/box prompts.

        For manual refinement of masks.

        Args:
            image: RGB image (H, W, 3).
            points: Click points as [(x, y), ...].
            point_labels: 1 for foreground, 0 for background.
            box: Optional bounding box [x1, y1, x2, y2].

        Returns:
            Binary mask (H, W).
        """
        self._ensure_models_loaded()

        self._sam_predictor.set_image(image)

        # Prepare inputs
        point_coords = np.array(points) if points else None
        point_labels_np = np.array(point_labels) if point_labels else None
        box_np = np.array(box) if box else None

        masks, scores, _ = self._sam_predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels_np,
            box=box_np,
            multimask_output=True
        )

        # Return best mask
        best_idx = np.argmax(scores)
        return masks[best_idx].astype(np.uint8)


def create_manual_mask(
    image_shape: Tuple[int, int],
    points: List[Tuple[int, int]]
) -> np.ndarray:
    """Create a polygon mask from points.

    Utility for manual mask creation.

    Args:
        image_shape: (H, W) of target mask.
        points: List of polygon vertices as (x, y).

    Returns:
        Binary mask (H, W).
    """
    import cv2

    mask = np.zeros(image_shape, dtype=np.uint8)
    pts = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)

    return mask
