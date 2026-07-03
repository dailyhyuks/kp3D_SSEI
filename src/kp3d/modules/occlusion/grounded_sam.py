"""Grounded SAM: Text-guided segmentation using Grounding DINO + SAM."""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
import warnings


class GroundedSAM:
    """Text-to-mask segmentation using Grounding DINO + SAM.

    Grounding DINO detects objects from text prompts,
    SAM generates precise masks from detected boxes.
    """

    def __init__(
        self,
        sam_model_type: str = "vit_b",
        grounding_model: str = "IDEA-Research/grounding-dino-tiny",
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        device: Optional[torch.device] = None
    ):
        """Initialize Grounded SAM.

        Args:
            sam_model_type: SAM model variant ("vit_b", "vit_l", "vit_h").
            grounding_model: Grounding DINO model from HuggingFace.
            box_threshold: Confidence threshold for box detection.
            text_threshold: Confidence threshold for text matching.
            device: Computation device.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sam_model_type = sam_model_type
        self.grounding_model_name = grounding_model
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        self._processor = None
        self._grounding_model = None
        self._sam = None
        self._predictor = None
        self._initialized = False

    def _load_grounding_dino(self) -> None:
        """Load Grounding DINO model from transformers."""
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

            print(f"Loading Grounding DINO from: {self.grounding_model_name}")
            self._processor = AutoProcessor.from_pretrained(self.grounding_model_name)
            self._grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.grounding_model_name
            )
            self._grounding_model.to(self.device)
            self._grounding_model.eval()

        except ImportError:
            raise ImportError(
                "transformers not found. Install with: pip install transformers"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Grounding DINO: {e}")

    def _load_sam(self) -> None:
        """Load SAM model and predictor."""
        try:
            from segment_anything import sam_model_registry, SamPredictor
            import os

            # Find checkpoint
            checkpoint_paths = [
                f"sam_{self.sam_model_type}.pth",
                f"checkpoints/sam_{self.sam_model_type}.pth",
                os.path.expanduser(f"~/.cache/sam/sam_{self.sam_model_type}.pth"),
                f"C:/Users/admin/.cache/sam/sam_{self.sam_model_type}.pth",
            ]

            checkpoint = None
            for path in checkpoint_paths:
                if os.path.exists(path):
                    checkpoint = path
                    break

            if checkpoint is None:
                raise FileNotFoundError(
                    f"SAM checkpoint not found. Download from: "
                    f"https://github.com/facebookresearch/segment-anything#model-checkpoints"
                )

            print(f"Loading SAM from: {checkpoint}")
            self._sam = sam_model_registry[self.sam_model_type](checkpoint=checkpoint)
            self._sam.to(self.device)

            self._predictor = SamPredictor(self._sam)

        except ImportError:
            raise ImportError(
                "segment_anything not found. Install with: pip install segment-anything"
            )

    def _ensure_loaded(self) -> None:
        """Ensure models are loaded."""
        if not self._initialized:
            self._load_grounding_dino()
            self._load_sam()
            self._initialized = True

    def detect_objects(
        self,
        image: np.ndarray,
        text_prompt: str
    ) -> List[Dict[str, Any]]:
        """Detect objects matching text prompt.

        Args:
            image: RGB image (H, W, 3), uint8.
            text_prompt: Object description (e.g., "white ceramic vase").

        Returns:
            List of detections with keys:
            - box: [x1, y1, x2, y2]
            - score: confidence score
            - label: matched text
        """
        self._ensure_loaded()

        h, w = image.shape[:2]

        # Prepare inputs
        inputs = self._processor(
            images=image,
            text=text_prompt,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Detect objects
        with torch.no_grad():
            outputs = self._grounding_model(**inputs)

        # Post-process results
        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            target_sizes=[(h, w)]
        )[0]

        # Convert to list of dicts and filter by thresholds
        detections = []
        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            score_val = score.item()
            # Filter by box threshold
            if score_val >= self.box_threshold:
                detections.append({
                    "box": box.cpu().numpy().tolist(),
                    "score": score_val,
                    "label": label
                })

        return detections

    def segment_with_text(
        self,
        image: np.ndarray,
        text_prompt: str,
        return_all: bool = False,
        select_strategy: str = "smallest"
    ) -> np.ndarray:
        """Segment objects matching text prompt.

        Args:
            image: RGB image (H, W, 3), uint8.
            text_prompt: Object description.
            return_all: If True, return all matching masks combined.
            select_strategy: How to select when multiple detections found.
                - "smallest": Most specific (good for foreground objects)
                - "largest": Background or larger objects
                - "best_score": Highest detection confidence

        Returns:
            Binary mask (H, W), uint8, 0 or 255.
        """
        self._ensure_loaded()

        # Detect objects
        detections = self.detect_objects(image, text_prompt)

        if not detections:
            warnings.warn(f"No objects detected for prompt: '{text_prompt}'")
            return np.zeros(image.shape[:2], dtype=np.uint8)

        print(f"Detected {len(detections)} object(s) for '{text_prompt}'")

        # Set image for SAM
        self._predictor.set_image(image)

        # Generate masks for each detection
        masks = []
        for i, det in enumerate(detections):
            box = det["box"]
            score = det["score"]

            # Convert box to [x1, y1, x2, y2] format
            box_array = np.array(box, dtype=np.float32)

            # Predict mask
            mask_outputs, scores, logits = self._predictor.predict(
                box=box_array,
                multimask_output=False
            )

            mask = mask_outputs[0]
            masks.append(mask)

            print(f"  Detection {i+1}: score={score:.3f}, mask_area={np.sum(mask)}")

        # Combine masks
        if return_all:
            # OR all masks together
            combined_mask = np.zeros_like(masks[0], dtype=bool)
            for mask in masks:
                combined_mask = np.logical_or(combined_mask, mask)
            final_mask = combined_mask
        else:
            # Select based on strategy
            mask_areas = [np.sum(m) for m in masks]
            scores = [det["score"] for det in detections]

            if select_strategy == "smallest":
                best_idx = np.argmin(mask_areas)
            elif select_strategy == "largest":
                best_idx = np.argmax(mask_areas)
            elif select_strategy == "best_score":
                best_idx = np.argmax(scores)
            else:
                best_idx = np.argmin(mask_areas)  # default to smallest

            print(f"  Selected ({select_strategy}): idx={best_idx}, area={mask_areas[best_idx]}")
            final_mask = masks[best_idx]

        return (final_mask.astype(np.uint8) * 255)

    def segment_multiple(
        self,
        image: np.ndarray,
        text_prompts: List[str]
    ) -> Dict[str, np.ndarray]:
        """Segment multiple objects by text prompts.

        Args:
            image: RGB image (H, W, 3), uint8.
            text_prompts: List of object descriptions.

        Returns:
            Dict mapping each prompt to its mask.
        """
        results = {}

        for prompt in text_prompts:
            mask = self.segment_with_text(image, prompt, return_all=True)
            results[prompt] = mask

        return results

    def segment_with_boxes(
        self,
        image: np.ndarray,
        boxes: List[Tuple[int, int, int, int]]
    ) -> List[np.ndarray]:
        """Segment objects from bounding boxes.

        Args:
            image: RGB image (H, W, 3), uint8.
            boxes: List of bounding boxes as (x1, y1, x2, y2).

        Returns:
            List of binary masks (H, W), uint8, 0 or 255.
        """
        self._ensure_loaded()

        # Set image for SAM
        self._predictor.set_image(image)

        masks = []
        for box in boxes:
            box_array = np.array(box, dtype=np.float32)

            mask_outputs, scores, logits = self._predictor.predict(
                box=box_array,
                multimask_output=False
            )

            mask = mask_outputs[0]
            masks.append(mask.astype(np.uint8) * 255)

        return masks


def grounded_segment(
    image: np.ndarray,
    text_prompt: str,
    sam_model_type: str = "vit_b",
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
    return_all: bool = False
) -> np.ndarray:
    """Convenience function for text-guided segmentation.

    Args:
        image: RGB image (H, W, 3), uint8.
        text_prompt: Object description.
        sam_model_type: SAM model variant.
        box_threshold: Detection confidence threshold.
        text_threshold: Text matching threshold.
        return_all: Return all matching masks combined.

    Returns:
        Binary mask (H, W), uint8, 0 or 255.
    """
    gsam = GroundedSAM(
        sam_model_type=sam_model_type,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    return gsam.segment_with_text(image, text_prompt, return_all=return_all)


if __name__ == "__main__":
    # Quick test
    import cv2
    import sys

    if len(sys.argv) < 3:
        print("Usage: python grounded_sam.py <image_path> <text_prompt>")
        print("Example: python grounded_sam.py test.png 'white ceramic vase'")
        sys.exit(1)

    image_path = sys.argv[1]
    text_prompt = sys.argv[2]

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        sys.exit(1)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Segment
    print(f"\nSegmenting '{text_prompt}' in {image_path}")
    gsam = GroundedSAM()
    mask = gsam.segment_with_text(image_rgb, text_prompt)

    print(f"\nMask area: {np.sum(mask > 0)} pixels")

    # Save result
    output_path = image_path.replace(".png", "_grounded_mask.png").replace(".jpg", "_grounded_mask.jpg")
    cv2.imwrite(output_path, mask)
    print(f"Saved mask to: {output_path}")

    # Create visualization
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = [0, 255, 0]  # Green
    overlay = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)

    vis_path = image_path.replace(".png", "_grounded_vis.png").replace(".jpg", "_grounded_vis.jpg")
    cv2.imwrite(vis_path, overlay)
    print(f"Saved visualization to: {vis_path}")
