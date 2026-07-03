"""Data loading and synthetic degradation/occlusion generation.

Provides utilities for:
- Loading painting images and LabelMe annotations
- Creating synthetic grid degradation (for Enhancement evaluation)
- Creating synthetic occlusions (for Inpainting evaluation)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def find_images(data_dir: str, min_size: int = 128) -> List[Path]:
    """Find all valid images in data directory.

    Args:
        data_dir: Directory to search.
        min_size: Minimum dimension (pixels).

    Returns:
        Sorted list of valid image paths.
    """
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    valid = []

    for p in Path(data_dir).iterdir():
        if p.suffix.lower() not in exts:
            continue
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        if h >= min_size and w >= min_size:
            valid.append(p)

    return sorted(valid)


def load_annotation(json_path: str) -> Dict:
    """Load LabelMe JSON annotation.

    Args:
        json_path: Path to LabelMe JSON file.

    Returns:
        Parsed annotation dict with shapes and image info.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def get_annotation_for_image(image_path: Path) -> Optional[Path]:
    """Find corresponding LabelMe JSON for an image.

    Args:
        image_path: Path to image file.

    Returns:
        Path to annotation JSON, or None if not found.
    """
    json_path = image_path.with_suffix(".json")
    if json_path.exists():
        return json_path
    return None


def extract_masks_from_annotation(
    annotation: Dict, image_shape: Tuple[int, int]
) -> List[Dict]:
    """Extract object masks from LabelMe annotation.

    Args:
        annotation: Parsed LabelMe JSON.
        image_shape: (height, width) of the image.

    Returns:
        List of dicts with 'label', 'mask' (binary H,W), 'polygon' keys.
    """
    h, w = image_shape
    objects = []

    for shape in annotation.get("shapes", []):
        label = shape.get("label", "unknown")
        points = np.array(shape.get("points", []), dtype=np.int32)

        if len(points) < 3:
            continue

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)

        objects.append({
            "label": label,
            "mask": mask,
            "polygon": points,
        })

    return objects


# =============================================================================
# Synthetic Degradation for Enhancement (Stage 1) Evaluation
# =============================================================================


def create_pseudo_clean(image_bgr: np.ndarray) -> np.ndarray:
    """Create pseudo-clean ground truth via heavy bilateral filtering.

    Applies 5 iterations of bilateral filter to remove existing grid
    while preserving large-scale structure. Used as approximate GT
    for grid removal evaluation.

    Args:
        image_bgr: Input BGR image (H, W, 3), uint8.

    Returns:
        Pseudo-clean image (H, W, 3), uint8.
    """
    result = image_bgr.copy()
    for _ in range(5):
        result = cv2.bilateralFilter(result, d=9, sigmaColor=25, sigmaSpace=9)
    return result


def add_synthetic_grid(
    image_bgr: np.ndarray,
    period_x: int = 9,
    period_y: int = 7,
    mod_b: float = 0.148,
    mod_g: float = 0.074,
    mod_r: float = 0.045,
) -> np.ndarray:
    """Add synthetic weave grid pattern matching real scan characteristics.

    Generates a periodic grid artifact similar to those caused by silk
    fabric mounting in digitized Korean paintings. Includes fundamental
    frequency and first harmonic (30% amplitude).

    Args:
        image_bgr: Clean or pseudo-clean image BGR uint8.
        period_x: Horizontal grid period (pixels).
        period_y: Vertical grid period (pixels).
        mod_b: Blue channel modulation depth.
        mod_g: Green channel modulation depth.
        mod_r: Red channel modulation depth.

    Returns:
        Image with synthetic grid added (H, W, 3), uint8.
    """
    h, w = image_bgr.shape[:2]
    result = image_bgr.astype(np.float32)

    y_coords = np.arange(h)[:, np.newaxis]
    x_coords = np.arange(w)[np.newaxis, :]

    # Fundamental + first harmonic (30%)
    grid_x = (
        np.sin(2 * np.pi * x_coords / period_x)
        + 0.3 * np.sin(4 * np.pi * x_coords / period_x)
    )
    grid_y = (
        np.sin(2 * np.pi * y_coords / period_y)
        + 0.3 * np.sin(4 * np.pi * y_coords / period_y)
    )
    grid = grid_x + grid_y

    modulations = [mod_b, mod_g, mod_r]
    for c in range(3):
        result[:, :, c] += grid * 255 * modulations[c]

    return np.clip(result, 0, 255).astype(np.uint8)


def create_synthetic_degradation(
    image_bgr: np.ndarray,
    degradation_type: str = "fading",
    intensity: float = 0.3,
) -> np.ndarray:
    """Create synthetic degradation for Restoration evaluation.

    Args:
        image_bgr: Input BGR image, uint8.
        degradation_type: One of "fading", "noise_spots", "color_shift".
        intensity: Degradation strength (0-1).

    Returns:
        Degraded image (H, W, 3), uint8.
    """
    if degradation_type == "fading":
        white = np.ones_like(image_bgr) * 255
        return cv2.addWeighted(
            image_bgr, 1 - intensity, white, intensity, 0
        )

    elif degradation_type == "noise_spots":
        result = image_bgr.copy()
        h, w = image_bgr.shape[:2]
        num_spots = int(h * w * intensity / 100)
        for _ in range(num_spots):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            r = np.random.randint(1, 4)
            color = np.random.randint(180, 220, 3).tolist()
            cv2.circle(result, (x, y), r, color, -1)
        return result

    elif degradation_type == "color_shift":
        # Shift in LAB space
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[:, :, 1] += intensity * 20  # a channel shift
        lab[:, :, 2] -= intensity * 15  # b channel shift
        lab = np.clip(lab, 0, 255).astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    else:
        raise ValueError(f"Unknown degradation type: {degradation_type}")


# =============================================================================
# Synthetic Occlusion for Inpainting (Stage 4) Evaluation
# =============================================================================


def create_synthetic_occlusion(
    image_shape: Tuple[int, int],
    occlusion_type: str = "center_ellipse",
    coverage: float = 0.15,
) -> np.ndarray:
    """Create synthetic occlusion mask for inpainting evaluation.

    Generates a binary mask where 255 indicates occluded (to-inpaint) region.

    Args:
        image_shape: (height, width) of target image.
        occlusion_type: Type of occlusion pattern.
        coverage: Approximate fraction of image area to occlude.

    Returns:
        Binary mask (H, W), uint8, 255=occluded.
    """
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)

    if occlusion_type == "center_ellipse":
        cx, cy = w // 2, h // 2
        # Compute axes from desired coverage
        area = h * w * coverage
        # area = pi * a * b, assume a/b ~ w/h ratio
        ratio = w / h
        b = int(np.sqrt(area / (np.pi * ratio)))
        a = int(b * ratio)
        cv2.ellipse(mask, (cx, cy), (a, b), 0, 0, 360, 255, -1)

    elif occlusion_type == "center_rect":
        rw = int(w * np.sqrt(coverage))
        rh = int(h * np.sqrt(coverage))
        x1 = (w - rw) // 2
        y1 = (h - rh) // 2
        mask[y1 : y1 + rh, x1 : x1 + rw] = 255

    elif occlusion_type == "random_blob":
        # Generate irregular blob using random walk + morphology
        cx, cy = w // 2, h // 2
        target_pixels = int(h * w * coverage)
        pts = [(cx, cy)]
        current = [cx, cy]

        while len(pts) < target_pixels // 10:
            dx = np.random.randint(-5, 6)
            dy = np.random.randint(-5, 6)
            nx = np.clip(current[0] + dx, 0, w - 1)
            ny = np.clip(current[1] + dy, 0, h - 1)
            pts.append((nx, ny))
            current = [nx, ny]

        for px, py in pts:
            r = np.random.randint(3, 8)
            cv2.circle(mask, (px, py), r, 255, -1)

        # Smooth the blob
        kernel_size = max(5, min(h, w) // 20)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    else:
        raise ValueError(f"Unknown occlusion type: {occlusion_type}")

    return mask
