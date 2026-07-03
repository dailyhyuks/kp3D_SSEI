"""Occlusion detection module for finding hidden regions.

Detects regions of background objects that are occluded (hidden)
by foreground objects, preparing masks for inpainting.
"""

from typing import Optional, Tuple
import numpy as np
import cv2
from scipy import ndimage


class OcclusionDetector:
    """Detect occluded regions between foreground and background.

    Uses mask intersection and convex hull analysis to find
    regions of the background hidden by the foreground.
    """

    def __init__(
        self,
        dilation_kernel_size: int = 1,
        dilation_iterations: int = 1,
        use_convex_hull: bool = False,
        margin_pixels: int = 0
    ):
        """Initialize occlusion detector with minimal expansion.

        Args:
            dilation_kernel_size: Kernel size for mask dilation (default: 1 = minimal).
            dilation_iterations: Number of dilation passes (default: 1).
            use_convex_hull: Use convex hull for background (default: False = exact polygon).
            margin_pixels: Extra margin around occlusion boundary (default: 0).
        """
        self.dilation_kernel_size = dilation_kernel_size
        self.dilation_iterations = dilation_iterations
        self.use_convex_hull = use_convex_hull
        self.margin_pixels = margin_pixels

    def detect_occlusion(
        self,
        foreground_mask: np.ndarray,
        background_mask: np.ndarray
    ) -> np.ndarray:
        """Detect occluded regions of background.

        The occluded region is where foreground overlaps with the
        estimated full extent of the background.

        Args:
            foreground_mask: Binary mask of foreground object (H, W).
            background_mask: Binary mask of visible background (H, W).

        Returns:
            Binary mask of occluded regions (H, W).
        """
        # Ensure binary masks
        fg_mask = (foreground_mask > 0).astype(np.uint8)
        bg_mask = (background_mask > 0).astype(np.uint8)

        if self.use_convex_hull:
            # Estimate full background extent using convex hull
            bg_hull = self._compute_convex_hull(bg_mask)
        else:
            # Use original mask
            bg_hull = bg_mask

        # Occluded region = foreground ∩ background_hull - visible_background
        # This captures where foreground covers what should be background
        overlap = np.logical_and(fg_mask, bg_hull)
        occlusion_mask = overlap.astype(np.uint8)

        # Add margin for better inpainting
        if self.margin_pixels > 0:
            occlusion_mask = self._add_margin(occlusion_mask)

        return occlusion_mask

    def detect_occlusion_with_dilation(
        self,
        foreground_mask: np.ndarray,
        background_mask: np.ndarray
    ) -> np.ndarray:
        """Detect occlusion with dilated foreground for edge coverage.

        Dilates foreground mask to ensure edge pixels are included
        in the occlusion region.

        Args:
            foreground_mask: Binary mask of foreground (H, W).
            background_mask: Binary mask of background (H, W).

        Returns:
            Dilated occlusion mask (H, W).
        """
        # Dilate foreground
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.dilation_kernel_size, self.dilation_kernel_size)
        )
        fg_dilated = cv2.dilate(
            foreground_mask.astype(np.uint8),
            kernel,
            iterations=self.dilation_iterations
        )

        # Detect occlusion with dilated foreground
        return self.detect_occlusion(fg_dilated, background_mask)

    def _compute_convex_hull(self, mask: np.ndarray) -> np.ndarray:
        """Compute convex hull of mask.

        Args:
            mask: Binary mask (H, W).

        Returns:
            Convex hull mask (H, W).
        """
        # Find contours
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return mask

        # Combine all contour points
        all_points = np.vstack(contours)

        # Compute convex hull
        hull = cv2.convexHull(all_points)

        # Draw hull mask
        hull_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.fillPoly(hull_mask, [hull], 255)

        return hull_mask

    def _add_margin(self, mask: np.ndarray) -> np.ndarray:
        """Add pixel margin around mask edges.

        Args:
            mask: Binary mask (H, W).

        Returns:
            Mask with added margin (H, W).
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.margin_pixels * 2 + 1, self.margin_pixels * 2 + 1)
        )
        return cv2.dilate(mask, kernel, iterations=1)

    def compute_occlusion_boundary(
        self,
        foreground_mask: np.ndarray,
        background_mask: np.ndarray,
        boundary_width: int = 3
    ) -> np.ndarray:
        """Compute the boundary zone between foreground and background.

        Useful for edge blending and seamless inpainting.

        Args:
            foreground_mask: Binary mask of foreground (H, W).
            background_mask: Binary mask of background (H, W).
            boundary_width: Width of boundary zone in pixels.

        Returns:
            Boundary mask (H, W).
        """
        # Get edges of foreground
        fg_edges = self._get_mask_edges(foreground_mask, boundary_width)

        # Boundary = foreground edges ∩ (background OR convex_hull)
        if self.use_convex_hull:
            bg_region = self._compute_convex_hull(background_mask.astype(np.uint8))
        else:
            bg_region = background_mask

        boundary = np.logical_and(fg_edges, bg_region > 0)

        return boundary.astype(np.uint8)

    def _get_mask_edges(self, mask: np.ndarray, width: int) -> np.ndarray:
        """Get edge region of a mask.

        Args:
            mask: Binary mask (H, W).
            width: Edge width in pixels.

        Returns:
            Edge mask (H, W).
        """
        mask_uint8 = (mask > 0).astype(np.uint8)

        # Erode to get interior
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (width * 2 + 1, width * 2 + 1)
        )
        interior = cv2.erode(mask_uint8, kernel, iterations=1)

        # Edge = original - interior
        edge = mask_uint8 - interior

        return edge

    def analyze_occlusion(
        self,
        foreground_mask: np.ndarray,
        background_mask: np.ndarray
    ) -> dict:
        """Analyze occlusion relationship between two masks.

        Args:
            foreground_mask: Binary mask of foreground (H, W).
            background_mask: Binary mask of background (H, W).

        Returns:
            Dictionary with analysis results:
            - occlusion_mask: Binary mask of occluded regions
            - occlusion_area: Pixel count of occluded region
            - occlusion_ratio: Ratio of occluded to total background
            - boundary_mask: Edge region mask
            - has_occlusion: Whether any occlusion exists
        """
        occlusion_mask = self.detect_occlusion_with_dilation(
            foreground_mask, background_mask
        )

        boundary_mask = self.compute_occlusion_boundary(
            foreground_mask, background_mask
        )

        occlusion_area = int(np.sum(occlusion_mask > 0))
        bg_area = int(np.sum(background_mask > 0))

        occlusion_ratio = occlusion_area / max(bg_area, 1)

        return {
            "occlusion_mask": occlusion_mask,
            "occlusion_area": occlusion_area,
            "occlusion_ratio": occlusion_ratio,
            "boundary_mask": boundary_mask,
            "has_occlusion": occlusion_area > 0
        }


def quick_occlusion_mask(
    foreground_mask: np.ndarray,
    background_mask: np.ndarray,
    use_convex_hull: bool = True
) -> np.ndarray:
    """Quick occlusion mask generation.

    Args:
        foreground_mask: Binary mask of foreground.
        background_mask: Binary mask of background.
        use_convex_hull: Use convex hull for background boundary.

    Returns:
        Binary occlusion mask.
    """
    detector = OcclusionDetector(use_convex_hull=use_convex_hull)
    return detector.detect_occlusion_with_dilation(foreground_mask, background_mask)


def find_plate_region(
    foreground_mask: np.ndarray,
    background_mask: np.ndarray
) -> Optional[Tuple[int, int]]:
    """Find where background is visible on both sides of foreground.

    This identifies the "plate" level where foreground sits on background.
    The rim/decorative pattern is typically just ABOVE this level.

    Args:
        foreground_mask: Binary mask of foreground (ceramic).
        background_mask: Binary mask of background (soban).

    Returns:
        Tuple of (plate_top_y, plate_bottom_y) or None if not found.
    """
    h, w = foreground_mask.shape[:2]
    fg = (foreground_mask > 0).astype(np.uint8)
    bg = (background_mask > 0).astype(np.uint8)

    plate_rows = []

    for y in range(h):
        fg_row = fg[y]
        bg_row = bg[y]

        fg_cols = np.where(fg_row > 0)[0]
        bg_cols = np.where(bg_row > 0)[0]

        if len(fg_cols) < 2 or len(bg_cols) < 2:
            continue

        fg_left = int(fg_cols[0])
        fg_right = int(fg_cols[-1])

        # Check if background is visible on BOTH sides of foreground
        left_bg = [c for c in bg_cols if c < fg_left]
        right_bg = [c for c in bg_cols if c > fg_right]

        if len(left_bg) > 3 and len(right_bg) > 3:
            plate_rows.append(y)

    if not plate_rows:
        return None

    return (min(plate_rows), max(plate_rows))


def detect_rim_occlusion(
    foreground_mask: np.ndarray,
    background_mask: np.ndarray,
    rim_height: int = 30,
    margin_below: int = 20
) -> np.ndarray:
    """Detect occlusion in the rim region using mask spatial relationship.

    Uses the spatial relationship between SAM-segmented masks:
    1. Find plate_top: where background is visible on both sides of foreground
    2. The rim pattern is ABOVE plate_top (hidden behind foreground)
    3. Occlusion = foreground pixels within background's x-range

    Args:
        foreground_mask: Binary mask of foreground object (ceramic).
        background_mask: Binary mask of visible background (soban).
        rim_height: How far ABOVE plate_top to include.
        margin_below: How far BELOW plate_top to include.

    Returns:
        Binary occlusion mask for the rim region.
    """
    h, w = foreground_mask.shape[:2]
    fg = (foreground_mask > 0).astype(np.uint8)
    bg = (background_mask > 0).astype(np.uint8)

    # Find plate region using mask relationship
    plate_region = find_plate_region(foreground_mask, background_mask)

    if plate_region is None:
        return np.zeros((h, w), dtype=np.uint8)

    plate_top, plate_bottom = plate_region

    # Rim region is ABOVE plate_top
    rim_start = max(0, plate_top - rim_height)
    rim_end = min(h - 1, plate_top + margin_below)

    # Get background's x-range at plate level
    bg_x_min, bg_x_max = 0, w - 1
    for y in range(plate_top, min(plate_top + 50, h)):
        bg_row = bg[y]
        bg_cols = np.where(bg_row > 0)[0]
        if len(bg_cols) >= 2:
            bg_x_min = int(bg_cols[0])
            bg_x_max = int(bg_cols[-1])
            break

    # Create occlusion mask - only within background's x-range
    occlusion = np.zeros((h, w), dtype=np.uint8)

    for y in range(rim_start, rim_end + 1):
        fg_row = fg[y]
        fg_cols = np.where(fg_row > 0)[0]

        if len(fg_cols) > 0:
            # Limit to background's x-range
            x_start = max(fg_cols[0], bg_x_min)
            x_end = min(fg_cols[-1], bg_x_max)
            if x_start <= x_end:
                occlusion[y, x_start:x_end+1] = 255

    # Keep only foreground pixels
    occlusion = cv2.bitwise_and(occlusion, fg * 255)

    # Smooth edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    occlusion = cv2.morphologyEx(occlusion, cv2.MORPH_CLOSE, kernel)

    return occlusion


def auto_detect_rim_region(
    foreground_mask: np.ndarray,
    background_mask: np.ndarray
) -> Optional[Tuple[int, int]]:
    """Automatically detect the Y range of the rim/junction region.

    Uses mask spatial relationship to find plate_top, then returns
    the rim region above it.

    Args:
        foreground_mask: Binary mask of foreground.
        background_mask: Binary mask of background.

    Returns:
        Tuple of (y_start, y_end) for the rim region, or None if not found.
    """
    plate_region = find_plate_region(foreground_mask, background_mask)

    if plate_region is None:
        return None

    plate_top, _ = plate_region

    # Rim is above plate_top
    rim_start = max(0, plate_top - 30)
    rim_end = plate_top + 20

    return (rim_start, rim_end)


def detect_true_occlusion(
    foreground_mask: np.ndarray,
    background_mask: np.ndarray,
    table_shape: str = "ellipse",
    margin: int = 5,
    image: np.ndarray = None
) -> np.ndarray:
    """Detect TRUE occlusion - the hidden region of background under foreground.

    Key insight: The table surface (soban) CONTINUES under the foreground object (ceramic).
    We estimate the hidden surface using various methods.

    Algorithm options:
    - "edge": Use Canny edge + ellipse fitting for structure prediction
    - "boundary": Extrapolate broken boundary using curve analysis
    - "ellipse": Fit ellipse to visible table contour
    - "contour": Use convex hull of visible contour
    - "projection": Simple horizontal projection

    Args:
        foreground_mask: Binary mask of foreground object (ceramic).
        background_mask: Binary mask of visible background (soban).
        table_shape: Estimation method.
        margin: Pixel margin for smoothing.
        image: Original image (required for "edge" method).

    Returns:
        Binary mask of TRUE occluded region (H, W).
    """
    h, w = foreground_mask.shape[:2]
    fg = (foreground_mask > 0).astype(np.uint8)
    bg = (background_mask > 0).astype(np.uint8)

    if np.sum(bg) == 0 or np.sum(fg) == 0:
        return np.zeros((h, w), dtype=np.uint8)

    if table_shape == "edge":
        # NEW: Use edge information to predict full structure
        if image is None:
            raise ValueError("image is required for 'edge' method")
        estimated_surface = _estimate_structure_from_edges(image, bg, fg, h, w)

    elif table_shape == "ellipse":
        # Fit ellipse to visible table contour and extrapolate hidden region
        estimated_surface = _fit_ellipse_to_table(bg, fg, h, w)

    elif table_shape == "boundary":
        # Extrapolate broken boundary using curve analysis
        estimated_surface = _extrapolate_broken_boundary(bg, fg, h, w)

    elif table_shape == "contour":
        # Use convex hull of visible contour
        estimated_surface = _extrapolate_contour(bg, fg, h, w)

    elif table_shape == "projection":
        # Simple horizontal projection
        estimated_surface = _project_horizontally(bg, fg, h, w)

    else:
        raise ValueError(f"Unknown table_shape: {table_shape}")

    # TRUE occlusion = estimated surface that is hidden by foreground
    # = (estimated_surface) AND (foreground) AND NOT (visible_background)
    true_occlusion = cv2.bitwise_and(estimated_surface, fg * 255)
    true_occlusion = cv2.bitwise_and(true_occlusion, cv2.bitwise_not(bg * 255))

    # Add margin for smooth inpainting
    if margin > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin*2+1, margin*2+1))
        true_occlusion = cv2.dilate(true_occlusion, kernel, iterations=1)

    return true_occlusion


def _estimate_structure_from_edges(
    image: np.ndarray,
    bg: np.ndarray,
    fg: np.ndarray,
    h: int,
    w: int
) -> np.ndarray:
    """Estimate hidden background region using spatial relationship.

    Key insight: The table top (soban) surface CONTINUES under the ceramic.
    We estimate this by:
    1. Finding where bg contour TOUCHES fg contour (contact zone)
    2. The hidden region is fg pixels within bg's extended boundary

    More specifically:
    - Find the TOP of visible soban (where it meets ceramic)
    - Extend this boundary THROUGH the ceramic region
    - The extended region under ceramic = hidden table surface

    Args:
        image: Original RGB image.
        bg: Background mask.
        fg: Foreground mask.
        h, w: Image dimensions.

    Returns:
        Estimated full background surface.
    """
    # Get background contour
    contours, _ = cv2.findContours(bg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return _extrapolate_contour(bg, fg, h, w)

    main_contour = max(contours, key=cv2.contourArea)
    contour_pts = main_contour.reshape(-1, 2)

    # Find where bg contour touches/is adjacent to fg
    # Dilate fg slightly to find contact points
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fg_dilated = cv2.dilate(fg, kernel, iterations=1)

    # Find contour points that touch foreground
    contact_indices = []
    for i, pt in enumerate(contour_pts):
        x, y = pt
        if 0 <= y < h and 0 <= x < w:
            if fg_dilated[y, x] > 0:
                contact_indices.append(i)

    if len(contact_indices) < 2:
        # No contact - fallback
        return _extrapolate_contour(bg, fg, h, w)

    # Get contact start and end points
    contact_start_idx = contact_indices[0]
    contact_end_idx = contact_indices[-1]

    # These are the two points where bg boundary "enters" the fg region
    pt_left = contour_pts[contact_start_idx]
    pt_right = contour_pts[contact_end_idx]

    # The hidden region is between these two points, covered by fg
    # Create a polygon: left contact point -> right contact point ->
    # connect through the fg region

    # Find the TOP-most y of the contact zone (where table meets ceramic)
    contact_pts = contour_pts[contact_indices]
    contact_top_y = contact_pts[:, 1].min()
    contact_bottom_y = contact_pts[:, 1].max()

    # The hidden table surface extends from contact points
    # THROUGH the foreground region
    # Create estimated surface by:
    # 1. Start with original bg
    # 2. Add the hidden region under fg

    estimated_surface = bg.copy() * 255

    # Get the x-range of the contact zone
    contact_x_min = contact_pts[:, 0].min()
    contact_x_max = contact_pts[:, 0].max()

    # The hidden region: fg pixels within the contact x-range
    # and within a reasonable y-range (table surface level)
    # Table surface is roughly at contact_top_y level

    # Create the hidden region polygon
    # Top edge: straight line connecting left and right contact points
    # (this represents where table surface SHOULD be)
    # Bottom edge: follows the actual visible bg boundary

    # Find the left and right extremes of contact
    left_contact = contact_pts[contact_pts[:, 0].argmin()]
    right_contact = contact_pts[contact_pts[:, 0].argmax()]

    # Create fill region: connect the two contact points and fill below
    # The idea: table surface forms a continuous line at contact level

    # Fill rectangle from contact line down to visible bg
    fill_region = np.zeros((h, w), dtype=np.uint8)

    # For each x in contact range, fill from contact_top_y to actual bg top
    for x in range(int(left_contact[0]), int(right_contact[0]) + 1):
        # Find where bg starts at this x
        col = bg[:, x]
        bg_rows = np.where(col > 0)[0]

        if len(bg_rows) > 0:
            bg_top_at_x = bg_rows[0]
            # Fill from contact level to bg top
            # But only if fg is present (it's actually hidden)
            if fg[contact_top_y:bg_top_at_x, x].any():
                fill_region[contact_top_y:bg_top_at_x, x] = 255

    # Alternative: create convex polygon connecting contact points
    # This handles perspective better

    # Use the contact points to create a filled polygon
    if len(contact_pts) >= 3:
        hull = cv2.convexHull(contact_pts)
        cv2.fillPoly(fill_region, [hull], 255)

    # Combine with original bg
    estimated_surface = cv2.bitwise_or(estimated_surface, fill_region)

    return estimated_surface


def _extrapolate_broken_boundary(bg: np.ndarray, fg: np.ndarray, h: int, w: int) -> np.ndarray:
    """Extrapolate the broken boundary of background where foreground occludes.

    Key insight: SAM can only detect VISIBLE parts of the background.
    Where foreground covers background, the boundary is "broken".
    We can extrapolate the broken boundary by:
    1. Finding where background contour meets foreground (break points)
    2. Analyzing the curve direction at these break points
    3. Extending the curves to estimate the hidden boundary

    Args:
        bg: Background mask (soban).
        fg: Foreground mask (ceramic).
        h, w: Image dimensions.

    Returns:
        Estimated full background surface including hidden region.
    """
    # Get background contour
    contours, _ = cv2.findContours(bg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros((h, w), dtype=np.uint8)

    main_contour = max(contours, key=cv2.contourArea)
    contour_pts = main_contour.reshape(-1, 2)

    # Dilate foreground slightly to find adjacency
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    fg_dilated = cv2.dilate(fg, kernel, iterations=1)

    # Find contour points that are near/touching foreground (break points)
    break_indices = []
    for i, pt in enumerate(contour_pts):
        x, y = pt
        if 0 <= y < h and 0 <= x < w:
            if fg_dilated[y, x] > 0:
                break_indices.append(i)

    if len(break_indices) < 2:
        # No clear break points, fall back to convex hull
        hull = cv2.convexHull(main_contour)
        result = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(result, [hull], 255)
        return result

    # Find the start and end of the break region
    # (consecutive indices where contour touches foreground)
    break_start = break_indices[0]
    break_end = break_indices[-1]

    # Get points just before and after the break region
    margin = 10  # points to sample for direction estimation
    n_pts = len(contour_pts)

    # Points before break (going backwards from break_start)
    pre_break_pts = []
    for i in range(margin):
        idx = (break_start - i - 1) % n_pts
        pre_break_pts.append(contour_pts[idx])
    pre_break_pts = np.array(pre_break_pts)

    # Points after break (going forward from break_end)
    post_break_pts = []
    for i in range(margin):
        idx = (break_end + i + 1) % n_pts
        post_break_pts.append(contour_pts[idx])
    post_break_pts = np.array(post_break_pts)

    # Estimate direction at break points using linear regression
    if len(pre_break_pts) >= 2 and len(post_break_pts) >= 2:
        # Direction before break
        pre_dir = pre_break_pts[0] - pre_break_pts[-1]
        pre_dir = pre_dir / (np.linalg.norm(pre_dir) + 1e-6)

        # Direction after break
        post_dir = post_break_pts[-1] - post_break_pts[0]
        post_dir = post_dir / (np.linalg.norm(post_dir) + 1e-6)

        # Break points
        pt_before = contour_pts[break_start]
        pt_after = contour_pts[break_end]

        # Extrapolate: extend lines and find intersection or midpoint
        # Simple approach: create arc connecting the two break points
        # through a control point estimated from directions

        # Control point: average of extended positions
        extend_dist = np.linalg.norm(pt_after - pt_before) * 0.5
        ctrl1 = pt_before + pre_dir * extend_dist
        ctrl2 = pt_after - post_dir * extend_dist
        ctrl_pt = ((ctrl1 + ctrl2) / 2).astype(int)

        # Create extrapolated boundary using quadratic Bezier-like curve
        extrapolated_pts = []
        for t in np.linspace(0, 1, 30):
            # Quadratic interpolation
            p = (1-t)**2 * pt_before + 2*(1-t)*t * ctrl_pt + t**2 * pt_after
            extrapolated_pts.append(p.astype(int))
        extrapolated_pts = np.array(extrapolated_pts)

        # Create the full boundary: original contour + extrapolated part
        # Remove the break region and add extrapolated curve
        if break_start < break_end:
            valid_pts = np.vstack([
                contour_pts[:break_start],
                extrapolated_pts,
                contour_pts[break_end+1:]
            ])
        else:
            valid_pts = np.vstack([
                contour_pts[break_end+1:break_start],
                extrapolated_pts
            ])

        # Fill the completed boundary
        result = np.zeros((h, w), dtype=np.uint8)
        if len(valid_pts) >= 3:
            hull = cv2.convexHull(valid_pts)
            cv2.fillPoly(result, [hull], 255)
        return result

    # Fallback: use convex hull
    hull = cv2.convexHull(main_contour)
    result = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(result, [hull], 255)
    return result


def _fit_ellipse_to_table(bg: np.ndarray, fg: np.ndarray, h: int, w: int) -> np.ndarray:
    """Estimate full table top surface from visible rim edges.

    Key insight from perspective view:
    - Camera looks DOWN at the table
    - Table rim (테두리) is visible around/below the ceramic
    - Table top (상판) extends BEHIND the ceramic (image UP direction)
    - The ceramic OCCLUDES the back portion of the table top

    Strategy:
    1. Find the visible rim line (where table rim appears)
    2. Find left and right endpoints of visible rim
    3. The table top extends from rim UPWARD (behind ceramic)
    4. Estimate how far back using ceramic position
    """
    # Get contours of background (visible table parts)
    contours, _ = cv2.findContours(bg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((h, w), dtype=np.uint8)

    all_points = np.vstack(contours).reshape(-1, 2)

    # Find the TOP of visible background (rim level)
    bg_rows = np.where(np.any(bg > 0, axis=1))[0]
    if len(bg_rows) == 0:
        return np.zeros((h, w), dtype=np.uint8)

    rim_y = bg_rows[0]  # Top of visible rim

    # Find foreground bounds
    fg_rows = np.where(np.any(fg > 0, axis=1))[0]
    fg_cols = np.where(np.any(fg > 0, axis=0))[0]
    if len(fg_rows) == 0 or len(fg_cols) == 0:
        return np.zeros((h, w), dtype=np.uint8)

    fg_top_y = fg_rows[0]
    fg_bottom_y = fg_rows[-1]
    fg_left_x = fg_cols[0]
    fg_right_x = fg_cols[-1]

    # Find rim endpoints (left and right edges of visible rim)
    rim_band = 30  # pixels
    rim_area_points = all_points[
        (all_points[:, 1] >= rim_y) &
        (all_points[:, 1] <= rim_y + rim_band)
    ]

    if len(rim_area_points) < 2:
        rim_area_points = all_points

    # Get extreme points of rim
    rim_left_x = rim_area_points[:, 0].min()
    rim_right_x = rim_area_points[:, 0].max()

    # TABLE TOP ESTIMATION:
    # The table top is an elliptical/rectangular surface
    # Visible rim shows the FRONT edge
    # Back edge is hidden BEHIND the ceramic

    # The back of table extends to approximately where ceramic BOTTOM is
    # (ceramic sits ON the table, so table is under/behind the bottom portion)
    table_back_y = fg_bottom_y - (fg_bottom_y - fg_top_y) // 2  # Table back is around ceramic middle-bottom

    # Table width at back might be slightly narrower (perspective)
    # Use rim width as reference
    rim_width = rim_right_x - rim_left_x
    back_width_ratio = 0.85  # Back is slightly narrower due to perspective

    back_left_x = int(rim_left_x + rim_width * (1 - back_width_ratio) / 2)
    back_right_x = int(rim_right_x - rim_width * (1 - back_width_ratio) / 2)

    # Create table top polygon (trapezoid shape for perspective)
    table_polygon = np.array([
        [rim_left_x, rim_y + 10],      # Front-left
        [rim_right_x, rim_y + 10],     # Front-right
        [back_right_x, table_back_y],  # Back-right
        [back_left_x, table_back_y],   # Back-left
    ], dtype=np.int32)

    estimated_surface = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(estimated_surface, [table_polygon], 255)

    return estimated_surface


def _extrapolate_contour(bg: np.ndarray, fg: np.ndarray, h: int, w: int) -> np.ndarray:
    """Extrapolate table surface using convex hull of visible contour."""
    contours, _ = cv2.findContours(bg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((h, w), dtype=np.uint8)

    main_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(main_contour)

    estimated_surface = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(estimated_surface, [hull], 255)
    return estimated_surface


def _project_horizontally(bg: np.ndarray, fg: np.ndarray, h: int, w: int) -> np.ndarray:
    """Simple horizontal projection of visible table edges."""
    bg_rows = np.where(np.any(bg > 0, axis=1))[0]
    if len(bg_rows) == 0:
        return np.zeros((h, w), dtype=np.uint8)

    bg_y_min, bg_y_max = bg_rows[0], bg_rows[-1]

    # Find X range at each row
    bg_x_ranges = {}
    for y in bg_rows:
        cols = np.where(bg[y] > 0)[0]
        if len(cols) >= 2:
            bg_x_ranges[y] = (cols[0], cols[-1])

    if not bg_x_ranges:
        return np.zeros((h, w), dtype=np.uint8)

    # Use widest visible row as reference
    ref_y = max(bg_x_ranges.keys(), key=lambda y: bg_x_ranges[y][1] - bg_x_ranges[y][0])
    ref_left, ref_right = bg_x_ranges[ref_y]

    estimated_surface = np.zeros((h, w), dtype=np.uint8)
    for y in range(bg_y_min, bg_y_max + 1):
        if y in bg_x_ranges:
            x_left, x_right = bg_x_ranges[y]
        else:
            x_left, x_right = ref_left, ref_right
        estimated_surface[y, x_left:x_right+1] = 255

    return estimated_surface


def estimate_table_surface(
    background_mask: np.ndarray,
    method: str = "ellipse"
) -> np.ndarray:
    """Estimate the full table surface from visible portions.

    Traditional Korean soban (소반) tables have elliptical or rectangular tops.
    This estimates the full surface even when partially occluded.

    Args:
        background_mask: Binary mask of visible table (소반).
        method: Estimation method ("ellipse", "rectangle", "contour").

    Returns:
        Binary mask of estimated full table surface.
    """
    h, w = background_mask.shape[:2]
    bg = (background_mask > 0).astype(np.uint8)

    contours, _ = cv2.findContours(bg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((h, w), dtype=np.uint8)

    # Get largest contour (main table body)
    main_contour = max(contours, key=cv2.contourArea)

    estimated = np.zeros((h, w), dtype=np.uint8)

    if method == "ellipse":
        if len(main_contour) >= 5:
            ellipse = cv2.fitEllipse(main_contour)
            cv2.ellipse(estimated, ellipse, 255, -1)
        else:
            # Fallback to bounding rect
            x, y, bw, bh = cv2.boundingRect(main_contour)
            cv2.rectangle(estimated, (x, y), (x+bw, y+bh), 255, -1)

    elif method == "rectangle":
        x, y, bw, bh = cv2.boundingRect(main_contour)
        cv2.rectangle(estimated, (x, y), (x+bw, y+bh), 255, -1)

    elif method == "contour":
        # Use convex hull of contour
        hull = cv2.convexHull(main_contour)
        cv2.fillPoly(estimated, [hull], 255)

    return estimated


def predict_inpaint_regions(
    foreground_mask: np.ndarray,
    background_mask: np.ndarray,
    dilation_size: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict which regions to inpaint vs fill with background color.

    Automatically determines:
    - inpaint_mask: Foreground regions overlapping background hull
                   (should be filled with background texture)
    - background_mask: Foreground regions outside background hull
                      (should be filled with plain background color)

    This is crucial for realistic occlusion handling:
    - Object on table: bottom part → inpaint with table texture
    - Object in air: top part → fill with background (white/sky)

    Args:
        foreground_mask: Binary mask of foreground object.
        background_mask: Binary mask of visible background.
        dilation_size: Kernel size for boundary smoothing.

    Returns:
        Tuple of (inpaint_mask, background_fill_mask).
    """
    h, w = foreground_mask.shape[:2]

    # Ensure uint8
    fg_mask = (foreground_mask > 0).astype(np.uint8)
    bg_mask = (background_mask > 0).astype(np.uint8)

    # Find background's top y coordinate
    bg_ys = np.where(np.any(bg_mask > 0, axis=1))[0]
    if len(bg_ys) == 0:
        # No background visible - all foreground becomes background fill
        return np.zeros_like(fg_mask), fg_mask * 255

    bg_top_y = bg_ys[0]

    # Compute background convex hull
    contours, _ = cv2.findContours(bg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(fg_mask), fg_mask * 255

    all_pts = np.vstack(contours)
    hull = cv2.convexHull(all_pts)
    hull_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(hull_mask, [hull], 255)

    # Y-range mask: from background top downward
    # (foreground above background top should be background-filled)
    y_range_mask = np.zeros((h, w), dtype=np.uint8)
    margin = 20  # Small margin above background top
    y_range_mask[max(0, bg_top_y - margin):, :] = 255

    # Inpaint mask = foreground AND background_hull AND y_range
    inpaint_mask = cv2.bitwise_and(fg_mask * 255, hull_mask)
    inpaint_mask = cv2.bitwise_and(inpaint_mask, y_range_mask)

    # Smooth boundaries
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
    inpaint_mask = cv2.dilate(inpaint_mask, kernel, iterations=1)

    # Background fill mask = foreground - inpaint
    background_fill_mask = cv2.bitwise_and(fg_mask * 255, cv2.bitwise_not(inpaint_mask))

    return inpaint_mask, background_fill_mask


# ============================================================================
# Layered Occlusion Detection (새 레이블 체계)
# ============================================================================

from dataclasses import dataclass, field
from typing import Dict, List
import re


# Layer 우선순위 (숫자가 클수록 앞에 있음 = 다른 것을 가림)
LAYER_PRIORITY = {
    'object_2': 3,   # 가장 앞 (object_2_1, object_2_2, object_2_3 등)
    'object_3': 2,   # 중간
    'object_1': 1,   # 가장 뒤 (object_2, object_3에 가려짐)
    'background': 0, # segmentation용 (occlusion 관계 X)
}


def get_layer_priority(label: str) -> int:
    """레이블에서 layer priority 추출.

    Args:
        label: 레이블 문자열 (object_1, object_2_1, object_3, background 등)

    Returns:
        우선순위 (높을수록 앞에 있음)
    """
    if label == 'background':
        return 0

    # object_2_1, object_2_2 등은 모두 object_2로 취급
    match = re.match(r'object_(\d+)', label)
    if match:
        base_num = int(match.group(1))
        if base_num == 2:
            return 3  # object_2_* 는 가장 앞
        elif base_num == 3:
            return 2  # object_3
        elif base_num == 1:
            return 1  # object_1

    return 0  # 알 수 없는 레이블


def get_base_layer(label: str) -> str:
    """레이블에서 기본 layer 추출 (object_2_1 -> object_2).

    Args:
        label: 레이블 문자열

    Returns:
        기본 layer 문자열
    """
    if label == 'background':
        return 'background'

    match = re.match(r'(object_\d+)', label)
    if match:
        return match.group(1)

    return label


@dataclass
class OcclusionRelation:
    """단일 occlusion 관계."""
    occluder_label: str       # 가리는 객체의 레이블
    occludee_label: str       # 가려지는 객체의 레이블
    occluder_mask: np.ndarray # 가리는 객체 마스크
    occludee_mask: np.ndarray # 가려지는 객체 마스크
    occlusion_mask: np.ndarray # 가려진 영역 마스크
    occlusion_ratio: float    # 가려진 비율


@dataclass
class LayeredOcclusionResult:
    """계층 기반 occlusion 감지 결과.

    레이블 체계:
    - object_1: 1단계 (object_2_*, object_3에 의해 가려짐)
    - object_2_1, object_2_2, object_2_3: 2단계 (가장 앞, 같은 층위 내 occlusion 없음)
    - object_3: 3단계 (object_2_*에 의해 가려짐)
    - background: segmentation용 (occlusion 관계 X)
    """
    # 레이어별 마스크
    layer_masks: Dict[str, List[np.ndarray]]  # label -> list of masks

    # Occlusion 관계들
    occlusion_relations: List[OcclusionRelation]

    # 통합 occlusion 마스크 (모든 가려진 영역)
    combined_occlusion_mask: np.ndarray

    # Background 마스크 (segmentation용)
    background_mask: np.ndarray


class LayeredOcclusionDetector:
    """계층 기반 occlusion 감지기.

    레이블 체계:
    - object_2_* (가장 앞) → object_3 가림
    - object_2_* (가장 앞) → object_1 가림
    - object_3 → object_1 가림
    - object_2_* 끼리는 같은 층위라 서로 가리지 않음
    - background는 segmentation용 (occlusion 관계 X)
    """

    def __init__(self, dilation_kernel_size: int = 1, use_convex_hull: bool = False):
        """Initialize detector with minimal expansion settings.

        Args:
            dilation_kernel_size: Kernel size for mask dilation (default: 1 = minimal)
            use_convex_hull: Use convex hull for background (default: False = exact polygon)
        """
        self.detector = OcclusionDetector(
            dilation_kernel_size=dilation_kernel_size,
            use_convex_hull=use_convex_hull
        )

    def detect_from_shapes(
        self,
        shapes: List[Dict],
        image_shape: Tuple[int, int]
    ) -> LayeredOcclusionResult:
        """
        Annotation shapes에서 occlusion 감지.

        Args:
            shapes: shape dict 리스트 (label, points 포함)
            image_shape: (height, width)

        Returns:
            LayeredOcclusionResult
        """
        h, w = image_shape

        # 레이블별 마스크 수집
        layer_masks: Dict[str, List[np.ndarray]] = {}
        background_masks = []

        for shape in shapes:
            label = shape.get('label', '')
            points = shape.get('points', [])

            if not points:
                continue

            mask = self._create_mask_from_points(points, h, w)

            if label == 'background':
                background_masks.append(mask)
            else:
                if label not in layer_masks:
                    layer_masks[label] = []
                layer_masks[label].append(mask)

        # Background 통합
        combined_bg = np.zeros((h, w), dtype=np.uint8)
        for mask in background_masks:
            combined_bg = np.maximum(combined_bg, mask)

        # Occlusion 관계 계산
        occlusion_relations = []
        combined_occlusion = np.zeros((h, w), dtype=np.uint8)

        # 모든 레이블 쌍에 대해 occlusion 확인
        labels = list(layer_masks.keys())

        for i, label_a in enumerate(labels):
            for label_b in labels[i+1:]:
                priority_a = get_layer_priority(label_a)
                priority_b = get_layer_priority(label_b)

                # 같은 기본 레이어(예: object_2_1, object_2_2)는 서로 가리지 않음
                base_a = get_base_layer(label_a)
                base_b = get_base_layer(label_b)
                if base_a == base_b:
                    continue

                # 우선순위가 높은 것이 낮은 것을 가림
                if priority_a > priority_b:
                    occluder_label, occludee_label = label_a, label_b
                elif priority_b > priority_a:
                    occluder_label, occludee_label = label_b, label_a
                else:
                    continue  # 같은 우선순위면 occlusion 없음

                # 마스크 통합
                occluder_combined = np.zeros((h, w), dtype=np.uint8)
                for m in layer_masks[occluder_label]:
                    occluder_combined = np.maximum(occluder_combined, m)

                occludee_combined = np.zeros((h, w), dtype=np.uint8)
                for m in layer_masks[occludee_label]:
                    occludee_combined = np.maximum(occludee_combined, m)

                # Occlusion 감지
                if np.sum(occluder_combined) > 0 and np.sum(occludee_combined) > 0:
                    occ_mask = self.detector.detect_occlusion_with_dilation(
                        foreground_mask=occluder_combined,
                        background_mask=occludee_combined
                    )

                    occ_area = int(np.sum(occ_mask > 0))
                    if occ_area > 0:
                        occludee_area = int(np.sum(occludee_combined > 0))
                        occ_ratio = occ_area / max(occludee_area, 1)

                        relation = OcclusionRelation(
                            occluder_label=occluder_label,
                            occludee_label=occludee_label,
                            occluder_mask=occluder_combined,
                            occludee_mask=occludee_combined,
                            occlusion_mask=occ_mask,
                            occlusion_ratio=occ_ratio
                        )
                        occlusion_relations.append(relation)
                        combined_occlusion = np.maximum(combined_occlusion, occ_mask)

        return LayeredOcclusionResult(
            layer_masks=layer_masks,
            occlusion_relations=occlusion_relations,
            combined_occlusion_mask=combined_occlusion,
            background_mask=combined_bg
        )

    def _create_mask_from_points(
        self,
        points: List[List[float]],
        height: int,
        width: int
    ) -> np.ndarray:
        """폴리곤 포인트에서 바이너리 마스크 생성."""
        mask = np.zeros((height, width), dtype=np.uint8)
        pts = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        return mask

    def get_occlusion_summary(self, result: LayeredOcclusionResult) -> str:
        """Occlusion 결과 요약 문자열 생성."""
        lines = ["=== Occlusion Summary ==="]
        lines.append(f"Layers: {list(result.layer_masks.keys())}")
        lines.append(f"Total occlusion relations: {len(result.occlusion_relations)}")

        for rel in result.occlusion_relations:
            lines.append(
                f"  {rel.occluder_label} → {rel.occludee_label}: "
                f"{rel.occlusion_ratio:.1%} occluded"
            )

        return "\n".join(lines)


# Legacy alias for compatibility
SimpleOcclusionResult = LayeredOcclusionResult
SimpleLabelOcclusionDetector = LayeredOcclusionDetector


def masks_overlap(mask1: np.ndarray, mask2: np.ndarray) -> bool:
    """
    Check if two masks overlap.

    Args:
        mask1: Binary mask (H, W)
        mask2: Binary mask (H, W)

    Returns:
        True if masks have any overlapping pixels, False otherwise
    """
    overlap = np.logical_and(mask1 > 0, mask2 > 0)
    return np.any(overlap)
