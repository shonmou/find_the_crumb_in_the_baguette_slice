"""
Identify and measure the bread area in an image using OpenCV.

Approach:
  1. Convert image to HSV color space
  2. Mask bread-coloured pixels (golden/brown tones)
  3. Refine mask with morphological operations
  4. Find the dominant bread contour
  5. Optionally refine boundary with GrabCut
  6. Extract and save only the bread area (crop + exact shape)

Requirements:
    pip install opencv-python numpy matplotlib pandas pillow

Usage:
    python identify_bread_area.py bread.jpg
    python identify_bread_area.py bread.jpg -o ./results
    python identify_bread_area.py bread.jpg --pixel-size 0.05 --unit mm
    python identify_bread_area.py bread.jpg --no-grabcut        # skip GrabCut (faster)
    python identify_bread_area.py bread.jpg --hue-low 8 --hue-high 40
    python identify_bread_area.py bread.jpg --bg transparent   # transparent background
    python identify_bread_area.py bread.jpg --bg white         # white background
    python identify_bread_area.py --folder ./images
    python identify_bread_area.py --folder ./images --recursive -o ./results
"""

import argparse
import concurrent.futures
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------
try:
    import cv2
    import numpy as np
except ImportError:
    print("Error: opencv-python is not installed. Run: pip install opencv-python numpy")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("Error: pandas is not installed. Run: pip install pandas")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    print("Error: matplotlib is not installed. Run: pip install matplotlib")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
}

# Default HSV thresholds for bread-coloured regions.
# Hue 8–38 covers light yellow through golden-brown.
# Adjust via --hue-low / --hue-high if your images differ.
DEFAULT_HUE_LOW  = 8
DEFAULT_HUE_HIGH = 38
DEFAULT_SAT_LOW  = 30    # exclude near-grey/white backgrounds
DEFAULT_VAL_LOW  = 60    # exclude very dark regions
DEFAULT_SAT_HIGH = 255
DEFAULT_VAL_HIGH = 255


# ---------------------------------------------------------------------------
# Core image processing
# ---------------------------------------------------------------------------
def load_image(image_path: Path) -> np.ndarray:
    """Load an image with OpenCV. Raises RuntimeError on failure."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Could not open image: {image_path}")
    return img                      # BGR format


def find_mat_region(bgr: np.ndarray, hsv: np.ndarray) -> np.ndarray:
    """
    Detect the green cutting mat and return a filled binary mask of its area.

    Because the bread sits on top of the mat it creates a large hole in the
    raw teal detection.  To recover a solid mat boundary we:
      1. Detect all dark-teal pixels.
      2. Collect ALL teal contour points into a single point cloud.
      3. Take the convex hull of that cloud — this spans across the bread
         hole and gives a clean polygon covering the whole mat surface.
      4. Fill the convex hull.

    Everything outside the returned mask (wood desktop, other surfaces) is
    discarded before any bread detection begins.

    Falls back to the full image when no mat is detected so the pipeline
    remains usable on images without a cutting mat.
    """
    h, w = bgr.shape[:2]

    teal = cv2.inRange(hsv,
                       np.array([78,  65,  10], dtype=np.uint8),
                       np.array([108, 255, 140], dtype=np.uint8))

    # Small closing to bridge thin gaps caused by ruler markings and shadows
    k = max(5, min(h, w) // 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    teal = cv2.morphologyEx(teal, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(teal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("  Warning: cutting mat not detected — using full image.")
        return np.full((h, w), 255, dtype=np.uint8)

    # Merge all significant teal contours into one point cloud then hull them.
    # This ensures disconnected mat fragments (visible around each bread edge)
    # are all included in the final boundary.
    min_area = (h * w) * 0.001          # ignore tiny specks
    pts = np.vstack([c for c in contours if cv2.contourArea(c) > min_area])
    hull = cv2.convexHull(pts)

    mat_region = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mat_region, [hull], -1, 255, thickness=cv2.FILLED)

    covered = int(np.sum(mat_region == 255))
    print(f"  Mat region   : {covered:,} px  ({100*covered/(h*w):.1f}% of frame)")
    return mat_region


def find_bread_within_mat(hsv: np.ndarray, mat_region: np.ndarray) -> np.ndarray:
    """
    Within the mat boundary, detect bread as everything that is NOT a
    green-family colour.

    Two exclusion masks are combined:

    1. Teal mat surface — H 78–108, S > 65, V < 140.
       The dark-teal cutting mat itself.

    2. Green tones — H 35–85, S > 40 (any brightness).
       Covers green, dark green, and forest green.  Bread is never green:
         - Crust sits at H 8–35 (warm golden-brown).
         - Crumb is near-white (S too low for hue to be meaningful).
       Any green-hued pixel within the mat is therefore definitively
       background and must be excluded.

    3. Blue / aqua-blue tones — H 85–140, S > 40 (any brightness).
       Covers aqua, cyan-blue, sky blue, and deep blue — including the
       cyan grid lines and ruler markings printed on the cutting mat.
       Bread is never blue or aqua.

    4. Black / shadow — V < 50 (any hue or saturation).
       Very dark pixels are shadow areas or background, not bread.
       Even the darkest bread crust or cast shadow on the bread stays
       well above V = 50 in these images.

    Result: mat_region AND NOT (teal OR green OR blue OR dark) = bread candidates.
    """
    mat_surface = cv2.inRange(hsv,
                              np.array([78,  65,  10], dtype=np.uint8),
                              np.array([108, 255, 140], dtype=np.uint8))

    green_tones = cv2.inRange(hsv,
                              np.array([35,  40,  10], dtype=np.uint8),
                              np.array([85, 255, 255], dtype=np.uint8))

    blue_tones = cv2.inRange(hsv,
                             np.array([ 85, 40,  10], dtype=np.uint8),
                             np.array([140, 255, 255], dtype=np.uint8))

    dark_pixels = cv2.inRange(hsv,
                              np.array([0,   0,   0], dtype=np.uint8),
                              np.array([179, 255, 50], dtype=np.uint8))

    non_bread = cv2.bitwise_or(
        cv2.bitwise_or(mat_surface, green_tones),
        cv2.bitwise_or(blue_tones,  dark_pixels),
    )
    return cv2.bitwise_and(mat_region, cv2.bitwise_not(non_bread))


def create_bread_mask(
    bgr: np.ndarray,
    hsv: np.ndarray,
    hue_low: int,
    hue_high: int,
    sat_low: int,
    sat_high: int,
    val_low: int,
    val_high: int,
) -> np.ndarray:
    """
    Build a binary mask of bread-coloured pixels.

    Two complementary channels are combined with OR:

    1. HSV inRange — the primary detector.  Captures well-lit bread pixels
       (golden-yellow through warm brown) using hue + saturation + value
       bounds supplied by the caller.

    2. Lab b-channel supplement — the b axis separates blue (low) from
       yellow (high).  Bread always has a warm/yellow bias so Lab b > 145
       reliably recovers bread pixels that HSV misses due to shadows
       (low value) or pale crust (low saturation).  The supplement is
       restricted to the caller's hue family so cool-coloured backgrounds
       are not included.

    Returns a uint8 mask: 255 = bread, 0 = background.
    """
    # Channel 1 — HSV crust: golden-yellow → warm-brown (caller-supplied range)
    lower = np.array([hue_low,  sat_low,  val_low],  dtype=np.uint8)
    upper = np.array([hue_high, sat_high, val_high], dtype=np.uint8)
    hsv_mask = cv2.inRange(hsv, lower, upper)

    # Channel 2 — Lab b-channel supplement (yellow bias, restricted to hue family).
    # Recovers shadowed or pale crust pixels that fall outside sat/val bounds.
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    lab_b_mask = cv2.inRange(lab[:, :, 2],
                             np.array([145], dtype=np.uint8),
                             np.array([255], dtype=np.uint8))
    hue_gate = cv2.inRange(hsv,
                           np.array([hue_low,  0,  15], dtype=np.uint8),
                           np.array([hue_high, 255, 255], dtype=np.uint8))
    lab_b_mask = cv2.bitwise_and(lab_b_mask, hue_gate)

    # Channel 3 — Pale crumb: very bright + near-white (low saturation).
    # Baguette/white-bread crumb is cream-white and has almost no colour;
    # it is completely invisible to hue-based detection above.
    crumb_mask = cv2.inRange(hsv,
                             np.array([0,   0,  170], dtype=np.uint8),
                             np.array([179, 65, 255], dtype=np.uint8))

    return cv2.bitwise_or(cv2.bitwise_or(hsv_mask, lab_b_mask), crumb_mask)


def suppress_background(bgr: np.ndarray, hsv: np.ndarray, bread_mask: np.ndarray) -> np.ndarray:
    """
    Remove confirmed non-bread pixels from the bread candidate mask.

    Two background regions are targeted:

    1. Green cutting mat — dark teal: H 78–108, S > 65, V < 140.
       Redundant with find_bread_within_mat but harmless as a safety net.

    2. Mat ruler/grid markings — white tick marks and numbers printed on the
       mat.  These are near-white (V > 175, S < 40) so they pass through the
       inverted teal-exclusion detector in find_bread_within_mat.
       Detected as: (near-white pixel) AND (within reach of teal surface).
       A narrow dilation (~10 px) is used so only marks that sit directly
       on the teal surface are removed; a large dilation would reach into
       near-white bread crumb whose HSV overlaps with the marking range.

    Note: wood-grain desktop removal is intentionally omitted here.  The
    wood-grain HSV range (H 8–40, S < 62, V > 120) overlaps heavily with
    warm-tinted near-white bread crumb, causing large false exclusions when
    this function is applied to the inverted-detection candidate mask.
    Wood-grain pixels that fall inside the mat convex hull are instead
    handled by the morphological opening step in refine_mask.
    """
    h, w = bgr.shape[:2]

    # 1. Green cutting mat
    green_mat = cv2.inRange(hsv,
                            np.array([78,  65,  10], dtype=np.uint8),
                            np.array([108, 255, 140], dtype=np.uint8))

    # 2. Ruler / grid line markings on the mat
    # Narrow dilation (~10 px for a 3000 px wide image) — enough to reach
    # marks printed directly on the teal surface without encroaching on crumb.
    k = max(3, min(h, w) // 300)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    green_expanded = cv2.dilate(green_mat, kernel, iterations=1)
    near_white = cv2.inRange(hsv,
                             np.array([0,   0,  175], dtype=np.uint8),
                             np.array([179, 40, 255], dtype=np.uint8))
    mat_markings = cv2.bitwise_and(near_white, green_expanded)

    bg_mask = cv2.bitwise_or(green_mat, mat_markings)
    return cv2.bitwise_and(bread_mask, cv2.bitwise_not(bg_mask))


def refine_mask(mask: np.ndarray) -> np.ndarray:
    """
    Clean the raw colour mask with morphological operations then fill any
    remaining interior holes.

    Steps:
      1. Close small holes inside the bread region
      2. Remove isolated noise pixels outside it
      3. One more closing pass to smooth the outline
      4. Flood-fill from outside the image to mark true background, then
         any un-reached zero pixels must be interior holes — fill them in.
         This closes dark air pockets / shadows inside the bread that
         morphological closing alone cannot reach.
    """
    h, w = mask.shape[:2]
    k = max(5, min(h, w) // 60)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Flood-fill hole removal:
    # Pad by 1 px so flood fill can always start from a background corner,
    # even when the bread region touches the image border.
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    cv2.floodFill(padded, None, (0, 0), 128)   # mark reachable background as 128
    interior = padded[1:-1, 1:-1]
    # Pixels still 0 after flood fill are enclosed holes — add them to the mask
    holes = np.where((mask == 0) & (interior == 0), 255, 0).astype(np.uint8)
    mask = cv2.bitwise_or(mask, holes)

    return mask


def find_bread_contour(mask: np.ndarray) -> tuple[np.ndarray | None, np.ndarray]:
    """
    Find all external contours in the mask and return the best bread candidate
    together with a filled contour mask.

    Scoring combines area (dominant) with centrality so that a large
    warm-coloured background blob on one side of the frame cannot beat a
    slightly smaller but centrally located bread region.

      score = area × (0.7 + 0.3 × centrality)

    centrality = 1 at the image centre, 0 at the farthest corner.

    Returns (contour, filled_mask).  contour is None when nothing is found.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, np.zeros_like(mask)

    h, w = mask.shape
    cx_img, cy_img = w / 2, h / 2
    max_dist = np.hypot(w / 2, h / 2)

    def score(c):
        area = cv2.contourArea(c)
        if area < 100:
            return 0.0
        m = cv2.moments(c)
        if m["m00"] == 0:
            return area
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
        dist = np.hypot(cx - cx_img, cy - cy_img)
        centrality = 1.0 - dist / max_dist
        return area * (0.7 + 0.3 * centrality)

    bread_contour = max(contours, key=score)

    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [bread_contour], -1, 255, thickness=cv2.FILLED)

    return bread_contour, filled


def refine_mask_edges(bgr: np.ndarray, filled_mask: np.ndarray) -> np.ndarray:
    """
    Snap the bread mask boundary to actual image edges using watershed.

    Morphological operations in refine_mask use a large kernel (~50 px) which
    over-smooths the boundary.  Watershed recovers the precise edge by:

      1. Eroding the filled mask  → definite-foreground seeds (safely inside bread).
      2. Dilating the filled mask → outer search limit.
      3. Marking everything beyond the search limit as definite background.
      4. Leaving a band between eroded and dilated as uncertain (marker = 0).
      5. Pre-blurring the image with a bilateral filter to suppress bread
         texture (air cells) so watershed follows the crust edge, not interior
         crumb structure.
      6. Running cv2.watershed; pixels labelled 2 become the refined mask.

    Returns a binary mask (255 = bread) with sub-morphology edge accuracy.
    """
    h, w = filled_mask.shape
    band = max(15, min(h, w) // 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band, band))

    sure_fg   = cv2.erode( filled_mask, kernel, iterations=2)
    sure_bg   = cv2.bitwise_not(cv2.dilate(filled_mask, kernel, iterations=1))

    # Watershed markers: 1 = background, 2 = foreground, 0 = unknown boundary
    markers = np.ones((h, w), dtype=np.int32)          # default: background
    markers[sure_fg == 255] = 2                         # definite bread
    markers[sure_bg == 255] = 1                         # definite background
    markers[(sure_fg == 0) & (sure_bg == 0)] = 0       # uncertain band

    # Bilateral filter: smooths flat regions (mat, crumb) while keeping the
    # sharp crust edge intact, guiding watershed to the correct boundary.
    smooth = cv2.bilateralFilter(bgr, d=9, sigmaColor=60, sigmaSpace=60)
    cv2.watershed(smooth, markers)

    refined = np.zeros((h, w), dtype=np.uint8)
    refined[markers == 2] = 255
    return refined


def apply_grabcut(bgr: np.ndarray, filled_mask: np.ndarray) -> np.ndarray:
    """
    Refine the bread boundary using GrabCut, initialised from the filled
    contour mask.  Returns an improved binary mask (255 = foreground).

    GrabCut iterates a GMM-based segmentation; it improves edge accuracy
    but is ~10× slower than pure morphology.
    """
    gc_mask = np.where(filled_mask == 255,
                       cv2.GC_PR_FGD,    # probable foreground
                       cv2.GC_PR_BGD).astype(np.uint8)  # probable background

    # Erode the filled mask to find a high-confidence core and mark it as
    # definite foreground.  This gives GrabCut a reliable anchor so it
    # doesn't erode the bread boundary inward during iterations.
    k = max(3, min(bgr.shape[:2]) // 50)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    core = cv2.erode(filled_mask, kernel, iterations=3)
    gc_mask[core == 255] = cv2.GC_FGD

    # Mark a slim border region as definite background
    border = max(3, min(bgr.shape[:2]) // 30)
    gc_mask[:border,  :] = cv2.GC_BGD
    gc_mask[-border:, :] = cv2.GC_BGD
    gc_mask[:,  :border] = cv2.GC_BGD
    gc_mask[:, -border:] = cv2.GC_BGD

    bgr_model  = np.zeros((1, 65), dtype=np.float64)
    fgrd_model = np.zeros((1, 65), dtype=np.float64)

    try:
        cv2.grabCut(bgr, gc_mask, None, bgr_model, fgrd_model, iterCount=5,
                    mode=cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        # GrabCut can fail on degenerate masks; fall back to filled_mask
        return filled_mask

    refined = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
                       255, 0).astype(np.uint8)
    return refined


def clip_to_ellipse(contour: np.ndarray, filled_mask: np.ndarray) -> np.ndarray:
    """Clip angular foreign-object protrusions by iteratively fitting a minimum-
    enclosing ellipse to the contour and intersecting the mask with it.

    Each pass refits the ellipse on the clipped result. When two cardboard wedges
    are present and pull the first ellipse in opposite directions, the first pass
    removes the most protruding wedge; the second pass — fitted to the cleaner
    contour — then exposes and removes the remaining wedge.

    Iteration stopping rules (applied in order after each pass):
    1. Global fallback: total clipping > 25% of original area → abort, return original.
    2. Circularity guard: if the post-clip circularity ≥ 0.68 the result is already
       clean enough; stop. (Clean baguettes reach ≥ 0.65 after one or two passes.
       A single remaining cardboard wedge keeps circularity < 0.65 after two passes,
       requiring a third pass to reach the 0.68 threshold.)
    3. Hard limit: at most 3 passes total.
    Other early exits: contour < 5 points, or find_bread_contour returns nothing.
    """
    if len(contour) < 5:
        return filled_mask
    orig_area = int(np.sum(filled_mask == 255))
    h, w = filled_mask.shape
    current_contour = contour
    current_mask = filled_mask

    for _ in range(3):
        if len(current_contour) < 5:
            break
        center, axes, angle = cv2.fitEllipse(current_contour)
        ellipse_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(ellipse_mask, (center, axes, angle), 255, cv2.FILLED)
        clipped = cv2.bitwise_and(current_mask, ellipse_mask)
        clip_area = int(np.sum(clipped == 255))

        # Global fallback: total clipping > 25% → abort, return original
        if orig_area > 0 and clip_area < 0.75 * orig_area:
            return filled_mask

        # Re-extract contour from the clipped mask
        new_contour, new_mask = find_bread_contour(clipped)
        if new_contour is None or cv2.contourArea(new_contour) < 100:
            return clipped
        current_contour, current_mask = new_contour, new_mask

        # Circularity guard: stop when the shape is already clean
        area = cv2.contourArea(current_contour)
        perimeter = cv2.arcLength(current_contour, True)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 1.0
        if circularity >= 0.68:
            break

    return current_mask


# ---------------------------------------------------------------------------
# Bread area extraction
# ---------------------------------------------------------------------------
def extract_bread_area(
    bgr: np.ndarray,
    contour: np.ndarray,
    filled_mask: np.ndarray,
    bg: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract only the bread pixels from the source image.

    Returns:
        crop    — tight bounding-box crop (BGR), background pixels included.
        extract — exact contour shape only; background replaced according to `bg`:
                    'transparent' → BGRA image with alpha=0 outside bread
                    'white'       → BGR image, background = white
                    'black'       → BGR image, background = black
    """
    x, y, w, h = cv2.boundingRect(contour)

    # --- Bounding-box crop (includes background inside the bbox) -------------
    crop = bgr[y:y + h, x:x + w].copy()

    # --- Exact shape extract (full original size) ----------------------------
    # The extract keeps the same dimensions as the input image so that the
    # bread position is preserved relative to the original frame.
    mask_3ch = cv2.merge([filled_mask, filled_mask, filled_mask])

    if bg == "transparent":
        bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = filled_mask          # alpha channel follows the mask
        extract = bgra
    else:
        bg_val = (255, 255, 255) if bg == "white" else (0, 0, 0)
        background = np.full_like(bgr, bg_val, dtype=np.uint8)
        extract = np.where(mask_3ch == 255, bgr, background).astype(np.uint8)

    return crop, extract


def save_bread_extracts(
    crop: np.ndarray,
    extract: np.ndarray,
    image_path: Path,
    output_dir: Path,
    extract_dir: Path,
    bg: str,
):
    """Save the bounding-box crop and the exact-shape extract.

    crop is written to output_dir (per-image sub-directory in folder mode).
    extract is written to extract_dir (top-level output folder in folder mode)
    so all extracted bread images land in one flat directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)

    crop_path = output_dir / f"{image_path.stem}_bread_crop.png"
    cv2.imwrite(str(crop_path), crop)
    print(f"  Crop saved    : {crop_path}")

    ext = "png"   # PNG preserves transparency for BGRA
    extract_path = extract_dir / f"{image_path.stem}_bread_exact.{ext}"
    cv2.imwrite(str(extract_path), extract)
    print(f"  Extract saved : {extract_path}  (bg={bg})")


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------
def measure_bread(
    contour: np.ndarray,
    filled_mask: np.ndarray,
    pixel_size: float,
    unit: str,
) -> dict:
    """
    Compute area, bounding box, circularity, convex hull coverage,
    and equivalent diameter for the detected bread region.
    """
    scale1 = pixel_size          # for linear measures
    scale2 = pixel_size ** 2     # for area measures

    px_area     = cv2.contourArea(contour)
    px_perim    = cv2.arcLength(contour, closed=True)
    mask_area   = int(np.sum(filled_mask == 255))   # filled pixel count

    circularity = (4 * np.pi * px_area / (px_perim ** 2)) if px_perim > 0 else 0.0
    equiv_diam  = float(np.sqrt(4 * px_area / np.pi)) * scale1

    hull        = cv2.convexHull(contour)
    hull_area   = cv2.contourArea(hull)
    solidity    = float(px_area / hull_area) if hull_area > 0 else 0.0

    x, y, w, h = cv2.boundingRect(contour)
    (cx, cy), (ma, Mi), angle = cv2.fitEllipse(contour) if len(contour) >= 5 else \
                                 ((0, 0), (0, 0), 0)

    return {
        f"area_{unit}2":           round(px_area   * scale2, 4),
        f"mask_area_{unit}2":      round(mask_area * scale2, 4),
        f"perimeter_{unit}":       round(px_perim  * scale1, 4),
        f"equiv_diameter_{unit}":  round(equiv_diam, 4),
        "circularity":             round(circularity, 4),
        "solidity":                round(solidity, 4),
        f"bbox_x_{unit}":          round(x  * scale1, 4),
        f"bbox_y_{unit}":          round(y  * scale1, 4),
        f"bbox_w_{unit}":          round(w  * scale1, 4),
        f"bbox_h_{unit}":          round(h  * scale1, 4),
        "contour_points":          len(contour),
        # raw pixel values for downstream use
        "area_px2":                int(px_area),
        "bbox_x_px":               x,
        "bbox_y_px":               y,
        "bbox_w_px":               w,
        "bbox_h_px":               h,
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def annotate_image(bgr: np.ndarray, contour: np.ndarray, filled_mask: np.ndarray) -> np.ndarray:
    """
    Return an annotated BGR image with:
      - Semi-transparent green overlay on the detected bread area
      - Yellow contour outline
      - Cyan bounding box
    """
    annotated = bgr.copy()

    # Semi-transparent fill
    overlay = bgr.copy()
    cv2.drawContours(overlay, [contour], -1, (0, 180, 0), thickness=cv2.FILLED)
    cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0, annotated)

    # Contour outline
    cv2.drawContours(annotated, [contour], -1, (0, 255, 255), thickness=2)

    # Bounding box
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 180, 0), 2)

    # Label
    area_px = int(cv2.contourArea(contour))
    label   = f"Bread  {area_px:,} px²"
    font    = cv2.FONT_HERSHEY_SIMPLEX
    scale   = max(0.5, min(bgr.shape[:2]) / 800)
    thick   = max(1, int(scale * 2))
    (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
    lx, ly = x, max(y - 8, th + 4)
    cv2.rectangle(annotated, (lx, ly - th - 4), (lx + tw + 4, ly + 4), (0, 0, 0), cv2.FILLED)
    cv2.putText(annotated, label, (lx + 2, ly), font, scale, (0, 255, 255), thick)

    return annotated


def save_figure(
    bgr: np.ndarray,
    annotated: np.ndarray,
    filled_mask: np.ndarray,
    image_path: Path,
    output_dir: Path,
    stats: dict,
    unit: str,
):
    """Save a three-panel figure: original | mask | annotated."""
    output_dir.mkdir(parents=True, exist_ok=True)

    rgb      = cv2.cvtColor(bgr,      cv2.COLOR_BGR2RGB)
    ann_rgb  = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Bread Area Identification — {image_path.name}", fontsize=13)

    axes[0].imshow(rgb);        axes[0].set_title("Original");    axes[0].axis("off")
    axes[1].imshow(filled_mask, cmap="gray")
    axes[1].set_title("Bread Mask"); axes[1].axis("off")
    axes[2].imshow(ann_rgb);    axes[2].set_title("Detected Area"); axes[2].axis("off")

    # Stats text on the right panel
    area_key = f"area_{unit}2"
    circ_key = "circularity"
    info = (
        f"Area: {stats.get(area_key, '—')} {unit}²\n"
        f"Perimeter: {stats.get(f'perimeter_{unit}', '—')} {unit}\n"
        f"Equiv. diam: {stats.get(f'equiv_diameter_{unit}', '—')} {unit}\n"
        f"Circularity: {stats.get(circ_key, '—')}\n"
        f"Solidity: {stats.get('solidity', '—')}"
    )
    axes[2].text(
        0.02, 0.02, info,
        transform=axes[2].transAxes,
        fontsize=8, color="white",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6),
    )

    plt.tight_layout()
    fig_path = output_dir / f"{image_path.stem}_bread_area.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  Figure saved  : {fig_path}")

    # Also save the clean mask
    mask_path = output_dir / f"{image_path.stem}_mask.png"
    cv2.imwrite(str(mask_path), filled_mask)
    print(f"  Mask saved    : {mask_path}")


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------
def print_statistics(image_path: Path, stats: dict, unit: str):
    print("\n" + "=" * 52)
    print("  BREAD AREA IDENTIFICATION — RESULTS")
    print("=" * 52)
    print(f"  Image          : {image_path.name}")
    print(f"  Area           : {stats.get(f'area_{unit}2', '—')} {unit}²")
    print(f"  Perimeter      : {stats.get(f'perimeter_{unit}', '—')} {unit}")
    print(f"  Equiv. diameter: {stats.get(f'equiv_diameter_{unit}', '—')} {unit}")
    print(f"  Circularity    : {stats.get('circularity', '—')}  (1 = perfect circle)")
    print(f"  Solidity       : {stats.get('solidity', '—')}  (1 = convex)")
    print(f"  Bounding box   : "
          f"x={stats.get(f'bbox_x_{unit}')}, y={stats.get(f'bbox_y_{unit}')}, "
          f"w={stats.get(f'bbox_w_{unit}')}, h={stats.get(f'bbox_h_{unit}')} {unit}")
    print("=" * 52 + "\n")


def save_combined_summary(rows: list[dict], output_dir: Path):
    """Write combined_summary.csv with one row per image (folder mode)."""
    if not rows:
        return
    path = output_dir / "combined_summary.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\nCombined summary saved: {path}")


# ---------------------------------------------------------------------------
# Folder helpers
# ---------------------------------------------------------------------------
def collect_images_from_folder(folder: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return [
        p for p in sorted(folder.glob(pattern))
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


# ---------------------------------------------------------------------------
# Single-image pipeline
# ---------------------------------------------------------------------------
def process_image(
    image_path: Path,
    output_dir: Path,
    extract_dir: Path,
    hue_low: int,
    hue_high: int,
    sat_low: int,
    val_low: int,
    pixel_size: float,
    unit: str,
    use_grabcut: bool,
    bg: str,
) -> dict | None:
    """
    Full pipeline for one image.
    Returns a stats dict (suitable for the combined CSV) or None on failure.
    """
    try:
        bgr = load_image(image_path)
    except RuntimeError as exc:
        print(f"  Error: {exc}", file=sys.stderr)
        return None

    # Pre-compute HSV once; all helpers that need it reuse this array.
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # 1. Detect the cutting mat boundary (convex hull of all teal pixels).
    #    Discards wood desktop and anything else outside the mat entirely.
    mat_region = find_mat_region(bgr, hsv)

    # 2. Within the mat: bread = everything that is NOT the teal mat surface.
    #    Inverting the mat-surface detection is more reliable than matching
    #    bread colour directly, because the teal is highly consistent while
    #    bread appearance varies across slices and lighting conditions.
    raw_mask = find_bread_within_mat(hsv, mat_region)

    # 2b. Remove known non-bread pixels that pass through the inverted detection:
    #     near-white ruler/grid markings on the mat, and warm low-saturation
    #     wood-grain pixels inside the convex hull.
    print("  Suppressing background …")
    raw_mask = suppress_background(bgr, hsv, raw_mask)

    # 3. Morphological refinement: closes holes in the bread, removes thin
    #    ruler/grid lines and small crumb fragments left by the inversion.
    refined_mask = refine_mask(raw_mask)

    # 4. Largest contour
    contour, filled_mask = find_bread_contour(refined_mask)
    if contour is None or cv2.contourArea(contour) < 100:
        print(
            f"  No bread region detected in '{image_path.name}'. "
            "Try adjusting --hue-low / --hue-high or --sat-low.",
            file=sys.stderr,
        )
        return None

    # 5. Watershed edge refinement — snaps the boundary to actual image edges.
    #    Runs on every image (not optional) because morphological operations
    #    always over-smooth the crust boundary.
    print("  Refining edges with watershed …")
    ws_mask = refine_mask_edges(bgr, filled_mask)
    ws_contour, ws_filled = find_bread_contour(ws_mask)
    if ws_contour is not None and cv2.contourArea(ws_contour) > 100:
        contour, filled_mask = ws_contour, ws_filled

    # 6. Optional GrabCut refinement (additional pass after watershed)
    if use_grabcut:
        print("  Refining boundary with GrabCut …")
        gc_mask = apply_grabcut(bgr, filled_mask)
        gc_contour, gc_filled = find_bread_contour(gc_mask)
        if gc_contour is not None and cv2.contourArea(gc_contour) > 100:
            contour, filled_mask = gc_contour, gc_filled

    # 6b. Ellipse protrusion filter — clips angular foreign objects (e.g., cardboard)
    print("  Clipping to ellipse …")
    clipped = clip_to_ellipse(contour, filled_mask)
    clip_contour, clip_filled = find_bread_contour(clipped)
    if clip_contour is not None and cv2.contourArea(clip_contour) > 100:
        contour, filled_mask = clip_contour, clip_filled

    # 7. Measure
    stats = measure_bread(contour, filled_mask, pixel_size, unit)
    stats["file"] = str(image_path)
    print_statistics(image_path, stats, unit)

    # 8. Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Bread area extracts (primary output) --------------------------------
    crop, extract = extract_bread_area(bgr, contour, filled_mask, bg)
    save_bread_extracts(crop, extract, image_path, output_dir, extract_dir, bg)

    # --- Annotated overview figure -------------------------------------------
    annotated = annotate_image(bgr, contour, filled_mask)
    save_figure(bgr, annotated, filled_mask, image_path, output_dir, stats, unit)

    # --- Per-image stats JSON ------------------------------------------------
    json_path = output_dir / f"{image_path.stem}_bread_stats.json"
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats saved   : {json_path}")

    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Identify and measure bread area in an image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "image",
        nargs="?",
        help="Path to a single bread image",
    )
    parser.add_argument(
        "--folder",
        metavar="DIR",
        help="Analyze all supported images in this folder",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="When using --folder, also search subdirectories",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="./bread_area_results",
        metavar="DIR",
        help="Output directory (default: ./bread_area_results)",
    )
    # HSV tuning
    parser.add_argument(
        "--hue-low",
        type=int, default=DEFAULT_HUE_LOW, metavar="N",
        help=f"Lower hue bound for bread colour (0–179, default: {DEFAULT_HUE_LOW})",
    )
    parser.add_argument(
        "--hue-high",
        type=int, default=DEFAULT_HUE_HIGH, metavar="N",
        help=f"Upper hue bound for bread colour (0–179, default: {DEFAULT_HUE_HIGH})",
    )
    parser.add_argument(
        "--sat-low",
        type=int, default=DEFAULT_SAT_LOW, metavar="N",
        help=f"Lower saturation bound (0–255, default: {DEFAULT_SAT_LOW})",
    )
    parser.add_argument(
        "--val-low",
        type=int, default=DEFAULT_VAL_LOW, metavar="N",
        help=f"Lower value/brightness bound (0–255, default: {DEFAULT_VAL_LOW})",
    )
    # Scale
    parser.add_argument(
        "--pixel-size",
        type=float, default=1.0, metavar="N",
        help="Physical size of one pixel in --unit (default: 1.0)",
    )
    parser.add_argument(
        "--unit",
        default="px", metavar="STR",
        help="Measurement unit label, e.g. mm (default: px)",
    )
    # GrabCut (on by default; --no-grabcut skips it for faster processing)
    parser.add_argument(
        "--no-grabcut",
        action="store_false",
        dest="use_grabcut",
        default=True,
        help="Skip GrabCut boundary refinement (faster, but boundary may be jagged)",
    )
    # Parallel workers
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Number of parallel worker processes in batch mode "
            "(default: os.cpu_count()). Ignored for single-image mode."
        ),
    )
    # Background for exact-shape extract
    parser.add_argument(
        "--bg",
        choices=["transparent", "white", "black"],
        default="transparent",
        help="Background for the exact-shape extract (default: transparent)",
    )

    args = parser.parse_args()

    if not args.image and not args.folder:
        parser.error("Provide an image file or use --folder <DIR>.")

    output_dir = Path(args.output_dir)

    # --- Build image list ---------------------------------------------------
    image_paths: list[Path] = []

    if args.image:
        p = Path(args.image)
        if not p.exists():
            print(f"Error: image not found: {p}", file=sys.stderr)
            sys.exit(1)
        if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
            print(
                f"Error: unsupported format '{p.suffix}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
                file=sys.stderr,
            )
            sys.exit(1)
        image_paths.append(p)

    if args.folder:
        folder = Path(args.folder)
        if not folder.is_dir():
            print(f"Error: '{folder}' is not a directory.", file=sys.stderr)
            sys.exit(1)
        found = collect_images_from_folder(folder, args.recursive)
        if not found:
            print(f"No supported image files found in '{folder}'.")
            sys.exit(0)
        print(f"Found {len(found)} image(s) in '{folder}'.")
        image_paths.extend(found)

    # --- Resolve worker count -----------------------------------------------
    if args.workers is not None and args.workers < 1:
        print("Error: --workers must be >= 1.", file=sys.stderr)
        sys.exit(1)

    # --- Process images -----------------------------------------------------
    summary_rows: list[dict] = []
    success_count = 0
    skip_count = 0

    is_batch = bool(args.folder and len(image_paths) > 1)
    workers = (args.workers or os.cpu_count() or 1) if is_batch else 1

    def _image_output_dir(image_path: Path) -> Path:
        return (output_dir / image_path.stem) if is_batch else output_dir

    common_kwargs: dict = dict(
        extract_dir=output_dir,
        hue_low=args.hue_low,
        hue_high=args.hue_high,
        sat_low=args.sat_low,
        val_low=args.val_low,
        pixel_size=args.pixel_size,
        unit=args.unit,
        use_grabcut=args.use_grabcut,
        bg=args.bg,
    )

    if workers == 1:
        # Serial path — no spawning overhead; also used for single-image mode.
        for i, image_path in enumerate(image_paths, 1):
            print(f"\n[{i}/{len(image_paths)}] {image_path.name}")
            print("-" * 52)
            stats = process_image(
                image_path=image_path,
                output_dir=_image_output_dir(image_path),
                **common_kwargs,
            )
            if stats is None:
                skip_count += 1
            else:
                summary_rows.append(stats)
                success_count += 1
    else:
        print(f"Processing {len(image_paths)} image(s) with {workers} workers …")
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            # Submit all images; preserve submission order for CSV ordering.
            futures = [
                executor.submit(
                    process_image,
                    image_path=image_path,
                    output_dir=_image_output_dir(image_path),
                    **common_kwargs,
                )
                for image_path in image_paths
            ]
            # Collect in submission order so combined_summary.csv rows match
            # the sorted input file list regardless of completion order.
            for i, (image_path, future) in enumerate(zip(image_paths, futures), 1):
                print(f"\n[{i}/{len(image_paths)}] {image_path.name}")
                print("-" * 52)
                try:
                    stats = future.result()
                except Exception as exc:
                    print(f"  Error: {exc}", file=sys.stderr)
                    stats = None
                if stats is None:
                    skip_count += 1
                else:
                    summary_rows.append(stats)
                    success_count += 1

    # --- Combined summary ---------------------------------------------------
    if len(image_paths) > 1 and summary_rows:
        save_combined_summary(summary_rows, output_dir)

    print(
        f"\nDone: {success_count} processed, {skip_count} skipped."
    )
    if success_count == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
