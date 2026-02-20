# Change: Ellipse-based protrusion filter for foreign objects

## Why

When a foreign object (e.g., cardboard wedge used to prop up the bread slice) is physically flush against the bread on the cutting mat, all existing filters fail to remove it:

- **Colour**: the cardboard has essentially the same warm-brown HSV as bread crust (H ≈ 22, S ≈ 57, V ≈ 192 vs. crust H ≈ 23, S ≈ 60+, V ≈ 150+). `find_bread_within_mat` cannot distinguish them by colour.
- **Connected-component separation**: with even a 40 px erosion the bread and cardboard remain a single connected component (the contact face is wide). Pre-close + component filters therefore do nothing.
- **Morphological opening**: the contact zone is too wide to be broken by any practical kernel size.

The only reliable discriminator is **shape**. Baguette cross-sections are smooth and roughly elliptical (solidity ≈ 0.95–0.97 on clean images). Angular cardboard wedges produce sharp protrusions that extend beyond the smooth ellipse that best fits the bread outline.

## What Changes

Add a new `clip_to_ellipse(contour, filled_mask)` function that:

1. Fits a minimum-enclosing ellipse to the final contour using `cv2.fitEllipse`.
2. Draws the fitted ellipse as a binary mask.
3. Returns `cv2.bitwise_and(filled_mask, ellipse_mask)` — the intersection clips anything that extends beyond the smooth elliptical boundary.

Insert this step in `process_image()` after GrabCut (step 6) and before the final measurement (step 7):

```
step 6  → GrabCut → contour, filled_mask
step 6b → clip_to_ellipse(contour, filled_mask)  ← NEW
           → find_bread_contour(clipped_mask) → contour, filled_mask
step 7  → measure_bread(...)
```

No new CLI flag is needed: for clean images the fitted ellipse is a tight fit to the actual crust so the intersection changes the mask by only a few pixels; for cardboard images it clips the angular protrusion.

## Why the ellipse works despite being fit to bread+cardboard

`cv2.fitEllipse` minimises the sum of squared distances to all contour points. A typical baguette contour has ~2,000 points tracing the ~2,600 px oval. The two cardboard wedges contribute ~200–400 points in two small angular regions. The fitted ellipse is dominated by the ~1,600 bread points; it remains a smooth oval, merely slightly elongated toward the cardboard directions. The cardboard tip — which extends sharply outward — lies beyond this smooth oval. The AND operation clips it.

## Impact

- Affected specs: `foreign-protrusion-removal` (new)
- Affected code: `identify_bread_area.py` — one new function `clip_to_ellipse`, one call in `process_image`
- No CLI changes, no output layout changes
- **Known limitation**: if the foreign object is very large relative to the bread (dominating the contour), the ellipse will be distorted and may not clip it cleanly. For typical small wedge supports this is not the case.
- **Known limitation**: if the bread is not approximately elliptical (e.g., a crescent-shaped slice), the ellipse clip may cut genuine bread area. The fallback condition (clip must not shrink the area by more than 25%) protects against catastrophic over-clipping.
