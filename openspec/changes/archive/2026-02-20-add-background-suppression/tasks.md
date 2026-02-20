## 1. Pipeline wiring

- [x] 1.1 In `process_image()`, after `raw_mask = find_bread_within_mat(hsv, mat_region)` (step 2), add `raw_mask = suppress_background(bgr, hsv, raw_mask)` before the `refine_mask` call (step 3)
- [x] 1.2 Add a `print("  Suppressing background …")` progress line before the call (consistent with watershed step style)

## 2. Validation

- [x] 2.1 Run single-image mode on `IMG_9740.jpeg` — confirm detected area and output files are produced and bread area is not smaller than baseline (stats should be within ~5% of previous values)
- [x] 2.2 Run batch with `--workers 1` on the full drive-download folder — confirm all 15 images succeed, no images regress to "no bread detected"
- [x] 2.3 Visually inspect at least one `_bread_area.png` figure to confirm mat markings are excluded and bread boundary is clean

## Implementation notes

Two additional corrections were made to `suppress_background` to make it safe for use with the inverted-detection `raw_mask`:

- **Dilation radius reduced**: from `max(20, min(h,w)//60) × 3 iterations` (~150 px) to `max(3, min(h,w)//300) × 1 iteration` (~10 px). Ruler marks sit directly on the teal surface; a 10 px spread is sufficient to catch them without reaching into the bread's crumb edge.
- **Wood-grain exclusion removed**: the range H 8–40, S < 62, V > 120 overlaps with 33% of the image (measured on IMG_9740), including the warm-tinted near-white crumb. It caused a 34% area regression and has been removed. Wood grain inside the mat hull is handled by the morphological opening in `refine_mask`.
