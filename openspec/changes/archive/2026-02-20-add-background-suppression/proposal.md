# Change: Wire background suppression into the segmentation pipeline

## Why

Near-white ruler tick marks and grid numbers printed on the cutting mat pass through every exclusion condition in `find_bread_within_mat`:

- Not teal (saturation < 40, below the S > 65 teal threshold)
- Not green/blue (saturation < 40)
- Not dark (value > 175)

So they land in `raw_mask` as "bread candidates". After morphological refinement, marks that are adjacent to the bread contour survive and inflate the detected boundary outward.

A `suppress_background` function already exists in `identify_bread_area.py` that handles all three background bleed sources:

1. Residual teal mat surface pixels
2. Near-white mat ruler/grid markings (dilated-mat AND near-white gate)
3. Warm low-saturation wood-grain desktop pixels

It is correctly implemented and accepts `bgr`, `hsv`, and `bread_mask` — but it is **not called anywhere in `process_image`**, making it dead code.

## What Changes

- Insert one call to `suppress_background(bgr, hsv, raw_mask)` in `process_image()` between step 2 (inverted mat detection) and step 3 (morphological refinement), replacing `raw_mask` in place.
- No new functions, no new parameters, no changed signatures.

## Impact

- Affected specs: `background-suppression` (new)
- Affected code: `identify_bread_area.py` — `process_image()` only (one line added)
- No change to CLI flags, output file layout, or any other function
- **Known trade-off**: the mat-marking dilation in `suppress_background` is ~150 px for a 3000 px wide image. Pale crumb pixels (S < 40, V > 175) that sit within ~150 px of the mat edge would be removed. In practice, baguette slices are centred on the mat so this boundary is rarely reached; any small gaps are recovered by the morphological closing that follows immediately in step 3.
