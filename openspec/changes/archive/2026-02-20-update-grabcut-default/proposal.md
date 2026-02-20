# Change: GrabCut boundary refinement on by default

## Why

The current pipeline always ends with a watershed step that produces pixel-level, gradient-guided boundaries. Watershed boundaries are inherently jagged: the algorithm snaps to the steepest local gradient at each pixel, producing staircase artifacts along smooth crust curves. This is visible on every output image as stepped/zigzag edges that don't follow the physical crust contour.

GrabCut (already implemented as `apply_grabcut`, currently opt-in via `--grabcut`) solves this directly: it fits Gaussian Mixture Models to the foreground (bread) and background (mat) colour distributions, then minimises an energy function via graph-cut. The result is a smooth, model-coherent boundary aligned to colour transitions rather than local gradients. It is well-known that GrabCut produces significantly cleaner contours than watershed on natural-image segmentation tasks.

The only reason GrabCut is currently opt-in is speed (~10× slower than morphology alone). Since the user's priority is accuracy and slower processing is acceptable, the flag polarity should be reversed.

## What Changes

- **BREAKING:** `--grabcut` (opt-in, `action="store_true"`) → `--no-grabcut` (opt-out, `action="store_false"`).  GrabCut now runs on every image by default. Users who need the faster pipeline can pass `--no-grabcut`.
- `process_image()` parameter name stays `use_grabcut: bool`; its call site flips from `args.grabcut` to `not args.no_grabcut`.
- Module-level docstring usage examples updated (remove `--grabcut` example, add `--no-grabcut` example).
- `argparse` help text for both the removed and added flags updated accordingly.

## Impact

- Affected specs: `boundary-refinement` (new)
- Affected code: `identify_bread_area.py` — `main()` argparse block, call site of `process_image`, module docstring
- **Breaking change for existing callers** who do not pass `--grabcut` today: their pipelines will now run GrabCut and be slower. Callers who currently pass `--grabcut` must drop the flag.
- No changes to output file layout, stats schema, or other scripts
