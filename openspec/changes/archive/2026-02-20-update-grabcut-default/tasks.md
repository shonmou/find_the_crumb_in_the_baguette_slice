## 1. Invert the GrabCut flag

- [x] 1.1 Replace the `--grabcut` argument block in `argparse` with `--no-grabcut` (`action="store_false"`, `dest="use_grabcut"`, `default=True`, updated help text)
- [x] 1.2 Update the `process_image()` call site: change `use_grabcut=args.grabcut` → `use_grabcut=args.use_grabcut`
- [x] 1.3 Update the module-level docstring: replace `--grabcut` usage example with `--no-grabcut`

## 2. Validation

- [x] 2.1 Run single-image mode without any flag — confirm GrabCut fires (progress line "Refining boundary with GrabCut …" appears) and outputs are produced
- [x] 2.2 Run single-image mode with `--no-grabcut` — confirm GrabCut is skipped and outputs are produced
- [x] 2.3 Run folder batch on ≥ 3 images (default, no flags) — confirm all images succeed with GrabCut active
- [x] 2.4 Visually compare `_bread_area.png` with vs. without `--no-grabcut` on the same image — confirm boundary is smoother with GrabCut

## Implementation notes

- `dest="use_grabcut"` with `default=True` and `action="store_false"` means: not passed → `use_grabcut=True`; `--no-grabcut` passed → `use_grabcut=False`. The call site uses `args.use_grabcut` directly (no negation needed).
- GrabCut boundary is tighter on the crust edge and produces a slightly lower area (5,306,234 vs 5,318,966 px²) because the watershed tends to bleed a few pixels onto the teal mat; GrabCut's colour model correctly classifies those mat-edge pixels as background.
