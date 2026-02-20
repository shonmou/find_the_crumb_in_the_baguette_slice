## 1. CLI Flag

- [x] 1.1 Add `--workers` argument to `argparse` (`type=int`, default `None`; resolved to `os.cpu_count()` at runtime in batch mode, minimum 1)
- [x] 1.2 Validate `--workers >= 1`; print a clear error and exit with a non-zero code on invalid values

## 2. HSV Deduplication

- [x] 2.1 Compute `hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)` once at the top of `process_image()`
- [x] 2.2 Update `find_mat_region(bgr)` → `find_mat_region(bgr, hsv)` to accept the pre-computed HSV array; remove its internal `cv2.cvtColor` call
- [x] 2.3 Update `find_bread_within_mat(bgr, mat_region)` → `find_bread_within_mat(hsv, mat_region)`; remove its internal `cv2.cvtColor` call
- [x] 2.4 Update `create_bread_mask` and `suppress_background` similarly if they are still called in the pipeline (currently unused in `process_image` but present in the file)
- [x] 2.5 Pass the cached `hsv` array through `process_image()` to every helper that needs it

## 3. Parallel Batch Loop

- [x] 3.1 Add `import concurrent.futures` and `import os` at the top of the file (if not already present)
- [x] 3.2 Replace the serial `for i, image_path in enumerate(image_paths, 1):` loop in `main()` with `ProcessPoolExecutor(max_workers=workers)`
- [x] 3.3 Submit all images as `executor.submit(process_image, ...)` futures preserving the original sorted order
- [x] 3.4 Collect results in submission order (iterate `futures` list, not `as_completed`) so CSV rows match input order
- [x] 3.5 Print `[i/n] filename` progress as each future's result is retrieved
- [x] 3.6 Ensure the `if __name__ == "__main__":` guard is present (required for `spawn` start method on Windows)

## 4. Validation

- [x] 4.1 Run single-image mode — confirm behaviour and outputs are unchanged
- [x] 4.2 Run folder batch on ≥ 3 images with `--workers 2` — confirm all output files are produced
- [x] 4.3 Run folder batch with `--workers 1` — confirm outputs match the prior serial baseline
- [x] 4.4 Confirm `combined_summary.csv` rows appear in sorted input order regardless of worker completion order
- [x] 4.5 Confirm `--grabcut` still works correctly (flag passed through to each worker)
