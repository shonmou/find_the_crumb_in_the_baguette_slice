# Change: Parallel batch processing for identify_bread_area.py

## Why

In folder batch mode the script processes images strictly serially. Every image spends the bulk of its runtime in CPU-bound OpenCV operations (HSV conversion, morphological operations, bilateral filter, watershed). On a quad-core or better machine all but one core sit idle, so batches of 10–20 baguette photos take several minutes when they could take under one.

Additionally, `cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)` is called independently by three separate helpers inside each `process_image()` invocation (`find_mat_region`, `find_bread_within_mat`, and `create_bread_mask`), wasting CPU time on identical conversions.

## What Changes

- Add `--workers N` CLI flag: number of concurrent worker processes in batch mode (default: `os.cpu_count()`; ignored for single-image mode).
- Replace the serial `for` loop in `main()` with `concurrent.futures.ProcessPoolExecutor` to process images in parallel.
- Deduplicate BGR→HSV conversion inside `process_image()`: compute once, pass the cached array to `find_mat_region`, `find_bread_within_mat`, and any other helper that currently recomputes it.
- `combined_summary.csv` row order is preserved to match the sorted input file list regardless of worker completion order.

## Impact

- Affected specs: `batch-parallelism` (new capability)
- Affected code: `identify_bread_area.py` — `main()`, `process_image()`, and helper signatures for `find_mat_region` / `find_bread_within_mat`
- No change to single-image mode behaviour or to the output file layout
- No new third-party dependencies (uses stdlib `concurrent.futures` and `os`)
