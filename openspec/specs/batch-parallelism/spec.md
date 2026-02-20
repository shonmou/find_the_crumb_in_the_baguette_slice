# batch-parallelism Specification

## Purpose
TBD - created by archiving change add-parallel-batch. Update Purpose after archive.
## Requirements
### Requirement: Parallel Worker Control

The script SHALL accept a `--workers N` integer argument in batch mode that controls how many images are processed concurrently via `concurrent.futures.ProcessPoolExecutor`. When `--workers` is omitted the default SHALL be `os.cpu_count()`. The argument SHALL be silently ignored (no error) when only a single image is processed. A value of `--workers 1` SHALL produce output equivalent to the previous serial loop.

#### Scenario: Default workers in batch mode

- **WHEN** the user runs `python identify_bread_area.py --folder ./images` without `--workers`
- **THEN** the script resolves the worker count to `os.cpu_count()` and processes images concurrently

#### Scenario: Explicit worker count

- **WHEN** the user runs `python identify_bread_area.py --folder ./images --workers 4`
- **THEN** at most 4 images are processed concurrently

#### Scenario: Single-worker equivalence

- **WHEN** the user runs `python identify_bread_area.py --folder ./images --workers 1`
- **THEN** all output files and `combined_summary.csv` are equivalent to the previous serial baseline

#### Scenario: Invalid worker count

- **WHEN** the user specifies `--workers 0` or a negative integer
- **THEN** the script prints a clear error message and exits with a non-zero code

### Requirement: CSV Row Order Preservation

In parallel batch mode, `combined_summary.csv` SHALL list rows in the same order as the sorted input file list, regardless of which images finish processing first.

#### Scenario: Out-of-order completion

- **WHEN** some images complete processing before others that were submitted earlier
- **THEN** `combined_summary.csv` rows still appear in the original sorted input file order

### Requirement: HSV Conversion Deduplication

Within `process_image()`, the BGR→HSV colour-space conversion SHALL be performed exactly once per image invocation. The resulting array SHALL be passed to every helper function that previously recomputed it independently (`find_mat_region`, `find_bread_within_mat`, `create_bread_mask`, `suppress_background`).

#### Scenario: Single conversion per image call

- **WHEN** `process_image()` is called for any image
- **THEN** `cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)` is called exactly once within that call stack, not once per helper

