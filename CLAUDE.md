<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Three independent Python CLI scripts for bread image analysis. Typical workflow runs them in order:

1. `convert_to_tiff.py` — convert HEIC/JPEG source photos to TIFF
2. `identify_bread_area.py` — detect and extract the bread region from each photo
3. `analyze_bread_crumb.py` — measure air cell sizes in bread cross-sections using Fiji/ImageJ

Input images are baguette slices photographed on a **dark teal/green cutting mat** placed on a **wood desktop**. The teal mat is the anchor for segmentation in `identify_bread_area.py`.

## Dependencies

```
pip install Pillow pillow-heif                              # convert_to_tiff.py (HEIC support)
pip install opencv-python numpy pandas matplotlib pillow   # identify_bread_area.py
pip install pyimagej scyjava pandas matplotlib pillow      # analyze_bread_crumb.py
# analyze_bread_crumb.py also requires Java JDK 11+ with JAVA_HOME set
```

## Running the Scripts

```bash
# Single image
python convert_to_tiff.py image.heic -o ./tiffs --compression lzw
python identify_bread_area.py bread.jpg -o ./results --bg transparent
python analyze_bread_crumb.py bread.tiff --output-dir ./results --pixel-size 0.05 --unit mm

# Batch folder
python convert_to_tiff.py --folder ./drive-download-* -o ./tiffs
python identify_bread_area.py --folder ./tiffs -o ./bread_area_results
python analyze_bread_crumb.py --folder ./tiffs --recursive -o ./crumb_results

# identify_bread_area.py optional flags
--grabcut                       # slower but more accurate boundary (runs after watershed)
--bg white|black|transparent    # background for _bread_exact.png
--hue-low / --hue-high          # HSV tuning if bread detection fails
```

## Architecture: `identify_bread_area.py`

### Detection pipeline (in order)

1. **`find_mat_region(bgr)`** — Detects dark-teal pixels (H 78–108, S>65, V<140), merges ALL significant teal contours into one point cloud, takes the convex hull. The hull spans across the bread-shaped hole in the teal detection. Falls back to full image if no mat found.

2. **`find_bread_within_mat(bgr, mat_region)`** — Inverted approach: bread = `mat_region AND NOT (teal OR green H35–85 OR blue/aqua H85–140 OR dark V<50)`. More reliable than colour-matching bread directly because the teal mat is consistent while bread appearance varies.

3. **`refine_mask(mask)`** — Morphological CLOSE→OPEN→CLOSE, then flood-fill hole removal (pad image, flood from corner, fill unreachable zeros as interior holes).

4. **`find_bread_contour(mask)`** — Scores contours by `area × (0.7 + 0.3 × centrality)` to prevent edge blobs from outscoring the bread.

5. **`refine_mask_edges(bgr, filled_mask)`** — Watershed with bilateral-filtered image as guide. Always runs. Seeds: eroded mask = foreground, dilated-then-inverted = background, band between = unknown (marker 0).

6. **`apply_grabcut(bgr, filled_mask)`** — Optional (`--grabcut`). Runs after watershed. Uses eroded mask core as definite-foreground anchor.

### Output file layout (folder mode)

```
<output-dir>/
  IMG_9740_bread_exact.png      ← extract lands here (flat, top-level)
  IMG_9741_bread_exact.png
  combined_summary.csv
  IMG_9740/                     ← per-image debug files
    IMG_9740_bread_crop.png
    IMG_9740_bread_area.png     ← 3-panel figure (original | mask | annotated)
    IMG_9740_mask.png
    IMG_9740_bread_stats.json
```

## Architecture: `analyze_bread_crumb.py`

Uses **PyImageJ in headless mode** to run Fiji's Analyze Particles on thresholded bread cross-sections.

### Key headless constraints

- **Never use `RoiManager`** — it extends AWT `Frame` and throws `HeadlessException` even with `RoiManager(False)`. Use PIL outline detection instead.
- **Save images via PIL**, not `ij.IJ.saveAsTiff()` — the macro engine raises `RuntimeException: Macro canceled` on low memory instead of a clean OOM.
- **Use `imp.close()` not `imp.flush()`** — `flush()` frees pixel data but leaves the `ImagePlus` in `WindowManager`, causing a ghost-reference leak. After ~34 images the heap fills up. `close()` removes from `WindowManager`.
- **Call `ij.IJ.run("Collect Garbage", "")` after every image** in the main loop — JVM GC doesn't run frequently enough on its own schedule.
- **Convert Java images one at a time**: call `np.array(ij.py.from_java(imp))` then immediately `imp.flush()` before converting the next image. Holding two large Java arrays simultaneously doubles peak heap usage.

### `ResultsTable` → DataFrame

`rt.getColumnAsDoubles(idx)` returns `None` for sparse columns. Skip `None` columns entirely rather than substituting `or []` — an empty list breaks `pd.DataFrame()` construction when other columns have data.