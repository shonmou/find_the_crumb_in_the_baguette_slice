# Project Context

## Purpose

Python-based image analysis pipeline for bread research. Extracts bread regions from photographs of sliced baguettes and measures air cell (crumb hole) sizes. Outputs include extracted bread images, binary masks, annotated figures, per-image JSON stats, and combined CSV summaries for downstream analysis.

## Tech Stack

- **Python 3.10+** (uses `X | Y` union type hints, `match` not required)
- **OpenCV (cv2)** — HSV/Lab color segmentation, morphology, watershed, GrabCut
- **NumPy** — array operations on masks and images
- **Pillow (PIL)** — all image file I/O (TIFF, PNG, JPEG); PIL also used for image saving in PyImageJ pipeline to avoid macro engine OOM
- **pillow-heif** — optional plugin for HEIC/HEIF input (Apple iPhone photos)
- **PyImageJ + Fiji 2.14.0** — headless Fiji JVM for Otsu threshold + Analyze Particles
- **scyjava** — JVM configuration (heap size via `-Xmx`)
- **pandas** — tabular results, CSV output
- **matplotlib (Agg backend)** — multi-panel figures, histograms; Agg required for headless operation

## Project Conventions

### Code Style

- **pathlib.Path** everywhere; no raw string paths
- **Type annotations** on all function signatures
- Dependency imports wrapped in `try/except ImportError` with actionable install message and `sys.exit(1)`
- Constants in `UPPER_SNAKE_CASE` at module level
- Internal helpers named with `snake_case` verbs: `find_`, `create_`, `refine_`, `save_`, `measure_`
- Print progress to stdout, errors to `sys.stderr`

### Architecture Patterns

Three independent single-file CLI scripts; no shared library. Each follows the same structure:

```
argparse → collect image list → for each image: process → save outputs → optional combined CSV
```

Each script supports **single-file mode** (positional arg) and **folder batch mode** (`--folder`). In folder batch mode, per-image debug files go to `<output-dir>/<stem>/`; top-level extracted results (e.g., `_bread_exact.png`) go directly to `<output-dir>/`.

`identify_bread_area.py` uses an **inverted detection** approach: detect the stable teal cutting mat via convex hull of all teal contours, then subtract known non-bread colors (teal, green, blue/aqua, dark/shadow) rather than trying to match bread color directly.

### Testing Strategy

No automated test suite. Scripts are validated by running against the reference image set in `drive-download-*/` and inspecting outputs in `test_out_*/`. The reference good result is `test_out_4/IMG_9740/IMG_9740_bread_exact.png`.

### Git Workflow

Not currently using git. No branching strategy defined.

## Domain Context

- **Subject**: Sliced baguette cross-sections photographed on a **dark teal/green cutting mat** on a **wood desktop**
- **Teal mat HSV range**: H 78–108, S > 65, V < 140 (OpenCV 0–179 hue scale)
- **Bread crust HSV**: H 8–35 (warm golden-brown), S 60–150, V 80–200
- **Bread crumb**: near-white, S < 65 (hue irrelevant); detected as pale pixels within the mat
- **Air cells**: dark holes in the bread cross-section; measured by Fiji's Analyze Particles on an Otsu-thresholded grayscale image
- **Pixel calibration**: `--pixel-size` (mm per pixel) and `--unit` flags scale all measurements; default is raw pixels
- **Reference image**: `IMG_9740.jpeg` produces the known-good segmentation result

## Important Constraints

- `analyze_bread_crumb.py` runs Fiji in **headless JVM mode** — no AWT/GUI allowed:
  - Never use `RoiManager` (extends AWT `Frame` → `HeadlessException`)
  - Never call `ij.IJ.saveAsTiff()` (macro engine raises `RuntimeException: Macro canceled` on low memory)
  - Always call `ImagePlus.close()` not `flush()` — `flush()` leaks ghost references in `WindowManager`
  - Call `ij.IJ.run("Collect Garbage", "")` after every image in the main loop
  - Convert Java images to numpy one at a time with immediate `flush()` before converting the next
- `matplotlib` must use the `Agg` backend (set before any pyplot import) to work without a display
- HEIC support requires `pillow-heif` installed separately; scripts degrade gracefully without it

## External Dependencies

- **Fiji/ImageJ JARs** (~300 MB): downloaded automatically by PyImageJ on first run to `~/.local/share/fiji` (or equivalent). No manual install required if internet is available. Specify a local Fiji installation with `--fiji /path/to/Fiji.app` to skip download.
- **Java JDK 11+**: must be installed and `JAVA_HOME` must be set before running `analyze_bread_crumb.py`
- No network APIs or cloud services used at runtime
