"""
Bread crumb size analyzer using Fiji/ImageJ via PyImageJ.

Workflow:
  1. Load bread image(s)
  2. Convert to 8-bit grayscale
  3. Denoise with Gaussian blur
  4. Threshold with Otsu's method to isolate air cells
  5. Run Analyze Particles to measure each air cell
  6. Report statistics and save per-image CSV + histogram
  7. (Folder mode) Save a combined summary CSV across all images

Requirements:
    pip install pyimagej scyjava pandas matplotlib pillow

    Java (JDK 11+) must be installed and JAVA_HOME must be set.
    On first run, Fiji JARs (~300 MB) are downloaded automatically.

Usage:
    python analyze_bread_crumb.py bread.jpg
    python analyze_bread_crumb.py bread.jpg --output-dir ./results
    python analyze_bread_crumb.py bread.jpg --min-size 50 --max-size 50000
    python analyze_bread_crumb.py bread.jpg --pixel-size 0.05 --unit mm
    python analyze_bread_crumb.py bread.jpg --fiji /path/to/Fiji.app
    python analyze_bread_crumb.py --folder ./bread_images
    python analyze_bread_crumb.py --folder ./bread_images --recursive -o ./results
"""

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------
try:
    import scyjava
except ImportError:
    print("Error: scyjava is not installed. Run: pip install pyimagej scyjava")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("Error: pandas is not installed. Run: pip install pandas")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend; safe in headless mode
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: matplotlib is not installed. Run: pip install matplotlib")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FIJI_VERSION = "sc.fiji:fiji:2.14.0"

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif",
}


# ---------------------------------------------------------------------------
# Fiji initialisation
# ---------------------------------------------------------------------------
def init_fiji(fiji_path: str | None, memory_gb: int = 4):
    """Start the Fiji/ImageJ JVM and return the ImageJ gateway object."""
    import imagej  # imported here so the import error is clear

    scyjava.config.add_options(f"-Xmx{memory_gb}g")

    if fiji_path:
        path = Path(fiji_path)
        if not path.exists():
            raise FileNotFoundError(f"Fiji not found at: {fiji_path}")
        print(f"Initializing Fiji from local installation: {fiji_path}")
        ij = imagej.init(str(path), mode="headless")
    else:
        print(f"Initializing Fiji ({FIJI_VERSION}) — first run downloads ~300 MB …")
        ij = imagej.init(FIJI_VERSION, mode="headless")

    print(f"Fiji ready  (ImageJ version: {ij.getVersion()})")
    return ij


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------
def preprocess(ij, image_path: Path):
    """
    Open and preprocess the bread image.

    Returns an 8-bit grayscale ImagePlus ready for thresholding.
    """
    print(f"Opening image: {image_path}")
    imp = ij.IJ.openImage(str(image_path))
    if imp is None:
        raise RuntimeError(f"Fiji could not open the image: {image_path}")

    # Convert to 8-bit grayscale
    ij.IJ.run(imp, "8-bit", "")

    # Gaussian blur to reduce imaging noise before thresholding
    ij.IJ.run(imp, "Gaussian Blur...", "sigma=1")

    return imp


# ---------------------------------------------------------------------------
# Thresholding
# ---------------------------------------------------------------------------
def threshold_otsu(ij, imp):
    """
    Apply Otsu's auto-threshold and convert to binary mask.

    Air cells (bright regions in crumb cross-sections) become white (255).
    Dough matrix becomes black (0).
    """
    print("Applying Otsu threshold …")
    ij.IJ.setAutoThreshold(imp, "Otsu dark")
    ij.IJ.run(imp, "Convert to Mask", "")
    return imp


# ---------------------------------------------------------------------------
# Particle analysis
# ---------------------------------------------------------------------------
def analyze_particles(ij, imp, min_size_px: float, max_size_px: float) -> pd.DataFrame:
    """
    Run ImageJ's Analyze Particles on the binary mask.

    Returns a DataFrame with one row per detected air cell.
    """
    # Import Java classes through scyjava
    ParticleAnalyzer = scyjava.jimport("ij.plugin.filter.ParticleAnalyzer")
    ResultsTable = scyjava.jimport("ij.measure.ResultsTable")
    Measurements = scyjava.jimport("ij.measure.Measurements")

    # Configure which measurements to collect
    measurements = (
        Measurements.AREA
        | Measurements.PERIMETER
        | Measurements.CIRCULARITY
        | Measurements.ELLIPSE
        | Measurements.SHAPE_DESCRIPTORS    # AR, roundness, solidity
        | Measurements.FERET
        | Measurements.CENTROID
    )

    options = (
        ParticleAnalyzer.CLEAR_WORKSHEET          # start with fresh ResultsTable
        | ParticleAnalyzer.EXCLUDE_EDGE_PARTICLES  # ignore cut-off cells at border
    )

    rt = ResultsTable()
    analyzer = ParticleAnalyzer(
        options,
        measurements,
        rt,
        min_size_px,   # minimum area in pixels²
        max_size_px,   # maximum area in pixels²
        0.0,           # min circularity (0 = any shape)
        1.0,           # max circularity (1 = perfect circle)
    )

    print("Running Analyze Particles …")
    analyzer.analyze(imp)

    n = rt.size()
    print(f"Detected {n} air cell(s).")

    if n == 0:
        return pd.DataFrame()

    # Pull columns out of the Java ResultsTable into a pandas DataFrame
    col_names = [rt.getColumnHeading(i) for i in range(rt.getLastColumn() + 1)
                 if rt.getColumnHeading(i) is not None]
    data = {}
    for col in col_names:
        col_data = rt.getColumnAsDoubles(rt.getColumnIndex(col))
        if col_data is not None:
            data[col] = list(col_data)
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Statistics & reporting
# ---------------------------------------------------------------------------
def print_statistics(df: pd.DataFrame, pixel_size: float, unit: str):
    """Print a formatted summary of crumb air-cell measurements."""
    scale = pixel_size ** 2   # px² → unit²

    area_col = "Area"
    circ_col = "Circ."

    areas = df[area_col] * scale if area_col in df else None
    circs = df[circ_col] if circ_col in df else None

    print("\n" + "=" * 52)
    print("  BREAD CRUMB ANALYSIS — SUMMARY")
    print("=" * 52)
    print(f"  Air cells detected         : {len(df)}")

    if areas is not None:
        print(f"\n  Air cell area ({unit}²):")
        print(f"    Mean                     : {areas.mean():.4f}")
        print(f"    Median                   : {areas.median():.4f}")
        print(f"    Std dev                  : {areas.std():.4f}")
        print(f"    Min                      : {areas.min():.4f}")
        print(f"    Max                      : {areas.max():.4f}")

    if circ_col in df:
        print(f"\n  Circularity  (0=line, 1=circle):")
        print(f"    Mean                     : {circs.mean():.3f}")
        print(f"    Std dev                  : {circs.std():.3f}")

    # Feret diameter (longest axis)
    if "Feret" in df:
        ferets = df["Feret"] * pixel_size
        print(f"\n  Max Feret diameter ({unit}):")
        print(f"    Mean                     : {ferets.mean():.4f}")
        print(f"    Max                      : {ferets.max():.4f}")

    print("=" * 52 + "\n")


def save_outputs(
    ij,
    df: pd.DataFrame,
    image_path: Path,
    output_dir: Path,
    pixel_size: float,
    unit: str,
    original_imp=None,
    mask_imp=None,
):
    """Save the CSV results table and histogram plot for one image."""
    import numpy as np
    from PIL import Image, ImageFilter

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    # --- CSV -----------------------------------------------------------------
    csv_path = output_dir / f"{stem}_crumb_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Results saved : {csv_path}")

    # --- Binary mask + annotated overlay (PIL-based, AWT-free) ---------------
    # Convert one Java image at a time and flush it immediately so the JVM heap
    # never holds more than one image copy alongside the Python arrays.
    if original_imp is not None and mask_imp is not None:
        # 1. Convert original → numpy, then free the Java object right away.
        orig_arr = np.array(ij.py.from_java(original_imp))
        original_imp.flush()
        if orig_arr.dtype != np.uint8:
            lo, hi = orig_arr.min(), orig_arr.max()
            orig_arr = ((orig_arr - lo) / max(hi - lo, 1) * 255).astype(np.uint8)

        # 2. Convert mask → numpy, then free the Java object right away.
        mask_arr = np.array(ij.py.from_java(mask_imp))
        mask_imp.flush()
        mask_binary = (mask_arr > 0).astype(np.uint8) * 255
        del mask_arr  # release before allocating overlay arrays

        # Save binary mask via PIL (avoids the ImageJ macro engine entirely).
        mask_path = output_dir / f"{stem}_mask.tiff"
        Image.fromarray(mask_binary).save(str(mask_path), format="TIFF")
        print(f"  Mask saved    : {mask_path}")

        # Build outline: dilated mask minus mask = one-pixel border.
        mask_pil = Image.fromarray(mask_binary)
        outline = np.array(mask_pil.filter(ImageFilter.MaxFilter(3))) - mask_binary
        del mask_binary, mask_pil

        # Compose RGB overlay: grayscale original + red outlines.
        rgb = np.stack([orig_arr, orig_arr, orig_arr], axis=-1)
        del orig_arr
        rgb[outline > 0] = [255, 0, 0]
        del outline

        overlay_path = output_dir / f"{stem}_overlay.tiff"
        Image.fromarray(rgb).save(str(overlay_path), format="TIFF")
        del rgb
        print(f"  Overlay saved : {overlay_path}")

    # --- Histogram of air-cell areas -----------------------------------------
    if "Area" not in df or df.empty:
        return

    areas = df["Area"] * (pixel_size ** 2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Bread Crumb Analysis — {image_path.name}", fontsize=13)

    # Area distribution
    axes[0].hist(areas, bins=30, color="steelblue", edgecolor="white")
    axes[0].set_xlabel(f"Air cell area ({unit}²)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Air Cell Area Distribution")

    # Circularity distribution
    if "Circ." in df:
        axes[1].hist(df["Circ."], bins=20, color="darkorange", edgecolor="white")
        axes[1].set_xlabel("Circularity  (0 = line, 1 = circle)")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Air Cell Circularity Distribution")

    plt.tight_layout()
    plot_path = output_dir / f"{stem}_crumb_histogram.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Histogram saved: {plot_path}")


def save_combined_summary(rows: list[dict], output_dir: Path):
    """Write a single summary CSV with one row per successfully analysed image."""
    if not rows:
        return
    summary_path = output_dir / "combined_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"\nCombined summary saved: {summary_path}")


# ---------------------------------------------------------------------------
# Folder helpers
# ---------------------------------------------------------------------------
def collect_images_from_folder(folder: Path, recursive: bool) -> list[Path]:
    """Return all supported image files in a folder."""
    pattern = "**/*" if recursive else "*"
    return [
        p for p in sorted(folder.glob(pattern))
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def build_summary_row(image_path: Path, df: pd.DataFrame, pixel_size: float, unit: str) -> dict:
    """Distil per-cell DataFrame into a single summary dict for the combined CSV."""
    scale = pixel_size ** 2
    row: dict = {"file": str(image_path), "air_cells": len(df)}

    if "Area" in df and not df.empty:
        areas = df["Area"] * scale
        row.update({
            f"area_mean_{unit}2": round(areas.mean(), 6),
            f"area_median_{unit}2": round(areas.median(), 6),
            f"area_std_{unit}2": round(areas.std(), 6),
            f"area_min_{unit}2": round(areas.min(), 6),
            f"area_max_{unit}2": round(areas.max(), 6),
        })

    if "Circ." in df and not df.empty:
        row.update({
            "circularity_mean": round(df["Circ."].mean(), 4),
            "circularity_std": round(df["Circ."].std(), 4),
        })

    if "Feret" in df and not df.empty:
        ferets = df["Feret"] * pixel_size
        row.update({
            f"feret_mean_{unit}": round(ferets.mean(), 6),
            f"feret_max_{unit}": round(ferets.max(), 6),
        })

    return row


# ---------------------------------------------------------------------------
# Single-image pipeline
# ---------------------------------------------------------------------------
def process_image(
    ij,
    image_path: Path,
    output_dir: Path,
    min_size: float,
    max_size: float,
    pixel_size: float,
    unit: str,
) -> dict | None:
    """
    Run the full analysis pipeline for one image.

    Returns a summary dict (for the combined CSV) or None if analysis failed.
    """
    imp = None
    original_imp = None
    mask_imp = None
    try:
        imp = preprocess(ij, image_path)
        original_imp = imp.duplicate()   # keep preprocessed image for the overlay
        imp = threshold_otsu(ij, imp)
        mask_imp = imp.duplicate()       # keep binary mask before particle analysis
        df = analyze_particles(ij, imp, min_size, max_size)
    except Exception as exc:
        print(f"  Error analysing '{image_path.name}': {exc}", file=sys.stderr)
        return None
    finally:
        # close() releases pixel data AND removes the image from the
        # WindowManager, preventing reference accumulation across a long batch.
        if imp is not None:
            imp.close()

    if df.empty:
        print(
            f"  No air cells detected in '{image_path.name}'. "
            "Try adjusting --min-size / --max-size."
        )
        if original_imp is not None:
            original_imp.close()
        if mask_imp is not None:
            mask_imp.close()
        return None

    try:
        print_statistics(df, pixel_size, unit)
        save_outputs(ij, df, image_path, output_dir, pixel_size, unit, original_imp, mask_imp)
        return build_summary_row(image_path, df, pixel_size, unit)
    finally:
        # save_outputs already flush()ed both objects; close() cleans up
        # any remaining WindowManager registration.
        if original_imp is not None:
            original_imp.close()
        if mask_imp is not None:
            mask_imp.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Analyze bread crumb structure (air-cell size) using Fiji/ImageJ.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "image",
        nargs="?",
        help="Path to a single bread cross-section image",
    )
    parser.add_argument(
        "--folder",
        metavar="DIR",
        help="Analyze all supported images in this folder",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="When using --folder, also search subdirectories",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="./crumb_results",
        metavar="DIR",
        help="Directory for CSV and histogram output (default: ./crumb_results)",
    )
    parser.add_argument(
        "--min-size",
        type=float,
        default=10.0,
        metavar="PX²",
        help="Minimum air-cell area in pixels² to count (default: 10)",
    )
    parser.add_argument(
        "--max-size",
        type=float,
        default=1_000_000.0,
        metavar="PX²",
        help="Maximum air-cell area in pixels² to count (default: 1000000)",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=1.0,
        metavar="N",
        help="Physical size of one pixel in --unit (default: 1.0 = pixel units)",
    )
    parser.add_argument(
        "--unit",
        default="px",
        metavar="STR",
        help="Measurement unit label, e.g. mm (default: px)",
    )
    parser.add_argument(
        "--memory",
        type=int,
        default=4,
        metavar="GB",
        help="JVM heap memory in GB (default: 4)",
    )
    parser.add_argument(
        "--fiji",
        metavar="PATH",
        help="Path to a local Fiji.app installation (skips auto-download)",
    )

    args = parser.parse_args()

    if not args.image and not args.folder:
        parser.error("Provide an image file or use --folder <DIR>.")

    try:
        import imagej  # noqa: F401 — validate import before heavy JVM init
    except ImportError:
        print(
            "Error: pyimagej is not installed.\n"
            "Install with: pip install pyimagej",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir = Path(args.output_dir)

    # --- Build list of images to process ------------------------------------
    image_paths: list[Path] = []

    if args.image:
        p = Path(args.image)
        if not p.exists():
            print(f"Error: image not found: {p}", file=sys.stderr)
            sys.exit(1)
        if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
            print(
                f"Error: unsupported format '{p.suffix}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
                file=sys.stderr,
            )
            sys.exit(1)
        image_paths.append(p)

    if args.folder:
        folder = Path(args.folder)
        if not folder.is_dir():
            print(f"Error: '{folder}' is not a directory.", file=sys.stderr)
            sys.exit(1)
        found = collect_images_from_folder(folder, args.recursive)
        if not found:
            print(f"No supported image files found in '{folder}'.")
            sys.exit(0)
        print(f"Found {len(found)} image(s) in '{folder}'.")
        image_paths.extend(found)

    # --- Initialise Fiji once for all images --------------------------------
    ij = init_fiji(args.fiji, memory_gb=args.memory)

    # --- Process each image -------------------------------------------------
    summary_rows: list[dict] = []
    success_count = 0
    skip_count = 0
    error_count = 0

    for i, image_path in enumerate(image_paths, 1):
        print(f"\n[{i}/{len(image_paths)}] {image_path.name}")
        print("-" * 52)

        # When processing a folder, put each image's results in its own
        # sub-directory so files from different images don't overwrite each other.
        if args.folder and len(image_paths) > 1:
            image_output_dir = output_dir / image_path.stem
        else:
            image_output_dir = output_dir

        row = process_image(
            ij,
            image_path,
            image_output_dir,
            args.min_size,
            args.max_size,
            args.pixel_size,
            args.unit,
        )

        if row is None:
            skip_count += 1
        else:
            summary_rows.append(row)
            success_count += 1

        # Force a JVM garbage-collection cycle after every image so heap
        # pressure does not compound across a large batch.
        ij.IJ.run("Collect Garbage", "")

    # --- Combined summary (folder mode) -------------------------------------
    if len(image_paths) > 1 and summary_rows:
        save_combined_summary(summary_rows, output_dir)

    print(
        f"\nDone: {success_count} analysed, "
        f"{skip_count} skipped (no cells found or error), "
        f"{error_count} failed."
    )
    if success_count == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
