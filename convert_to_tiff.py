"""
Convert image files to TIFF format.

Supported input formats: JPEG, PNG, BMP, GIF, WebP, HEIC, and other
formats supported by Pillow.

Usage:
    python convert_to_tiff.py <input_file> [<input_file2> ...]
    python convert_to_tiff.py <input_file> -o <output_dir>
    python convert_to_tiff.py <input_file> --compression lzw
    python convert_to_tiff.py *.jpg --dpi 300
    python convert_to_tiff.py --folder <input_dir>
    python convert_to_tiff.py --folder <input_dir> --recursive
"""

import argparse
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow is not installed. Run: pip install Pillow")
    sys.exit(1)

# HEIC/HEIF support requires the pillow-heif plugin.
# Install with: pip install pillow-heif
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC_SUPPORTED = True
except ImportError:
    HEIC_SUPPORTED = False

HEIC_EXTENSIONS = {".heic", ".heif"}

COMPRESSION_OPTIONS = ["none", "lzw", "jpeg", "packbits", "deflate", "tiff_lzw"]

VALID_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp",
    ".heic", ".heif", ".tga", ".ico", ".ppm", ".pgm",
    ".pbm", ".tiff", ".tif",
}


def convert_to_tiff(
    input_path: Path,
    output_dir: Path | None,
    compression: str,
    dpi: int | None,
) -> Path:
    """Convert a single image file to TIFF and return the output path."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.suffix.lower() not in VALID_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format: '{input_path.suffix}'. "
            f"Supported: {', '.join(sorted(VALID_EXTENSIONS))}"
        )

    dest_dir = output_dir if output_dir else input_path.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    output_path = dest_dir / (input_path.stem + ".tiff")

    is_heic = input_path.suffix.lower() in HEIC_EXTENSIONS
    if is_heic and not HEIC_SUPPORTED:
        raise RuntimeError(
            f"HEIC/HEIF support is not available for '{input_path.name}'. "
            "Install the plugin with: pip install pillow-heif"
        )

    with Image.open(input_path) as img:
        save_kwargs = {}

        if compression != "none":
            save_kwargs["compression"] = compression

        if dpi:
            save_kwargs["dpi"] = (dpi, dpi)

        # Normalize image mode for TIFF compatibility.
        # HEIC images may be in YCbCr or 16-bit (I;16) color spaces.
        if is_heic:
            if img.mode in ("YCbCr", "CMYK"):
                img = img.convert("RGB")
            elif img.mode == "I;16":
                # 16-bit grayscale — keep as-is, TIFF supports it natively
                pass
            elif img.mode not in ("RGB", "RGBA", "L", "LA"):
                img = img.convert("RGB")
        elif img.mode == "P":
            # Palette mode (e.g. GIF) — expand to RGBA to preserve transparency
            img = img.convert("RGBA")

        img.save(output_path, format="TIFF", **save_kwargs)

    return output_path


def collect_files_from_folder(folder: Path, recursive: bool) -> list[Path]:
    """Return all image files in a folder, optionally searching recursively."""
    pattern = "**/*" if recursive else "*"
    return [
        p for p in folder.glob(pattern)
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Convert image files to TIFF format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Compression options: {', '.join(COMPRESSION_OPTIONS)}",
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        metavar="FILE",
        help="Input image file(s) to convert",
    )
    parser.add_argument(
        "--folder",
        metavar="DIR",
        help="Convert all supported images in this folder",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="When using --folder, also search subdirectories",
    )
    parser.add_argument(
        "-o", "--output-dir",
        metavar="DIR",
        help="Output directory (default: same directory as each input file)",
    )
    parser.add_argument(
        "--compression",
        default="lzw",
        choices=COMPRESSION_OPTIONS,
        help="TIFF compression method (default: lzw)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        metavar="N",
        help="Set DPI metadata in the output TIFF (e.g. 300)",
    )

    args = parser.parse_args()

    if not args.inputs and not args.folder:
        parser.error("Provide at least one FILE or use --folder <DIR>.")

    output_dir = Path(args.output_dir) if args.output_dir else None

    # Build the full list of files to process
    input_paths: list[Path] = [Path(f) for f in args.inputs]

    if args.folder:
        folder = Path(args.folder)
        if not folder.is_dir():
            print(f"Error: '{folder}' is not a directory.", file=sys.stderr)
            sys.exit(1)
        folder_files = collect_files_from_folder(folder, args.recursive)
        if not folder_files:
            print(f"No supported image files found in '{folder}'.")
            sys.exit(0)
        heic_count = sum(1 for f in folder_files if f.suffix.lower() in HEIC_EXTENSIONS)
        print(f"Found {len(folder_files)} image(s) in '{folder}'"
              + (f" ({heic_count} HEIC/HEIF)" if heic_count else "") + ".")
        if heic_count and not HEIC_SUPPORTED:
            print(
                "Warning: pillow-heif is not installed — "
                f"{heic_count} HEIC/HEIF file(s) will be skipped. "
                "Install with: pip install pillow-heif",
                file=sys.stderr,
            )
        input_paths.extend(folder_files)

    success_count = 0
    error_count = 0

    for input_path in input_paths:
        # When converting a folder recursively, mirror the subfolder structure
        # inside the output directory so files don't collide.
        effective_output_dir = output_dir
        if args.folder and output_dir and args.recursive:
            folder = Path(args.folder)
            try:
                relative = input_path.parent.relative_to(folder)
                effective_output_dir = output_dir / relative
            except ValueError:
                pass  # file was specified directly, not from --folder

        try:
            output_path = convert_to_tiff(
                input_path,
                effective_output_dir,
                args.compression,
                args.dpi,
            )
            print(f"Converted: {input_path} -> {output_path}")
            success_count += 1
        except (FileNotFoundError, ValueError, OSError, RuntimeError) as e:
            print(f"Error: {e}", file=sys.stderr)
            error_count += 1

    print(f"\nDone: {success_count} converted, {error_count} failed.")
    if error_count:
        sys.exit(1)


if __name__ == "__main__":
    main()
