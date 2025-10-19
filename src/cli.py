"""
Command-line interface for reproducing figures.

Usage:
    python -m src.cli reproduce --in runs/affordable --out runs/affordable/figures --dpi 600
    python -m src.cli reproduce --in runs/affordable --out runs/affordable/figures --dpi 300 --smoke
"""

import argparse
import sys
from pathlib import Path

from .figures import generate_all_figures


def main():
    parser = argparse.ArgumentParser(
        description='Reproduce figures for "No Evidence for Epistemic Entropy Collapse in Small Open Language Models"'
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Reproduce command
    reproduce_parser = subparsers.add_parser(
        "reproduce", help="Reproduce all figures from paper"
    )
    reproduce_parser.add_argument(
        "--in",
        dest="input_dir",
        type=str,
        default="runs/affordable",
        help="Path to run directory containing metrics CSVs (default: runs/affordable)",
    )
    reproduce_parser.add_argument(
        "--out",
        dest="output_dir",
        type=str,
        default="runs/affordable/figures",
        help="Directory to save generated figures (default: runs/affordable/figures)",
    )
    reproduce_parser.add_argument(
        "--dpi", type=int, default=600, help="DPI for output figures (default: 600)"
    )
    reproduce_parser.add_argument(
        "--smoke", action="store_true", help="Run smoke test with 5%% subsample"
    )

    # Legacy smoke command for backward compatibility
    smoke_parser = subparsers.add_parser(
        "smoke", help="Run fast smoke test with subsampled data"
    )
    smoke_parser.add_argument(
        "--run-dir",
        type=str,
        default="runs/affordable",
        help="Path to run directory containing metrics CSVs (default: runs/affordable)",
    )
    smoke_parser.add_argument(
        "--output-dir",
        type=str,
        default="figures/smoke/",
        help="Directory to save smoke test figures (default: figures/smoke/)",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Handle both new and legacy CLI formats
    if args.command == "reproduce":
        run_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        dpi = args.dpi
        smoke = args.smoke
    else:  # legacy 'smoke' command
        run_dir = Path(args.run_dir)
        output_dir = Path(args.output_dir)
        dpi = 300
        smoke = True

    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        sys.exit(1)

    try:
        generate_all_figures(run_dir, output_dir, smoke=smoke, dpi=dpi)
        print("\n✓ Success!")
        return 0
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
