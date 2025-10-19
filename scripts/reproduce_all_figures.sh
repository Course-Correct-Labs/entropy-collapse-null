#!/usr/bin/env bash
#
# Reproduce all figures from the paper in one command.
#
# Usage:
#   bash scripts/reproduce_all_figures.sh
#

set -euo pipefail

echo "=== Reproducing all figures from paper ==="
echo ""

# Check if data exists
if [ ! -f "runs/affordable/metrics_internal.csv" ]; then
    echo "Error: Data files not found in runs/affordable/"
    echo "Please ensure metrics_internal.csv and metrics_external.csv are present."
    exit 1
fi

# Create output directory
mkdir -p runs/affordable/figures

# Run reproduction
python -m src.cli reproduce --in runs/affordable --out runs/affordable/figures --dpi 600

echo ""
echo "=== Figures generated successfully ==="
echo "Output directory: runs/affordable/figures/"
echo ""
echo "Generated files:"
ls -lh runs/affordable/figures/*.png

echo ""
echo "✓ Done! All figures ready for inspection."
