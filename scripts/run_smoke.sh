#!/usr/bin/env bash
#
# Run smoke test with subsampled data (5% sample, ~30 sequences).
# Should complete in <5 minutes without GPU.
#
# Usage:
#   bash scripts/run_smoke.sh
#

set -euo pipefail

echo "=== Running smoke test (5% subsample) ==="
echo "Expected runtime: <5 minutes"
echo ""

# Create output directory
mkdir -p runs/affordable/figures

# Run smoke test
time python3 -m src.cli reproduce --in runs/affordable --out runs/affordable/figures --dpi 300 --smoke

echo ""
echo "=== Smoke test completed ==="
echo "Output directory: runs/affordable/figures/"
echo ""
echo "Generated files:"
ls -lh runs/affordable/figures/*_smoke.png

echo ""
echo "✓ Smoke test passed! Full reproduction should work correctly."
