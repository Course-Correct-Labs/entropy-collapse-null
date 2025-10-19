#!/usr/bin/env bash
#
# Run code quality checks with ruff.
#
# Usage:
#   bash scripts/lint_check.sh
#

set -euo pipefail

echo "=== Running code quality checks ==="
echo ""

if ! command -v ruff &> /dev/null; then
    echo "Error: ruff not found. Install with: pip install ruff"
    exit 1
fi

echo "Running ruff linter..."
ruff check . --select E,F,W,I --ignore E501

echo ""
echo "Running black formatter check..."
if command -v black &> /dev/null; then
    black --check .
else
    echo "Warning: black not found, skipping black check"
fi

echo ""
echo "✓ All checks passed!"
