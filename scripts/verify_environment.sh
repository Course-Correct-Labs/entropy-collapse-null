#!/usr/bin/env bash
#
# Verify that the conda/pip environment is correctly set up.
#
# Usage:
#   bash scripts/verify_environment.sh
#

set -euo pipefail

echo "=== Verifying environment setup ==="
echo ""

REQUIRED_PACKAGES=(
    "numpy"
    "pandas"
    "matplotlib"
    "scipy"
    "sklearn:scikit-learn"
)

MISSING=()

for pkg_spec in "${REQUIRED_PACKAGES[@]}"; do
    if [[ "$pkg_spec" == *":"* ]]; then
        import_name="${pkg_spec%%:*}"
        display_name="${pkg_spec##*:}"
    else
        import_name="$pkg_spec"
        display_name="$pkg_spec"
    fi

    echo -n "Checking for $display_name... "
    if python -c "import $import_name" 2>/dev/null; then
        VERSION=$(python -c "import $import_name; print($import_name.__version__)" 2>/dev/null || echo "unknown")
        echo "✓ ($VERSION)"
    else
        echo "✗ MISSING"
        MISSING+=("$display_name")
    fi
done

echo ""

if [ ${#MISSING[@]} -eq 0 ]; then
    echo "✓ All required packages installed!"
    echo ""
    echo "Python version:"
    python --version
    exit 0
else
    echo "✗ Missing packages: ${MISSING[*]}"
    echo ""
    echo "Install with:"
    echo "  conda env create -f environment.yml"
    echo "  conda activate eec-null"
    echo ""
    echo "Or with pip:"
    echo "  pip install -r requirements.txt"
    exit 1
fi
