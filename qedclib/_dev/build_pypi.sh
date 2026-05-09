#!/bin/bash
# Build and publish qedclib to PyPI
# Usage: ./build_pypi.sh          (build only)
#        ./build_pypi.sh upload   (build and upload)
#
# Can run from anywhere — auto-navigates to repo root.
# Temporarily swaps pyproject.toml to build qedclib only.

# Navigate to repo root (two levels up from qedclib/_dev/)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

echo "=== Building qedclib package ==="

# Clean previous builds
rm -rf dist/ build/ qedclib.egg-info/

# Swap in qedclib config (save main)
cp pyproject.toml pyproject-main-save.toml
cp pyproject-qedclib.toml pyproject.toml

# Ensure restore happens even on failure
restore_pyproject() {
    mv pyproject-main-save.toml pyproject.toml
}
trap restore_pyproject EXIT

# Build
python -m build

echo ""
echo "=== Build complete ==="
ls -l dist/

if [ "$1" = "upload" ]; then
    echo ""
    echo "=== Uploading to PyPI ==="
    twine upload dist/*
else
    echo ""
    echo "To upload: ./build_pypi.sh upload"
fi
