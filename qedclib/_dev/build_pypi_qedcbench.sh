#!/bin/bash
# Build and publish qedcbench (full package: qedclib + qedcbench) to PyPI
# Usage: ./build_pypi_qedcbench.sh          (build only)
#        ./build_pypi_qedcbench.sh upload   (build and upload)
#
# Builds directly from the top-level pyproject.toml (no swap needed).

# Navigate to repo root (two levels up from qedclib/_dev/)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

echo "=== Building qedcbench package ==="

# Clean previous builds
rm -rf dist/ build/ qedcbench.egg-info/

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
    echo "To upload: ./build_pypi_qedcbench.sh upload"
fi
