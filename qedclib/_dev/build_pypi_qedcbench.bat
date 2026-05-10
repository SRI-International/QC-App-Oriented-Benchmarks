@echo off
REM Build and publish qedcbench (full package: qedclib + qedcbench) to PyPI
REM Usage: build_pypi_qedcbench.bat          (build only)
REM        build_pypi_qedcbench.bat upload   (build and upload)
REM
REM Builds directly from the top-level pyproject.toml (no swap needed).

REM Navigate to repo root (two levels up from qedclib/_dev/)
pushd "%~dp0..\.."

echo === Building qedcbench package ===

REM Clean previous builds
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build
if exist qedcbench.egg-info rmdir /s /q qedcbench.egg-info

REM Build
python -m build
set BUILD_RC=%errorlevel%

if %BUILD_RC% neq 0 (
    echo.
    echo === Build FAILED ===
    popd
    exit /b %BUILD_RC%
)

echo.
echo === Build complete ===
dir dist\

if "%1"=="upload" (
    echo.
    echo === Uploading to PyPI ===
    twine upload dist/*
) else (
    echo.
    echo To upload: build_pypi_qedcbench.bat upload
)

popd
