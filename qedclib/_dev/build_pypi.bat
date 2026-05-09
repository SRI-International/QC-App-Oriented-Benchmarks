@echo off
REM Build and publish qedclib to PyPI
REM Usage: build_pypi.bat          (build only)
REM        build_pypi.bat upload   (build and upload)
REM
REM Can run from anywhere — auto-navigates to repo root.
REM Temporarily swaps pyproject.toml to build qedclib only.

REM Navigate to repo root (two levels up from qedclib/_dev/)
pushd "%~dp0..\.."

echo === Building qedclib package ===

REM Clean previous builds
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build
if exist qedclib.egg-info rmdir /s /q qedclib.egg-info

REM Swap in qedclib config (save main)
copy pyproject.toml pyproject-main-save.toml >nul
copy pyproject-qedclib.toml pyproject.toml >nul

REM Build
python -m build
set BUILD_RC=%errorlevel%

REM Restore main pyproject.toml (always, even if build failed)
copy pyproject-main-save.toml pyproject.toml >nul
del pyproject-main-save.toml

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
    echo To upload: build_pypi.bat upload
)

popd
