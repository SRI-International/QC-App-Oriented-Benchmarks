# Release Checklist

Step-by-step instructions for publishing a new version. All build scripts are in `qedclib/_dev/`.

---

## 1. Bump Version Numbers

Update the version string in **all 5 files**:

| File | Field |
|------|-------|
| `pyproject.toml` | `version = "X.Y.Z"` |
| `pyproject-qedclib.toml` | `version = "X.Y.Z"` |
| `qedclib/pyproject.toml` | `version = "X.Y.Z"` |
| `qedclib/__init__.py` | `__version__ = "X.Y.Z"` |
| `doc/docs/about.md` | `Current version: **X.Y.Z**` |

**Important:** `pyproject-qedclib.toml` and `qedclib/pyproject.toml` must stay in sync.
If you change package metadata (authors, description, URLs, classifiers), update **both** files.

## 2. Update Release Notes

Add a new section to `doc/docs/release_notes.md` under the current major version heading:

```markdown
### Release X.Y.Z - DD Mon YYYY

- Change description
- Change description
```

## 3. Commit and Push to GitHub

```bash
git add -A
git commit -m "Bump version to X.Y.Z"
git push
git tag vX.Y.Z
git push --tags
```

## 4. Rebuild and Deploy Documentation

```bash
cd doc
python -m mkdocs build
python -m mkdocs gh-deploy
cd ..
```

Then commit the updated `doc/site/` if it changed:

```bash
git add doc/site/
git commit -m "Update docs for vX.Y.Z"
git push
```

## 5. Build PyPI Packages

Two packages are published separately. Both scripts auto-navigate to the repo root.

### qedclib (standalone execution engine)

Uses the swap trick: temporarily replaces `pyproject.toml` with `pyproject-qedclib.toml`, builds, then restores.

```bash
qedclib\_dev\build_pypi.bat              # Windows (build only)
qedclib\_dev\build_pypi.bat upload       # Windows (build and upload)
./qedclib/_dev/build_pypi.sh             # Linux
./qedclib/_dev/build_pypi.sh upload      # Linux
```

Verify before uploading:

- [ ] `dist/` contains `qedclib-X.Y.Z-py3-none-any.whl` and `qedclib-X.Y.Z.tar.gz`
- [ ] Package name is `qedclib` (not `qedcbench`)
- [ ] `pyproject.toml` was restored to `qedcbench` after build
- [ ] Version in the filename matches what you bumped

Quick metadata check:

```bash
python -c "import zipfile; z=zipfile.ZipFile('dist/qedclib-X.Y.Z-py3-none-any.whl'); print(z.read('qedclib-X.Y.Z.dist-info/METADATA').decode()[:500])"
```

### qedcbench (full suite: qedclib + qedcbench + benchmarks)

Builds directly from the top-level `pyproject.toml` — no swap needed.

```bash
qedclib\_dev\build_pypi_qedcbench.bat              # Windows (build only)
qedclib\_dev\build_pypi_qedcbench.bat upload       # Windows (build and upload)
./qedclib/_dev/build_pypi_qedcbench.sh             # Linux
./qedclib/_dev/build_pypi_qedcbench.sh upload      # Linux
```

Verify before uploading:

- [ ] `dist/` contains `qedcbench-X.Y.Z-py3-none-any.whl` and `qedcbench-X.Y.Z.tar.gz`
- [ ] Package contains both `qedclib/` and `qedcbench/` directories

## 6. Verify on PyPI

- https://pypi.org/project/qedclib/ — standalone engine
- https://pypi.org/project/qedcbench/ — full benchmark suite

Check that version, author, description, and README render correctly on both.

## 7. Update Local Install

```bash
pip install -e .
python -c "import qedclib; print(qedclib.__version__)"
```

---

## Notes

- **Versioning:** Use semantic versioning. Patch (X.Y.Z+1) for bug fixes and small additions. Minor (X.Y+1.0) for new features. Major (X+1.0.0) for breaking changes.
- **Two PyPI packages:** `qedclib` (standalone engine) and `qedcbench` (full suite including qedclib). Each has its own build script. Version numbers should stay in sync.
- **Two pyproject files for qedclib:** `qedclib/pyproject.toml` is the source of truth. `pyproject-qedclib.toml` is a copy at the repo root needed because `python -m build` must run from the top level to find the `qedclib/` package directory. The build script swaps it in temporarily.
- **Build order:** Build qedclib first, verify, then build qedcbench. If either fails, don't upload the other.
- **TestPyPI:** To test before a real publish: `twine upload --repository testpypi dist/*`
