# QED-C Benchmarks Documentation

This directory contains the documentation source files and build tools for the QED-C Application-Oriented Benchmarks.

The documentation is built using [mkdocs](https://www.mkdocs.org/), which converts markdown files into a themed HTML site. Source files are in the `docs/` folder.

## Building the Documentation

Install mkdocs (once):
```
pip install mkdocs
```

From this `doc/` directory, build the documentation:
```
mkdocs build
```

This produces a `site/` folder containing the complete HTML documentation. Open `site/index.html` to view.

## Live Preview

For development with automatic reload:
```
mkdocs serve
```

Then open [http://localhost:8000/](http://localhost:8000/) in your browser. Changes to files in `docs/` are reflected immediately.

## Convenience Script

On Windows, run:
```
make_docs
```

## Documentation Files

| File | Content |
|------|---------|
| `docs/index.md` | Documentation overview and navigation |
| `docs/quick_start.md` | First-time user walkthrough |
| `docs/user_guide.md` | Complete feature reference |
| `docs/release_notes.md` | Version history |
| `docs/known_issues.md` | Problems, anomalies, and limitations |
| `docs/about.md` | Project background and credits |
| `docs/setup/` | Platform-specific setup guides |
| `_design/` | Internal design documents (not in nav) |
