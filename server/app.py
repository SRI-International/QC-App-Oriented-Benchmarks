"""
QED-C Benchmarks Server

A simple FastAPI server that serves the documentation site.
Future: will serve a full benchmark execution app.

Usage:
    cd server
    uvicorn app:app --reload --port 8088

Or use the start_server scripts at the repo root.
"""

from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

app = FastAPI(title="QED-C Benchmarks")

# Serve the mkdocs-generated documentation site
site_dir = Path(__file__).parent.parent / "doc" / "site"
if site_dir.exists():
    app.mount("/site", StaticFiles(directory=str(site_dir), html=True), name="site")

@app.get("/")
async def root():
    """Redirect root to documentation site."""
    return RedirectResponse(url="/site/")
