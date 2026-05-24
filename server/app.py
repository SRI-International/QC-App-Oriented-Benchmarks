"""
QED-C Benchmarks Server

FastAPI server that serves documentation and a benchmark execution UI.

Usage:
    cd server
    uvicorn app:app --reload --port 8088

Or use the start_server scripts at the repo root.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — no plt.show() popups

import os
import sys
import io
import time
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import json

# Add repo root to path so we can import qedcbench
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from qedcbench.run_all import (
    import_benchmark, configure_backend,
    ALL_UNIFORM_BENCHMARKS_QISKIT, ALL_UNIFORM_BENCHMARKS_CUDAQ,
    DEFAULT_BENCHMARKS_QISKIT, DEFAULT_BENCHMARKS_CUDAQ,
)
from qedclib import metrics

# Disable interactive plot display — save to files only
metrics.show_plot_images = False

app = FastAPI(title="QED-C Benchmarks")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# --- State for active run ---

active_run = None  # dict with run state, or None
run_lock = threading.Lock()


class RunRequest(BaseModel):
    api: str = "qiskit"
    backend_id: str = "qasm_simulator"
    min_qubits: int = 2
    max_qubits: int = 8
    max_circuits: int = 3
    num_shots: int = 100
    method: int = 1
    benchmarks: list[str] = []


class OutputCapture(io.TextIOBase):
    """Captures stdout and stores complete lines in a thread-safe buffer."""

    def __init__(self, original):
        self.original = original
        self.buffer = []       # complete lines ready to send
        self._partial = ""     # incomplete line being accumulated
        self.lock = threading.Lock()

    def write(self, s):
        if s:
            self.original.write(s)
            self.original.flush()
            with self.lock:
                self._partial += s
                # Emit complete lines (terminated by \n), keep any trailing partial
                while '\n' in self._partial:
                    line, self._partial = self._partial.split('\n', 1)
                    self.buffer.append(line)
        return len(s) if s else 0

    def flush(self):
        self.original.flush()

    def get_new_lines(self, start):
        """Return buffer entries from index start onward."""
        with self.lock:
            return self.buffer[start:]


def run_benchmarks_thread(run_state):
    """Execute benchmarks in a background thread, updating run_state."""
    capture = run_state["capture"]
    old_stdout = sys.stdout
    sys.stdout = capture

    try:
        benchmarks = run_state["benchmarks"]
        run_args = run_state["run_args"]
        total = len(benchmarks)

        for i, name in enumerate(benchmarks):
            if run_state["cancelled"]:
                run_state["events"].append({
                    "event": "log", "data": "Run cancelled by user."
                })
                # Mark all remaining benchmarks as cancelled
                for remaining in benchmarks[i:]:
                    run_state["results"][remaining] = {"status": "CANCELLED", "elapsed": 0}
                    run_state["events"].append({
                        "event": "result",
                        "data": json.dumps({"benchmark": remaining, "status": "CANCELLED", "elapsed": 0})
                    })
                break

            run_state["current_benchmark"] = name
            run_state["current_index"] = i
            run_state["events"].append({
                "event": "progress",
                "data": json.dumps({
                    "benchmark": name, "index": i, "total": total, "status": "running"
                })
            })

            print(f"\n--- [{i+1}/{total}] {name} ---\n")

            bm = import_benchmark(name)
            if bm is None:
                run_state["results"][name] = {"status": "SKIPPED", "elapsed": 0}
                run_state["events"].append({
                    "event": "result",
                    "data": json.dumps({"benchmark": name, "status": "SKIPPED", "elapsed": 0})
                })
                continue

            t0 = time.time()
            try:
                bm.run(**run_args)
                elapsed = round(time.time() - t0, 1)
                run_state["results"][name] = {"status": "OK", "elapsed": elapsed}
                run_state["events"].append({
                    "event": "result",
                    "data": json.dumps({"benchmark": name, "status": "OK", "elapsed": elapsed})
                })
                print(f"\n  completed in {elapsed}s")

            except KeyboardInterrupt:
                elapsed = round(time.time() - t0, 1)
                run_state["results"][name] = {"status": "CANCELLED", "elapsed": elapsed}
                run_state["cancelled"] = True

            except Exception as e:
                elapsed = round(time.time() - t0, 1)
                run_state["results"][name] = {"status": f"FAILED: {e}", "elapsed": elapsed}
                run_state["events"].append({
                    "event": "result",
                    "data": json.dumps({"benchmark": name, "status": f"FAILED: {e}", "elapsed": elapsed})
                })
                print(f"\n  FAILED after {elapsed}s: {e}")

                # If the first benchmark fails, likely a backend issue — abort the run
                if i == 0:
                    run_state["cancelled"] = True
                    print("  First benchmark failed — cancelling remaining benchmarks.")

        # Generate combined volumetric plot
        images = []
        try:
            backend_id = run_state["run_args"].get("backend_id", "qasm_simulator")
            metrics.plot_all_app_metrics(backend_id)
            import matplotlib.pyplot as plt
            plt.close('all')

            # Find generated images
            img_dir = os.path.join(os.getcwd(), "__images", backend_id.replace("/", "_"))
            if os.path.isdir(img_dir):
                for f in sorted(os.listdir(img_dir)):
                    if f.endswith(".jpg"):
                        images.append(f"/api/images/{backend_id.replace('/', '_')}/{f}")
        except Exception as e:
            print(f"  Plot generation failed: {e}")

        total_elapsed = round(time.time() - run_state["start_time"], 1)
        run_state["events"].append({
            "event": "done",
            "data": json.dumps({
                "total_elapsed": total_elapsed,
                "results": run_state["results"],
                "images": images,
            })
        })

    finally:
        sys.stdout = old_stdout
        run_state["finished"] = True


# === API Endpoints ===

@app.get("/api/benchmarks")
async def get_benchmarks(api: str = "qiskit"):
    """Return available benchmarks for the given API, with defaults marked."""
    if api == "cudaq":
        all_bms = ALL_UNIFORM_BENCHMARKS_CUDAQ
        defaults = DEFAULT_BENCHMARKS_CUDAQ
    else:
        all_bms = ALL_UNIFORM_BENCHMARKS_QISKIT
        defaults = DEFAULT_BENCHMARKS_QISKIT

    return {
        "api": api,
        "benchmarks": [
            {"name": name, "default": name in defaults}
            for name in all_bms
        ]
    }


@app.post("/api/run")
async def start_run(req: RunRequest):
    """Start a benchmark run. Returns a run ID."""
    global active_run

    with run_lock:
        if active_run and not active_run["finished"]:
            raise HTTPException(409, "A run is already in progress")

        run_id = str(uuid.uuid4())[:8]

        # Build run_args like run_all.py does
        run_args = {
            "min_qubits": req.min_qubits,
            "max_qubits": req.max_qubits,
            "max_circuits": req.max_circuits,
            "num_shots": req.num_shots,
            "method": req.method,
            "api": req.api,
            "backend_id": req.backend_id,
            "draw_circuits": False,
            "plot_results": False,
        }

        # Use batch_size=1 for local simulators so cancel can take effect between circuits
        # (don't apply to cloud simulators like ionq_simulator where batching matters)
        local_simulators = ["qasm_simulator", "statevector_simulator", "aer_sampler", "statevector_sampler", "nvidia"]
        if req.backend_id in local_simulators:
            run_args["max_batch_size"] = 1

        # Configure hardware backend
        configure_backend(req.backend_id, run_args)

        # Determine benchmarks
        if req.benchmarks:
            benchmarks = req.benchmarks
        else:
            benchmarks = (DEFAULT_BENCHMARKS_CUDAQ if req.api == "cudaq"
                          else DEFAULT_BENCHMARKS_QISKIT)

        capture = OutputCapture(sys.stdout)
        run_state = {
            "id": run_id,
            "benchmarks": benchmarks,
            "run_args": run_args,
            "capture": capture,
            "results": {},
            "events": [],
            "current_benchmark": None,
            "current_index": 0,
            "start_time": time.time(),
            "finished": False,
            "cancelled": False,
        }

        active_run = run_state
        thread = threading.Thread(target=run_benchmarks_thread, args=(run_state,), daemon=True)
        thread.start()

    return {"run_id": run_id, "benchmarks": benchmarks}


@app.get("/api/run/{run_id}/stream")
async def stream_run(run_id: str, request: Request):
    """SSE stream of run output — log lines, progress, results, done."""
    if not active_run or active_run["id"] != run_id:
        # Run not found or already finished — return 204 to prevent EventSource reconnect
        raise HTTPException(204)

    run_state = active_run

    def format_sse(event, data):
        """Format a single SSE message."""
        return f"event: {event}\ndata: {data}\n\n"

    async def event_generator():
        log_cursor = 0
        event_cursor = 0
        done = False

        try:
            while not done:
                # Check if client disconnected (prevents socket.send() warnings on Windows)
                if await request.is_disconnected():
                    break

                # If this run was replaced by a new one, stop streaming
                if active_run is not run_state:
                    break

                # Send new log lines first (before events, so "done" is the last thing sent)
                new_lines = run_state["capture"].get_new_lines(log_cursor)
                for line in new_lines:
                    yield format_sse("log", line)
                log_cursor += len(new_lines)

                # Send new structured events (progress, result, done)
                new_events = run_state["events"][event_cursor:]
                for ev in new_events:
                    yield format_sse(ev["event"], ev["data"])
                    event_cursor += 1
                    if ev["event"] == "done":
                        done = True
                        break  # stop yielding immediately after "done"

                if not done and run_state["finished"] and event_cursor >= len(run_state["events"]):
                    break

                if not done:
                    await asyncio.sleep(0.3)

        except asyncio.CancelledError:
            pass  # Client disconnected — normal during SSE
        except Exception as e:
            import traceback
            print(f"\n[SSE stream error] {type(e).__name__}: {e}")
            traceback.print_exc()

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/run/{run_id}/stop")
async def stop_run(run_id: str):
    """Cancel an active run."""
    if not active_run or active_run["id"] != run_id:
        raise HTTPException(404, "Run not found")
    active_run["cancelled"] = True
    # Signal the execute module to stop between circuits/batches
    import qedclib
    ex = qedclib.execute
    if ex and hasattr(ex, 'request_cancel'):
        ex.request_cancel()
    return {"status": "cancelling"}


@app.get("/api/run/{run_id}/status")
async def run_status(run_id: str):
    """Poll current run status."""
    if not active_run or active_run["id"] != run_id:
        raise HTTPException(404, "Run not found")
    return {
        "finished": active_run["finished"],
        "cancelled": active_run["cancelled"],
        "current_benchmark": active_run["current_benchmark"],
        "current_index": active_run["current_index"],
        "total": len(active_run["benchmarks"]),
        "results": active_run["results"],
    }


@app.get("/api/data_files")
async def list_data_files():
    """List all DATA-*.json files in __data/."""
    data_dir = Path(os.getcwd()) / "__data"
    if not data_dir.exists():
        return {"files": []}
    files = sorted(f.name for f in data_dir.glob("DATA-*.json"))
    return {"files": files}


@app.post("/api/clear_data")
async def clear_data(files: list[str]):
    """Delete specified data files from __data/."""
    data_dir = Path(os.getcwd()) / "__data"
    cleared = []
    for name in files:
        # Only allow DATA-*.json files, no path traversal
        if not name.startswith("DATA-") or not name.endswith(".json") or "/" in name or "\\" in name:
            continue
        filepath = data_dir / name
        if filepath.exists():
            filepath.unlink()
            cleared.append(name)
    return {"cleared": cleared}


@app.get("/api/images/{backend_id}/{filename}")
async def serve_image(backend_id: str, filename: str):
    """Serve a generated plot image."""
    filepath = Path(os.getcwd()) / "__images" / backend_id / filename
    if not filepath.exists() or not filepath.suffix in (".jpg", ".png", ".pdf"):
        raise HTTPException(404, "Image not found")
    media = "image/jpeg" if filepath.suffix == ".jpg" else f"image/{filepath.suffix[1:]}"
    return StreamingResponse(open(filepath, "rb"), media_type=media)


# === Static files and UI ===

# Serve the benchmark runner UI
@app.get("/", response_class=HTMLResponse)
async def ui():
    """Serve the benchmark runner UI."""
    ui_path = Path(__file__).parent / "index.html"
    if ui_path.exists():
        return HTMLResponse(ui_path.read_text(encoding="utf-8"))
    return RedirectResponse(url="/site/")

# Serve the mkdocs-generated documentation site
site_dir = Path(__file__).parent.parent / "doc" / "site"
if site_dir.exists():
    app.mount("/site", StaticFiles(directory=str(site_dir), html=True), name="site")
