'''
Job Store - Persist and resume benchmark jobs across sessions.

Saves job records to a JSON file so that long-running jobs can be submitted
in one session and results retrieved in a later session. Backend-agnostic:
the caller is responsible for reconnecting to the backend and retrieving results.

(C) Quantum Economic Development Consortium (QED-C) 2024.
'''

import json
import os
from datetime import datetime

JOBS_FILE = "__jobs/pending_jobs.json"


def _jobs_path():
    """Return absolute path to the jobs file, relative to cwd."""
    return os.path.abspath(JOBS_FILE)


def _load_store():
    """Load the full job store dict from disk."""
    path = _jobs_path()
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def _save_store(store):
    """Write the full job store dict to disk."""
    path = _jobs_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(store, f, indent=2)


def save_job(benchmark_id, job_id, backend_id, get_circuits_params, num_shots):
    """Save a submitted job record. Overwrites any existing entry for this benchmark."""
    store = _load_store()
    if benchmark_id in store and store[benchmark_id].get("status") == "pending":
        print(f"WARNING: Replacing existing pending job for '{benchmark_id}' "
              f"(old job_id: {store[benchmark_id]['job_id']})")
    store[benchmark_id] = {
        "job_id": str(job_id),
        "backend_id": backend_id,
        "num_shots": num_shots,
        "get_circuits_params": get_circuits_params,
        "submitted_at": datetime.now().isoformat(),
        "status": "pending"
    }
    _save_store(store)
    print(f"Job saved: benchmark='{benchmark_id}', job_id='{job_id}', backend='{backend_id}'")


def load_job(benchmark_id):
    """Load a single job record by benchmark_id. Returns dict or None."""
    store = _load_store()
    return store.get(benchmark_id)


def load_all_jobs():
    """Load all job records."""
    return _load_store()


def mark_complete(benchmark_id):
    """Mark a job as completed."""
    store = _load_store()
    if benchmark_id in store:
        store[benchmark_id]["status"] = "completed"
        store[benchmark_id]["completed_at"] = datetime.now().isoformat()
        _save_store(store)


def mark_failed(benchmark_id, error=""):
    """Mark a job as failed."""
    store = _load_store()
    if benchmark_id in store:
        store[benchmark_id]["status"] = "failed"
        store[benchmark_id]["error"] = error
        store[benchmark_id]["failed_at"] = datetime.now().isoformat()
        _save_store(store)


def remove_job(benchmark_id):
    """Remove a job record entirely."""
    store = _load_store()
    if benchmark_id in store:
        del store[benchmark_id]
        _save_store(store)
