#!/bin/bash
echo "Starting QED-C Benchmarks Server ..."
echo "Documentation will be available at http://localhost:8088/site/"
echo
cd "$(dirname "$0")/server"
python -m uvicorn app:app --reload --port 8088
