#!/bin/bash
# Script to launch the ANPR Web Studio
cd "$(dirname "$0")"
source .venv/bin/activate
echo "Starting ANPR Web Studio on http://localhost:8000 ..."
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
