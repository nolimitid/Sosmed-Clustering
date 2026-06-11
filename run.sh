#!/bin/bash
source ~/pyvenvs/cluster-venv/bin/activate
cd ~/app/cluster
uvicorn api:app --host 0.0.0.0 --port 8880