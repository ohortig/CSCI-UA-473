#!/bin/bash

# Deactivate conda (if active)
conda deactivate 2>/dev/null || true

# Deactivate uv (if active)
deactivate 2>/dev/null || true

# Activate the uv environment
source .venv/bin/activate 2>/dev/null || true
