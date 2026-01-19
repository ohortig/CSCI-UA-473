#!/bin/bash

# Deactivate conda (if active)
conda deactivate 2>/dev/null || true

# Deactivate uv (if active)
deactivate 2>/dev/null || true

# Install uv (if not installed). WARNING: the next line pipes a script from the web to sh.
which uv || curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv --python 3.11
source .venv/bin/activate 2>/dev/null || true

# Install pre-commit
uv pip install pre-commit

# Install pre-commit hooks
pre-commit install
