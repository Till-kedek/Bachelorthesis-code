#!/bin/zsh
set -eu
cd "${0:A:h}"
export JUPYTER_RUNTIME_DIR="$PWD/.venv/jupyter-runtime"
export MPLCONFIGDIR="$PWD/.venv/matplotlib"
exec "$PWD/.venv/bin/jupyter" lab "$PWD/Thesis_analysis_v2.ipynb"
