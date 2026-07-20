#!/usr/bin/env bash
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    echo "[haikyuu-chat] Activating conda environment: unsloth_1"
    conda activate unsloth_1
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    echo "[haikyuu-chat] Activating conda environment: unsloth_1"
    conda activate unsloth_1
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    echo "[haikyuu-chat] Activating conda environment: unsloth_1"
    conda activate unsloth_1
else
    echo "[haikyuu-chat] Conda was not found; using the current Python environment."
fi

cd "$PROJECT_DIR"
python run_haikyuu_chat.py "$@"
