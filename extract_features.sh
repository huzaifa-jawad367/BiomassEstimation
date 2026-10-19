#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Configuration: edit these defaults or set the corresponding environment variables.
# For Colab/WSL, replace the E:/ paths with your mounted Linux dataset paths.
CHECKPOINT="${CHECKPOINT:-$ROOT/models/tf_efficientnetv2_l_in21k_f0_b8x2_e100_nrmse_devscse_attnlin_augs_decplus7_plus800eb_200ft/modelo_best.pth}"
TRAIN_FEATURES_DIR="${TRAIN_FEATURES_DIR:-E:/Datasets/Biomassters/train_features/Volumes/Samsung_T5/BIOMASS/BioMasster_Dataset/v1/DrivenData/train_features}"
# Resume into the extracted_features directory beside this repository.
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/../extracted_features}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-2}"
DEVICE="${DEVICE:-cuda}"
MAX_SAMPLES="${MAX_SAMPLES:-0}" # 0 extracts all samples.
OVERWRITE="${OVERWRITE:-false}" # true recomputes selected samples.
PYTHON="${PYTHON:-}" # Empty selects a local .venv or system python3.

if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_BIN="$PYTHON"
elif [[ -x "$ROOT/.venv/Scripts/python.exe" ]]; then
    PYTHON_BIN="$ROOT/.venv/Scripts/python.exe"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
else
    echo "No Python interpreter found. Set PYTHON to your Python executable." >&2
    exit 1
fi

ARGS=(
    --checkpoint "$CHECKPOINT"
    --train-features-dir "$TRAIN_FEATURES_DIR"
    --output-dir "$OUTPUT_DIR"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --device "$DEVICE"
    --max-samples "$MAX_SAMPLES"
)
case "${OVERWRITE,,}" in
    true|1|yes|on) ARGS+=(--overwrite) ;;
    false|0|no|off) ;;
    *) echo "OVERWRITE must be true or false (also accepts 1/0, yes/no, on/off)." >&2; exit 1 ;;
esac

# Explicit scalar CLI arguments override the configuration above.
exec "$PYTHON_BIN" -u "$ROOT/src/extract_features.py" "${ARGS[@]}" "$@"
