#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Colab configuration: all values can be overridden through environment variables.
CHECKPOINT="${CHECKPOINT:-/content/modelo_best.pth}"
TRAIN_FEATURES_DIR="${TRAIN_FEATURES_DIR:-/content/drive/MyDrive/projects/Biomass/Biomassters/train_features/Volumes/Samsung_T5/BIOMASS/BioMasster_Dataset/v1/DrivenData/train_features}"
OUTPUT_DIR="${OUTPUT_DIR:-/content/drive/MyDrive/projects/Biomass/Biomassters/extracted_features}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-2}"
DEVICE="${DEVICE:-cuda}"
MAX_SAMPLES="${MAX_SAMPLES:-0}" # 0 extracts all samples.
OVERWRITE="${OVERWRITE:-false}" # true recomputes selected samples.
ALLOW_SOURCE_RELOCATION="${ALLOW_SOURCE_RELOCATION:-true}" # Same TIFFs, different machine/path.
# Drive's mounted free-space report is not a reliable Google account quota check.
if [[ "$OUTPUT_DIR" == /content/drive/* ]]; then
    SKIP_DISK_SPACE_CHECK="${SKIP_DISK_SPACE_CHECK:-true}"
else
    SKIP_DISK_SPACE_CHECK="${SKIP_DISK_SPACE_CHECK:-false}"
fi
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
add_boolean_flag() {
    local value="${!1}"
    case "${value,,}" in
        true|1|yes|on) ARGS+=("$2") ;;
        false|0|no|off) ;;
        *) echo "$1 must be true or false (also accepts 1/0, yes/no, on/off)." >&2; exit 1 ;;
    esac
}
add_boolean_flag OVERWRITE --overwrite
add_boolean_flag ALLOW_SOURCE_RELOCATION --allow-source-relocation
add_boolean_flag SKIP_DISK_SPACE_CHECK --skip-disk-space-check

# Explicit scalar CLI arguments override the configuration above.
exec "$PYTHON_BIN" -u "$ROOT/src/extract_features.py" "${ARGS[@]}" "$@"
