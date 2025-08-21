#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_mmap.sh <splits_dir> [options]

Options:
  --device {cuda|cpu}      (default: auto-detect)
  --max-frames N           (default: 100)
  --sample-rate N          (default: 1)
  --cache-size N           (default: 2)
  --out-dir DIR            (default: sibling 'glb_out' next to splits_dir)
  -h, --help               Show this help
EOF
}

if [[ $# -eq 0 ]] || [[ "${1:-}" == -h ]] || [[ "${1:-}" == --help ]]; then
  usage
  exit 0
fi

SPLITS_DIR="$1"; shift || true
if [[ ! -d "$SPLITS_DIR" ]]; then
  echo "Error: splits_dir '$SPLITS_DIR' not found" >&2
  exit 1
fi

# Defaults
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  DEF_DEVICE="cuda"
else
  DEF_DEVICE="cpu"
fi
DEF_MAX_FRAMES=100
DEF_SAMPLE_RATE=1
DEF_CACHE_SIZE=2
DEF_OUT_DIR="$(dirname "$SPLITS_DIR")/glb_out"

DEVICE=""
MAX_FRAMES=""
SAMPLE_RATE=""
CACHE_SIZE=""
OUT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) DEVICE="${2:?}"; shift 2;;
    --max-frames) MAX_FRAMES="${2:?}"; shift 2;;
    --sample-rate) SAMPLE_RATE="${2:?}"; shift 2;;
    --cache-size) CACHE_SIZE="${2:?}"; shift 2;;
    --out-dir) OUT_DIR="${2:?}"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1" >&2; usage; exit 1;;
  esac
done

# Apply defaults if unset
DEVICE="${DEVICE:-$DEF_DEVICE}"
MAX_FRAMES="${MAX_FRAMES:-$DEF_MAX_FRAMES}"
SAMPLE_RATE="${SAMPLE_RATE:-$DEF_SAMPLE_RATE}"
CACHE_SIZE="${CACHE_SIZE:-$DEF_CACHE_SIZE}"
OUT_DIR="${OUT_DIR:-$DEF_OUT_DIR}"

mkdir -p "$OUT_DIR"

echo "[run_mmap] splits_dir=$SPLITS_DIR"
echo "[run_mmap] device=$DEVICE max_frames=$MAX_FRAMES sample_rate=$SAMPLE_RATE cache_size=$CACHE_SIZE out_dir=$OUT_DIR"

exec python /workspace/demo_mmap.py \
  "$SPLITS_DIR" \
  --device "$DEVICE" \
  --max-frames "$MAX_FRAMES" \
  --sample-rate "$SAMPLE_RATE" \
  --cache-size "$CACHE_SIZE" \
  --out-dir "$OUT_DIR"