#!/usr/bin/env bash
set -euo pipefail

RAW_ROOT=${1:?usage: examples/run_c000.sh /path/to/abacus_lightcones}
PYTHON=${PYTHON:-python}

"$PYTHON" build_lensplane_dataset.py \
  --stages direct-patches \
  --phases 0-24 \
  --raw-root "$RAW_ROOT" \
  --patch-root ./lp_dataset_patched \
  --metadata-root ./metadata \
  --dataset-name lp_dataset_patched \
  --step-source any \
  --step-min 408 \
  --step-max 1105 \
  --no-assert-overlaps \
  --skip-existing
