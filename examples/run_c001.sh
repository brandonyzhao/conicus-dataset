#!/usr/bin/env bash
set -euo pipefail

RAW_ROOT=${1:?usage: examples/run_c001.sh /path/to/abacus_lightcones}
PYTHON=${PYTHON:-python}

"$PYTHON" build_lensplane_dataset.py \
  --stages direct-patches \
  --phases 0-4 \
  --phase-prefix c001_ \
  --raw-root "$RAW_ROOT" \
  --patch-root ./lp_dataset_patched_c001 \
  --metadata-root ./metadata \
  --dataset-name lp_dataset_patched_c001 \
  --step-source any \
  --no-assert-overlaps \
  --skip-existing
