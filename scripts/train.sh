#!/usr/bin/env bash
set -euo pipefail
# DEVICE=npu|cuda NPROC=8 bash scripts/train.sh ...
sc_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$sc_root"
sc_device="${DEVICE:-npu}"
sc_nproc="${NPROC:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export TOKENIZERS_PARALLELISM=false
# Weights and datasets are supplied locally; training does not download them.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec torchrun --standalone --nnodes=1 --nproc_per_node="$sc_nproc" \
  -m sc_embedding.train --device "$sc_device" "$@"
