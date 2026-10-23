#!/bin/bash

# Train base -> SFT -> RL while saving 6 checkpoints in total, then evaluate each one,
# to see how capabilities develop over the course of training.
#
# Checkpoint budget (6 total, see nanochat.checkpoint_manager.split_checkpoint_budget):
#   base: 4  log-spaced (~3%, ~10%, ~31%, 100% of pretraining)
#   sft : 1  (final)
#   rl  : 1  (final)
#
# Assumes the tokenizer and dataset are already prepared (see runs/speedrun.sh).
#
# Usage:
#   bash runs/checkpoint_series.sh
#   DEPTH=12 NPROC=1 MODEL_TAG=series_d12 bash runs/checkpoint_series.sh
#   WANDB_RUN=series bash runs/checkpoint_series.sh

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
[ -d ".venv" ] && source .venv/bin/activate

DEPTH="${DEPTH:-24}"
NPROC="${NPROC:-8}"
MODEL_TAG="${MODEL_TAG:-series_d${DEPTH}}"
WANDB_RUN="${WANDB_RUN:-dummy}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"
RESULTS_DIR="${RESULTS_DIR:-runs/checkpoint_series_results/$MODEL_TAG}"
mkdir -p "$RESULTS_DIR"

RUN="torchrun --standalone --nproc_per_node=$NPROC -m"

# -----------------------------------------------------------------------------
# 1) Base pretraining: 4 log-spaced checkpoints (the last one is the final model)
$RUN scripts.base_train -- --depth=$DEPTH --target-param-data-ratio=8 \
    --device-batch-size=$DEVICE_BATCH_SIZE --fp8 --run=$WANDB_RUN --model-tag=$MODEL_TAG \
    --num-checkpoints=4 --checkpoint-spacing=log --checkpoint-min-frac=0.03

# 2) SFT: final checkpoint only
$RUN scripts.chat_sft -- --run=$WANDB_RUN --model-tag=$MODEL_TAG --num-checkpoints=1

# 3) RL: final checkpoint only
$RUN scripts.chat_rl -- --run=$WANDB_RUN --model-tag=$MODEL_TAG --num-checkpoints=1

# -----------------------------------------------------------------------------
# Evaluate every saved checkpoint
BASE_CKPT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG"
for f in "$BASE_CKPT_DIR"/model_*.pt; do
    step=$((10#$(basename "$f" .pt | sed 's/model_//')))
    echo "=== base step $step ==="
    $RUN scripts.base_eval -- --model-tag=$MODEL_TAG --step=$step --device-batch-size=$DEVICE_BATCH_SIZE \
        2>&1 | tee "$RESULTS_DIR/base_step_$(printf '%06d' $step).log"
done

# Chat-stage checkpoints (final only; chat_eval loads the last step by default)
for source in sft rl; do
    echo "=== $source (final) ==="
    $RUN scripts.chat_eval -- -i $source -g $MODEL_TAG 2>&1 | tee "$RESULTS_DIR/${source}_final.log"
done

echo "Done. Logs in $RESULTS_DIR"
