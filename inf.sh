#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CKPT_DIR="${CKPT_DIR:-}"
WAV2VEC_DIR="${WAV2VEC_DIR:-}"
LORA_DIR="${LORA_DIR:-}"
AUDIO_PROJ="${AUDIO_PROJ:-}"
COMPILE="${COMPILE:-off}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-1}"
LITE="${LITE:-1}"
COND_IMAGE="${COND_IMAGE:-./assets/photo.jpg}"
AUDIO_PATH="${AUDIO_PATH:-./assets/news.wav}"

ARGS=(
  --ckpt_dir "$CKPT_DIR"
  --wav2vec_dir "$WAV2VEC_DIR"
  --lora_dir "$LORA_DIR"
  --compile "$COMPILE"
  --num_inference_steps "$NUM_INFERENCE_STEPS"
  --cond_image "$COND_IMAGE"
  --audio_path "$AUDIO_PATH"
)

if [[ -n "$AUDIO_PROJ" ]]; then
  ARGS+=(--audio_proj "$AUDIO_PROJ")
fi

if [[ "$LITE" == "1" || "$LITE" == "true" || "$LITE" == "on" ]]; then
  ARGS+=(--lite)
else
  ARGS+=(--no_lite)
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  torchrun --nproc_per_node=1 inference.py "${ARGS[@]}" "$@"
