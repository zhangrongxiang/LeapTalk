#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-leaptalk}"

CKPT_DIR="${CKPT_DIR:-./models/SoulX-FlashHead-1_3B}"
WAV2VEC_DIR="${WAV2VEC_DIR:-./models/wav2vec2-base-960h}"
VIDEO_DIR="${VIDEO_DIR:-./data/VividHead/videos}"
AUDIO_DIR="${AUDIO_DIR:-./data/VividHead/audios}"
SAVE_DIR="${SAVE_DIR:-./outputs/train}"

TEACHER_MODEL_DIR="${TEACHER_MODEL_DIR:-}"
TEACHER_LORA_DIR="${TEACHER_LORA_DIR:-}"
TEACHER_AUDIO_PROJ="${TEACHER_AUDIO_PROJ:-}"
RESUME_LORA_DIR="${RESUME_LORA_DIR:-}"
RESUME_AUDIO_PROJ="${RESUME_AUDIO_PROJ:-}"

EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LR="${LR:-1e-4}"
VAL_EVERY="${VAL_EVERY:-200}"
MAX_FRAMES="${MAX_FRAMES:-125}"
STUDENT_STEPS="${STUDENT_STEPS:-1}"
TEACHER_STEPS="${TEACHER_STEPS:-4}"
ROLLOUT_CHUNKS="${ROLLOUT_CHUNKS:-2}"
LORA_RANK="${LORA_RANK:-64}"

BRIDGE_LOSS_WEIGHT="${BRIDGE_LOSS_WEIGHT:-1.0}"
GT_LOSS_WEIGHT="${GT_LOSS_WEIGHT:-0.0}"
KD_LOSS_WEIGHT="${KD_LOSS_WEIGHT:-0.0}"
BRIDGE_NOISE_SCALE="${BRIDGE_NOISE_SCALE:-1.0}"
USE_SPATIAL_WEIGHTING="${USE_SPATIAL_WEIGHTING:-1}"
LAMBDA_FACE="${LAMBDA_FACE:-2.0}"
LAMBDA_LIP="${LAMBDA_LIP:-5.0}"
GAMMA_TEMPORAL="${GAMMA_TEMPORAL:-1.0}"
MASK_CACHE_DIR="${MASK_CACHE_DIR:-}"
MASK_DILATE_LIP="${MASK_DILATE_LIP:-5}"
MASK_DILATE_FACE="${MASK_DILATE_FACE:-10}"

USE_DMD="${USE_DMD:-1}"
DMD_LOSS_WEIGHT="${DMD_LOSS_WEIGHT:-1.0}"
CRITIC_LOSS_WEIGHT="${CRITIC_LOSS_WEIGHT:-1.0}"
CRITIC_LR="${CRITIC_LR:-8e-5}"
FAKE_SCORE_LORA_RANK="${FAKE_SCORE_LORA_RANK:-64}"
RESUME_FAKE_SCORE_LORA_DIR="${RESUME_FAKE_SCORE_LORA_DIR:-}"
RESUME_FAKE_SCORE_AUDIO_PROJ="${RESUME_FAKE_SCORE_AUDIO_PROJ:-}"
LPIPS_LOSS_WEIGHT="${LPIPS_LOSS_WEIGHT:-4.0}"
LPIPS_NET="${LPIPS_NET:-vgg}"
LPIPS_MODEL_PATH="${LPIPS_MODEL_PATH:-/cache/vgg.pth}"
LPIPS_CROP_SIZE="${LPIPS_CROP_SIZE:-256}"



ARGS=(
    --ckpt_dir "$CKPT_DIR"
    --wav2vec_dir "$WAV2VEC_DIR"
    --video_dir "$VIDEO_DIR"
    --audio_dir "$AUDIO_DIR"
    --save_dir "$SAVE_DIR"
    --epochs "$EPOCHS"
    --batch_size "$BATCH_SIZE"
    --lr "$LR"
    --val_every "$VAL_EVERY"
    --max_frames "$MAX_FRAMES"
    --student_steps "$STUDENT_STEPS"
    --teacher_steps "$TEACHER_STEPS"
    --rollout_chunks "$ROLLOUT_CHUNKS"
    --lora_rank "$LORA_RANK"
    --bridge_loss_weight "$BRIDGE_LOSS_WEIGHT"
    --gt_loss_weight "$GT_LOSS_WEIGHT"
    --kd_loss_weight "$KD_LOSS_WEIGHT"
    --bridge_noise_scale "$BRIDGE_NOISE_SCALE"
    --lambda_face "$LAMBDA_FACE"
    --lambda_lip "$LAMBDA_LIP"
    --gamma_temporal "$GAMMA_TEMPORAL"
    --mask_dilate_lip "$MASK_DILATE_LIP"
    --mask_dilate_face "$MASK_DILATE_FACE"
    --dmd_loss_weight "$DMD_LOSS_WEIGHT"
    --critic_loss_weight "$CRITIC_LOSS_WEIGHT"
    --critic_lr "$CRITIC_LR"
    --fake_score_lora_rank "$FAKE_SCORE_LORA_RANK"
    --lpips_loss_weight "$LPIPS_LOSS_WEIGHT"
    --lpips_net "$LPIPS_NET"
    --lpips_crop_size "$LPIPS_CROP_SIZE"
    --lpips_model_path "$LPIPS_MODEL_PATH"
    --dmd_step_list 1000 750 500 250
    --vis_step_list 1000
    --use_bridge_step_list
    --disable_alpha_stabilization
    --freeze_audio_proj
)

if [[ "$USE_SPATIAL_WEIGHTING" == "1" ]]; then
    ARGS+=(--use_spatial_weighting)
fi

if [[ -n "$MASK_CACHE_DIR" ]]; then
    ARGS+=(--mask_cache_dir "$MASK_CACHE_DIR")
fi

if [[ "$USE_DMD" == "1" ]]; then
    ARGS+=(--use_dmd)
fi

if [[ -n "$TEACHER_MODEL_DIR" ]]; then
    ARGS+=(--teacher_model_dir "$TEACHER_MODEL_DIR")
fi
if [[ -n "$TEACHER_LORA_DIR" ]]; then
    ARGS+=(--teacher_lora_dir "$TEACHER_LORA_DIR")
fi
if [[ -n "$TEACHER_AUDIO_PROJ" ]]; then
    ARGS+=(--teacher_audio_proj "$TEACHER_AUDIO_PROJ")
fi
if [[ -n "$RESUME_LORA_DIR" ]]; then
    ARGS+=(--resume_lora_dir "$RESUME_LORA_DIR")
fi
if [[ -n "$RESUME_AUDIO_PROJ" ]]; then
    ARGS+=(--resume_audio_proj "$RESUME_AUDIO_PROJ")
fi
if [[ -n "$RESUME_FAKE_SCORE_LORA_DIR" ]]; then
    ARGS+=(--resume_fake_score_lora_dir "$RESUME_FAKE_SCORE_LORA_DIR")
fi
if [[ -n "$RESUME_FAKE_SCORE_AUDIO_PROJ" ]]; then
    ARGS+=(--resume_fake_score_audio_proj "$RESUME_FAKE_SCORE_AUDIO_PROJ")
fi
python "train.py" "${ARGS[@]}" "$@"

# python "train_flashhead_vibt_ardmd.py" "${ARGS[@]}" "$@"
