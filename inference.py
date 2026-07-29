



import argparse
import contextlib
import glob
import os
import subprocess
import sys
import time
from dataclasses import dataclass

import imageio
import librosa
import numpy as np
import torch
import torch.distributed as dist
from loguru import logger
from peft import PeftModel


# Ensure we import ViBT-local packages
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
VIBT_ROOT = os.path.dirname(THIS_DIR)
SOULX_ROOT = os.path.join(os.path.dirname(VIBT_ROOT), "SoulX-FlashHead")
# NOTE: insert(0, ...) makes the *last* inserted path highest priority.
# We want: THIS_DIR > VIBT_ROOT > SOULX_ROOT so that `flash_head` resolves to ViBT's copy.
for path in (SOULX_ROOT, VIBT_ROOT, THIS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from flash_head.src.pipeline.flash_head_pipeline import FlashHeadPipeline, timestep_transform  # noqa: E402
from flash_head.utils.utils import match_and_blend_colors_torch  # noqa: E402
from vibt.scheduler import ViBTScheduler  # noqa: E402


def _build_infer_timesteps(
    *,
    step_list: list[int] | None,
    num_inference_steps: int,
    shift_gamma: float,
    device: str,
    num_timesteps: int = 1000,
) -> torch.Tensor:
    """Return 1D float tensor of timesteps for ViBTScheduler.

    Mirrors training-style warping with `timestep_transform(shift=shift_gamma)`.
    Does not include a terminal 0 step to avoid an extra Euler update at t=0.
    """

    if step_list is not None and len(step_list) > 0:
        base_steps = [int(x) for x in step_list if int(x) > 0]
        if not base_steps:
            raise ValueError("--step_list must contain at least one positive timestep")
        is_strict_desc = all(base_steps[i] > base_steps[i + 1] for i in range(len(base_steps) - 1))
        if not is_strict_desc:
            base_steps = sorted(set(base_steps), reverse=True)
            logger.warning(f"step_list is not strictly descending; using sorted unique list: {base_steps}")
    else:
        n = int(num_inference_steps)
        if n <= 0:
            raise ValueError("--num_inference_steps must be > 0")
        if n == 2:
            base_steps = [1000, 500]
        elif n == 4:
            base_steps = [1000, 750, 500, 250]
        else:
            base_steps = list(np.linspace(num_timesteps, 1, n, dtype=np.float32))
            base_steps = [int(round(float(x))) for x in base_steps]
            base_steps = [x for x in base_steps if x > 0]
            base_steps = sorted(set(base_steps), reverse=True)
            if not base_steps:
                raise ValueError("Derived empty timestep list; check --num_inference_steps")
    logger.info(f"Using timesteps: {base_steps} (derived from num_inference_steps={num_inference_steps})")
    ts = [torch.tensor([float(t)], device=device) for t in base_steps]
    ts = [timestep_transform(t, shift=float(shift_gamma), num_timesteps=num_timesteps) for t in ts]
    values = [t.reshape(-1)[0].to(device=device, dtype=torch.float32) for t in ts]
    return torch.stack(values, dim=0)


def _pick_latest_audio_proj(path_or_dir: str | None) -> str | None:
    if not path_or_dir:
        return None
    if os.path.isfile(path_or_dir):
        return path_or_dir
    candidates = glob.glob(os.path.join(path_or_dir, "audio_proj_step_*.pt"))
    if not candidates:
        return None

    def _step_num(p: str) -> int:
        base = os.path.basename(p)
        try:
            return int(base.split("audio_proj_step_")[-1].split(".pt")[0])
        except Exception:
            return -1

    candidates.sort(key=_step_num)
    return candidates[-1]


def _pick_latest_lora_dir(path_or_dir: str | None) -> str | None:
    if not path_or_dir:
        return None
    if os.path.isfile(os.path.join(path_or_dir, "adapter_config.json")):
        return path_or_dir

    release_lora_dir = os.path.join(path_or_dir, "lora")
    if os.path.isfile(os.path.join(release_lora_dir, "adapter_config.json")):
        return release_lora_dir

    candidates = glob.glob(os.path.join(path_or_dir, "lora_step_*"))
    candidates = [p for p in candidates if os.path.isfile(os.path.join(p, "adapter_config.json"))]
    if not candidates:
        return None

    def _step_num(p: str) -> int:
        base = os.path.basename(os.path.normpath(p))
        try:
            return int(base.split("lora_step_")[-1])
        except Exception:
            return -1

    candidates.sort(key=_step_num)
    return candidates[-1]


def _resolve_lite_tae_path(
    *,
    lora_dir: str,
    resolved_lora_dir: str,
    ckpt_dir: str,
    explicit_tae_path: str | None,
) -> str:
    if explicit_tae_path:
        if os.path.isfile(explicit_tae_path):
            return explicit_tae_path
        raise SystemExit(f"TAE checkpoint not found: {explicit_tae_path}")

    search_roots = [
        lora_dir,
        resolved_lora_dir,
        os.path.dirname(os.path.abspath(resolved_lora_dir)),
        os.path.join(ckpt_dir, "VAE_Wan"),
    ]
    seen = set()
    for root in search_roots:
        if not root:
            continue
        root = os.path.abspath(root)
        if root in seen:
            continue
        seen.add(root)
        for name in ("taew2_1.pth", "taew2_1.safetensors", "taew2_2.pth", "taew2_2.safetensors"):
            candidate = os.path.join(root, name)
            if os.path.isfile(candidate):
                return candidate

    raise SystemExit(
        "Lite mode requires a TAE checkpoint. Expected taew2_1.pth under --lora_dir "
        "or its parent directory after downloading https://huggingface.co/z-rx/leaptalk."
    )


def _get_inner_flashhead_model(model):
    # After PEFT wrapping, training code uses model.base_model.model
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model
    return model


def _read_num_heads_from_infer_params(vibt_root: str) -> int:
    """Best-effort parse of flash_head/configs/infer_params.yaml without requiring PyYAML."""
    cfg_path = os.path.join(vibt_root, "flash_head", "configs", "infer_params.yaml")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("num_heads"):
                    # e.g. "num_heads: 12"
                    _, value = line.split(":", 1)
                    return int(value.strip())
    except Exception:
        pass
    return 12


def _lora_checkpoint_needs_compiled_base(lora_dir: str) -> bool:
    """Detect whether adapter weights were saved from a torch.compile()'d base model."""
    weights_path = os.path.join(lora_dir, "adapter_model.safetensors")
    if not os.path.exists(weights_path):
        return False

    try:
        from safetensors.torch import safe_open

        with safe_open(weights_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if "._orig_mod." in k:
                    return True
        return False
    except Exception:
        # Best-effort: if we cannot inspect, do not force compile.
        return False


@dataclass
class StreamParams:
    frame_num: int = 33
    motion_frames_latent_num: int = 2
    tgt_fps: int = 25
    sample_rate: int = 16000
    cached_audio_duration: int = 8

    def init_with_stride(self, vae_stride_t: int) -> "InitializedStreamParams":
        # motion_frames_num is in *pixel frames*.
        motion_frames_num = (self.motion_frames_latent_num - 1) * vae_stride_t + 1
        return InitializedStreamParams(
            frame_num=self.frame_num,
            motion_frames_latent_num=self.motion_frames_latent_num,
            tgt_fps=self.tgt_fps,
            sample_rate=self.sample_rate,
            cached_audio_duration=self.cached_audio_duration,
            motion_frames_num=motion_frames_num,
        )


@dataclass
class InitializedStreamParams:
    frame_num: int
    motion_frames_latent_num: int
    tgt_fps: int
    sample_rate: int
    cached_audio_duration: int
    motion_frames_num: int

    @property
    def slice_len(self) -> int:
        return self.frame_num - self.motion_frames_num


def _open_video_writer(tmp_mp4: str, fps: int):
    os.makedirs(os.path.dirname(os.path.abspath(tmp_mp4)) or ".", exist_ok=True)
    return imageio.get_writer(
        tmp_mp4,
        format="mp4",
        mode="I",
        fps=fps,
        codec="h264",
        ffmpeg_params=["-bf", "0"],
    )


@torch.no_grad()
def _preprocess_audio_embeddings(
    pipeline: FlashHeadPipeline,
    audio_array: np.ndarray,
    sr: int,
    fps: int,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return per-frame wav2vec2 embeddings.

    Output: [T, 12, 768] where T ~= len(audio_array) * fps / sr.
    """
    video_len = int(len(audio_array) * fps / sr)
    if video_len <= 0:
        raise RuntimeError("audio too short: computed video_len <= 0")

    inputs = pipeline.wav2vec_feature_extractor(audio_array, sampling_rate=sr, return_tensors="pt", padding=True)
    audio_values = inputs.input_values.to(device)
    audio_out = pipeline.audio_encoder(audio_values, seq_len=video_len, output_hidden_states=True)

    if not getattr(audio_out, "hidden_states", None):
        raise RuntimeError("wav2vec2 returned no hidden_states")

    hs = audio_out.hidden_states[-12:]
    audio_emb = torch.stack(hs, dim=2).squeeze(0)  # [T, 12, 768]
    return audio_emb.to(device=device, dtype=dtype)


@torch.no_grad()
def _audio_context_from_embeddings_range(
    audio_emb: torch.Tensor,
    start_idx: int,
    end_idx: int,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build [1, F, 5, 12, 768] from audio_emb [T, 12, 768] for [start_idx, end_idx).

    This mirrors the official implementation in SoulX-FlashHead (vectorized gather)
    and avoids Python-side loops.
    """

    if audio_emb.dim() != 3:
        raise RuntimeError(f"Expected audio_emb [T,12,768], got shape: {tuple(audio_emb.shape)}")

    T = int(audio_emb.shape[0])
    start_idx = int(start_idx)
    end_idx = int(end_idx)
    if end_idx <= start_idx:
        raise RuntimeError(f"Invalid audio index range: start={start_idx}, end={end_idx}")

    # Clamp to valid range.
    start_idx = max(0, start_idx)
    end_idx = min(T, end_idx)
    F = end_idx - start_idx
    if F <= 0:
        raise RuntimeError(f"Invalid clamped range: start={start_idx}, end={end_idx}, T={T}")

    # Official: indices = [-2,-1,0,1,2]
    dev = audio_emb.device
    base = torch.arange(start_idx, end_idx, device=dev, dtype=torch.long)  # [F]
    offsets = torch.arange(-2, 3, device=dev, dtype=torch.long)  # [5]
    center = base.unsqueeze(1) + offsets.unsqueeze(0)  # [F,5]

    # Match official clamping behavior (max to end_idx-1).
    center = torch.clamp(center, min=0, max=end_idx - 1)
    ctx = audio_emb[center]  # [F,5,12,768]
    ctx = ctx.unsqueeze(0).contiguous()  # [1,F,5,12,768]
    return ctx.to(device=device, dtype=dtype)


@torch.no_grad()
def _decode_to_cthw(pipeline: FlashHeadPipeline, latents_cthw: torch.Tensor) -> torch.Tensor:
    decoded = pipeline.vae.decode(latents_cthw)
    if isinstance(decoded, (tuple, list)):
        decoded = decoded[0]
    if decoded.dim() == 5:
        decoded = decoded[0]
    if decoded.dim() != 4:
        raise RuntimeError(f"Unexpected decoded video shape: {tuple(decoded.shape)}")

    # Expect [C,T,H,W]
    if decoded.shape[0] not in (1, 3, 4):
        raise RuntimeError(f"Cannot interpret decoded layout as [C,T,H,W]: {tuple(decoded.shape)}")
    return decoded


@torch.no_grad()
def _maybe_apply_color_correction(
    pipeline: FlashHeadPipeline,
    decoded_video_cthw: torch.Tensor,
) -> torch.Tensor:
    strength = float(getattr(pipeline, "color_correction_strength", 0.0))
    if strength <= 0.0:
        return decoded_video_cthw

    reference = getattr(pipeline, "original_color_reference", None)
    if reference is None:
        return decoded_video_cthw

    corrected = match_and_blend_colors_torch(
        decoded_video_cthw.unsqueeze(0),
        reference,
        strength,
    )
    return corrected[0]


@torch.no_grad()
def _encode_motion_prefix_from_decoded(
    pipeline: FlashHeadPipeline,
    decoded_video_cthw: torch.Tensor,
    motion_frames_num: int,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    # decoded_video_cthw: [C,T,H,W] in [-1,1]
    cond = decoded_video_cthw[:, -motion_frames_num:]
    cond = cond.unsqueeze(0).to(device=device, dtype=dtype)  # [1,C,T,H,W]
    return pipeline.vae.encode(cond).squeeze(0)


@torch.no_grad()
def _motion_prefix_from_latent_tail(latent_chunk_cthw: torch.Tensor, motion_frames_latent_num: int) -> torch.Tensor:
    if motion_frames_latent_num <= 0:
        raise RuntimeError(f"motion_frames_latent_num must be positive, got {motion_frames_latent_num}")
    total_latent_len = int(latent_chunk_cthw.shape[1])
    prefix_len = min(int(motion_frames_latent_num), total_latent_len)
    return latent_chunk_cthw[:, -prefix_len:].contiguous()


@torch.no_grad()
def _source_suffix_from_ref_latent(ref_latent: torch.Tensor, prefix_len: int, suffix_len: int) -> torch.Tensor:
    if suffix_len <= 0:
        return ref_latent[:, 0:0, :, :].contiguous()
    return ref_latent[:, prefix_len:prefix_len + suffix_len, :, :].contiguous()


@torch.no_grad()
def _bridge_sample_one_chunk(
    pipeline: FlashHeadPipeline,
    scheduler: ViBTScheduler,
    ref_latent: torch.Tensor,
    audio_context: torch.Tensor,
    guidance_scale: float,
    latent_motion_frames: torch.Tensor | None,
    clamp_latent_len: int,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Run one chunk with the same source construction used in training.

    Training defines the source suffix from the fixed initial reference latent, while
    keeping the history prefix hard-clamped for autoregressive continuity.
    """
    guidance_scale = float(guidance_scale)
    y = ref_latent.unsqueeze(0).to(device=device, dtype=dtype)

    if latent_motion_frames is None or clamp_latent_len <= 0:
        x_t = ref_latent.unsqueeze(0).to(device=device, dtype=dtype).clone()
    else:
        history = latent_motion_frames[:, :, :clamp_latent_len].to(device=device, dtype=dtype)
        total_latent_len = int(ref_latent.shape[1])
        suffix_len = total_latent_len - clamp_latent_len
        if suffix_len < 0:
            raise RuntimeError(f"Invalid latent lengths: total={total_latent_len}, clamp={clamp_latent_len}")

        if suffix_len > 0:
            source_suffix = _source_suffix_from_ref_latent(ref_latent, clamp_latent_len, suffix_len).unsqueeze(0).to(device=device, dtype=dtype)
            x_t = torch.cat([history, source_suffix], dim=2)
        else:
            x_t = history.clone()

    for t in scheduler.timesteps:
        if latent_motion_frames is not None and clamp_latent_len > 0:
            x_t[:, :, :clamp_latent_len] = latent_motion_frames[:, :, :clamp_latent_len]
        timestep = t.unsqueeze(0).to(device=device, dtype=x_t.dtype)
        v_pred_cond = pipeline.model(x=x_t, timestep=timestep, context=audio_context, y=y)
        if isinstance(v_pred_cond, (tuple, list)):
            v_pred_cond = v_pred_cond[0]

        if guidance_scale == 1.0:
            v_pred = v_pred_cond
        else:
            # Audio-driven classifier-free guidance (CFG)
            # v_final = v_uncond + s * (v_cond - v_uncond)
            uncond_audio_context = torch.zeros_like(audio_context)
            v_pred_uncond = pipeline.model(x=x_t, timestep=timestep, context=uncond_audio_context, y=y)
            if isinstance(v_pred_uncond, (tuple, list)):
                v_pred_uncond = v_pred_uncond[0]
            v_pred = v_pred_uncond + guidance_scale * (v_pred_cond - v_pred_uncond)
        x_t = scheduler.step(v_pred, t, x_t)[0]

        if latent_motion_frames is not None and clamp_latent_len > 0:
            x_t[:, :, :clamp_latent_len] = latent_motion_frames[:, :, :clamp_latent_len]

    return x_t[0]


def _mux_audio(tmp_mp4: str, audio_path: str, out_path: str):
    cmd = [
        "ffmpeg",
        "-i",
        tmp_mp4,
        "-i",
        audio_path,
        "-c:v",
        "copy",
        "-c:a",
        "mp3",
        "-shortest",
        out_path,
        "-y",
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


def main():
    parser = argparse.ArgumentParser(description="Streaming inference for ViBT-finetuned FlashHead (chunked AR).")
    parser.add_argument("--ckpt_dir", type=str, default="")
    parser.add_argument("--wav2vec_dir", type=str, default="")
    parser.add_argument("--lora_dir", type=str, default="")
    parser.add_argument(
        "--audio_proj",
        type=str,
        default="",
        help="Path to audio_proj_step_*.pt (or directory containing them). If omitted, picks latest in the LeapTalk model directory.",
    )

    parser.add_argument("--cond_image", type=str, default="")
    parser.add_argument("--audio_path", type=str, default="")
    parser.add_argument("--out", type=str, default="outputs/leaptalk.mp4")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument(
        "--compile",
        type=str,
        default="off",
        choices=["auto", "on", "off"],
        help="torch.compile base model: auto uses LoRA key inspection.",
    )

    parser.add_argument(
        "--usp",
        type=str,
        default="on",
        choices=["auto", "on", "off"],
        help="Enable USP multi-GPU (same forward split across ranks). auto enables when WORLD_SIZE>1.",
    )

    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--sample_rate", type=int, default=16000)

    # Stream params (match SoulX defaults)
    parser.add_argument("--frame_num", type=int, default=33)
    parser.add_argument("--motion_frames_latent_num", type=int, default=2)
    parser.add_argument("--cached_audio_duration", type=int, default=8)
    parser.add_argument("--audio_encode_mode", type=str, default="stream", choices=["stream", "once"])
    parser.add_argument("--max_chunks", type=int, default=0, help="0 means run all chunks.")
    parser.add_argument(
        "--history_update_mode",
        type=str,
        default="roundtrip",
        choices=["roundtrip", "latent"],
        help="How to update next-chunk history: SoulX-style VAE round-trip or direct latent tail reuse.",
    )

    # Bridge sampler params
    parser.add_argument("--num_inference_steps", type=int, default=1)
    parser.add_argument(
        "--step_list",
        type=int,
        nargs="+",
        default=None,
        help="Explicit discrete timesteps (e.g. --step_list 1000 750 500 250). Overrides --num_inference_steps.",
    )
    parser.add_argument("--noise_scale", type=float, default=1.0)
    parser.add_argument("--shift_gamma", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="Audio CFG guidance scale; 1.0 disables guidance.",
    )

    parser.add_argument("--use_face_crop", action="store_true")
    parser.add_argument(
        "--color_correction_strength",
        type=float,
        default=1.0,
        help="Blend decoded chunk colors toward the reference image; 0 disables correction and 1 applies full correction.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="pro"
    )

    # Lite mode uses the TAE backend for fast VAE encode/decode.
    parser.add_argument(
        "--lite",
        dest="lite",
        action="store_true",
        default=True,
        help="Use the TAE checkpoint shipped with the LeapTalk model directory for VAE encode/decode.",
    )
    parser.add_argument(
        "--no_lite",
        dest="lite",
        action="store_false",
        help="Use WanVAE instead of the Lite TAE backend.",
    )

    # Backward-compatible aliases for older evaluation scripts.
    parser.add_argument(
        "--use_tae",
        dest="lite",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no_tae",
        dest="lite",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--tae_path",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--tae_model_type",
        type=str,
        default="wan21",
        choices=["wan21", "wan22", "hy15"],
        help="TAEHV model_type (affects latent_channels/patch_size/clamp behavior).",
    )
    args = parser.parse_args()

    missing_paths = [
        name
        for name in ("ckpt_dir", "wav2vec_dir", "lora_dir", "cond_image", "audio_path")
        if not getattr(args, name)
    ]
    if missing_paths:
        raise SystemExit("Missing required path argument(s): " + ", ".join(f"--{name}" for name in missing_paths))

    resolved_lora_dir = _pick_latest_lora_dir(args.lora_dir)
    if not resolved_lora_dir:
        raise SystemExit(
            "LoRA weights not found; pass --lora_dir /path/to/leaptalk, "
            "/path/to/leaptalk/lora, or a directory containing lora_step_*."
        )

    tae_path = _resolve_lite_tae_path(
        lora_dir=args.lora_dir,
        resolved_lora_dir=resolved_lora_dir,
        ckpt_dir=args.ckpt_dir,
        explicit_tae_path=args.tae_path,
    ) if args.lite else None

    # USP init must happen before pipeline construction.
    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    use_usp = (args.usp == "on") or (args.usp == "auto" and world_size_env > 1)
    did_init_dist = False
    if use_usp and world_size_env > 1:
        if not torch.cuda.is_available():
            raise SystemExit("USP requested but CUDA is not available")

        if dist.is_initialized():
            logger.info("torch.distributed already initialized; reusing existing process group.")
        else:
            num_heads = _read_num_heads_from_infer_params(VIBT_ROOT)
            from flash_head.src.distributed.usp_device import get_device, get_parallel_degree

            ulysses_degree, ring_degree = get_parallel_degree(world_size_env, num_heads)
            logger.info(
                f"Initializing USP: WORLD_SIZE={world_size_env}, num_heads={num_heads}, "
                f"ulysses_degree={ulysses_degree}, ring_degree={ring_degree}"
            )
            args.device = str(get_device(ulysses_degree, ring_degree))
            did_init_dist = True
    else:
        use_usp = False

    rank = dist.get_rank() if dist.is_initialized() else 0
    is_rank0 = rank == 0

    # Under torchrun/USP, every rank executes the same script.
    # To avoid duplicated progress logs and stdout prints, keep rank0 verbose
    # and silence non-rank0 stdout while still surfacing errors on stderr.
    if not is_rank0:
        try:
            sys.stdout = open(os.devnull, "w")
        except Exception:
            pass
        try:
            logger.remove()
            logger.add(sys.stderr, level="ERROR")
        except Exception:
            pass

    if use_usp and args.history_update_mode != "roundtrip":
        logger.warning("USP enabled: forcing --history_update_mode=roundtrip to match official behavior.")
        args.history_update_mode = "roundtrip"

    if args.dtype == "bf16":
        amp_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.float32

    # Decide compile mode BEFORE creating pipeline (pipeline module reads globals)
    needs_compiled = _lora_checkpoint_needs_compiled_base(resolved_lora_dir)
    if args.compile == "auto":
        compile_model = bool(needs_compiled)
    elif args.compile == "on":
        compile_model = True
    else:
        compile_model = False

    # If checkpoint was saved from a compiled base (keys contain '._orig_mod.'), we must compile
    # BEFORE loading LoRA so module names match. Otherwise, compiling before loading may
    # introduce a prefix mismatch; compile after loading is safer.
    compile_before_lora = bool(needs_compiled)
    compile_after_lora = bool(compile_model and not needs_compiled)

    import flash_head.src.pipeline.flash_head_pipeline as fh_pipe_mod

    fh_pipe_mod.COMPILE_MODEL = bool(compile_before_lora)
    # TAEHV uses Python-level block loops; torch.compile often provides little benefit and can break.
    fh_pipe_mod.COMPILE_VAE = bool(not args.lite)

    logger.info(
        f"Init pipeline: device={args.device}, dtype={args.dtype}, compile_model={fh_pipe_mod.COMPILE_MODEL}, "
        f"usp={use_usp}, lite={args.lite}, rank={rank}"
    )
    pipeline = FlashHeadPipeline(
        checkpoint_dir=args.ckpt_dir,
        model_type=args.model_type,
        wav2vec_dir=args.wav2vec_dir,
        device=args.device,
        param_dtype=amp_dtype,
        use_usp=use_usp,
        use_tae=bool(args.lite),
        tae_path=tae_path,
        tae_model_type=args.tae_model_type,
    )
    logger.info(f"Pipeline initialized. Model dtype: {args.model_type}")
    # Load LoRA
    logger.info(f"Loading LoRA: {resolved_lora_dir}")
    pipeline.model = PeftModel.from_pretrained(pipeline.model, resolved_lora_dir, is_trainable=False)
    # Merge LoRA weights into base model to remove PEFT overhead during inference
    pipeline.model = pipeline.model.merge_and_unload()
    pipeline.model.eval().requires_grad_(False)

    if compile_after_lora:
        logger.info("Compiling model after loading LoRA (--compile on; LoRA keys do not require compiled base).")
        pipeline.model = torch.compile(pipeline.model)

    # Load audio_proj weights
    audio_proj_path = (
        _pick_latest_audio_proj(args.audio_proj)
        or _pick_latest_audio_proj(args.lora_dir)
        or _pick_latest_audio_proj(resolved_lora_dir)
        or _pick_latest_audio_proj(os.path.dirname(resolved_lora_dir))
    )
    if not audio_proj_path:
        raise SystemExit("audio_proj weights not found; pass --audio_proj /path/to/audio_proj_step_XXXX.pt")

    logger.info(f"Loading audio_proj: {audio_proj_path}")
    audio_proj_state = torch.load(audio_proj_path, map_location="cpu")
    inner = _get_inner_flashhead_model(pipeline.model)
    inner.audio_proj.load_state_dict(audio_proj_state, strict=True)

    # Stream params derived from VAE stride
    vae_stride_t = int(pipeline.config.vae_stride[0])
    sp = StreamParams(
        frame_num=int(args.frame_num),
        motion_frames_latent_num=int(args.motion_frames_latent_num),
        tgt_fps=int(args.fps),
        sample_rate=int(args.sample_rate),
        cached_audio_duration=int(args.cached_audio_duration),
    ).init_with_stride(vae_stride_t)

    frame_num = sp.frame_num
    motion_frames_num = sp.motion_frames_num
    slice_len = sp.slice_len

    if slice_len <= 0:
        raise SystemExit(f"Invalid streaming params: frame_num={frame_num}, motion_frames_num={motion_frames_num}")

    logger.info(f"Streaming: frame_num={frame_num}, motion_frames_num={motion_frames_num}, slice_len={slice_len}")
    logger.info(f"History update mode: {args.history_update_mode}")

    # Build X0 using pipeline preprocessing (resizes to H/W and repeats to frame_num)
    pipeline.prepare_params(
        cond_image_path_or_dir=args.cond_image,
        target_size=(args.height, args.width),
        frame_num=frame_num,
        motion_frames_num=0,
        sampling_steps=4,
        seed=args.seed,
        shift=args.shift_gamma,
        color_correction_strength=args.color_correction_strength,
        use_face_crop=args.use_face_crop,
    )
    X0 = pipeline.ref_img_latent.to(device=args.device, dtype=amp_dtype)  # [C_lat,T_lat,h,w]
    logger.info(f"Initialized static anchor latent with shape: {tuple(X0.shape)}")

    # Scheduler
    scheduler = ViBTScheduler(num_train_timesteps=1000)
    scheduler.timesteps = _build_infer_timesteps(
        step_list=args.step_list,
        num_inference_steps=args.num_inference_steps,
        shift_gamma=args.shift_gamma,
        device=args.device,
        num_timesteps=1000,
    )
    scheduler.num_inference_steps = int(scheduler.timesteps.numel())
    scheduler.set_parameters(noise_scale=args.noise_scale, shift_gamma=args.shift_gamma, seed=args.seed)

    # Init motion prefix (match pipeline.reset_person_name): only 1 latent frame at start
    latent_motion_frames = X0[:, :1].unsqueeze(0).clone()  # [1,C_lat,1,h,w]
    clamp_latent_len = int(latent_motion_frames.shape[2])
    # Audio preparation
    audio_all, _ = librosa.load(args.audio_path, sr=args.sample_rate, mono=True)
    human_speech_array_slice_len = slice_len * args.sample_rate // args.fps
    human_speech_array_frame_num = frame_num * args.sample_rate // args.fps

    # pad audio with silence to avoid re
    if args.audio_encode_mode == "once":
        remainder = (len(audio_all) - human_speech_array_frame_num) % human_speech_array_slice_len
    else:
        remainder = len(audio_all) % human_speech_array_slice_len
    if remainder > 0:
        pad_len = human_speech_array_slice_len - remainder
        audio_all = np.concatenate([audio_all, np.zeros(pad_len, dtype=audio_all.dtype)])

    # ensure audio is long enough for at least one window
    if len(audio_all) < human_speech_array_frame_num:
        audio_all = np.concatenate([audio_all, np.zeros(human_speech_array_frame_num - len(audio_all), dtype=audio_all.dtype)])

    tmp_mp4 = args.out.replace(".mp4", "_tmp.mp4")
    # Match official generate_video.py: cache per-chunk video tensors on CPU,
    # then write tmp mp4 once and mux audio at the end.
    generated_list: list[torch.Tensor] = []

    total_frames = 0
    total_time = 0.0
    total_chunks = 0

    try:
        if args.audio_encode_mode == "once":
            # Match official path: use pipeline.preprocess_audio + vectorized gather windowing.
            audio_emb_all = pipeline.preprocess_audio(audio_all, sr=args.sample_rate, fps=args.fps)
            if audio_emb_all is None:
                raise RuntimeError("Failed to extract audio embeddings")
            audio_emb_all = audio_emb_all.to(device=args.device, dtype=amp_dtype)
            total_frames = int(audio_emb_all.shape[0])
            audio_ctx_all = _audio_context_from_embeddings_range(
                audio_emb_all,
                start_idx=0,
                end_idx=total_frames,
                device=args.device,
                dtype=amp_dtype,
            )

            # Always run at least one chunk.
            num_chunks = 1 + max(0, (total_frames - frame_num) // slice_len)
            if args.max_chunks and args.max_chunks > 0:
                num_chunks = min(num_chunks, int(args.max_chunks))

            for chunk_idx in range(num_chunks):
                use_cuda_timing = args.device.startswith("cuda") and torch.cuda.is_available()
                if use_cuda_timing:
                    torch.cuda.synchronize()
                chunk_start = time.perf_counter()
                start = chunk_idx * slice_len
                end = start + frame_num
                if end > total_frames:
                    # For the last partial chunk, clamp; audio windows will pad at edges.
                    end = total_frames
                audio_ctx = audio_ctx_all[:, start:end].contiguous()

                if use_cuda_timing:
                    denoise_evt_s = torch.cuda.Event(enable_timing=True)
                    denoise_evt_e = torch.cuda.Event(enable_timing=True)
                    denoise_evt_s.record()
                x_final = _bridge_sample_one_chunk(
                    pipeline,
                    scheduler=scheduler,
                    ref_latent=X0,
                    audio_context=audio_ctx,
                    guidance_scale=args.guidance_scale,
                    latent_motion_frames=latent_motion_frames,
                    clamp_latent_len=clamp_latent_len,
                    device=args.device,
                    dtype=amp_dtype,
                )
                if use_cuda_timing:
                    denoise_evt_e.record()

                if use_cuda_timing:
                    decode_evt_s = torch.cuda.Event(enable_timing=True)
                    decode_evt_e = torch.cuda.Event(enable_timing=True)
                    decode_evt_s.record()
                decoded_cthw = _decode_to_cthw(pipeline, x_final)
                if use_cuda_timing:
                    decode_evt_e.record()

                if use_cuda_timing:
                    color_evt_s = torch.cuda.Event(enable_timing=True)
                    color_evt_e = torch.cuda.Event(enable_timing=True)
                    color_evt_s.record()
                decoded_cthw = _maybe_apply_color_correction(pipeline, decoded_cthw)
                if use_cuda_timing:
                    color_evt_e.record()

                if args.history_update_mode == "roundtrip":
                    if use_cuda_timing:
                        hist_evt_s = torch.cuda.Event(enable_timing=True)
                        hist_evt_e = torch.cuda.Event(enable_timing=True)
                        hist_evt_s.record()
                    latent_motion_frames = _encode_motion_prefix_from_decoded(
                        pipeline,
                        decoded_video_cthw=decoded_cthw,
                        motion_frames_num=motion_frames_num,
                        device=args.device,
                        dtype=amp_dtype,
                    ).unsqueeze(0)
                    if use_cuda_timing:
                        hist_evt_e.record()
                else:
                    if use_cuda_timing:
                        hist_evt_s = torch.cuda.Event(enable_timing=True)
                        hist_evt_e = torch.cuda.Event(enable_timing=True)
                        hist_evt_s.record()
                    latent_motion_frames = _motion_prefix_from_latent_tail(
                        x_final,
                        motion_frames_latent_num=sp.motion_frames_latent_num,
                    ).unsqueeze(0)
                    if use_cuda_timing:
                        hist_evt_e.record()
                clamp_latent_len = int(latent_motion_frames.shape[2])

                # Write frames: drop overlap on later chunks
                if chunk_idx != 0:
                    decoded_cthw = decoded_cthw[:, motion_frames_num:]

                # Match official run_pipeline(): build THWC frames in [0,255] on GPU.
                video_thwc = (
                    ((decoded_cthw + 1.0) / 2.0)
                    .permute(1, 2, 3, 0)
                    .clamp(0.0, 1.0)
                    .mul(255.0)
                    .contiguous()
                )

                # End chunk_time at GPU boundary (match official timing semantics).
                if use_cuda_timing:
                    torch.cuda.synchronize()
                chunk_time = time.perf_counter() - chunk_start

                # Post-processing / IO time (excluded from chunk_time)
                io_start = time.perf_counter()
                video_cpu = video_thwc.detach().cpu()
                if is_rank0:
                    generated_list.append(video_cpu)
                io_time = time.perf_counter() - io_start

                if chunk_idx >= 2:
                    total_frames += int(video_thwc.shape[0])
                    total_time += float(chunk_time)
                    total_chunks += 1

                if use_cuda_timing:
                    denoise_time = float(denoise_evt_s.elapsed_time(denoise_evt_e)) / 1000.0
                    decode_time = float(decode_evt_s.elapsed_time(decode_evt_e)) / 1000.0
                    color_time = float(color_evt_s.elapsed_time(color_evt_e)) / 1000.0
                    hist_time = float(hist_evt_s.elapsed_time(hist_evt_e)) / 1000.0
                else:
                    denoise_time = float("nan")
                    decode_time = float("nan")
                    color_time = float("nan")
                    hist_time = float("nan")

                if is_rank0:
                    logger.info(
                        f"Chunk {chunk_idx + 1}/{num_chunks} timings | "
                        f"denoise={denoise_time:.3f}s | decode={decode_time:.3f}s | color={color_time:.3f}s | "
                        f"hist={hist_time:.3f}s | io={io_time:.3f}s"
                    )
                logger.info(f"Chunk {chunk_idx + 1}/{num_chunks} done")

        else:
            # stream: cached ring buffer, recompute embeddings for cached audio each chunk
            from collections import deque

            cached_len = args.sample_rate * sp.cached_audio_duration
            audio_end_idx = sp.cached_audio_duration * args.fps
            audio_start_idx = audio_end_idx - frame_num

            if human_speech_array_slice_len <= 0:
                raise SystemExit("Invalid audio slice length")

            # Ensure divisible for reshape
            remainder = len(audio_all) % human_speech_array_slice_len
            if remainder != 0:
                audio_all = np.concatenate([audio_all, np.zeros(human_speech_array_slice_len - remainder, dtype=audio_all.dtype)])

            slices = audio_all.reshape(-1, human_speech_array_slice_len)
            if slices.shape[0] == 0:
                raise SystemExit("audio too short after padding")

            audio_dq = deque([0.0] * cached_len, maxlen=cached_len)

            num_chunks = int(slices.shape[0])
            if args.max_chunks and args.max_chunks > 0:
                num_chunks = min(num_chunks, int(args.max_chunks))

            for chunk_idx in range(num_chunks):
                use_cuda_timing = args.device.startswith("cuda") and torch.cuda.is_available()
                if use_cuda_timing:
                    torch.cuda.synchronize()
                chunk_start = time.perf_counter()
                audio_dq.extend(slices[chunk_idx].tolist())
                audio_cache = np.array(audio_dq, dtype=np.float32)
                audio_emb_cache = pipeline.preprocess_audio(audio_cache, sr=args.sample_rate, fps=args.fps)
                if audio_emb_cache is None:
                    raise RuntimeError("Failed to extract audio embeddings")
                audio_emb_cache = audio_emb_cache.to(device=args.device, dtype=amp_dtype)
                audio_ctx = _audio_context_from_embeddings_range(
                    audio_emb_cache,
                    start_idx=audio_start_idx,
                    end_idx=audio_end_idx,
                    device=args.device,
                    dtype=amp_dtype,
                )

                if use_cuda_timing:
                    denoise_evt_s = torch.cuda.Event(enable_timing=True)
                    denoise_evt_e = torch.cuda.Event(enable_timing=True)
                    denoise_evt_s.record()
                x_final = _bridge_sample_one_chunk(
                    pipeline,
                    scheduler=scheduler,
                    ref_latent=X0,
                    audio_context=audio_ctx,
                    guidance_scale=args.guidance_scale,
                    latent_motion_frames=latent_motion_frames,
                    clamp_latent_len=clamp_latent_len,
                    device=args.device,
                    dtype=amp_dtype,
                )
                if use_cuda_timing:
                    denoise_evt_e.record()

                if use_cuda_timing:
                    decode_evt_s = torch.cuda.Event(enable_timing=True)
                    decode_evt_e = torch.cuda.Event(enable_timing=True)
                    decode_evt_s.record()
                decoded_cthw = _decode_to_cthw(pipeline, x_final)
                if use_cuda_timing:
                    decode_evt_e.record()

                if use_cuda_timing:
                    color_evt_s = torch.cuda.Event(enable_timing=True)
                    color_evt_e = torch.cuda.Event(enable_timing=True)
                    color_evt_s.record()
                decoded_cthw = _maybe_apply_color_correction(pipeline, decoded_cthw)
                if use_cuda_timing:
                    color_evt_e.record()

                if args.history_update_mode == "roundtrip":
                    if use_cuda_timing:
                        hist_evt_s = torch.cuda.Event(enable_timing=True)
                        hist_evt_e = torch.cuda.Event(enable_timing=True)
                        hist_evt_s.record()
                    latent_motion_frames = _encode_motion_prefix_from_decoded(
                        pipeline,
                        decoded_video_cthw=decoded_cthw,
                        motion_frames_num=motion_frames_num,
                        device=args.device,
                        dtype=amp_dtype,
                    ).unsqueeze(0)
                    if use_cuda_timing:
                        hist_evt_e.record()
                else:
                    if use_cuda_timing:
                        hist_evt_s = torch.cuda.Event(enable_timing=True)
                        hist_evt_e = torch.cuda.Event(enable_timing=True)
                        hist_evt_s.record()
                    latent_motion_frames = _motion_prefix_from_latent_tail(
                        x_final,
                        motion_frames_latent_num=sp.motion_frames_latent_num,
                    ).unsqueeze(0)
                    if use_cuda_timing:
                        hist_evt_e.record()
                clamp_latent_len = int(latent_motion_frames.shape[2])

                # In stream mode, match generate_video.py: always drop overlap
                decoded_cthw = decoded_cthw[:, motion_frames_num:]

                # Match official run_pipeline(): build THWC frames in [0,255] on GPU.
                video_thwc = (
                    ((decoded_cthw + 1.0) / 2.0)
                    .permute(1, 2, 3, 0)
                    .clamp(0.0, 1.0)
                    .mul(255.0)
                    .contiguous()
                )

                # End chunk_time at GPU boundary (match official timing semantics).
                if use_cuda_timing:
                    torch.cuda.synchronize()
                chunk_time = time.perf_counter() - chunk_start

                # Post-processing / IO time (excluded from chunk_time)
                io_start = time.perf_counter()
                video_cpu = video_thwc.detach().cpu()
                if is_rank0:
                    generated_list.append(video_cpu)
                io_time = time.perf_counter() - io_start

                chunk_frames = int(video_thwc.shape[0])
                chunk_fps = chunk_frames / chunk_time

                if chunk_idx >= 2:
                    total_frames += chunk_frames
                    total_time += chunk_time
                    total_chunks += 1

                if use_cuda_timing:
                    denoise_time = float(denoise_evt_s.elapsed_time(denoise_evt_e)) / 1000.0
                    decode_time = float(decode_evt_s.elapsed_time(decode_evt_e)) / 1000.0
                    color_time = float(color_evt_s.elapsed_time(color_evt_e)) / 1000.0
                    hist_time = float(hist_evt_s.elapsed_time(hist_evt_e)) / 1000.0
                else:
                    denoise_time = float("nan")
                    decode_time = float("nan")
                    color_time = float("nan")
                    hist_time = float("nan")

                logger.info(
                    f"Chunk {chunk_idx+1}/{num_chunks} | "
                    f"time={chunk_time:.3f}s | "
                    f"frames={chunk_frames} | "
                    f"FPS={chunk_fps:.2f} | "
                    f"denoise={denoise_time:.3f}s | decode={decode_time:.3f}s | color={color_time:.3f}s | "
                    f"hist={hist_time:.3f}s | io={io_time:.3f}s"
                )
                logger.info(f"Chunk {chunk_idx + 1}/{num_chunks} done")

    finally:
        pass

    if dist.is_initialized():
        dist.barrier()

    if is_rank0:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

        # Save video once (official-style) then mux audio.
        with _open_video_writer(tmp_mp4, fps=args.fps) as writer:
            for video_cpu in generated_list:
                # Match official save_video(): numpy().astype(np.uint8) at write time.
                # NumPy does not support bfloat16; cast at write time (outside chunk_time).
                if video_cpu.dtype == torch.bfloat16:
                    video_cpu = video_cpu.to(torch.float16)
                frames_np = video_cpu.numpy().astype(np.uint8)
                for fr in frames_np:
                    writer.append_data(fr)
        _mux_audio(tmp_mp4, args.audio_path, args.out)
        try:
            os.remove(tmp_mp4)
        except OSError:
            pass

        if total_time > 0:
            avg_fps = total_frames / total_time
            avg_chunk_time = total_time / total_chunks if total_chunks > 0 else float("nan")
            logger.info(
                f"Average generation FPS (excl first 2 chunks): {avg_fps:.2f} "
                f"(total_frames={total_frames}, total_time={total_time:.2f}s)"
            )
            logger.info(f"Average chunk time (excl first 2 chunks): {avg_chunk_time:.3f}s")
        logger.info(f"Saved: {args.out}")

    if dist.is_initialized():
        dist.barrier()
    if did_init_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
