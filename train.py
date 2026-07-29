
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("..")
import argparse
import gc
import glob
import json
import datetime
import random
import re
import subprocess
from dataclasses import dataclass

import imageio
import librosa
import lpips
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from peft import LoraConfig, PeftModel, get_peft_model

import flash_head.src.pipeline.flash_head_pipeline as flash_head_pipeline_module
from flash_head.src.modules.flash_head_model import WanModelAudioProject
from flash_head.src.pipeline.flash_head_pipeline import FlashHeadPipeline, timestep_transform
from train.spatial_mask_utils import FaceLipMaskExtractor, MEDIAPIPE_AVAILABLE
from vibt.scheduler import ViBTScheduler


# Training should not wrap the model/VAE in torch.compile by default.
flash_head_pipeline_module.COMPILE_MODEL = False
flash_head_pipeline_module.COMPILE_VAE = False


def _unwrap_wan_model(m: torch.nn.Module) -> torch.nn.Module:
    if hasattr(m, "base_model"):
        m = m.base_model
    if hasattr(m, "model"):
        m = m.model
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    return m


def _parse_step_from_path(path: str | None) -> int | None:
    if not path:
        return None
    name = os.path.basename(os.path.normpath(path))
    m = re.search(r"(?:^|_)(?:step|global_step)[_-]?(\d+)$", name)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)", name)
    if m:
        return int(m.group(1))
    return None


def _auto_find_audio_proj_ckpt(resume_lora_dir: str) -> str | None:
    step = _parse_step_from_path(resume_lora_dir)
    candidates: list[str] = []
    search_dirs = [resume_lora_dir, os.path.dirname(os.path.normpath(resume_lora_dir))]
    for d in search_dirs:
        if not d:
            continue
        if step is not None:
            exact = os.path.join(d, f"audio_proj_step_{step}.pt")
            if os.path.exists(exact):
                return exact
        candidates.extend(glob.glob(os.path.join(d, "audio_proj_step_*.pt")))

    if not candidates:
        return None

    candidates.sort(key=lambda p: _parse_step_from_path(p) or -1)
    return candidates[-1]


def _load_audio_proj_if_present(model: torch.nn.Module, ckpt_path: str | None, prefix: str = "") -> None:
    if not ckpt_path:
        return
    try:
        state_dict = torch.load(ckpt_path, map_location="cpu")
        wan_model = _unwrap_wan_model(model)
        missing, unexpected = wan_model.audio_proj.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[WARN] {prefix}audio_proj missing={len(missing)} unexpected={len(unexpected)}")
        print(f"[INFO] Loaded {prefix}audio_proj from: {ckpt_path}")
    except Exception as exc:
        print(f"[WARN] Failed to load {prefix}audio_proj from {ckpt_path}: {exc}")


def _enable_audio_proj_training(model: torch.nn.Module) -> None:
    wan_model = _unwrap_wan_model(model)
    if hasattr(wan_model, "audio_proj"):
        for param in wan_model.audio_proj.parameters():
            param.requires_grad = True


def _set_audio_proj_trainable(model: torch.nn.Module, trainable: bool, prefix: str = "") -> None:
    wan_model = _unwrap_wan_model(model)
    if not hasattr(wan_model, "audio_proj"):
        return
    for param in wan_model.audio_proj.parameters():
        param.requires_grad = trainable
    state = "trainable" if trainable else "frozen"
    print(f"[INFO] {prefix}audio_proj set to {state}")


def _masked_mse(prediction: torch.Tensor, target: torch.Tensor, prefix_len: int) -> torch.Tensor:
    prediction_suffix = prediction[:, :, prefix_len:, :, :].float()
    target_suffix = target[:, :, prefix_len:, :, :].float()
    if prediction_suffix.numel() == 0:
        return prediction.new_zeros(())
    return F.mse_loss(prediction_suffix, target_suffix, reduction="mean")


def _normalize_video_to_thwc(video: torch.Tensor) -> torch.Tensor:
    if isinstance(video, (tuple, list)):
        video = video[0]
    if video.dim() == 5:
        video = video[0]
    if video.dim() == 4:
        if video.shape[0] in (1, 3, 4):
            return video.permute(1, 2, 3, 0)
        if video.shape[1] in (1, 3, 4):
            return video.permute(0, 2, 3, 1)
    if video.dim() == 3:
        return video.unsqueeze(-1)
    raise RuntimeError(f"Unexpected decoded video shape: {tuple(video.shape)}")


def save_video_with_audio(frames_thwc: torch.Tensor, video_path: str, audio_path: str | None, fps: int) -> None:
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    temp_path = video_path.replace('.mp4', '_tmp.mp4')
    frames_np = ((frames_thwc.detach().float().cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    imageio.mimsave(temp_path, frames_np, fps=fps)

    if audio_path and os.path.exists(audio_path):
        subprocess.run(
            ['ffmpeg', '-i', temp_path, '-i', audio_path, '-c:v', 'copy', '-c:a', 'mp3', '-shortest', video_path, '-y'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        try:
            os.remove(temp_path)
        except OSError:
            pass
    else:
        os.replace(temp_path, video_path)


@dataclass
class ChunkSpec:
    chunk_idx: int
    raw_start: int
    raw_end: int
    lat_start: int
    lat_end: int
    prefix_len: int


class AudioVideoSequenceDataset(Dataset):
    def __init__(
        self,
        video_dir: str,
        audio_dir: str,
        vae,
        audio_encoder,
        wav2vec_extractor,
        max_frames: int = 125,
        ref_frame_num: int = 33,
        dtype: torch.dtype = torch.bfloat16,
        use_spatial_weighting: bool = False,
        mask_cache_dir: str | None = None,
        mask_dilate_lip: int = 5,
        mask_dilate_face: int = 10,
    ):
        self.video_dir = video_dir
        self.audio_dir = audio_dir
        self.vae = vae
        self.audio_encoder = audio_encoder
        self.wav2vec_extractor = wav2vec_extractor
        self.max_frames = max_frames
        self.ref_frame_num = ref_frame_num
        self.dtype = dtype
        self.use_spatial_weighting = use_spatial_weighting
        self.mask_cache_dir = mask_cache_dir
        self.mask_dilate_lip = mask_dilate_lip
        self.mask_dilate_face = mask_dilate_face
        self.mask_extractor: FaceLipMaskExtractor | None = None
        self._mask_warning_printed = False

        video_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.mp4')])
        self.pairs = [(v, v.replace('.mp4', '.wav')) for v in video_files if os.path.exists(os.path.join(audio_dir, v.replace('.mp4', '.wav')))]

        if self.mask_cache_dir:
            os.makedirs(self.mask_cache_dir, exist_ok=True)

        if self.use_spatial_weighting and not MEDIAPIPE_AVAILABLE and not self._mask_warning_printed:
            print("[WARN] Spatial weighting requested but MediaPipe is unavailable. Falling back to zero masks.")
            self._mask_warning_printed = True

    @staticmethod
    def _zero_masks(frame_count: int, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
        zero = torch.zeros((frame_count, height, width), dtype=torch.float32)
        return zero, zero.clone()

    def _mask_cache_path(self, video_file: str, frame_count: int, height: int, width: int) -> str | None:
        if not self.mask_cache_dir:
            return None
        stem = os.path.splitext(video_file)[0]
        cache_name = f"{stem}_f{frame_count}_h{height}_w{width}.npz"
        return os.path.join(self.mask_cache_dir, cache_name)

    def _get_mask_extractor(self) -> FaceLipMaskExtractor | None:
        if not self.use_spatial_weighting or not MEDIAPIPE_AVAILABLE:
            return None
        if self.mask_extractor is None:
            self.mask_extractor = FaceLipMaskExtractor(device="cpu")
        return self.mask_extractor

    def _extract_spatial_masks(self, frames: np.ndarray, video_file: str) -> tuple[torch.Tensor, torch.Tensor]:
        frame_count, height, width = frames.shape[:3]
        if not self.use_spatial_weighting:
            return self._zero_masks(frame_count, height, width)

        cache_path = self._mask_cache_path(video_file, frame_count, height, width)
        if cache_path and os.path.exists(cache_path):
            try:
                cached = np.load(cache_path)
                face_mask = torch.from_numpy(cached["face_mask"]).float()
                lip_mask = torch.from_numpy(cached["lip_mask"]).float()
                if tuple(face_mask.shape) == (frame_count, height, width) and tuple(lip_mask.shape) == (frame_count, height, width):
                    return face_mask, lip_mask
            except Exception as exc:
                print(f"[WARN] Failed to read mask cache {cache_path}: {exc}")

        extractor = self._get_mask_extractor()
        if extractor is None:
            return self._zero_masks(frame_count, height, width)

        try:
            face_mask, lip_mask = extractor.extract_masks(
                frames,
                dilate_lip=self.mask_dilate_lip,
                dilate_face=self.mask_dilate_face,
            )
            face_mask = face_mask.cpu().float()
            lip_mask = lip_mask.cpu().float()
        except Exception as exc:
            if not self._mask_warning_printed:
                print(f"[WARN] MediaPipe mask extraction failed. Falling back to zero masks. Error: {exc}")
                self._mask_warning_printed = True
            return self._zero_masks(frame_count, height, width)

        if cache_path:
            try:
                np.savez_compressed(
                    cache_path,
                    face_mask=face_mask.numpy().astype(np.uint8),
                    lip_mask=lip_mask.numpy().astype(np.uint8),
                )
            except Exception as exc:
                print(f"[WARN] Failed to write mask cache {cache_path}: {exc}")
        return face_mask, lip_mask

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        import cv2

        video_file, audio_file = self.pairs[idx]
        video_path = os.path.join(self.video_dir, video_file)
        audio_path = os.path.join(self.audio_dir, audio_file)

        cap = cv2.VideoCapture(video_path)
        frames = []
        while len(frames) < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        if not frames:
            raise RuntimeError(f"No frames found in {video_path}")

        if len(frames) < self.max_frames:
            frames = frames + [frames[-1]] * (self.max_frames - len(frames))

        frames = np.stack(frames[:self.max_frames])
        h, w = frames.shape[1], frames.shape[2]
        h2, w2 = (h // 16) * 16, (w // 16) * 16
        if h2 > 0 and w2 > 0:
            y0, x0 = (h - h2) // 2, (w - w2) // 2
            frames = frames[:, y0:y0 + h2, x0:x0 + w2, :]

        face_mask, lip_mask = self._extract_spatial_masks(frames, video_file)

        frames_t = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 127.5 - 1.0
        device = getattr(self.vae, "device", "cuda")

        with torch.no_grad():
            frames_gpu = frames_t.unsqueeze(0).to(device, dtype=self.dtype)
            video_latent = self.vae.encode(frames_gpu).squeeze(0).cpu()

            ref_src = frames_gpu[:, :, :1, :, :].repeat(1, 1, self.ref_frame_num, 1, 1)
            ref_latent = self.vae.encode(ref_src).squeeze(0).cpu()

        audio, _ = librosa.load(audio_path, sr=16000, mono=True)
        inputs = self.wav2vec_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            audio_values = inputs.input_values.to(device)
            audio_out = self.audio_encoder(audio_values, seq_len=int(self.max_frames), output_hidden_states=True)
            audio_emb = torch.stack(audio_out.hidden_states[-12:], dim=2).squeeze(0).cpu()

        seq_len = audio_emb.shape[0]
        audio_windows = []
        for frame_idx in range(self.max_frames):
            window = []
            for offset in range(-2, 3):
                src_idx = max(0, min(seq_len - 1, frame_idx + offset))
                window.append(audio_emb[src_idx])
            audio_windows.append(torch.stack(window))
        audio_context = torch.stack(audio_windows)

        return {
            "video_latent": video_latent,
            "ref_latent": ref_latent,
            "audio_context": audio_context,
            "face_mask": face_mask,
            "lip_mask": lip_mask,
            "audio_path": audio_path,
        }


class FlashHeadARBridgeKDTrainer:
    def __init__(
        self,
        ckpt_dir: str,
        wav2vec_dir: str,
        video_dir: str,
        audio_dir: str,
        device: str = "cuda",
        lora_rank: int = 64,
        max_frames: int = 125,
        student_steps: int = 4,
        teacher_steps: int = 4,
        bridge_loss_weight: float = 1.0,
        gt_loss_weight: float = 0.0,
        kd_loss_weight: float = 0.0,
        rollout_chunks: int = 2,
        resume_lora_dir: str | None = None,
        resume_audio_proj: str | None = None,
        freeze_audio_proj: bool = False,
        teacher_model_dir: str | None = None,
        teacher_lora_dir: str | None = None,
        teacher_audio_proj: str | None = None,
        teacher_audio_guidance_scale: float = 1.5,
        use_bridge_step_list: bool = False,
        dmd_step_list: list[int] | None = None,
        vis_step_list: list[int] | None = None,
        bridge_noise_scale: float = 1.0,
        use_alpha_stabilization: bool = True,
        use_gt_target: bool = False,
        use_dmd: bool = False,
        dmd_loss_weight: float = 0.0,
        critic_loss_weight: float = 1.0,
        critic_lr: float = 1e-4,
        dfake_gen_update_ratio: int = 5,
        fake_score_lora_rank: int = 64,
        resume_fake_score_lora_dir: str | None = None,
        resume_fake_score_audio_proj: str | None = None,
        use_spatial_weighting: bool = False,
        lambda_face: float = 2.0,
        lambda_lip: float = 5.0,
        gamma_temporal: float = 1.0,
        mask_cache_dir: str | None = None,
        mask_dilate_lip: int = 5,
        mask_dilate_face: int = 10,
        lpips_loss_weight: float = 0.5,
        lpips_net: str = "vgg",
        lpips_model_path: str ="/cache/vgg.pth",
        lpips_crop_size: int = 256,
    ):
        self.device = device
        self.amp_dtype = torch.bfloat16
        self.scaler = torch.amp.GradScaler('cuda', enabled=False)
        self.writer: SummaryWriter | None = None

        self.bridge_loss_weight = bridge_loss_weight
        self.gt_loss_weight = gt_loss_weight
        self.kd_loss_weight = kd_loss_weight
        self.rollout_chunks = rollout_chunks
        self.student_steps = student_steps
        self.teacher_steps = teacher_steps
        self.bridge_noise_scale = bridge_noise_scale
        self.use_alpha_stabilization = use_alpha_stabilization
        self.use_gt_target = use_gt_target
        self.use_dmd = use_dmd
        self.teacher_audio_guidance_scale = float(teacher_audio_guidance_scale)
        self.dmd_loss_weight = dmd_loss_weight
        self.critic_loss_weight = critic_loss_weight
        self.critic_lr = critic_lr
        self.dfake_gen_update_ratio = max(1, int(dfake_gen_update_ratio))
        self.use_spatial_weighting = use_spatial_weighting
        self.lambda_face = lambda_face
        self.lambda_lip = lambda_lip
        self.gamma_temporal = gamma_temporal
        self.lpips_loss_weight = lpips_loss_weight
        self.lpips_net = lpips_net
        self.lpips_model_path = lpips_model_path
        self.lpips_crop_size = lpips_crop_size
        self.freeze_audio_proj = bool(freeze_audio_proj)
        self.num_timesteps = 1000
        self.sample_shift = 5.0
        self.dmd_step_list = [int(x) for x in (dmd_step_list or [])]
        self.vis_step_list = [int(x) for x in (vis_step_list or [])]
        # DMD timestep sampling should match official behavior: sample from a
        # fixed denoising-step list (not continuous random timesteps).
        self.dmd_train_timesteps = self._build_dmd_train_timesteps()
        self.use_bridge_step_list = bool(use_bridge_step_list)
        self.writer = None
        self.lpips_loss: lpips.LPIPS | None = None

        if self.lpips_loss_weight > 0.0:
            self.lpips_loss = lpips.LPIPS(
                net=self.lpips_net,
                model_path=self.lpips_model_path,
                verbose=False,
            )
            self.lpips_loss.eval().requires_grad_(False)
            self.lpips_loss.to(device=self.device)

        self.student_pipeline = FlashHeadPipeline(
            checkpoint_dir=ckpt_dir,
            model_type="pro",
            wav2vec_dir=wav2vec_dir,
            device=device,
            param_dtype=self.amp_dtype,
        )

        student_model = self.student_pipeline.model
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            target_modules=["q", "k", "v", "o"],
            lora_dropout=0.0,
        )
        if resume_lora_dir:
            student_model = PeftModel.from_pretrained(student_model, resume_lora_dir, is_trainable=True)
        else:
            student_model = get_peft_model(student_model, lora_config)
        self.student_pipeline.model = student_model
        self.student_model = self.student_pipeline.model

        _set_audio_proj_trainable(self.student_model, trainable=not self.freeze_audio_proj)
        if resume_lora_dir and resume_audio_proj is None:
            resume_audio_proj = _auto_find_audio_proj_ckpt(resume_lora_dir)
        _load_audio_proj_if_present(self.student_model, resume_audio_proj)

        teacher_model_dir = teacher_model_dir or os.path.join(ckpt_dir, "Model_Pro")
        self.teacher_model = WanModelAudioProject.from_pretrained(teacher_model_dir)
        self.teacher_model.to(device="cpu", dtype=self.amp_dtype)
        self.teacher_model.eval().requires_grad_(False)
        if teacher_lora_dir:
            self.teacher_model = PeftModel.from_pretrained(self.teacher_model, teacher_lora_dir, is_trainable=False)
            self.teacher_model.eval().requires_grad_(False)
        _load_audio_proj_if_present(self.teacher_model, teacher_audio_proj, prefix="teacher ")

        self.fake_score = None
        if self.use_dmd:
            fake_score_base = WanModelAudioProject.from_pretrained(teacher_model_dir)
            if resume_fake_score_lora_dir:
                self.fake_score = PeftModel.from_pretrained(fake_score_base, resume_fake_score_lora_dir, is_trainable=True)
            elif teacher_lora_dir:
                self.fake_score = PeftModel.from_pretrained(fake_score_base, teacher_lora_dir, is_trainable=True)
            else:
                fake_score_lora_config = LoraConfig(
                    r=fake_score_lora_rank,
                    lora_alpha=fake_score_lora_rank,
                    target_modules=["q", "k", "v", "o"],
                    lora_dropout=0.0,
                )
                self.fake_score = get_peft_model(fake_score_base, fake_score_lora_config)

            _set_audio_proj_trainable(self.fake_score, trainable=not self.freeze_audio_proj, prefix="fake_score ")
            if resume_fake_score_lora_dir and resume_fake_score_audio_proj is None:
                resume_fake_score_audio_proj = _auto_find_audio_proj_ckpt(resume_fake_score_lora_dir)
            _load_audio_proj_if_present(
                self.fake_score,
                resume_fake_score_audio_proj or teacher_audio_proj,
                prefix="fake_score ",
            )
            self.fake_score.to(device=device, dtype=self.amp_dtype)
            self.fake_score.train()

        self.config = self.student_pipeline.model.config
        self.vae_stride_t = int(self.config.vae_stride[0])
        self.raw_chunk_len = 33
        self.raw_history_len = 5
        self.raw_new_len = self.raw_chunk_len - self.raw_history_len
        self.lat_chunk_len = (self.raw_chunk_len - 1) // self.vae_stride_t + 1
        self.lat_history_len = 2
        self.init_history_len = 1
        self.lat_new_len = self.lat_chunk_len - self.lat_history_len
        self.dataset = AudioVideoSequenceDataset(
            video_dir=video_dir,
            audio_dir=audio_dir,
            vae=self.student_pipeline.vae,
            audio_encoder=self.student_pipeline.audio_encoder,
            wav2vec_extractor=self.student_pipeline.wav2vec_feature_extractor,
            max_frames=max_frames,
            ref_frame_num=self.raw_chunk_len,
            dtype=self.amp_dtype,
            use_spatial_weighting=use_spatial_weighting,
            mask_cache_dir=mask_cache_dir,
            mask_dilate_lip=mask_dilate_lip,
            mask_dilate_face=mask_dilate_face,
        )

    def _teacher_flow_with_audio_cfg(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        audio_chunk: torch.Tensor,
        ref_latent: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher forward with audio classifier-free guidance.

        v_final = v_uncond + s * (v_cond - v_uncond)
        where uncond uses a zero audio context.
        """

        s = float(self.teacher_audio_guidance_scale)
        v_cond = self.teacher_model(x=x, timestep=timestep, context=audio_chunk, y=ref_latent)
        if isinstance(v_cond, (tuple, list)):
            v_cond = v_cond[0]
        if s == 1.0:
            return v_cond

        uncond_audio = torch.zeros_like(audio_chunk)
        v_uncond = self.teacher_model(x=x, timestep=timestep, context=uncond_audio, y=ref_latent)
        if isinstance(v_uncond, (tuple, list)):
            v_uncond = v_uncond[0]
        return v_uncond + s * (v_cond - v_uncond)

    def _sample_dmd_timestep(self, batch_size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        choices = self.dmd_train_timesteps.to(device=device)
        if choices.numel() == 0:
            raise RuntimeError("Empty DMD timestep list")
        idx = torch.randint(0, int(choices.numel()), (int(batch_size),), device=device, dtype=torch.long)
        return choices[idx].to(dtype=dtype)

    def _build_dmd_train_timesteps(self) -> torch.Tensor:
        """Return 1D tensor of fixed denoising timesteps for DMD.

        If `self.dmd_step_list` is provided, use it (excluding any 0 step).
        Otherwise match the teacher denoising schedule (excluding the final 0 step).
        Values are already warped by `timestep_transform`.
        """

        if self.dmd_step_list:
            values: list[torch.Tensor] = []
            seen: set[int] = set()
            for step in self.dmd_step_list:
                step_int = int(step)
                if step_int <= 0:
                    continue
                if step_int > self.num_timesteps:
                    continue
                if step_int in seen:
                    continue
                seen.add(step_int)
                t = torch.tensor([float(step_int)], device=self.device)
                t = timestep_transform(t, shift=self.sample_shift, num_timesteps=self.num_timesteps)
                values.append(t.reshape(-1)[0].to(dtype=torch.float32))
            if not values:
                return torch.empty((0,), device=self.device, dtype=torch.float32)
            return torch.stack(values, dim=0)

        ts = self._build_teacher_timesteps()
        if len(ts) <= 1:
            return torch.empty((0,), device=self.device, dtype=torch.float32)

        # Exclude the terminal 0 step.
        values = [t.reshape(-1)[0].to(device=self.device, dtype=torch.float32) for t in ts[:-1]]
        return torch.stack(values, dim=0)

    # Bridge-loss timestep sharing: when enabled, reuse `self.dmd_train_timesteps`.

    def _add_teacher_noise(self, clean_chunk: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        sigma = (timestep / self.num_timesteps).view(-1, 1, 1, 1, 1)
        return (1.0 - sigma) * clean_chunk + sigma * noise

    def _phi_tau_to_t(self, tau: torch.Tensor) -> torch.Tensor:
        """Time conversion t = Phi(tau).

        Phi(tau) = 1 / (1 + sqrt( 1 - tau / ( tau)))
        where tau in (0, 1). This maps student-time (tau) to teacher-time (t).
        """

        tau = tau.clamp(1e-4, 1.0 - 1e-4)
        ratio = (1.0 - tau) / tau
        return 1.0 / (1.0 + torch.sqrt(ratio))

    def _map_student_timestep_to_teacher_timestep(self, student_timestep: torch.Tensor) -> torch.Tensor:
        """Map student timestep (0..T) to teacher timestep (0..T) via Phi."""

        tau = (student_timestep.float() / float(self.num_timesteps)).clamp(1e-4, 1.0 - 1e-4)
        t = self._phi_tau_to_t(tau)
        return (t * float(self.num_timesteps)).to(device=student_timestep.device, dtype=student_timestep.dtype)

    def _build_brownian_bridge_xt(
        self,
        history: torch.Tensor,
        target_chunk: torch.Tensor,
        ref_latent: torch.Tensor,
        timestep: torch.Tensor,
        prefix_len: int,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Construct x_tau via Brownian bridge between ref (source) and target.

        This matches the bridge construction used elsewhere in this file:
        x_tau = tau * x_src + (1 - tau) * x_tgt + sqrt(tau * (1 - tau)) * eps
        where tau = timestep / num_timesteps.

        Only the suffix (after prefix_len) is bridged; the prefix is kept as history.
        """

        target_suffix = target_chunk[:, :, prefix_len:, :, :]
        if target_suffix.numel() == 0:
            return target_chunk

        tau = (timestep.float() / float(self.num_timesteps)).clamp(1e-4, 1.0 - 1e-4)
        tau_view = tau.view(-1, 1, 1, 1, 1)

        source_suffix = self._source_suffix_from_ref_latent(ref_latent, prefix_len, target_suffix.shape[2])
        eps = noise if noise is not None else torch.randn_like(target_suffix)
        suffix_tau = tau_view * source_suffix + (1.0 - tau_view) * target_suffix + (tau_view * (1.0 - tau_view)).sqrt() * eps

        return torch.cat([history, suffix_tau.to(dtype=history.dtype)], dim=2)

    def _flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        sigma = (timestep / self.num_timesteps).view(-1, 1, 1, 1, 1)
        return xt - sigma * flow_pred

    def _bridge_pred_to_x0(self, v_pred: torch.Tensor, x_tau: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        tau = (timestep.float() / float(self.num_timesteps)).clamp(1e-4, 1.0 - 1e-4)
        tau_view = tau.view(-1, 1, 1, 1, 1)
        return x_tau - tau_view.to(dtype=x_tau.dtype) * v_pred

    def _compute_dmd_kl_grad(
        self,
        teacher_noisy_chunk: torch.Tensor,
        fake_xt_chunk: torch.Tensor,
        estimated_clean_chunk: torch.Tensor,
        student_timestep: torch.Tensor,
        teacher_timestep: torch.Tensor,
        audio_chunk: torch.Tensor,
        ref_latent: torch.Tensor,
        prefix_len: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.fake_score is None:
            zero = estimated_clean_chunk.new_zeros(())
            return torch.zeros_like(estimated_clean_chunk), {
                "dmdtrain_gradient_norm": zero,
                "dmd_timestep": student_timestep.detach().mean(),
            }

        with torch.no_grad():
            fake_flow = self.fake_score(
                x=fake_xt_chunk,
                timestep=student_timestep,
                context=audio_chunk,
                y=ref_latent,
            )
            fake_x0 = self._bridge_pred_to_x0(fake_flow, fake_xt_chunk, student_timestep)
            fake_x0[:, :, :prefix_len] = estimated_clean_chunk[:, :, :prefix_len]

            self._move_teacher_to_device(self.device)
            try:
                real_flow = self._teacher_flow_with_audio_cfg(
                    x=teacher_noisy_chunk,
                    timestep=teacher_timestep,
                    audio_chunk=audio_chunk,
                    ref_latent=ref_latent,
                )
            finally:
                pass
                # self._move_teacher_to_device("cpu")
            real_x0 = self._flow_pred_to_x0(real_flow, teacher_noisy_chunk, teacher_timestep)
            real_x0[:, :, :prefix_len] = estimated_clean_chunk[:, :, :prefix_len]

            grad = fake_x0 - real_x0
            real_residual = (estimated_clean_chunk[:, :, prefix_len:, :, :] - real_x0[:, :, prefix_len:, :, :]).abs()
            normalizer = real_residual.mean(dim=[1, 2, 3, 4], keepdim=True).clamp_min(1e-6)
            grad = grad / normalizer.view(-1, 1, 1, 1, 1)
            grad = torch.nan_to_num(grad)

        return grad, {
            "dmdtrain_gradient_norm": grad[:, :, prefix_len:, :, :].abs().mean().detach(),
            "dmd_timestep": student_timestep.detach().mean(),
            "teacher_timestep": teacher_timestep.detach().mean(),
        }

    def _compute_dmd_generator_loss(
        self,
        student_chunk: torch.Tensor,
        audio_chunk: torch.Tensor,
        ref_latent: torch.Tensor,
        prefix_len: int,
        face_mask: torch.Tensor | None = None,
        lip_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if (not self.use_dmd) or self.dmd_loss_weight <= 0.0 or self.fake_score is None:
            zero = student_chunk.new_zeros(())
            return zero, {
                "dmdtrain_gradient_norm": zero.detach(),
                "dmd_timestep": zero.detach(),
            }

        timestep = self._sample_dmd_timestep(student_chunk.shape[0], student_chunk.dtype, student_chunk.device)
        teacher_timestep = self._map_student_timestep_to_teacher_timestep(timestep)
        teacher_noise = torch.randn_like(student_chunk)
        teacher_noisy = self._add_teacher_noise(student_chunk.detach(), teacher_noise, teacher_timestep)

        history = student_chunk.detach()[:, :, :prefix_len, :, :]
        bridge_noise = torch.randn_like(student_chunk[:, :, prefix_len:, :, :])
        fake_xt = self._build_brownian_bridge_xt(
            history=history,
            target_chunk=student_chunk.detach(),
            ref_latent=ref_latent,
            timestep=timestep,
            prefix_len=prefix_len,
            noise=bridge_noise,
        )
        grad, dmd_log_dict = self._compute_dmd_kl_grad(
            teacher_noisy_chunk=teacher_noisy,
            fake_xt_chunk=fake_xt,
            estimated_clean_chunk=student_chunk.detach(),
            student_timestep=timestep,
            teacher_timestep=teacher_timestep,
            audio_chunk=audio_chunk,
            ref_latent=ref_latent,
            prefix_len=prefix_len,
        )
        target = (student_chunk.double() - grad.double()).detach()
        temporal_progress = 1.0 - (timestep.float() / self.num_timesteps).clamp(0.0, 1.0)
        dmd_loss = 0.5 * self._weighted_suffix_mse(
            student_chunk[:, :, prefix_len:, :, :].double(),
            target[:, :, prefix_len:, :, :],
            face_mask,
            lip_mask,
            temporal_progress,
        )
        return dmd_loss.to(dtype=student_chunk.dtype), dmd_log_dict

    def _compute_dmd_critic_loss(
        self,
        generated_chunk: torch.Tensor,
        audio_chunk: torch.Tensor,
        ref_latent: torch.Tensor,
        prefix_len: int,
        face_mask: torch.Tensor | None = None,
        lip_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.fake_score is None:
            zero = generated_chunk.new_zeros(())
            return zero, {"critic_timestep": zero.detach()}

        timestep = self._sample_dmd_timestep(generated_chunk.shape[0], generated_chunk.dtype, generated_chunk.device)

        history = generated_chunk.detach()[:, :, :prefix_len, :, :]
        bridge_noise = torch.randn_like(generated_chunk[:, :, prefix_len:, :, :])
        x_tau = self._build_brownian_bridge_xt(
            history=history,
            target_chunk=generated_chunk.detach(),
            ref_latent=ref_latent,
            timestep=timestep,
            prefix_len=prefix_len,
            noise=bridge_noise,
        )
        fake_flow = self.fake_score(
            x=x_tau,
            timestep=timestep,
            context=audio_chunk,
            y=ref_latent,
        )

        # Brownian-bridge target: for x0 reconstruction x0 = x_tau - tau * v,
        # the matching target velocity is v = (x_tau - x0) / tau.
        tau = (timestep.float() / float(self.num_timesteps)).clamp(1e-4, 1.0 - 1e-4)
        tau_view = tau.view(-1, 1, 1, 1, 1).to(dtype=generated_chunk.dtype, device=generated_chunk.device)
        target_flow = (x_tau - generated_chunk.detach()) / tau_view
        temporal_progress = 1.0 - (timestep.float() / self.num_timesteps).clamp(0.0, 1.0)
        critic_loss = self._weighted_suffix_mse(
            fake_flow[:, :, prefix_len:, :, :],
            target_flow[:, :, prefix_len:, :, :],
            face_mask,
            lip_mask,
            temporal_progress,
        )
        return critic_loss, {"critic_timestep": timestep.detach().mean()}

    def _move_data_modules_to_device(self, device: str) -> None:
        """No-op helper kept for API compatibility.

        WanVAE is a wrapper (not an nn.Module) and does not implement ``to``.
        The dataset also assumes a fixed device derived from ``vae.device``.
        To avoid inconsistent states and attribute errors, we keep data
        modules on their original device and only optionally clear CUDA
        cache when asked to move to CPU.
        """

        if device == "cpu" and torch.cuda.is_available():
            # torch.cuda.empty_cache()
            gc.collect()

    def _move_teacher_to_device(self, device: str) -> None:
        self.teacher_model.to(device=device, dtype=self.amp_dtype)
        if device == "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    def _build_teacher_timesteps(self):
        if self.teacher_steps == 2:
            timesteps = [1000, 500]
        elif self.teacher_steps == 4:
            timesteps = [1000, 750, 500, 250]
        else:
            timesteps = list(np.linspace(self.num_timesteps, 1, self.teacher_steps, dtype=np.float32))
        timesteps.append(0.0)
        ts = [torch.tensor([t], device=self.device) for t in timesteps]
        ts = [timestep_transform(t, shift=self.sample_shift, num_timesteps=self.num_timesteps) for t in ts]
        return ts

    def _build_visualize_timesteps(self) -> torch.Tensor:
        """Return a 1D tensor of explicit visualization timesteps (warped).

        Used only for validate_and_visualize so visualization uses a passed
        discrete step list (official-style) instead of UniPC's linspace steps.
        The terminal 0 step is excluded to avoid an extra Euler update at t=0.
        """

        if self.vis_step_list:
            base = [int(x) for x in self.vis_step_list if int(x) > 0]
            if not base:
                raise RuntimeError("Empty --vis_step_list after filtering non-positive steps")
            if not all(base[i] > base[i + 1] for i in range(len(base) - 1)):
                base = sorted(set(base), reverse=True)
        else:
            n = int(self.student_steps)
            if n <= 0:
                raise RuntimeError("student_steps must be > 0")
            if n == 2:
                base = [1000, 500]
            elif n == 4:
                base = [1000, 750, 500, 250]
            else:
                base = list(np.linspace(self.num_timesteps, 1, n, dtype=np.float32))
                base = [int(round(float(x))) for x in base]
                base = [x for x in base if x > 0]
                base = sorted(set(base), reverse=True)
                if not base:
                    raise RuntimeError("Derived empty visualize timestep list")

        ts = [torch.tensor([float(t)], device=self.device) for t in base]
        ts = [timestep_transform(t, shift=self.sample_shift, num_timesteps=self.num_timesteps) for t in ts]
        values = [t.reshape(-1)[0].to(device=self.device, dtype=torch.float32) for t in ts]
        return torch.stack(values, dim=0)

    def _build_chunk_specs(self, video_latent: torch.Tensor, audio_context: torch.Tensor) -> list[ChunkSpec]:
        total_lat = video_latent.shape[2]
        total_raw = audio_context.shape[1]
        specs: list[ChunkSpec] = []
        chunk_idx = 0
        while True:
            lat_start = chunk_idx * self.lat_new_len
            lat_end = lat_start + self.lat_chunk_len
            raw_start = chunk_idx * self.raw_new_len
            raw_end = raw_start + self.raw_chunk_len
            if lat_end > total_lat or raw_end > total_raw:
                break
            prefix_len = self.init_history_len if chunk_idx == 0 else self.lat_history_len
            specs.append(ChunkSpec(chunk_idx, raw_start, raw_end, lat_start, lat_end, prefix_len))
            chunk_idx += 1
        return specs

    def _source_suffix_from_ref_latent(self, ref_latent: torch.Tensor, prefix_len: int, suffix_len: int) -> torch.Tensor:
        if suffix_len <= 0:
            return ref_latent[:, :, 0:0, :, :].contiguous()
        return ref_latent[:, :, prefix_len:prefix_len + suffix_len, :, :].contiguous()

    def _initial_history(self, ref_latent: torch.Tensor, gt_chunk: torch.Tensor, spec: ChunkSpec) -> torch.Tensor:
        if spec.chunk_idx == 0:
            return ref_latent[:, :, :self.init_history_len].contiguous()
        return gt_chunk[:, :, :spec.prefix_len].contiguous()

    def _sample_bridge_state(self, history: torch.Tensor, gt_chunk: torch.Tensor, ref_latent: torch.Tensor) -> dict[str, torch.Tensor] | None:
        prefix_len = history.shape[2]
        target_suffix = gt_chunk[:, :, prefix_len:, :, :]
        if target_suffix.shape[2] == 0:
            return None

        if self.use_bridge_step_list:
            choices = self.dmd_train_timesteps.to(device=gt_chunk.device)
            if choices.numel() == 0:
                raise RuntimeError("Empty shared timestep list (check --dmd_step_list / teacher_steps)")
            idx = torch.randint(0, int(choices.numel()), (int(gt_chunk.shape[0]),), device=gt_chunk.device, dtype=torch.long)
            timestep = choices[idx].to(dtype=gt_chunk.dtype)
            tau = (timestep / float(self.num_timesteps)).clamp(1e-4, 1.0 - 1e-4)
        else:
            tau = torch.rand(gt_chunk.shape[0], device=gt_chunk.device, dtype=gt_chunk.dtype).clamp_(1e-4, 1.0 - 1e-4)
            timestep = (tau * self.num_timesteps).to(dtype=gt_chunk.dtype)

        tau_view = tau.view(-1, 1, 1, 1, 1)
        source_suffix = self._source_suffix_from_ref_latent(ref_latent, prefix_len, target_suffix.shape[2])
        noise = torch.randn_like(target_suffix)
        suffix_tau = tau_view * source_suffix + (1.0 - tau_view) * target_suffix + (tau_view * (1.0 - tau_view)).sqrt() * noise
        x_tau = torch.cat([history, suffix_tau], dim=2)

        return {
            "prefix_len": torch.tensor(prefix_len, device=gt_chunk.device),
            "target_suffix": target_suffix,
            "source_suffix": source_suffix,
            "tau": tau,
            "tau_view": tau_view,
            "suffix_tau": suffix_tau,
            "x_tau": x_tau,
            "timestep": timestep,
        }

    def _apply_alpha_stabilization(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        tau: torch.Tensor,
        target_suffix: torch.Tensor,
        source_suffix: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_alpha_stabilization:
            return prediction, target

        suffix_dim = float(target_suffix[0].numel())
        diff_norm_sq = ((target_suffix - source_suffix) ** 2).sum(dim=[1, 2, 3, 4])
        alpha_tau = (1.0 + (1.0 - tau) * suffix_dim / (tau * diff_norm_sq + 1e-8)).sqrt()
        alpha_tau_view = alpha_tau.view(-1, 1, 1, 1, 1)
        return prediction / alpha_tau_view, target / alpha_tau_view

    def _compute_endpoint_weight(
        self,
        tau_view: torch.Tensor,
        tau: torch.Tensor,
        target_suffix: torch.Tensor,
        source_suffix: torch.Tensor,
    ) -> torch.Tensor:
        weight = 1.0 / (tau_view.square() + 1e-8)
        return weight
        # if not self.use_alpha_stabilization:
        #     return weight

        # suffix_dim = float(target_suffix[0].numel())
        # diff_norm_sq = ((target_suffix - source_suffix) ** 2).sum(dim=[1, 2, 3, 4])
        # alpha_tau = (1.0 + (1.0 - tau) * suffix_dim / (tau * diff_norm_sq + 1e-8)).sqrt()
        # alpha_tau_view = alpha_tau.view(-1, 1, 1, 1, 1)
        # return weight / (alpha_tau_view.square() + 1e-8)

    def _chunk_raw_prefix_len(self, spec: ChunkSpec) -> int:
        return 1 if spec.chunk_idx == 0 else self.raw_history_len

    def _build_spatial_loss_weight(
        self,
        squared_error: torch.Tensor,
        temporal_progress: torch.Tensor,
        face_mask: torch.Tensor | None,
        lip_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if not self.use_spatial_weighting or face_mask is None or lip_mask is None:
            return None

        _, _, latent_t, latent_h, latent_w = squared_error.shape
        face_mask_down = F.interpolate(
            face_mask.unsqueeze(1),
            size=(latent_t, latent_h, latent_w),
            mode="nearest",
        )
        lip_mask_down = F.interpolate(
            lip_mask.unsqueeze(1),
            size=(latent_t, latent_h, latent_w),
            mode="nearest",
        )
        spatial_weight = 1.0 + self.lambda_face * face_mask_down + self.lambda_lip * lip_mask_down
        temporal_weight = 1.0 + self.gamma_temporal * temporal_progress.view(-1, 1, 1, 1, 1)
        return spatial_weight * temporal_weight

    def _weighted_suffix_mse(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        face_mask: torch.Tensor | None,
        lip_mask: torch.Tensor | None,
        temporal_progress: torch.Tensor,
    ) -> torch.Tensor:
        squared_error = (prediction.float() - target.float()).square()
        spatial_weight = self._build_spatial_loss_weight(
            squared_error,
            temporal_progress,
            face_mask,
            lip_mask,
        )
        if spatial_weight is None:
            return squared_error.mean()
        return (squared_error * spatial_weight.float()).mean()

    def _select_chunk_suffix_masks(
        self,
        face_mask: torch.Tensor | None,
        lip_mask: torch.Tensor | None,
        spec: ChunkSpec,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if face_mask is None or lip_mask is None:
            return None, None

        raw_prefix_len = self._chunk_raw_prefix_len(spec)
        face_suffix = face_mask[:, spec.raw_start + raw_prefix_len:spec.raw_end, :, :]
        lip_suffix = lip_mask[:, spec.raw_start + raw_prefix_len:spec.raw_end, :, :]
        return face_suffix, lip_suffix

    def _decode_latent_batch(self, latent_batch: torch.Tensor) -> torch.Tensor:
        decoded_batches = []
        for sample in latent_batch:
            decoded = self.student_pipeline.vae.decode(sample)
            if isinstance(decoded, (tuple, list)):
                decoded = decoded[0]
            if decoded.dim() == 5:
                if decoded.shape[0] != 1:
                    raise RuntimeError(f"Unexpected decoded batch shape: {tuple(decoded.shape)}")
            elif decoded.dim() == 4:
                decoded = decoded.unsqueeze(0)
            else:
                raise RuntimeError(f"Unexpected decoded ndim={decoded.dim()} shape={tuple(decoded.shape)}")
            decoded_batches.append(decoded)
        return torch.cat(decoded_batches, dim=0)

    def _compute_lpips_loss(
        self,
        prediction_latent: torch.Tensor,
        target_latent: torch.Tensor,
    ) -> torch.Tensor:
        if self.lpips_loss is None or self.lpips_loss_weight <= 0.0:
            return prediction_latent.new_zeros(())

        if prediction_latent.shape[2] == 0:
            return prediction_latent.new_zeros(())

        stride_h = int(self.config.vae_stride[1])
        stride_w = int(self.config.vae_stride[2])
        latent_crop_h = max(1, self.lpips_crop_size // stride_h)
        latent_crop_w = max(1, self.lpips_crop_size // stride_w)

        crop_h = min(prediction_latent.shape[3], latent_crop_h)
        crop_w = min(prediction_latent.shape[4], latent_crop_w)

        max_offset_h = max(prediction_latent.shape[3] - crop_h, 0)
        max_offset_w = max(prediction_latent.shape[4] - crop_w, 0)
        offset_h = 0 if max_offset_h == 0 else random.randint(0, max_offset_h)
        offset_w = 0 if max_offset_w == 0 else random.randint(0, max_offset_w)

        pred_crop = prediction_latent[
            :,
            :,
            :,
            offset_h:offset_h + crop_h,
            offset_w:offset_w + crop_w,
        ]
        target_crop = target_latent[
            :,
            :,
            :,
            offset_h:offset_h + crop_h,
            offset_w:offset_w + crop_w,
        ]

        decoded_pred = self._decode_latent_batch(pred_crop)
        with torch.no_grad():
            decoded_target = self._decode_latent_batch(target_crop.detach())

        decoded_pred = decoded_pred.float().clamp(-1.0, 1.0)
        decoded_target = decoded_target.float().clamp(-1.0, 1.0)

        batch_size, channels, time_steps, height, width = decoded_pred.shape
        decoded_pred = decoded_pred.permute(0, 2, 1, 3, 4).reshape(batch_size * time_steps, channels, height, width)

        target_batch, target_channels, target_time_steps, target_height, target_width = decoded_target.shape
        decoded_target = decoded_target.permute(0, 2, 1, 3, 4).reshape(
            target_batch * target_time_steps,
            target_channels,
            target_height,
            target_width,
        )
        if prediction_latent.device.type == "cuda":
            with torch.amp.autocast("cuda", enabled=False):
                lpips_value = self.lpips_loss(decoded_pred, decoded_target).mean()
        else:
            lpips_value = self.lpips_loss(decoded_pred, decoded_target).mean()

        return lpips_value.to(dtype=prediction_latent.dtype)

    def _compute_single_step_losses(
        self,
        history: torch.Tensor,
        gt_chunk: torch.Tensor,
        audio_chunk: torch.Tensor,
        ref_latent: torch.Tensor,
        face_mask: torch.Tensor | None = None,
        lip_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        if not self.use_gt_target:
            return self._compute_teacher_guided_losses(history, audio_chunk, ref_latent, face_mask, lip_mask)

        state = self._sample_bridge_state(history, gt_chunk, ref_latent)
        if state is None:
            zero = gt_chunk.new_zeros(())
            return {
                "bridge_loss": zero,
                "gt_loss": zero,
                "kd_loss": zero,
                "student_chunk": gt_chunk.detach(),
                "teacher_chunk": None,
                "bridge_timestep": zero,
            }

        prefix_len = int(state["prefix_len"].item())
        target_suffix = state["target_suffix"]
        source_suffix = state["source_suffix"]
        tau = state["tau"]
        tau_view = state["tau_view"]
        suffix_tau = state["suffix_tau"]
        x_tau = state["x_tau"]
        timestep = state["timestep"]

        v_pred = self.student_model(x=x_tau, timestep=timestep, context=audio_chunk, y=ref_latent)
        v_pred_suffix = v_pred[:, :, prefix_len:, :, :]
        student_x0_suffix = suffix_tau - tau_view * v_pred_suffix
        student_chunk = torch.cat([history, student_x0_suffix], dim=2)
        endpoint_weight = self._compute_endpoint_weight(
            tau_view,
            tau,
            target_suffix,
            source_suffix,
        ).float()
        endpoint_error = (student_x0_suffix.float() - target_suffix.float()).square()
        spatial_weight = self._build_spatial_loss_weight(endpoint_error, 1.0 - tau, face_mask, lip_mask)
        bridge_weight = endpoint_weight if spatial_weight is None else endpoint_weight * spatial_weight.float()
        bridge_loss = (spatial_weight * endpoint_error).mean()
        gt_loss = F.l1_loss(student_x0_suffix.float(), target_suffix.float(), reduction="mean")
        lpips_loss = self._compute_lpips_loss(student_x0_suffix, target_suffix)

        kd_loss = gt_chunk.new_zeros(())
        teacher_chunk = None
        if self.kd_loss_weight > 0:
            self._move_teacher_to_device(self.device)
            try:
                with torch.no_grad():
                    teacher_v = self._teacher_flow_with_audio_cfg(
                        x=x_tau.detach(),
                        timestep=timestep.detach(),
                        audio_chunk=audio_chunk,
                        ref_latent=ref_latent,
                    )
                teacher_v_suffix = teacher_v[:, :, prefix_len:, :, :]
                teacher_x0_suffix = suffix_tau -  teacher_v_suffix
                kd_error = (student_x0_suffix.float() - teacher_x0_suffix.float().detach()).square()
                kd_loss = (endpoint_weight * kd_error).mean()
                teacher_chunk = torch.cat([history, teacher_x0_suffix], dim=2)
            finally:
                pass
                # self._move_teacher_to_device("cpu")

        return {
            "bridge_loss": bridge_loss,
            "gt_loss": gt_loss,
            "kd_loss": kd_loss,
            "lpips_loss": lpips_loss,
            "student_chunk": student_chunk,
            "teacher_chunk": teacher_chunk,
            "bridge_timestep": timestep.detach().mean(),
        }

    def _compute_teacher_guided_losses(
        self,
        history: torch.Tensor,
        audio_chunk: torch.Tensor,
        ref_latent: torch.Tensor,
        face_mask: torch.Tensor | None = None,
        lip_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        self._move_teacher_to_device(self.device)
        try:
            timesteps = self._build_teacher_timesteps()
            batch_size, channels, _, height, width = history.shape
            prefix_len = history.shape[2]
            teacher_chunk = torch.randn(
                batch_size,
                channels,
                self.lat_chunk_len,
                height,
                width,
                device=self.device,
                dtype=history.dtype,
            )
            teacher_chunk[:, :, :prefix_len] = history

            with torch.no_grad():
                for i in range(len(timesteps) - 1):
                    timestep_roll = torch.full(
                        (batch_size,),
                        float(timesteps[i].item()),
                        device=self.device,
                        dtype=history.dtype,
                    )
                    flow_pred = self._teacher_flow_with_audio_cfg(
                        x=teacher_chunk,
                        timestep=timestep_roll,
                        audio_chunk=audio_chunk,
                        ref_latent=ref_latent,
                    )
                    t_i = timestep_roll.view(batch_size, 1, 1, 1, 1) / self.num_timesteps
                    t_i_1 = torch.full(
                        (batch_size, 1, 1, 1, 1),
                        float(timesteps[i + 1].item()) / self.num_timesteps,
                        device=self.device,
                        dtype=history.dtype,
                    )
                    x0 = teacher_chunk - flow_pred * t_i
                    teacher_chunk = (1.0 - t_i_1) * x0 + t_i_1 * torch.randn_like(x0)
                    teacher_chunk[:, :, :prefix_len] = history

            teacher_chunk[:, :, :prefix_len] = history
            state = self._sample_bridge_state(history, teacher_chunk, ref_latent)
            if state is None:
                zero = history.new_zeros(())
                return {
                    "bridge_loss": zero,
                    "gt_loss": zero,
                    "kd_loss": zero,
                    "lpips_loss": zero,
                    "student_chunk": teacher_chunk.detach(),
                    "teacher_chunk": teacher_chunk.detach(),
                    "bridge_timestep": zero,
                }

            prefix_len = int(state["prefix_len"].item())
            teacher_suffix = state["target_suffix"]
            source_suffix = state["source_suffix"]
            tau = state["tau"]
            tau_view = state["tau_view"]
            suffix_tau = state["suffix_tau"]
            x_tau = state["x_tau"]
            timestep = state["timestep"]

            v_pred = self.student_model(x=x_tau, timestep=timestep, context=audio_chunk, y=ref_latent)
            v_pred_suffix = v_pred[:, :, prefix_len:, :, :]

            student_x0_suffix = suffix_tau - tau_view * v_pred_suffix
            student_chunk = torch.cat([history, student_x0_suffix], dim=2)
            endpoint_weight = self._compute_endpoint_weight(
                tau_view,
                tau,
                teacher_suffix,
                source_suffix,
            ).float()
            endpoint_error = (student_x0_suffix.float() - teacher_suffix.float()).square()
            spatial_weight = self._build_spatial_loss_weight(endpoint_error, 1.0 - tau, face_mask, lip_mask)
            bridge_weight = endpoint_weight if spatial_weight is None else endpoint_weight * spatial_weight.float()
            bridge_loss = (bridge_weight * endpoint_error).mean()
            endpoint_loss = history.new_zeros(())
            kd_loss = history.new_zeros(())
            lpips_loss = self._compute_lpips_loss(student_x0_suffix, teacher_suffix)

            return {
                "bridge_loss": bridge_loss,
                "gt_loss": endpoint_loss,
                "kd_loss": kd_loss,
                "lpips_loss": lpips_loss,
                "student_chunk": student_chunk,
                "teacher_chunk": teacher_chunk.detach(),
                "bridge_timestep": timestep.detach().mean(),
            }
        finally:
            pass
            # self._move_teacher_to_device("cpu")  # Keep teacher on GPU for potential reuse in next step to save transfer time

    def _run_bridge_chunk(
        self,
        model: torch.nn.Module,
        history: torch.Tensor,
        audio_chunk: torch.Tensor,
        ref_latent: torch.Tensor,
        num_steps: int,
        with_grad: bool,
        timesteps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        prefix_len = history.shape[2]
        suffix_len = self.lat_chunk_len - prefix_len
        suffix = self._source_suffix_from_ref_latent(ref_latent, prefix_len, suffix_len)
        x_t = torch.cat([history, suffix], dim=2)

        scheduler = ViBTScheduler(num_train_timesteps=self.num_timesteps)
        if timesteps is None:
            scheduler.set_timesteps(num_steps, device=self.device)
        else:
            scheduler.timesteps = timesteps.to(device=self.device, dtype=torch.float32)
            scheduler.num_inference_steps = int(scheduler.timesteps.numel())
        scheduler.set_parameters(noise_scale=self.bridge_noise_scale, shift_gamma=self.sample_shift, seed=42)

        context_manager = torch.enable_grad if with_grad else torch.no_grad
        with context_manager():
            for t in scheduler.timesteps:
                timestep = t.unsqueeze(0).to(device=self.device, dtype=x_t.dtype).repeat(x_t.shape[0])
                v_pred = model(x=x_t, timestep=timestep, context=audio_chunk, y=ref_latent)
                x_t = scheduler.step(v_pred, t, x_t)[0]
                x_t[:, :, :prefix_len] = history
        x_t[:, :, :prefix_len] = history
        return x_t

    def _run_teacher_chunk(self, history: torch.Tensor, audio_chunk: torch.Tensor, ref_latent: torch.Tensor) -> torch.Tensor:
        self._move_teacher_to_device(self.device)
        try:
            timesteps = self._build_teacher_timesteps()
            batch_size, channels, _, height, width = history.shape
            prefix_len = history.shape[2]
            sample = torch.randn(
                batch_size,
                channels,
                self.lat_chunk_len,
                height,
                width,
                device=self.device,
                dtype=history.dtype,
            )
            sample[:, :, :prefix_len] = history

            with torch.no_grad():
                for i in range(len(timesteps) - 1):
                    timestep = torch.full((batch_size,), float(timesteps[i].item()), device=self.device, dtype=history.dtype)
                    flow_pred = self._teacher_flow_with_audio_cfg(
                        x=sample,
                        timestep=timestep,
                        audio_chunk=audio_chunk,
                        ref_latent=ref_latent,
                    )
                    t_i = timestep.view(batch_size, 1, 1, 1, 1) / self.num_timesteps
                    t_i_1 = torch.full((batch_size, 1, 1, 1, 1), float(timesteps[i + 1].item()) / self.num_timesteps, device=self.device, dtype=history.dtype)
                    x0 = sample - flow_pred * t_i
                    sample = (1.0 - t_i_1) * x0 + t_i_1 * torch.randn_like(x0)
                    sample[:, :, :prefix_len] = history
            sample[:, :, :prefix_len] = history
            return sample
        finally:
            self._move_teacher_to_device("cpu")

    def _compute_rollout_losses(self, batch: dict, global_step: int) -> dict[str, torch.Tensor]:
        video_latent = batch["video_latent"].to(self.device, dtype=self.amp_dtype)
        ref_latent = batch["ref_latent"].to(self.device, dtype=self.amp_dtype)
        audio_context = batch["audio_context"].to(self.device, dtype=self.amp_dtype)
        face_mask = batch["face_mask"].to(self.device, dtype=self.amp_dtype)
        lip_mask = batch["lip_mask"].to(self.device, dtype=self.amp_dtype)

        if video_latent.shape[0] != 1:
            raise RuntimeError("This training script currently expects batch_size=1 for AR rollout.")

        specs = self._build_chunk_specs(video_latent, audio_context)
        if not specs:
            raise RuntimeError("No valid chunk could be built from the current sample.")

        rollout_len = min(self.rollout_chunks, len(specs))
        start_idx = 0 if len(specs) == rollout_len else random.randint(0, len(specs) - rollout_len)
        specs = specs[start_idx:start_idx + rollout_len]

        total_bridge = video_latent.new_zeros(())
        total_gt = video_latent.new_zeros(())
        total_kd = video_latent.new_zeros(())
        total_lpips = video_latent.new_zeros(())

        current_history = None
        last_student_chunk = None
        last_audio_chunk = None
        last_prefix_len = None
        last_bridge_timestep = None
        for local_idx, spec in enumerate(specs):
            gt_chunk = video_latent[:, :, spec.lat_start:spec.lat_end, :, :]
            audio_chunk = audio_context[:, spec.raw_start:spec.raw_end, :, :, :]
            face_mask_chunk, lip_mask_chunk = self._select_chunk_suffix_masks(face_mask, lip_mask, spec)
            if local_idx == 0:
                current_history = self._initial_history(ref_latent, gt_chunk, spec)

            chunk_loss_dict = self._compute_single_step_losses(
                current_history,
                gt_chunk,
                audio_chunk,
                ref_latent,
                face_mask_chunk,
                lip_mask_chunk,
            )
            total_bridge = total_bridge + chunk_loss_dict["bridge_loss"]
            total_gt = total_gt + chunk_loss_dict["gt_loss"]
            total_kd = total_kd + chunk_loss_dict["kd_loss"]
            total_lpips = total_lpips + chunk_loss_dict["lpips_loss"]

            if "bridge_timestep" in chunk_loss_dict and chunk_loss_dict["bridge_timestep"] is not None:
                last_bridge_timestep = chunk_loss_dict["bridge_timestep"]

            student_chunk = chunk_loss_dict["student_chunk"]
            last_student_chunk = student_chunk
            last_audio_chunk = audio_chunk
            last_prefix_len = int(current_history.shape[2])

            if local_idx != len(specs) - 1:
                current_history = student_chunk[:, :, -self.lat_history_len:, :, :].detach()

        denom = float(len(specs))
        loss = (
            self.bridge_loss_weight * (total_bridge / denom)
            + self.gt_loss_weight * (total_gt / denom)
            + self.kd_loss_weight * (total_kd / denom)
            + self.lpips_loss_weight * (total_lpips / denom)
        )
        return {
            "loss": loss,
            "bridge_loss": total_bridge / denom,
            "gt_loss": total_gt / denom,
            "kd_loss": total_kd / max(1.0, denom),
            "lpips_loss": total_lpips / denom,
            "last_student_chunk": last_student_chunk,
            "last_audio_chunk": last_audio_chunk,
            "last_face_mask_chunk": face_mask_chunk if last_student_chunk is not None else None,
            "last_lip_mask_chunk": lip_mask_chunk if last_student_chunk is not None else None,
            "last_prefix_len": torch.tensor(last_prefix_len or 0, device=video_latent.device),
            "bridge_timestep": last_bridge_timestep if last_bridge_timestep is not None else video_latent.new_zeros(()),
            "ref_latent": ref_latent,
        }

    @torch.no_grad()
    def _generate_full_video_latent(self, sample: dict) -> torch.Tensor:
        video_latent = sample["video_latent"].unsqueeze(0).to(self.device, dtype=self.amp_dtype)
        ref_latent = sample["ref_latent"].unsqueeze(0).to(self.device, dtype=self.amp_dtype)
        audio_context = sample["audio_context"].unsqueeze(0).to(self.device, dtype=self.amp_dtype)
        specs = self._build_chunk_specs(video_latent, audio_context)
        if not specs:
            raise RuntimeError("No validation chunks available.")

        generated_suffixes = []
        history = ref_latent[:, :, :self.init_history_len].contiguous()
        viz_timesteps = self._build_visualize_timesteps()
        for spec in specs:
            audio_chunk = audio_context[:, spec.raw_start:spec.raw_end, :, :, :]
            student_chunk = self._run_bridge_chunk(
                self.student_model,
                history,
                audio_chunk,
                ref_latent,
                self.student_steps,
                with_grad=False,
                timesteps=viz_timesteps,
            )
            prefix_len = history.shape[2]
            generated_suffixes.append(student_chunk[:, :, prefix_len:, :, :])
            history = student_chunk[:, :, -self.lat_history_len:, :, :]

        return torch.cat([ref_latent[:, :, :self.init_history_len], *generated_suffixes], dim=2)

    @torch.no_grad()
    def validate_and_visualize(self, save_dir: str, step: int) -> None:
        os.makedirs(save_dir, exist_ok=True)
        self.student_model.eval()
        try:
            self._move_data_modules_to_device(self.device)
            sample = self.dataset[0]
            latent_video = self._generate_full_video_latent(sample)
            decoded = self.student_pipeline.vae.decode(latent_video[0])
            video_thwc = _normalize_video_to_thwc(decoded)
            save_path = os.path.join(save_dir, f"val_step_{step}.mp4")
            save_video_with_audio(video_thwc, save_path, sample["audio_path"], fps=25)
        finally:
            self._move_data_modules_to_device(self.device)
            self.student_model.train()

    def train(
        self,
        epochs: int = 100,
        batch_size: int = 1,
        lr: float = 1e-4,
        save_dir: str = "/cache/flashhead_vibt_ar_ckpt",
        val_every: int = 100,
        start_global_step: int = 0,
    ):
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "videos"), exist_ok=True)

        try:
            self.writer = SummaryWriter(log_dir=os.path.join(save_dir, "tb"))
        except Exception as exc:
            self.writer = None
            print(f"[WARN] TensorBoard init failed: {exc}")

        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        optimizer = AdamW([p for p in self.student_model.parameters() if p.requires_grad], lr=lr)
        critic_optimizer = None
        if self.use_dmd and self.fake_score is not None:
            critic_optimizer = AdamW(
                [p for p in self.fake_score.parameters() if p.requires_grad],
                lr=self.critic_lr,
            )
        global_step = int(start_global_step)

        try:
            for epoch in range(epochs):
                total_loss = 0.0
                pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
                for batch in pbar:
                    self._move_data_modules_to_device("cpu")

                    train_generator = True
                    if self.use_dmd and critic_optimizer is not None and self.dfake_gen_update_ratio > 1:
                        train_generator = (global_step % self.dfake_gen_update_ratio) == 0

                    if train_generator:
                        optimizer.zero_grad(set_to_none=True)
                        with torch.amp.autocast('cuda', dtype=self.amp_dtype):
                            loss_dict = self._compute_rollout_losses(batch, global_step)
                    else:
                        with torch.no_grad(), torch.amp.autocast('cuda', dtype=self.amp_dtype):
                            loss_dict = self._compute_rollout_losses(batch, global_step)

                    critic_loss = None
                    critic_log_dict: dict[str, torch.Tensor] = {}
                    if critic_optimizer is not None:
                        critic_optimizer.zero_grad(set_to_none=True)
                        with torch.amp.autocast('cuda', dtype=self.amp_dtype):
                            critic_loss, critic_log_dict = self._compute_dmd_critic_loss(
                                generated_chunk=loss_dict["last_student_chunk"].detach(),
                                audio_chunk=loss_dict["last_audio_chunk"],
                                ref_latent=loss_dict["ref_latent"],
                                prefix_len=int(loss_dict["last_prefix_len"].item()),
                                face_mask=loss_dict["last_face_mask_chunk"],
                                lip_mask=loss_dict["last_lip_mask_chunk"],
                            )
                            scaled_critic_loss = self.critic_loss_weight * critic_loss
                        scaled_critic_loss.backward()
                        clip_grad_norm_([p for p in self.fake_score.parameters() if p.requires_grad], max_norm=1.0)
                        critic_optimizer.step()

                    dmd_loss = loss_dict["loss"].new_zeros(())
                    dmd_log_dict: dict[str, torch.Tensor] = {
                        "dmdtrain_gradient_norm": loss_dict["loss"].detach().new_zeros(()),
                        "dmd_timestep": loss_dict["loss"].detach().new_zeros(()),
                    }
                    if self.use_dmd and train_generator:
                        with torch.amp.autocast('cuda', dtype=self.amp_dtype):
                            dmd_loss, dmd_log_dict = self._compute_dmd_generator_loss(
                                student_chunk=loss_dict["last_student_chunk"],
                                audio_chunk=loss_dict["last_audio_chunk"],
                                ref_latent=loss_dict["ref_latent"],
                                prefix_len=int(loss_dict["last_prefix_len"].item()),
                                face_mask=loss_dict["last_face_mask_chunk"],
                                lip_mask=loss_dict["last_lip_mask_chunk"],
                            )

                    loss = loss_dict["loss"] + (self.dmd_loss_weight * dmd_loss)
                    if train_generator:
                        loss.backward()
                        clip_grad_norm_([p for p in self.student_model.parameters() if p.requires_grad], max_norm=1.0)
                        optimizer.step()

                    global_step += 1
                    loss_item = float(loss.detach().item())
                    total_loss += loss_item

                    dmd_timestep_item = None
                    dmd_sampled_this_step = bool(
                        self.use_dmd
                        and train_generator
                        and (self.dmd_loss_weight > 0.0)
                        and (self.fake_score is not None)
                    )
                    if dmd_sampled_this_step and isinstance(dmd_log_dict, dict) and ("dmd_timestep" in dmd_log_dict):
                        try:
                            dmd_timestep_item = float(dmd_log_dict["dmd_timestep"].detach().item())
                        except Exception:
                            dmd_timestep_item = None

                    critic_timestep_item = None
                    if (critic_optimizer is not None) and isinstance(critic_log_dict, dict) and ("critic_timestep" in critic_log_dict):
                        try:
                            critic_timestep_item = float(critic_log_dict["critic_timestep"].detach().item())
                        except Exception:
                            critic_timestep_item = None

                    bridge_timestep_item = None
                    if "bridge_timestep" in loss_dict:
                        try:
                            bridge_timestep_item = float(loss_dict["bridge_timestep"].detach().item())
                        except Exception:
                            bridge_timestep_item = None

                    pbar.set_postfix({
                        "loss": f"{loss_item:.4f}",
                        "bridge": f"{float(loss_dict['bridge_loss'].detach().item()):.4f}",
                        "br_t": ("-" if bridge_timestep_item is None else f"{bridge_timestep_item:.1f}"),
                        "gt": f"{float(loss_dict['gt_loss'].detach().item()):.4f}",
                        "kd": f"{float(loss_dict['kd_loss'].detach().item()):.4f}",
                        "lpips": f"{float(loss_dict['lpips_loss'].detach().item()):.4f}",
                        "dmd": f"{float(dmd_loss.detach().item()):.4f}",
                        "dmd_t": ("-" if dmd_timestep_item is None else f"{dmd_timestep_item:.1f}"),
                        "crit": f"{float(critic_loss.detach().item() if critic_loss is not None else 0.0):.4f}",
                        "crit_t": ("-" if critic_timestep_item is None else f"{critic_timestep_item:.1f}"),
                        "gen": int(train_generator),
                        "step": global_step,
                    })

                    if self.writer is not None:
                        self.writer.add_scalar("train/loss", loss_item, global_step)
                        self.writer.add_scalar("train/bridge_loss", float(loss_dict["bridge_loss"].detach().item()), global_step)
                        self.writer.add_scalar("train/gt_loss", float(loss_dict["gt_loss"].detach().item()), global_step)
                        self.writer.add_scalar("train/kd_loss", float(loss_dict["kd_loss"].detach().item()), global_step)
                        self.writer.add_scalar("train/lpips_loss", float(loss_dict["lpips_loss"].detach().item()), global_step)
                        self.writer.add_scalar("train/dmd_loss", float(dmd_loss.detach().item()), global_step)
                        self.writer.add_scalar("train/dmd_gradient_norm", float(dmd_log_dict["dmdtrain_gradient_norm"].detach().item()), global_step)
                        if critic_loss is not None:
                            self.writer.add_scalar("train/critic_loss", float(critic_loss.detach().item()), global_step)

                    if global_step % val_every == 0:
                        print(f"\n[Step {global_step}] Running validation...")
                        self.validate_and_visualize(os.path.join(save_dir, "videos"), global_step)
                        self.student_model.save_pretrained(os.path.join(save_dir, f"lora_step_{global_step}"))
                        student_wan = _unwrap_wan_model(self.student_model)
                        if hasattr(student_wan, "audio_proj"):
                            torch.save(student_wan.audio_proj.state_dict(), os.path.join(save_dir, f"audio_proj_step_{global_step}.pt"))
                        if self.use_dmd and self.fake_score is not None:
                            self.fake_score.save_pretrained(os.path.join(save_dir, f"fake_score_lora_step_{global_step}"))
                            fake_score_wan = _unwrap_wan_model(self.fake_score)
                            if hasattr(fake_score_wan, "audio_proj"):
                                torch.save(fake_score_wan.audio_proj.state_dict(), os.path.join(save_dir, f"fake_score_audio_proj_step_{global_step}.pt"))

                    self._move_data_modules_to_device(self.device)

                avg_loss = total_loss / max(1, len(dataloader))
                print(f"Epoch {epoch + 1}: Loss = {avg_loss:.6f}")
                if self.writer is not None:
                    self.writer.add_scalar("train/epoch_loss", avg_loss, epoch + 1)
                    self.writer.flush()
        finally:
            self._move_data_modules_to_device(self.device)
            if self.writer is not None:
                self.writer.close()


def _parse_args():
    parser = argparse.ArgumentParser(description="Autoregressive Brownian-bridge FlashHead training with teacher latent distillation")
    parser.add_argument("--ckpt_dir", type=str, default="/cache/SoulX-FlashHead-1_3B")
    parser.add_argument("--wav2vec_dir", type=str, default="/cache/wav2vec2-base-960h")
    parser.add_argument("--video_dir", type=str, default="/cache/VividHead/videos")
    parser.add_argument("--audio_dir", type=str, default="/cache/VividHead/audios")
    parser.add_argument("--teacher_model_dir", type=str, default=None)
    parser.add_argument("--teacher_lora_dir", type=str, default=None)
    parser.add_argument("--teacher_audio_proj", type=str, default=None)
    parser.add_argument(
        "--teacher_audio_guidance_scale",
        type=float,
        default=1.6,
        help="Teacher audio CFG guidance scale (1.0 disables guidance).",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default="/cache/flashhead_vibt_ar_ckpt")
    parser.add_argument("--val_every", type=int, default=200)
    parser.add_argument("--max_frames", type=int, default=125)
    parser.add_argument("--student_steps", type=int, default=1)
    parser.add_argument("--teacher_steps", type=int, default=4)
    parser.add_argument("--rollout_chunks", type=int, default=2)
    parser.add_argument("--bridge_loss_weight", type=float, default=1.0)
    parser.add_argument("--gt_loss_weight", type=float, default=0.0)
    parser.add_argument("--kd_loss_weight", type=float, default=0.0)
    parser.add_argument("--bridge_noise_scale", type=float, default=1.0)
    parser.add_argument(
        "--use_bridge_step_list",
        action="store_true",
        help="Make bridge-loss sample timesteps from the same (warped) discrete list as DMD (from --dmd_step_list or teacher-derived list), instead of continuous random tau.",
    )
    parser.add_argument("--disable_alpha_stabilization", action="store_true")
    parser.add_argument("--use_gt_target", action="store_true", help="Use dataset video latent as the training target instead of teacher-guided targets")
    parser.add_argument("--use_spatial_weighting", action="store_true")
    parser.add_argument("--lambda_face", type=float, default=2.0)
    parser.add_argument("--lambda_lip", type=float, default=5.0)
    parser.add_argument("--gamma_temporal", type=float, default=0.0)
    parser.add_argument(
        "--mask_cache_dir",
        type=str,
        default=None,
        help="Optional directory for cached MediaPipe masks; defaults to <video_dir>/.mediapipe_mask_cache when omitted.",
    )
    parser.add_argument("--mask_dilate_lip", type=int, default=5)
    parser.add_argument("--mask_dilate_face", type=int, default=10)
    parser.add_argument("--use_dmd", action="store_true", help="Enable DMD with a trainable fake score network on the last rollout chunk suffix")
    parser.add_argument("--dmd_loss_weight", type=float, default=0.0)
    parser.add_argument(
        "--dmd_step_list",
        type=int,
        nargs="+",
        default=None,
        help="Explicit discrete DMD timestep list (e.g. --dmd_step_list 1000 750 500 250). Overrides teacher-derived list.",
    )
    parser.add_argument(
        "--vis_step_list",
        type=int,
        nargs="+",
        default=None,
        help="Explicit discrete timesteps for validate_and_visualize sampling (e.g. --vis_step_list 1000 750 500 250).",
    )
    parser.add_argument("--critic_loss_weight", type=float, default=1.0)
    parser.add_argument("--critic_lr", type=float, default=8e-6)
    parser.add_argument(
        "--dfake_gen_update_ratio",
        type=int,
        default=5,
        help="Update generator every N steps while critic updates every step (official default: 5)",
    )
    parser.add_argument("--fake_score_lora_rank", type=int, default=64)
    parser.add_argument("--lpips_loss_weight", type=float, default=1.0)
    parser.add_argument("--lpips_net", type=str, default="vgg", choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--lpips_model_path", type=str, default=None)
    parser.add_argument("--lpips_crop_size", type=int, default=256)

    parser.add_argument("--resume_lora_dir", type=str, default=None)
    parser.add_argument("--resume_audio_proj", type=str, default=None)
    parser.add_argument("--freeze_audio_proj", action="store_true", help="Freeze audio_proj so it does not receive gradients or optimizer updates.")
    parser.add_argument("--resume_fake_score_lora_dir", type=str, default=None)
    parser.add_argument("--resume_fake_score_audio_proj", type=str, default=None)
    parser.add_argument("--resume_step", type=int, default=None)
    return parser.parse_args()


def _dump_run_hparams(save_dir: str, args: argparse.Namespace) -> None:
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    git_commit = None
    try:
        git_commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__))
            .decode("utf-8")
            .strip()
        )
    except Exception:
        pass

    payload = {
        "timestamp": timestamp,
        "argv": sys.argv,
        "git_commit": git_commit,
        "hparams": vars(args),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

    main_path = os.path.join(save_dir, "hparams.json")
    stamped_path = os.path.join(save_dir, f"hparams_{timestamp}.json")
    for path in (main_path, stamped_path):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            print(f"[WARN] Failed to write hparams to {path}: {exc}")


if __name__ == "__main__":
    args = _parse_args()

    _dump_run_hparams(args.save_dir, args)

    inferred_step = _parse_step_from_path(args.resume_lora_dir) if args.resume_lora_dir else None
    start_step = args.resume_step if args.resume_step is not None else (inferred_step or 0)
    if args.resume_lora_dir:
        print(f"[INFO] Resuming LoRA from: {args.resume_lora_dir}")
        print(f"[INFO] Start global_step: {start_step}")

    trainer = FlashHeadARBridgeKDTrainer(
        ckpt_dir=args.ckpt_dir,
        wav2vec_dir=args.wav2vec_dir,
        video_dir=args.video_dir,
        audio_dir=args.audio_dir,
        device=args.device,
        lora_rank=args.lora_rank,
        max_frames=args.max_frames,
        student_steps=args.student_steps,
        teacher_steps=args.teacher_steps,
        bridge_loss_weight=args.bridge_loss_weight,
        gt_loss_weight=args.gt_loss_weight,
        kd_loss_weight=args.kd_loss_weight,
        rollout_chunks=args.rollout_chunks,
        resume_lora_dir=args.resume_lora_dir,
        resume_audio_proj=args.resume_audio_proj,
        freeze_audio_proj=args.freeze_audio_proj,
        teacher_model_dir=args.teacher_model_dir,
        teacher_lora_dir=args.teacher_lora_dir,
        teacher_audio_proj=args.teacher_audio_proj,
        teacher_audio_guidance_scale=args.teacher_audio_guidance_scale,
        use_bridge_step_list=args.use_bridge_step_list,
        dmd_step_list=args.dmd_step_list,
        vis_step_list=args.vis_step_list,
        bridge_noise_scale=args.bridge_noise_scale,
        use_alpha_stabilization=not args.disable_alpha_stabilization,
        use_gt_target=args.use_gt_target,
        use_dmd=args.use_dmd,
        dmd_loss_weight=args.dmd_loss_weight,
        critic_loss_weight=args.critic_loss_weight,
        critic_lr=args.critic_lr,
        dfake_gen_update_ratio=args.dfake_gen_update_ratio,
        fake_score_lora_rank=args.fake_score_lora_rank,
        resume_fake_score_lora_dir=args.resume_fake_score_lora_dir,
        resume_fake_score_audio_proj=args.resume_fake_score_audio_proj,
        lpips_loss_weight=args.lpips_loss_weight,
        lpips_net=args.lpips_net,
        lpips_model_path=args.lpips_model_path,
        lpips_crop_size=args.lpips_crop_size,
        use_spatial_weighting=args.use_spatial_weighting,
        lambda_face=args.lambda_face,
        lambda_lip=args.lambda_lip,
        gamma_temporal=args.gamma_temporal,
        mask_cache_dir=args.mask_cache_dir or os.path.join(args.video_dir, ".mediapipe_mask_cache"),
        mask_dilate_lip=args.mask_dilate_lip,
        mask_dilate_face=args.mask_dilate_face,
    )
    trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=args.save_dir,
        val_every=args.val_every,
        start_global_step=start_step,
    )
