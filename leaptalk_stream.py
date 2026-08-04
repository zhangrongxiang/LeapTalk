from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import os
import subprocess
import time
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

import imageio
import numpy as np


@dataclass
class LeapTalkStreamSettings:
    ckpt_dir: str
    wav2vec_dir: str
    lora_dir: str
    audio_proj: str
    compile: str
    steps: int
    lite: bool
    cuda_visible_devices: str
    output_sample_rate: int
    root_dir: Path
    web_dir: Path
    audio_dir: Path
    video_dir: Path
    device: str = "cuda"
    dtype: str = "bf16"
    height: int = 512
    width: int = 512
    fps: int = 25
    sample_rate: int = 16000
    frame_num: int = 33
    motion_frames_latent_num: int = 2
    cached_audio_duration: int = 8
    history_update_mode: str = "roundtrip"
    guidance_scale: float = 1.0
    noise_scale: float = 1.0
    shift_gamma: float = 5.0
    seed: int = 42
    model_type: str = "pro"
    tae_model_type: str = "wan21"
    color_correction_strength: float = 1.0
    use_face_crop: bool = False


@dataclass
class _TurnState:
    x0: Any
    latent_motion_frames: Any
    clamp_latent_len: int
    audio_deque: deque[float]


class LeapTalkEngine:
    def __init__(self, settings: LeapTalkStreamSettings) -> None:
        self.settings = settings
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="leaptalk")
        self._load_lock = asyncio.Lock()
        self._turn_lock = asyncio.Lock()
        self._loaded = False
        self._pipeline = None
        self._scheduler = None
        self._sp = None
        self._amp_dtype = None
        self._helpers: dict[str, Any] = {}

    @property
    def slice_len(self) -> int:
        if self._sp is None:
            return self.settings.frame_num - 5
        return int(self._sp.slice_len)

    @property
    def source_samples_per_slice(self) -> int:
        return int(round(self.settings.output_sample_rate * self.slice_len / self.settings.fps))

    @property
    def source_bytes_per_slice(self) -> int:
        return self.source_samples_per_slice * 2

    async def ensure_loaded(self) -> None:
        if self._loaded:
            return
        async with self._load_lock:
            if self._loaded:
                return
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, self._load_sync)

    def _load_sync(self) -> None:
        s = self.settings
        os.environ["CUDA_VISIBLE_DEVICES"] = s.cuda_visible_devices

        missing = [
            (name, path)
            for name, path in (
                ("LEAPTALK_CKPT_DIR", s.ckpt_dir),
                ("LEAPTALK_WAV2VEC_DIR", s.wav2vec_dir),
                ("LEAPTALK_LORA_DIR", s.lora_dir),
            )
            if not path or not Path(path).exists()
        ]
        if missing:
            details = ", ".join(f"{name}={path}" for name, path in missing)
            raise RuntimeError(f"LeapTalk model path missing: {details}")

        import torch
        from peft import PeftModel

        from inference import (
            StreamParams,
            _build_infer_timesteps,
            _get_inner_flashhead_model,
            _lora_checkpoint_needs_compiled_base,
            _pick_latest_audio_proj,
            _pick_latest_lora_dir,
            _resolve_lite_tae_path,
        )
        import flash_head.src.pipeline.flash_head_pipeline as fh_pipe_mod
        from flash_head.src.pipeline.flash_head_pipeline import FlashHeadPipeline
        from vibt.scheduler import ViBTScheduler

        if s.dtype == "bf16":
            amp_dtype = torch.bfloat16
        elif s.dtype == "fp16":
            amp_dtype = torch.float16
        else:
            amp_dtype = torch.float32

        resolved_lora_dir = _pick_latest_lora_dir(s.lora_dir)
        if not resolved_lora_dir:
            raise RuntimeError(f"LoRA weights not found under {s.lora_dir}")

        tae_path = (
            _resolve_lite_tae_path(
                lora_dir=s.lora_dir,
                resolved_lora_dir=resolved_lora_dir,
                ckpt_dir=s.ckpt_dir,
                explicit_tae_path=None,
            )
            if s.lite
            else None
        )

        needs_compiled = _lora_checkpoint_needs_compiled_base(resolved_lora_dir)
        compile_model = needs_compiled if s.compile == "auto" else s.compile == "on"
        compile_before_lora = bool(needs_compiled)
        compile_after_lora = bool(compile_model and not needs_compiled)
        fh_pipe_mod.COMPILE_MODEL = bool(compile_before_lora)
        fh_pipe_mod.COMPILE_VAE = bool(not s.lite)

        self._pipeline = FlashHeadPipeline(
            checkpoint_dir=s.ckpt_dir,
            model_type=s.model_type,
            wav2vec_dir=s.wav2vec_dir,
            device=s.device,
            param_dtype=amp_dtype,
            use_usp=False,
            use_tae=bool(s.lite),
            tae_path=tae_path,
            tae_model_type=s.tae_model_type,
        )
        self._pipeline.model = PeftModel.from_pretrained(self._pipeline.model, resolved_lora_dir, is_trainable=False)
        self._pipeline.model = self._pipeline.model.merge_and_unload()
        self._pipeline.model.eval().requires_grad_(False)
        self._pipeline.audio_encoder.eval().requires_grad_(False)

        if compile_after_lora:
            self._pipeline.model = torch.compile(self._pipeline.model)

        audio_proj_path = (
            _pick_latest_audio_proj(s.audio_proj)
            or _pick_latest_audio_proj(s.lora_dir)
            or _pick_latest_audio_proj(resolved_lora_dir)
            or _pick_latest_audio_proj(os.path.dirname(resolved_lora_dir))
        )
        if not audio_proj_path:
            raise RuntimeError("audio_proj weights not found")

        audio_proj_state = torch.load(audio_proj_path, map_location="cpu")
        inner = _get_inner_flashhead_model(self._pipeline.model)
        inner.audio_proj.load_state_dict(audio_proj_state, strict=True)

        self._sp = StreamParams(
            frame_num=s.frame_num,
            motion_frames_latent_num=s.motion_frames_latent_num,
            tgt_fps=s.fps,
            sample_rate=s.sample_rate,
            cached_audio_duration=s.cached_audio_duration,
        ).init_with_stride(int(self._pipeline.config.vae_stride[0]))

        if self._sp.slice_len <= 0:
            raise RuntimeError(f"Invalid LeapTalk stream params: frame_num={s.frame_num}, slice_len={self._sp.slice_len}")

        scheduler = ViBTScheduler(num_train_timesteps=1000)
        scheduler.timesteps = _build_infer_timesteps(
            step_list=None,
            num_inference_steps=int(s.steps),
            shift_gamma=float(s.shift_gamma),
            device=s.device,
            num_timesteps=1000,
        )
        scheduler.num_inference_steps = int(scheduler.timesteps.numel())
        scheduler.set_parameters(noise_scale=s.noise_scale, shift_gamma=s.shift_gamma, seed=s.seed)

        from inference import (
            _audio_context_from_embeddings_range,
            _bridge_sample_one_chunk,
            _decode_to_cthw,
            _encode_motion_prefix_from_decoded,
            _maybe_apply_color_correction,
            _motion_prefix_from_latent_tail,
        )

        self._scheduler = scheduler
        self._amp_dtype = amp_dtype
        self._helpers = {
            "torch": torch,
            "audio_context": _audio_context_from_embeddings_range,
            "bridge_sample": _bridge_sample_one_chunk,
            "decode": _decode_to_cthw,
            "color": _maybe_apply_color_correction,
            "encode_motion": _encode_motion_prefix_from_decoded,
            "motion_tail": _motion_prefix_from_latent_tail,
        }
        self._loaded = True

    async def prepare_turn(self, image_path: Path) -> _TurnState:
        await self.ensure_loaded()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._prepare_turn_sync, str(image_path))

    def _prepare_turn_sync(self, image_path: str) -> _TurnState:
        s = self.settings
        torch = self._helpers["torch"]
        pipeline = self._pipeline
        assert pipeline is not None and self._sp is not None and self._amp_dtype is not None

        with torch.no_grad():
            pipeline.prepare_params(
                cond_image_path_or_dir=image_path,
                target_size=(s.height, s.width),
                frame_num=s.frame_num,
                motion_frames_num=0,
                sampling_steps=max(1, int(s.steps)),
                seed=s.seed,
                shift=s.shift_gamma,
                color_correction_strength=s.color_correction_strength,
                use_face_crop=s.use_face_crop,
            )
            x0 = pipeline.ref_img_latent.to(device=s.device, dtype=self._amp_dtype)
            latent_motion_frames = x0[:, :1].unsqueeze(0).clone()

        cached_len = s.sample_rate * self._sp.cached_audio_duration
        return _TurnState(
            x0=x0,
            latent_motion_frames=latent_motion_frames,
            clamp_latent_len=int(latent_motion_frames.shape[2]),
            audio_deque=deque([0.0] * cached_len, maxlen=cached_len),
        )

    async def generate_chunk(
        self,
        state: _TurnState,
        *,
        pcm16_segment: bytes,
        turn_dir: Path,
        chunk_index: int,
    ) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._generate_chunk_sync,
            state,
            pcm16_segment,
            str(turn_dir),
            int(chunk_index),
        )

    def _generate_chunk_sync(
        self,
        state: _TurnState,
        pcm16_segment: bytes,
        turn_dir: str,
        chunk_index: int,
    ) -> dict[str, Any]:
        s = self.settings
        torch = self._helpers["torch"]
        pipeline = self._pipeline
        assert pipeline is not None and self._sp is not None and self._scheduler is not None and self._amp_dtype is not None

        target_audio = self._pcm16_to_model_audio(pcm16_segment)
        state.audio_deque.extend(target_audio.tolist())
        audio_cache = np.array(state.audio_deque, dtype=np.float32)

        audio_end_idx = self._sp.cached_audio_duration * s.fps
        audio_start_idx = audio_end_idx - self._sp.frame_num

        use_cuda_timing = s.device.startswith("cuda") and torch.cuda.is_available()
        if use_cuda_timing:
            torch.cuda.synchronize()
        started = time.perf_counter()

        with torch.no_grad():
            audio_emb_cache = pipeline.preprocess_audio(audio_cache, sr=s.sample_rate, fps=s.fps)
            if audio_emb_cache is None:
                raise RuntimeError("Failed to extract LeapTalk audio embeddings")
            audio_emb_cache = audio_emb_cache.to(device=s.device, dtype=self._amp_dtype)
            audio_ctx = self._helpers["audio_context"](
                audio_emb_cache,
                start_idx=audio_start_idx,
                end_idx=audio_end_idx,
                device=s.device,
                dtype=self._amp_dtype,
            )
            x_final = self._helpers["bridge_sample"](
                pipeline,
                scheduler=self._scheduler,
                ref_latent=state.x0,
                audio_context=audio_ctx,
                guidance_scale=s.guidance_scale,
                latent_motion_frames=state.latent_motion_frames,
                clamp_latent_len=state.clamp_latent_len,
                device=s.device,
                dtype=self._amp_dtype,
            )
            decoded_cthw = self._helpers["decode"](pipeline, x_final)
            decoded_cthw = self._helpers["color"](pipeline, decoded_cthw)

            if s.history_update_mode == "roundtrip":
                state.latent_motion_frames = self._helpers["encode_motion"](
                    pipeline,
                    decoded_video_cthw=decoded_cthw,
                    motion_frames_num=self._sp.motion_frames_num,
                    device=s.device,
                    dtype=self._amp_dtype,
                ).unsqueeze(0)
            else:
                state.latent_motion_frames = self._helpers["motion_tail"](
                    x_final,
                    motion_frames_latent_num=self._sp.motion_frames_latent_num,
                ).unsqueeze(0)
            state.clamp_latent_len = int(state.latent_motion_frames.shape[2])

            decoded_cthw = decoded_cthw[:, self._sp.motion_frames_num :]
            video_thwc = (
                ((decoded_cthw + 1.0) / 2.0)
                .permute(1, 2, 3, 0)
                .clamp(0.0, 1.0)
                .mul(255.0)
                .contiguous()
            )
            if use_cuda_timing:
                torch.cuda.synchronize()
            chunk_seconds = time.perf_counter() - started
            if video_thwc.dtype == torch.bfloat16:
                video_thwc = video_thwc.to(torch.float16)
            frames = video_thwc.detach().cpu().numpy().astype(np.uint8)

        out_dir = Path(turn_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"chunk_{chunk_index:04d}.mp4"
        self._write_av_chunk(frames, pcm16_segment, out_path)

        frame_count = int(frames.shape[0])
        return {
            "path": out_path,
            "frames": frame_count,
            "fps": s.fps,
            "generation_fps": frame_count / chunk_seconds if chunk_seconds > 0 else None,
            "chunk_seconds": chunk_seconds,
        }

    def _pcm16_to_model_audio(self, pcm16_segment: bytes) -> np.ndarray:
        s = self.settings
        audio = np.frombuffer(pcm16_segment, dtype="<i2").astype(np.float32) / 32768.0
        target_samples = int(round(s.sample_rate * self.slice_len / s.fps))
        if s.output_sample_rate != s.sample_rate:
            import librosa

            audio = librosa.resample(audio, orig_sr=s.output_sample_rate, target_sr=s.sample_rate).astype(np.float32)
        if audio.shape[0] < target_samples:
            audio = np.pad(audio, (0, target_samples - int(audio.shape[0])))
        elif audio.shape[0] > target_samples:
            audio = audio[:target_samples]
        return audio.astype(np.float32, copy=False)

    def _write_av_chunk(self, frames: np.ndarray, pcm16_segment: bytes, out_path: Path) -> None:
        tmp_video = out_path.with_name(out_path.stem + "_video.mp4")
        tmp_wav = out_path.with_name(out_path.stem + ".wav")
        duration = float(frames.shape[0]) / float(self.settings.fps)
        with imageio.get_writer(
            str(tmp_video),
            format="mp4",
            mode="I",
            fps=self.settings.fps,
            codec="h264",
            ffmpeg_params=["-bf", "0"],
        ) as writer:
            for frame in frames:
                writer.append_data(frame)
        _write_wav(tmp_wav, pcm16_segment, sample_rate=self.settings.output_sample_rate)
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(tmp_video),
            "-i",
            str(tmp_wav),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            "-af",
            "apad",
            "-t",
            f"{duration:.6f}",
            "-movflags",
            "+empty_moov+default_base_moof+frag_keyframe+omit_tfhd_offset",
            str(out_path),
            "-y",
        ]
        subprocess.run(cmd, cwd=str(self.settings.root_dir), check=True)
        with contextlib.suppress(OSError):
            tmp_video.unlink()
        with contextlib.suppress(OSError):
            tmp_wav.unlink()

    def runtime_url(self, path: Path) -> str:
        return "/" + path.relative_to(self.settings.web_dir).as_posix()


class LeapTalkTurnStreamer:
    def __init__(
        self,
        engine: LeapTalkEngine,
        *,
        turn_id: str,
        image_path: Path,
        send_client: Callable[[dict], Awaitable[None]],
        send_video_chunk: Callable[[dict, bytes], Awaitable[None]] | None = None,
    ) -> None:
        self.engine = engine
        self.turn_id = turn_id
        self.image_path = image_path
        self.send_client = send_client
        self.send_video_chunk = send_video_chunk
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._started_at = time.perf_counter()

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def feed(self, pcm16_chunk: bytes) -> None:
        if pcm16_chunk:
            await self._queue.put(pcm16_chunk)

    async def finish(self) -> None:
        await self._queue.put(None)
        if self._task is not None:
            await self._task

    async def cancel(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def _run(self) -> None:
        settings = self.engine.settings
        pending = bytearray()
        chunk_index = 0
        turn_dir = settings.video_dir / self.turn_id
        final_seen = False
        async with self.engine._turn_lock:
            await self.send_client({"type": "video_stream_start", "message": "LeapTalk is preparing the streaming video."})
            state = await self.engine.prepare_turn(self.image_path)
            slice_bytes = self.engine.source_bytes_per_slice
            await self.send_client(
                {
                    "type": "video_stream_ready",
                    "fps": settings.fps,
                    "frames_per_chunk": self.engine.slice_len,
                    "seconds_per_chunk": self.engine.slice_len / settings.fps,
                }
            )
            while True:
                item = await self._queue.get()
                if item is None:
                    final_seen = True
                else:
                    pending.extend(item)

                while len(pending) >= slice_bytes or (final_seen and pending):
                    segment = bytes(pending[:slice_bytes])
                    del pending[:slice_bytes]
                    if len(segment) < slice_bytes:
                        segment += b"\x00" * (slice_bytes - len(segment))
                    chunk_index += 1
                    result = await self.engine.generate_chunk(
                        state,
                        pcm16_segment=segment,
                        turn_dir=turn_dir,
                        chunk_index=chunk_index,
                    )
                    chunk_url = self.engine.runtime_url(result["path"])
                    payload = {
                        "type": "video_chunk",
                        "transport": "websocket_fmp4",
                        "container": "fmp4",
                        "mime_type": 'video/mp4; codecs="avc1.64001e, mp4a.40.2"',
                        "url": chunk_url,
                        "chunk_index": chunk_index,
                        "fps": result["fps"],
                        "generation_fps": result["generation_fps"],
                        "frames": result["frames"],
                        "chunk_seconds": result["chunk_seconds"],
                        "elapsed_seconds": time.perf_counter() - self._started_at,
                    }
                    if self.send_video_chunk is not None:
                        await self.send_video_chunk(payload, Path(result["path"]).read_bytes())
                    else:
                        await self.send_client(payload)
                    await self.send_client(
                        {
                            "type": "video_chunk_ready",
                            "transport": "websocket_fmp4",
                            "container": "fmp4",
                            "mime_type": 'video/mp4; codecs="avc1.64001e, mp4a.40.2"',
                            "url": chunk_url,
                            "chunk_index": chunk_index,
                            "fps": result["fps"],
                            "generation_fps": result["generation_fps"],
                            "frames": result["frames"],
                            "chunk_seconds": result["chunk_seconds"],
                            "elapsed_seconds": time.perf_counter() - self._started_at,
                        }
                    )

                if final_seen:
                    break

            await self.send_client(
                {
                    "type": "video_stream_end",
                    "chunks": chunk_index,
                    "seconds": time.perf_counter() - self._started_at,
                }
            )


def _write_wav(path: Path, pcm: bytes, *, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
