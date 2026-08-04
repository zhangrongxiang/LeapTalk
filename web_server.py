from __future__ import annotations

import asyncio
import contextlib
import gzip
import json
import logging
import os
import shutil
import struct
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Awaitable, Callable

import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from leaptalk_stream import LeapTalkEngine, LeapTalkStreamSettings, LeapTalkTurnStreamer

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
STATIC_DIR = WEB_DIR / "static"
RUNTIME_DIR = WEB_DIR / "runtime"
AVATAR_DIR = RUNTIME_DIR / "avatars"
AUDIO_DIR = RUNTIME_DIR / "audio"
VIDEO_DIR = RUNTIME_DIR / "videos"

for _path in (STATIC_DIR, AVATAR_DIR, AUDIO_DIR, VIDEO_DIR):
    _path.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("leaptalk.web")


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(ROOT / ".env")


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@dataclass
class Settings:
    host: str = _env("LEAPTALK_WEB_HOST", "0.0.0.0")
    port: int = int(_env("LEAPTALK_WEB_PORT", "7860") or 7860)
    ckpt_dir: str = _env("LEAPTALK_CKPT_DIR", "./models/SoulX-FlashHead-1_3B")
    wav2vec_dir: str = _env("LEAPTALK_WAV2VEC_DIR", "./models/wav2vec2-base-960h")
    lora_dir: str = _env("LEAPTALK_LORA_DIR", "./models/leaptalk")
    audio_proj: str = _env("LEAPTALK_AUDIO_PROJ", "")
    compile: str = _env("LEAPTALK_COMPILE", "off")
    steps: str = _env("LEAPTALK_NUM_INFERENCE_STEPS", "1")
    lite: bool = _env("LEAPTALK_LITE", "1").lower() in {"1", "true", "on", "yes"}
    cuda_visible_devices: str = _env("CUDA_VISIBLE_DEVICES", _env("LEAPTALK_CUDA_VISIBLE_DEVICES", "0"))
    doubao_api_key: str = _env("DOUBAO_API_KEY")
    doubao_access_token: str = _env("DOUBAO_ACCESS_TOKEN") or _env("DOUBAO_ACCESS_KEY") or _env("DOUBAO_TOKEN")
    doubao_app_id: str = _env("DOUBAO_APP_ID")
    doubao_ws_url: str = _env("DOUBAO_WS_URL", "wss://openspeech.bytedance.com/api/v3/realtime/dialogue")
    doubao_voice: str = _env("DOUBAO_VOICE_TYPE", "温柔文雅")
    doubao_bot_name: str = _env("DOUBAO_BOT_NAME", "LeapTalk")
    doubao_system_prompt: str = _env("DOUBAO_SYSTEM_PROMPT", "你是一个自然、简洁、友好的数字人助手。")
    doubao_speaking_style: str = _env("DOUBAO_SPEAKING_STYLE", "你的回答要口语化，适合语音播报，通常控制在三句话以内。")
    doubao_model: str = _env("DOUBAO_MODEL", "2.2.0.0")
    input_sample_rate: int = int(_env("DOUBAO_INPUT_SAMPLE_RATE", "16000") or 16000)
    output_sample_rate: int = int(_env("DOUBAO_OUTPUT_SAMPLE_RATE", "24000") or 24000)


settings = Settings()

leaptalk_engine = LeapTalkEngine(
    LeapTalkStreamSettings(
        ckpt_dir=settings.ckpt_dir,
        wav2vec_dir=settings.wav2vec_dir,
        lora_dir=settings.lora_dir,
        audio_proj=settings.audio_proj,
        compile=settings.compile,
        steps=int(settings.steps or 1),
        lite=settings.lite,
        cuda_visible_devices=settings.cuda_visible_devices,
        output_sample_rate=settings.output_sample_rate,
        root_dir=ROOT,
        web_dir=WEB_DIR,
        audio_dir=AUDIO_DIR,
        video_dir=VIDEO_DIR,
    )
)

def _doubao_auth_mode() -> str:
    if settings.doubao_app_id and settings.doubao_access_token:
        return "app_id_access_token"
    if settings.doubao_api_key:
        return "api_key"
    return "missing"


SC20_VOICES: dict[str, str] = {
    "傲娇女友": "saturn_zh_female_aojiaonvyou_tob",
    "冰娇姐姐": "saturn_zh_female_bingjiaojiejie_tob",
    "成熟姐姐": "saturn_zh_female_chengshujiejie_tob",
    "可爱女生": "saturn_zh_female_keainvsheng_tob",
    "暖心学姐": "saturn_zh_female_nuanxinxuejie_tob",
    "贴心女友": "saturn_zh_female_tiexinnvyou_tob",
    "温柔文雅": "saturn_zh_female_wenrouwenya_tob",
    "妩媚御姐": "saturn_zh_female_wumeiyujie_tob",
    "性感御姐": "saturn_zh_female_xingganyujie_tob",
    "磁性男嗓": "saturn_zh_male_cixingnansang_tob",
    "风发少年": "saturn_zh_male_fengfashaonian_tob",
    "成熟总裁": "saturn_zh_male_chengshuzongcai_tob",
}


class DoubaoEvent:
    START_CONNECTION = 1
    CONNECTION_STARTED = 50
    START_SESSION = 100
    FINISH_SESSION = 102
    SESSION_STARTED = 150
    SESSION_FINISHED = 152
    SESSION_FAILED = 153
    TASK_REQUEST = 200
    CHAT_TEXT_QUERY = 501
    TTS_SENTENCE_DONE = 351
    AUDIO_DATA = 352
    REPLY_DONE = 359
    ASR_START = 450
    ASR_RESULT = 451
    TURN_FINISHED = 459
    LLM_TOKEN = 550


MSGTYPE_FULL_CLIENT = 0x10
MSGTYPE_AUDIO_ONLY_CLIENT = 0x20
MSGTYPE_FULL_SERVER = 0x90
MSGTYPE_AUDIO_ONLY_SERVER = 0xB0
MSGTYPE_ERROR = 0xF0
MSGTYPE_FLAG_WITH_EVENT = 0x04
SERIALIZATION_RAW = 0x00
SERIALIZATION_JSON = 0x10
COMPRESSION_NONE = 0x00
COMPRESSION_GZIP = 0x01


@dataclass
class DecodedFrame:
    msg_type_bits: int
    msg_flags: int
    serialization_bits: int
    compression_bits: int
    event: int | None
    session_id: str | None
    connect_id: str | None
    error_code: int | None
    payload: bytes

    def is_audio(self) -> bool:
        return self.msg_type_bits == MSGTYPE_AUDIO_ONLY_SERVER

    def is_full_server(self) -> bool:
        return self.msg_type_bits == MSGTYPE_FULL_SERVER

    def is_error(self) -> bool:
        return self.msg_type_bits == MSGTYPE_ERROR


def _compress(payload: bytes, compression_bits: int) -> bytes:
    return gzip.compress(payload) if compression_bits == COMPRESSION_GZIP else payload


def _decompress(payload: bytes, compression_bits: int) -> bytes:
    return gzip.decompress(payload) if compression_bits == COMPRESSION_GZIP else payload


def encode_frame(
    *,
    msg_type_bits: int,
    serialization_bits: int,
    event: int,
    session_id: str | None,
    payload: bytes,
    compression_bits: int = COMPRESSION_NONE,
    connect_id: str | None = None,
) -> bytes:
    header = bytearray([0x11, msg_type_bits | MSGTYPE_FLAG_WITH_EVENT, serialization_bits | compression_bits, 0x00])
    header += struct.pack(">i", int(event))
    if event not in (1, 2, 50, 51, 52):
        if not session_id:
            raise ValueError(f"session_id is required for event={event}")
        sid = session_id.encode("utf-8")
        header += struct.pack(">I", len(sid))
        header += sid
    if event in (50, 51, 52):
        cid = (connect_id or "").encode("utf-8")
        header += struct.pack(">I", len(cid))
        header += cid
    header += struct.pack(">I", len(payload))
    header += payload
    return bytes(header)


def decode_frame(frame: bytes) -> DecodedFrame:
    if len(frame) < 4:
        raise ValueError("frame too short")
    version_and_header_size = frame[0]
    type_and_flag = frame[1]
    serialization_and_compression = frame[2]
    header_size_bytes = 4 * (version_and_header_size & 0x0F)
    offset = max(4, header_size_bytes)
    msg_type_bits = type_and_flag & 0xF0
    msg_flags = type_and_flag & 0x0F
    serialization_bits = serialization_and_compression & 0xF0
    compression_bits = serialization_and_compression & 0x0F
    contains_event = (msg_flags & MSGTYPE_FLAG_WITH_EVENT) == MSGTYPE_FLAG_WITH_EVENT
    contains_sequence = (msg_flags & 0x01) == 0x01 or (msg_flags & 0x03) == 0x03
    event = None
    session_id = None
    connect_id = None
    error_code = None
    if msg_type_bits == MSGTYPE_ERROR:
        error_code = struct.unpack(">I", frame[offset : offset + 4])[0]
        offset += 4
    if contains_sequence and msg_type_bits in (MSGTYPE_AUDIO_ONLY_CLIENT, MSGTYPE_AUDIO_ONLY_SERVER):
        offset += 4
    if contains_event:
        event = struct.unpack(">i", frame[offset : offset + 4])[0]
        offset += 4
        if event not in (1, 2, 50, 51, 52):
            sid_len = struct.unpack(">I", frame[offset : offset + 4])[0]
            offset += 4
            session_id = frame[offset : offset + sid_len].decode("utf-8") if sid_len else ""
            offset += sid_len
        if event in (50, 51, 52):
            cid_len = struct.unpack(">I", frame[offset : offset + 4])[0]
            offset += 4
            connect_id = frame[offset : offset + cid_len].decode("utf-8") if cid_len else ""
            offset += cid_len
    payload_len = struct.unpack(">I", frame[offset : offset + 4])[0]
    offset += 4
    return DecodedFrame(
        msg_type_bits=msg_type_bits,
        msg_flags=msg_flags,
        serialization_bits=serialization_bits,
        compression_bits=compression_bits,
        event=event,
        session_id=session_id,
        connect_id=connect_id,
        error_code=error_code,
        payload=frame[offset : offset + payload_len],
    )


@dataclass
class InputEvent:
    text: str = ""
    audio: bytes = b""


@dataclass
class DoubaoConfig:
    input_mod: str
    conversation_id: str

    @property
    def compression_bits(self) -> int:
        return COMPRESSION_GZIP

    def headers(self, connect_id: str) -> dict[str, str]:
        headers = {
            "X-Api-Resource-Id": "volc.speech.dialog",
            "X-Api-Connect-Id": connect_id,
        }
        if settings.doubao_app_id and settings.doubao_access_token:
            headers["X-Api-App-ID"] = settings.doubao_app_id
            headers["X-Api-Access-Key"] = settings.doubao_access_token
            headers["X-Api-App-Key"] = "PlgvMymc7f3tQnJ6"
        elif settings.doubao_api_key:
            headers["X-Api-Key"] = settings.doubao_api_key
        return headers

    def start_session_payload(self) -> dict:
        speaker = SC20_VOICES.get(settings.doubao_voice, settings.doubao_voice)
        return {
            "asr": {
                "extra": {
                    "end_smooth_window_ms": 1500,
                    "enable_custom_vad": False,
                },
            },
            "tts": {
                "speaker": speaker,
                "audio_config": {
                    "channel": 1,
                    "format": "pcm_s16le",
                    "sample_rate": settings.output_sample_rate,
                },
            },
            "dialog": {
                "character_manifest": "\n".join(
                    [
                        f"名字：{settings.doubao_bot_name}",
                        settings.doubao_system_prompt,
                        f"说话风格：{settings.doubao_speaking_style}",
                    ]
                ),
                "extra": {
                    "strict_audit": False,
                    "recv_timeout": 120,
                    "input_mod": self.input_mod,
                    "model": settings.doubao_model,
                },
            },
        }


def _decode_payload_text(decoded: DecodedFrame) -> str:
    if not decoded.payload:
        return ""
    try:
        payload = _decompress(decoded.payload, decoded.compression_bits)
    except Exception:
        payload = decoded.payload
    return payload.decode("utf-8", errors="ignore")


async def _send_full_client_event(ws, *, event: int, session_id: str | None, config: DoubaoConfig, payload: dict | bytes) -> None:
    if isinstance(payload, (bytes, bytearray)):
        payload_bytes = bytes(payload)
    else:
        payload_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    await ws.send(
        encode_frame(
            msg_type_bits=MSGTYPE_FULL_CLIENT,
            serialization_bits=SERIALIZATION_JSON,
            event=event,
            session_id=session_id,
            payload=_compress(payload_bytes, config.compression_bits),
            compression_bits=config.compression_bits,
        )
    )


async def _recv_expected(ws, *, expected_event: int, stage: str) -> DecodedFrame:
    frame = await ws.recv()
    if isinstance(frame, str):
        raise RuntimeError(f"Doubao {stage} returned an unexpected text frame")
    decoded = decode_frame(frame)
    if decoded.is_error() or (decoded.is_full_server() and decoded.event == DoubaoEvent.SESSION_FAILED):
        raise RuntimeError(f"Doubao {stage} failed: {_decode_payload_text(decoded)}")
    if not decoded.is_full_server() or decoded.event != expected_event:
        raise RuntimeError(f"Doubao {stage} returned unexpected event={decoded.event}")
    return decoded


async def _connect_doubao(websockets, headers: dict[str, str]):
    try:
        return websockets.connect(settings.doubao_ws_url, additional_headers=headers, proxy=None)
    except TypeError:
        return websockets.connect(settings.doubao_ws_url, extra_headers=headers)


async def _start_doubao_session(ws, *, session_id: str, config: DoubaoConfig) -> None:
    await _send_full_client_event(
        ws,
        event=DoubaoEvent.START_CONNECTION,
        session_id=None,
        config=config,
        payload=b"{}",
    )
    await _recv_expected(ws, expected_event=DoubaoEvent.CONNECTION_STARTED, stage="connection handshake")
    await _send_full_client_event(
        ws,
        event=DoubaoEvent.START_SESSION,
        session_id=session_id,
        config=config,
        payload=config.start_session_payload(),
    )
    await _recv_expected(ws, expected_event=DoubaoEvent.SESSION_STARTED, stage="start session")


async def _send_doubao_inputs(
    ws,
    input_stream: AsyncIterator[InputEvent],
    session_id: str,
    config: DoubaoConfig,
) -> None:
    sent_text_query = False
    async for event in input_stream:
        if event.text:
            sent_text_query = True
            payload = json.dumps({"content": event.text}, ensure_ascii=False).encode("utf-8")
            await ws.send(
                encode_frame(
                    msg_type_bits=MSGTYPE_FULL_CLIENT,
                    serialization_bits=SERIALIZATION_JSON,
                    event=DoubaoEvent.CHAT_TEXT_QUERY,
                    session_id=session_id,
                    payload=_compress(payload, config.compression_bits),
                    compression_bits=config.compression_bits,
                )
            )
            continue
        if not event.audio:
            continue
        await ws.send(
            encode_frame(
                msg_type_bits=MSGTYPE_AUDIO_ONLY_CLIENT,
                serialization_bits=SERIALIZATION_RAW,
                event=DoubaoEvent.TASK_REQUEST,
                session_id=session_id,
                payload=_compress(event.audio, config.compression_bits),
                compression_bits=config.compression_bits,
            )
        )
    return


async def run_doubao_turn(
    input_stream: AsyncIterator[InputEvent],
    *,
    text_mode: bool,
    send_client: Callable[[dict], Awaitable[None]],
    conversation_id: str,
    audio_sink: Callable[[bytes], Awaitable[None]] | None = None,
    input_mod: str | None = None,
) -> tuple[str, str, bytes]:
    auth_mode = _doubao_auth_mode()
    if auth_mode == "missing":
        raise RuntimeError("Missing Doubao credentials in .env. Set DOUBAO_APP_ID and DOUBAO_ACCESS_TOKEN.")

    import websockets

    session_id = str(uuid.uuid4())
    connect_id = str(uuid.uuid4())
    config = DoubaoConfig(input_mod=input_mod or ("text" if text_mode else "audio"), conversation_id=conversation_id)
    pcm_parts: list[bytes] = []
    user_text = ""
    assistant_text = ""
    reply_done = False

    headers = config.headers(connect_id)
    ws_context = await _connect_doubao(websockets, headers)
    try:
        ws = await ws_context.__aenter__()
    except Exception as exc:
        if "401" in str(exc):
            raise RuntimeError(
                "Doubao WebSocket authentication failed with HTTP 401. "
                f"Current auth mode is {auth_mode}; for realtime dialogue, set DOUBAO_APP_ID and DOUBAO_ACCESS_TOKEN in .env."
            ) from exc
        raise
    try:
        await _start_doubao_session(ws, session_id=session_id, config=config)
        sender_task = asyncio.create_task(_send_doubao_inputs(ws, input_stream, session_id, config))
        try:
            while True:
                if sender_task.done() and not text_mode and not pcm_parts and not assistant_text:
                    timeout = 20.0
                else:
                    timeout = 120.0
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=timeout)
                except asyncio.TimeoutError:
                    if not text_mode and sender_task.done():
                        logger.warning(
                            "Doubao audio turn timed out after input finished: user_text=%r assistant_text=%r audio_bytes=%d",
                            user_text,
                            assistant_text,
                            sum(len(part) for part in pcm_parts),
                        )
                        break
                    raise
                if isinstance(message, str):
                    continue
                decoded = decode_frame(message)
                if decoded.is_audio():
                    audio_payload = _decompress(decoded.payload, decoded.compression_bits)
                    if audio_payload:
                        logger.info("Doubao audio frame: bytes=%d", len(audio_payload))
                        pcm_parts.append(audio_payload)
                        if audio_sink is not None:
                            await audio_sink(audio_payload)
                    continue
                if decoded.is_error():
                    payload = _decode_payload_text(decoded)
                    raise RuntimeError(f"Doubao error code={decoded.error_code}: {payload}")
                if not decoded.is_full_server():
                    continue
                try:
                    data = json.loads(_decompress(decoded.payload, decoded.compression_bits) or b"{}")
                except Exception:
                    data = {}
                logger.info("Doubao full event=%s payload_keys=%s", decoded.event, sorted(data.keys()))
                if decoded.event == DoubaoEvent.ASR_RESULT:
                    results = data.get("results", [])
                    if results:
                        text = str(results[0].get("text", "") or "")
                        is_interim = bool(results[0].get("is_interim", True))
                        logger.info("Doubao ASR result: interim=%s text=%r", is_interim, text)
                        if text and not is_interim:
                            user_text = text
                            await send_client({"type": "user_transcript", "text": user_text})
                elif decoded.event == DoubaoEvent.TURN_FINISHED:
                    logger.info("Doubao ASR ended / turn finished: user_text=%r", user_text)
                elif decoded.event == DoubaoEvent.LLM_TOKEN:
                    token = str(data.get("content", "") or "")
                    if token:
                        assistant_text += token
                        await send_client({"type": "assistant_delta", "text": token})
                elif decoded.event == DoubaoEvent.TTS_SENTENCE_DONE and not assistant_text:
                    assistant_text = str(data.get("text", "") or assistant_text)
                    if assistant_text:
                        await send_client({"type": "assistant_delta", "text": assistant_text})
                elif decoded.event == DoubaoEvent.REPLY_DONE:
                    reply_done = True
                    break
                elif decoded.event in (DoubaoEvent.SESSION_FINISHED, DoubaoEvent.SESSION_FAILED):
                    break
        finally:
            if not sender_task.done():
                sender_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await sender_task
    finally:
        if reply_done:
            with contextlib.suppress(Exception):
                await _send_full_client_event(
                    ws,
                    event=DoubaoEvent.FINISH_SESSION,
                    session_id=session_id,
                    config=config,
                    payload=b"{}",
                )
                await _recv_expected(ws, expected_event=DoubaoEvent.SESSION_FINISHED, stage="finish session")
        with contextlib.suppress(Exception):
            await ws_context.__aexit__(None, None, None)

    return user_text, assistant_text, b"".join(pcm_parts)


def _safe_name(filename: str) -> str:
    suffix = Path(filename or "avatar.png").suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        suffix = ".png"
    return f"{uuid.uuid4().hex}{suffix}"


def _runtime_url(path: Path) -> str:
    return "/" + path.relative_to(WEB_DIR).as_posix()


def _write_wav(path: Path, pcm: bytes, *, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


def _pcm16_stats(pcm: bytes) -> dict[str, float | int]:
    if len(pcm) < 2:
        return {"samples": 0, "rms": 0.0, "peak": 0}
    usable = pcm[: len(pcm) - (len(pcm) % 2)]
    audio = np.frombuffer(usable, dtype="<i2").astype("float32")
    if audio.size == 0:
        return {"samples": 0, "rms": 0.0, "peak": 0}
    return {
        "samples": int(audio.size),
        "rms": float(np.sqrt(np.mean(audio * audio))),
        "peak": int(np.max(np.abs(audio))),
    }


@dataclass
class Pcm16Stats:
    chunks: int = 0
    nonzero_chunks: int = 0
    samples: int = 0
    sum_squares: float = 0.0
    peak: int = 0

    def add(self, pcm: bytes) -> None:
        if len(pcm) < 2:
            self.chunks += 1
            return
        usable = pcm[: len(pcm) - (len(pcm) % 2)]
        audio = np.frombuffer(usable, dtype="<i2").astype("float64")
        self.chunks += 1
        if audio.size == 0:
            return
        peak = int(np.max(np.abs(audio)))
        if peak > 0:
            self.nonzero_chunks += 1
        self.samples += int(audio.size)
        self.sum_squares += float(np.sum(audio * audio))
        self.peak = max(self.peak, peak)

    def snapshot(self) -> dict[str, float | int]:
        rms = float(np.sqrt(self.sum_squares / self.samples)) if self.samples else 0.0
        return {
            "chunks": self.chunks,
            "nonzero_chunks": self.nonzero_chunks,
            "samples": self.samples,
            "rms": rms,
            "peak": self.peak,
        }


async def _webm_to_pcm16(webm_chunks: list[bytes]) -> bytes:
    if not webm_chunks:
        return b""
    process = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(settings.input_sample_rate),
        "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate(b"".join(webm_chunks))
    if process.returncode != 0:
        detail = stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"Failed to decode microphone audio with ffmpeg: {detail}")
    return stdout


def _resolve_runtime_path(path_or_url: str) -> Path:
    raw = (path_or_url or "").strip()
    if not raw:
        return ROOT / "assets" / "photo.jpg"
    if raw.startswith("/runtime/"):
        return WEB_DIR / raw.lstrip("/")
    if raw.startswith("/assets/"):
        return ROOT / raw.lstrip("/")
    return Path(raw).expanduser().resolve()


class TurnMessageGate:
    _SYNCED_TYPES = {"user_transcript", "assistant_delta", "assistant_final"}

    def __init__(self, send_client: Callable[[dict], Awaitable[None]]) -> None:
        self._send_client = send_client
        self._pending: list[dict] = []
        self._released = False
        self._lock = asyncio.Lock()

    async def send(self, payload: dict) -> None:
        if payload.get("type") in self._SYNCED_TYPES:
            async with self._lock:
                if not self._released:
                    self._pending.append(dict(payload))
                    return
        await self._send_client(payload)

    async def release(self) -> None:
        async with self._lock:
            if self._released and not self._pending:
                return
            self._released = True
            pending = self._pending
            self._pending = []
        for payload in pending:
            await self._send_client(payload)


app = FastAPI(title="LeapTalk Web")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/runtime", StaticFiles(directory=str(RUNTIME_DIR)), name="runtime")
app.mount("/assets", StaticFiles(directory=str(ROOT / "assets")), name="assets")


@app.on_event("startup")
async def _startup() -> None:
    for path in (AVATAR_DIR, AUDIO_DIR, VIDEO_DIR):
        path.mkdir(parents=True, exist_ok=True)
    logger.info("Doubao auth mode: %s", _doubao_auth_mode())
    logger.info("Warming up LeapTalk model before accepting web connections...")
    await leaptalk_engine.ensure_loaded()
    logger.info("LeapTalk model warmup completed; web server is ready.")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/config")
async def config() -> JSONResponse:
    return JSONResponse(
        {
            "doubao_ready": _doubao_auth_mode() != "missing",
            "doubao_auth_mode": _doubao_auth_mode(),
            "default_avatar_url": "/assets/photo.jpg",
            "lite": settings.lite,
            "steps": settings.steps,
        }
    )


@app.post("/api/avatar")
async def upload_avatar(file: UploadFile = File(...)) -> JSONResponse:
    path = AVATAR_DIR / _safe_name(file.filename or "")
    with path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return JSONResponse({"url": _runtime_url(path), "path": str(path)})


@app.websocket("/ws/chat")
async def chat_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    conversation_id = uuid.uuid4().hex
    avatar_url = "/assets/photo.jpg"
    audio_chunks: list[bytes] = []
    audio_format = "webm"
    audio_turn_task: asyncio.Task | None = None
    audio_input_queue: asyncio.Queue[bytes | None] | None = None
    audio_received_bytes = 0
    recording = False
    send_lock = asyncio.Lock()

    async def send_client(payload: dict) -> None:
        async with send_lock:
            await websocket.send_json(payload)

    async def send_video_chunk(meta: dict, data: bytes) -> None:
        meta_bytes = json.dumps(meta, ensure_ascii=False).encode("utf-8")
        packet = b"LTF4" + struct.pack(">I", len(meta_bytes)) + meta_bytes + data
        async with send_lock:
            await websocket.send_bytes(packet)

    async def run_dialog_turn(
        input_stream: AsyncIterator[InputEvent],
        *,
        text_mode: bool,
        audio_sink: Callable[[bytes], Awaitable[None]],
        send_dialogue: Callable[[dict], Awaitable[None]],
    ) -> tuple[str, str, bytes]:
        return await run_doubao_turn(
            input_stream,
            text_mode=text_mode,
            send_client=send_dialogue,
            conversation_id=conversation_id,
            audio_sink=audio_sink,
            input_mod="text" if text_mode else "audio",
        )

    async def process_turn_text(text: str, image_url: str) -> None:
        async def inputs() -> AsyncIterator[InputEvent]:
            yield InputEvent(text=text)

        message_gate = TurnMessageGate(send_client)

        async def send_synced_video_chunk(meta: dict, data: bytes) -> None:
            await send_video_chunk(meta, data)
            await message_gate.release()

        streamer = LeapTalkTurnStreamer(
            leaptalk_engine,
            turn_id=uuid.uuid4().hex,
            image_path=_resolve_runtime_path(image_url),
            send_client=send_client,
            send_video_chunk=send_synced_video_chunk,
        )
        await streamer.start()
        await send_client({"type": "status", "state": "thinking", "message": "Doubao is replying."})
        try:
            user_text, assistant_text, pcm = await run_dialog_turn(
                inputs(),
                text_mode=True,
                audio_sink=streamer.feed,
                send_dialogue=message_gate.send,
            )
            await finish_turn(user_text or text, assistant_text, pcm, image_url, streamer, message_gate)
        except Exception as exc:
            await streamer.cancel()
            await message_gate.release()
            logger.exception("Text turn failed")
            await send_client({"type": "error", "message": str(exc)})
            await send_client({"type": "status", "state": "idle", "message": "Turn failed. Ready for the next turn."})

    async def process_turn_audio_stream(input_queue: asyncio.Queue[bytes | None], image_url: str) -> None:
        async def inputs() -> AsyncIterator[InputEvent]:
            chunk_size = settings.input_sample_rate // 5 * 2
            pending = bytearray()
            while True:
                item = await input_queue.get()
                if item is None:
                    break
                pending.extend(item)
                while len(pending) >= chunk_size:
                    segment = bytes(pending[:chunk_size])
                    del pending[:chunk_size]
                    yield InputEvent(audio=segment)
                    await asyncio.sleep(0.2)
            if pending:
                yield InputEvent(audio=bytes(pending))
                await asyncio.sleep(max(0.02, len(pending) / 2 / settings.input_sample_rate))
            for _ in range(8):
                yield InputEvent(audio=b"\x00" * chunk_size)
                await asyncio.sleep(0.2)

        message_gate = TurnMessageGate(send_client)

        async def send_synced_video_chunk(meta: dict, data: bytes) -> None:
            await send_video_chunk(meta, data)
            await message_gate.release()

        streamer = LeapTalkTurnStreamer(
            leaptalk_engine,
            turn_id=uuid.uuid4().hex,
            image_path=_resolve_runtime_path(image_url),
            send_client=send_client,
            send_video_chunk=send_synced_video_chunk,
        )
        await streamer.start()
        await send_client({"type": "status", "state": "listening", "message": "Doubao is transcribing and replying."})
        try:
            user_text, assistant_text, pcm = await run_dialog_turn(
                inputs(),
                text_mode=False,
                audio_sink=streamer.feed,
                send_dialogue=message_gate.send,
            )
            await finish_turn(user_text, assistant_text, pcm, image_url, streamer, message_gate)
        except Exception as exc:
            await streamer.cancel()
            await message_gate.release()
            logger.exception("Audio turn failed")
            await send_client({"type": "error", "message": str(exc)})
            await send_client({"type": "status", "state": "idle", "message": "Turn failed. Ready for the next turn."})

    async def finish_turn(
        user_text: str,
        assistant_text: str,
        pcm: bytes,
        image_url: str,
        streamer: LeapTalkTurnStreamer,
        message_gate: TurnMessageGate,
    ) -> None:
        await message_gate.send({"type": "assistant_final", "text": assistant_text})
        if not pcm:
            await streamer.cancel()
            await message_gate.release()
            await send_client({"type": "status", "state": "idle", "message": "No audio returned from Doubao."})
            return
        wav_path = AUDIO_DIR / f"{streamer.turn_id}.wav"
        _write_wav(wav_path, pcm, sample_rate=settings.output_sample_rate)
        try:
            await streamer.finish()
        except Exception as exc:
            await message_gate.release()
            logger.exception("LeapTalk video generation failed")
            await send_client({"type": "error", "message": str(exc)})
            await send_client({"type": "status", "state": "idle", "message": "Video generation failed. Ready for the next turn."})
            return
        await message_gate.release()
        await send_client(
            {
                "type": "turn_complete",
                "audio_url": _runtime_url(wav_path),
                "user_text": user_text,
                "assistant_text": assistant_text,
            }
        )
        await send_client({"type": "status", "state": "idle", "message": "Ready."})

    await send_client({"type": "ready", "message": "LeapTalk WebSocket connected."})
    try:
        while True:
            message = await websocket.receive()
            chunk = message.get("bytes")
            if chunk is not None:
                if recording and audio_input_queue is not None:
                    audio_received_bytes += len(chunk)
                    if audio_received_bytes <= len(chunk):
                        logger.info("First microphone PCM chunk stats=%s", _pcm16_stats(chunk))
                    await audio_input_queue.put(chunk)
                continue
            text_payload = message.get("text")
            if text_payload is None:
                continue
            data = json.loads(text_payload)
            msg_type = data.get("type")
            if data.get("avatar_url"):
                avatar_url = str(data.get("avatar_url"))
            if msg_type == "init":
                avatar_url = str(data.get("avatar_url") or avatar_url)
                await send_client({"type": "status", "state": "idle", "message": "Ready."})
            elif msg_type == "text":
                text = str(data.get("text", "")).strip()
                if text:
                    await process_turn_text(
                        text,
                        str(data.get("avatar_url") or avatar_url),
                    )
            elif msg_type == "audio_start":
                if audio_turn_task is not None and not audio_turn_task.done():
                    await send_client({"type": "status", "state": "busy", "message": "Previous audio turn is still running."})
                    continue
                recording = True
                audio_chunks = []
                audio_format = str(data.get("format") or "webm")
                audio_received_bytes = 0
                audio_input_queue = asyncio.Queue()
                audio_turn_task = asyncio.create_task(
                    process_turn_audio_stream(
                        audio_input_queue,
                        str(data.get("avatar_url") or avatar_url),
                    )
                )
                await send_client({"type": "status", "state": "recording", "message": "Recording microphone audio."})
            elif msg_type == "audio_end":
                recording = False
                chunks = audio_chunks
                audio_chunks = []
                logger.info(
                    "Received microphone turn: format=%s chunks=%d bytes=%d",
                    audio_format,
                    len(chunks),
                    audio_received_bytes if audio_format == "pcm_s16le" else sum(len(chunk) for chunk in chunks),
                )
                if audio_input_queue is not None:
                    await audio_input_queue.put(None)
                    audio_input_queue = None
                if audio_turn_task is not None:
                    task = audio_turn_task
                    audio_turn_task = None
                    await task
                elif chunks:
                    await send_client({"type": "status", "state": "idle", "message": "Legacy microphone path is disabled."})
                else:
                    await send_client({"type": "status", "state": "idle", "message": "No microphone audio was received."})
            elif msg_type == "ping":
                await send_client({"type": "pong"})
    except WebSocketDisconnect:
        if audio_input_queue is not None:
            with contextlib.suppress(Exception):
                await audio_input_queue.put(None)
        if audio_turn_task is not None:
            audio_turn_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await audio_turn_task
        return
    except Exception as exc:
        logger.exception("WebSocket error")
        with contextlib.suppress(Exception):
            await send_client({"type": "error", "message": str(exc)})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web_server:app", host=settings.host, port=settings.port, reload=False)
