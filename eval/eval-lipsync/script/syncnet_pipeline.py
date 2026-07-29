import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import ffmpeg
import numpy as np
import torch
from scipy import signal
from scipy.interpolate import interp1d
from scenedetect import ContentDetector, SceneManager, StatsManager
from scenedetect.video_manager import VideoManager

from detectors.s3fd import S3FD
from detectors.s3fd.nets import S3FDNet
from SyncNetInstance import SyncNetInstance
from SyncNetModel import S

# ---------------------------------------------------------------------- #
# Configuration                                                          #
# ---------------------------------------------------------------------- #
@dataclass
class PipelineConfig:
    # Face-detection / tracking
    facedet_scale: float = 0.25
    crop_scale: float = 0.40
    min_track: int = 50
    frame_rate: int = 25
    num_failed_det: int = 25
    min_face_size: int = 100

    # SyncNet
    batch_size: int = 20
    vshift: int = 15

    # Local weight paths
    s3fd_weights: str = "sfd_face.pth"
    syncnet_weights: str = "syncnet_v2.model"

    # Tools
    ffmpeg_bin: str = "ffmpeg"  # assumes ffmpeg in $PATH
    audio_sample_rate: int = 16000  # resample rate for speech

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})


# ---------------------------------------------------------------------- #
# Pipeline                                                               #
# ---------------------------------------------------------------------- #
class SyncNetPipeline:
    def __init__(
        self,
        cfg: Union[PipelineConfig, Dict[str, Any], None] = None,
        *,
        device: str = "cuda",
        **override,
    ):
        base = cfg if isinstance(cfg, PipelineConfig) else PipelineConfig.from_dict(cfg or {})
        for k, v in override.items():
            if hasattr(base, k):
                setattr(base, k, v)
        self.cfg = base
        self.device = device

        self.s3fd = self._load_s3fd(self.cfg.s3fd_weights)
        self.syncnet = self._load_syncnet(self.cfg.syncnet_weights)

    # ---------------------------- model loading ---------------------------- #
    def _load_s3fd(self, path: str) -> S3FD:
        logging.info(f"Loading S3FD from {path}")
        net = S3FDNet(device=self.device)
        net.load_state_dict(torch.load(path, map_location=self.device))
        net.eval()
        return S3FD(net=net, device=self.device)

    def _load_syncnet(self, path: str) -> SyncNetInstance:
        logging.info(f"Loading SyncNet from {path}")
        model = S()
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.eval()
        return SyncNetInstance(net=model, device=self.device)

    # ---------------------------- helpers ---------------------------------- #
    @staticmethod
    def _iou(a, b):
        xA, yA = max(a[0], b[0]), max(a[1], b[1])
        xB, yB = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = (a[2] - a[0]) * (a[3] - a[1])
        areaB = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (areaA + areaB - inter + 1e-8)

    def _track(self, dets):
        cfg = self.cfg
        tracks = []
        while True:
            t = []
            for faces in dets:
                for f in faces:
                    if not t:
                        t.append(f)
                        faces.remove(f)
                    elif (
                        f["frame"] - t[-1]["frame"] <= cfg.num_failed_det
                        and self._iou(f["bbox"], t[-1]["bbox"]) > 0.5
                    ):
                        t.append(f)
                        faces.remove(f)
                        continue
                    else:
                        break
            if not t:
                break
            if len(t) > cfg.min_track:
                fr = np.array([d["frame"] for d in t])
                bb = np.array([d["bbox"] for d in t])
                full_f = np.arange(fr[0], fr[-1] + 1)
                bb_i = np.stack([interp1d(fr, bb[:, i])(full_f) for i in range(4)], 1)
                if max(
                    np.mean(bb_i[:, 2] - bb_i[:, 0]),
                    np.mean(bb_i[:, 3] - bb_i[:, 1]),
                ) > cfg.min_face_size:
                    tracks.append({"frame": full_f, "bbox": bb_i})
        return tracks
    
    def _crop(self, track, frames, audio_wav, base):
        cfg = self.cfg
        base.parent.mkdir(parents=True, exist_ok=True)
        tmp_avi = f"{base}t.avi"
        vw = cv2.VideoWriter(tmp_avi, cv2.VideoWriter_fourcc(*"XVID"), cfg.frame_rate, (224, 224))

        s, x, y = [], [], []
        for b in track["bbox"]:
            s.append(max(b[3] - b[1], b[2] - b[0]) / 2)
            x.append((b[0] + b[2]) / 2)
            y.append((b[1] + b[3]) / 2)
        s, x, y = map(lambda v: signal.medfilt(v, 13), (s, x, y))

        for i, fidx in enumerate(track["frame"]):
            img = cv2.imread(frames[fidx])
            if img is None:
                continue
            bs = s[i]
            cs = cfg.crop_scale
            pad = int(bs * (1 + 2 * cs))
            img_p = cv2.copyMakeBorder(
                img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(110, 110, 110)
            )
            my, mx = y[i] + pad, x[i] + pad
            y1, y2 = int(my - bs), int(my + bs * (1 + 2 * cs))
            x1, x2 = int(mx - bs * (1 + cs)), int(mx + bs * (1 + cs))
            crop = cv2.resize(img_p[y1:y2, x1:x2], (224, 224))
            vw.write(crop)
        vw.release()

        slice_wav = f"{base}.wav"
        ss = track["frame"][0] / cfg.frame_rate
        to = (track["frame"][-1] + 1) / cfg.frame_rate
        subprocess.call(
            f'{cfg.ffmpeg_bin} -y -i "{audio_wav}" -ss {ss:.3f} -to {to:.3f} "{slice_wav}"',
            shell=True,
        )

        final_avi = f"{base}.avi"
        subprocess.call(
            f'{cfg.ffmpeg_bin} -y -i "{tmp_avi}" -i "{slice_wav}" -c:v copy -c:a copy "{final_avi}"',
            shell=True,
        )
        os.remove(tmp_avi)
        return final_avi

    # ---------------------------- inference -------------------------------- #
    def inference(
        self,
        video_path: str,  # We do not extract audio from video_path!
        audio_path: str,
        *,
        cache_dir: Optional[str] = None,
    ) -> Tuple[List[int], List[float], List[float], float, float, str, bool]:
        cfg = self.cfg
        work = Path(cache_dir) if cache_dir else Path(tempfile.mkdtemp())
        if cache_dir:
            work.mkdir(parents=True, exist_ok=True)

        try:
            # 1) Convert video to constant-fps AVI
            avi = work / "video.avi"
            (
                ffmpeg.input(video_path)
                .output(str(avi), **{"q:v": 2}, r=cfg.frame_rate, **{"async": 1})
                .overwrite_output()
                .run()
            )

            # 2) Extract frames
            frames_dir = work / "frames"
            frames_dir.mkdir(exist_ok=True)
            (
                ffmpeg.input(str(avi))
                .output(str(frames_dir / "%06d.jpg"), **{"q:v": 2}, f="image2", threads=1)
                .overwrite_output()
                .run()
            )
            frames = sorted(glob(str(frames_dir / "*.jpg")))

            # 3) Resample speech
            audio_wav = work / "speech.wav"
            (
                ffmpeg.input(audio_path)
                .output(str(audio_wav), ac=1, ar=cfg.audio_sample_rate, format="wav")
                .overwrite_output()
                .run()
            )

            # 4) Face detection
            detections = []
            for i, fp in enumerate(frames):
                img = cv2.imread(fp)
                boxes = (
                    self.s3fd.detect_faces(
                        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                        conf_th=0.9,
                        scales=[cfg.facedet_scale],
                    )
                    if img is not None
                    else []
                )
                detections.append(
                    [
                        {"frame": i, "bbox": b[:-1].tolist(), "conf": float(b[-1])}
                        for b in boxes
                    ]
                )

            flat = [f for fs in detections for f in fs]
            s3fd_json = json.dumps(flat) if flat else ""
            has_face = bool(flat)

            # 5) Scene detection
            vm = VideoManager([str(avi)])
            sm = SceneManager(StatsManager())
            sm.add_detector(ContentDetector())
            vm.start()
            sm.detect_scenes(frame_source=vm)
            scenes = sm.get_scene_list(vm.get_base_timecode()) or [
                (vm.get_base_timecode(), vm.get_current_timecode())
            ]

            # 6) Track faces
            tracks = []
            for sc in scenes:
                s, e = sc[0].frame_num, sc[1].frame_num
                if e - s >= cfg.min_track:
                    tracks.extend(self._track([lst.copy() for lst in detections[s:e]]))

            # 7) Crop tracks
            crops = [
                self._crop(t, frames, str(audio_wav), Path(work) / "cropped" / f"{i:05d}") for i, t in enumerate(tracks)
            ]
            # AV offset:      5
            # Min dist:       5.370
            # Confidence:     9.892

            # crops = [work / ".." / ".."/ "data" / "example.avi"]
            # AV offset:      3
            # Min dist:       5.348
            # Confidence:     10.081
            
            # crops = [work / "video.avi"]
            # AV offset:      3
            # Min dist:       6.668
            # Confidence:     8.337

            # 8) SyncNet evaluation
            offsets, confs, dists = [], [], []
            class Opt: ...
            for i, cp in enumerate(crops):
                crop_dir = work / "cropped" / f"crop_{i:05d}"
                frames_dir = crop_dir
                frames_dir.mkdir(parents=True, exist_ok=True)
                audio_path = crop_dir / "audio.wav"

                # Extract frames
                (
                    ffmpeg.input(cp)
                    .output(str(frames_dir / "%06d.jpg"), f="image2", threads=1)
                    .overwrite_output()
                    .run()
                )
                
                # Extract audio
                (
                    ffmpeg.input(cp)
                    .output(
                        str(audio_path),
                        ac=1,
                        vn=None,
                        acodec="pcm_s16le",
                        ar=16000,
                        af="aresample=async=1",
                    )
                    .overwrite_output()
                    .run()
                )

                opt = Opt()
                opt.tmp_dir = str(crop_dir)
                opt.batch_size = cfg.batch_size
                opt.vshift = cfg.vshift

                off, conf, dist = self.syncnet.evaluate(opt=opt)
                offsets.append(off)
                confs.append(conf)
                dists.append(dist)

            if not offsets:
                return ([], [], [], 0.0, 0.0, "", False)

            return offsets, confs, dists, max(confs), min(dists), s3fd_json, has_face

        finally:
            if not cache_dir:
                shutil.rmtree(work, ignore_errors=True)
