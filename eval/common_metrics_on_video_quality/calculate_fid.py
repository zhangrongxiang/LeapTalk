import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.functional import adaptive_avg_pool2d
from tqdm import tqdm


THIS_DIR = Path(__file__).resolve().parent
PYTORCH_FID_SRC = THIS_DIR.parent / "pytorch-fid" / "src"
if str(PYTORCH_FID_SRC) not in sys.path:
    sys.path.insert(0, str(PYTORCH_FID_SRC))

from pytorch_fid.fid_score import calculate_frechet_distance  # type: ignore
from pytorch_fid.inception import InceptionV3  # type: ignore


def _resolve_path(value, base_dir: Path) -> Path:
    p = Path(str(value))
    if p.is_absolute():
        return p

    candidates = [(base_dir / p).resolve()]
    for parent in base_dir.parents:
        candidates.append((parent / p).resolve())
    candidates.append((Path.cwd() / p).resolve())

    seen = set()
    ordered = []
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            ordered.append(candidate)
    for candidate in ordered:
        if candidate.exists():
            return candidate
    return ordered[0]


def _read_manifest(path: Path):
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise RuntimeError("Manifest JSON must be a list of row dicts")
        return data
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    raise RuntimeError(f"Unsupported manifest format: {path}")


def _list_video_files(video_dir: Path):
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")
    files = sorted(video_dir.rglob("*.mp4"))
    if not files:
        raise RuntimeError(f"No mp4 files found in {video_dir}")
    return files


def _load_video_rgb(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return np.stack(frames, axis=0)


def _sample_frames(video_rgb: np.ndarray, frame_stride: int, max_frames: int) -> np.ndarray:
    frames = video_rgb[:: max(1, int(frame_stride))]
    if frames.shape[0] > max_frames:
        idx = np.linspace(0, frames.shape[0] - 1, max_frames).round().astype(np.int64)
        frames = frames[idx]
    return frames


def _build_inception(device):
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    model.eval()
    return model


def _extract_features(model, frames_rgb: np.ndarray, batch_size: int, device) -> torch.Tensor:
    if frames_rgb.size == 0:
        return torch.empty((0, 2048), dtype=torch.float32)

    tensor = torch.from_numpy(frames_rgb).permute(0, 3, 1, 2).float() / 255.0
    tensor = F.interpolate(tensor, size=(299, 299), mode="bilinear", align_corners=False)

    feats = []
    with torch.no_grad():
        for start in range(0, tensor.shape[0], batch_size):
            batch = tensor[start:start + batch_size].to(device)
            pred = model(batch)[0]
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
            pred = pred.squeeze(3).squeeze(2).cpu()
            feats.append(pred.float())
    return torch.cat(feats, dim=0)


def _compute_stats(features: torch.Tensor):
    feats = features.cpu().numpy()
    mu = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def calculate_fid(feats_source: torch.Tensor, feats_generated: torch.Tensor) -> float:
    mu1, sigma1 = _compute_stats(feats_source)
    mu2, sigma2 = _compute_stats(feats_generated)
    return float(calculate_frechet_distance(mu1, sigma1, mu2, sigma2))


def _parse_args():
    parser = argparse.ArgumentParser(description="Calculate frame-level FID from a ViBT batch-inference manifest.")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--real_videos_dir", type=str, default="/mnt/public_2/liusonghua/rxcache/HDTF/videos")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--frame_stride", type=int, default=5, help="Sample every Nth frame from each video.")
    parser.add_argument("--max_frames_per_video", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0, help="0 means use all valid rows in the manifest.")
    parser.add_argument("--out_json", type=str, default=None)
    return parser.parse_args()


def _cli_main():
    args = _parse_args()
    manifest_path = Path(args.manifest).resolve()
    real_videos_dir = Path(args.real_videos_dir).resolve()
    rows = _read_manifest(manifest_path)
    rows = [r for r in rows if str(r.get("status", "")).lower() == "ok"]
    if args.limit and int(args.limit) > 0:
        rows = rows[: int(args.limit)]
    if not rows:
        raise RuntimeError("No valid manifest rows with status=ok")

    device = torch.device(args.device)
    model = _build_inception(device)

    source_features = []
    generated_features = []
    meta = []

    real_paths = _list_video_files(real_videos_dir)

    for real_path in tqdm(real_paths, desc="loading real videos"):
        source_rgb = _load_video_rgb(real_path)
        source_frames = _sample_frames(source_rgb, args.frame_stride, args.max_frames_per_video)
        source_features.append(_extract_features(model, source_frames, args.batch_size, device))

    for row in tqdm(rows, desc="loading generated videos"):
        sample_id = str(row.get("sample_id", "")).strip() or Path(str(row.get("generated_video", ""))).stem
        generated_path = _resolve_path(row.get("generated_video", ""), manifest_path.parent)

        if not generated_path.exists():
            raise FileNotFoundError(f"Generated video not found: {generated_path}")

        gen_rgb = _load_video_rgb(generated_path)

        gen_frames = _sample_frames(gen_rgb, args.frame_stride, args.max_frames_per_video)
        generated_features.append(_extract_features(model, gen_frames, args.batch_size, device))
        meta.append(
            {
                "sample_id": sample_id,
                "generated_video": str(generated_path),
                "generated_frames": int(gen_rgb.shape[0]),
                "fid_generated_sampled_frames": int(gen_frames.shape[0]),
            }
        )

    source_all = torch.cat(source_features, dim=0)
    generated_all = torch.cat(generated_features, dim=0)
    fid_value = calculate_fid(source_all, generated_all)

    payload = {
        "manifest": str(manifest_path),
        "real_videos_dir": str(real_videos_dir),
        "num_real_videos": len(real_paths),
        "num_fake_videos": len(rows),
        "frame_stride": int(args.frame_stride),
        "max_frames_per_video": int(args.max_frames_per_video),
        "fid_feature_count_source": int(source_all.shape[0]),
        "fid_feature_count_generated": int(generated_all.shape[0]),
        "fid": fid_value,
        "meta": meta,
        "notes": {
            "implementation": "Uses local pytorch-fid InceptionV3 feature extraction and calculate_frechet_distance.",
            "real_distribution": "All mp4 videos found recursively under real_videos_dir are used as the real distribution.",
            "fake_distribution": "Generated videos come from manifest rows with status=ok; --limit only truncates fake/manifest rows.",
            "sample_count_constraint": "Source/generated feature counts may differ; FID compares distributions, not paired samples.",
        },
    }

    if args.out_json:
        out_path = Path(args.out_json).resolve()
    else:
        out_path = manifest_path.with_name("fid.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"num_real_videos            : {payload['num_real_videos']}")
    print(f"num_fake_videos            : {payload['num_fake_videos']}")
    print(f"fid_feature_count_source   : {payload['fid_feature_count_source']}")
    print(f"fid_feature_count_generated: {payload['fid_feature_count_generated']}")
    print(f"fid                        : {payload['fid']}")
    print(f"saved_json                 : {out_path}")


if __name__ == "__main__":
    _cli_main()
