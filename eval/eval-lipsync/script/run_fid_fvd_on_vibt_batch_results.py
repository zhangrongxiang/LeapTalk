from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import Inception_V3_Weights, inception_v3
from torchvision.models.video import R3D_18_Weights, r3d_18


SCRIPT_DIR = Path(__file__).resolve().parent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute dataset-level FID and video Fréchet distance for ViBT batch inference results.",
    )
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest.csv or manifest.json from batch_infer_flashhead_vibt_hdtf.py")
    parser.add_argument("--out_dir", type=str, default=None, help="Default: <manifest_dir>/fid_fvd_eval")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit", type=int, default=0, help="0 means use all valid rows")
    parser.add_argument("--frame_stride", type=int, default=5, help="Sample every N frames per video for FID")
    parser.add_argument("--max_frames_per_video_fid", type=int, default=32)
    parser.add_argument("--video_num_frames", type=int, default=16, help="Uniformly sampled frames per video for FVD backbone")
    parser.add_argument("--video_size", type=int, default=112, help="Spatial size for FVD backbone input")
    parser.add_argument("--fid_batch_size", type=int, default=32)
    parser.add_argument("--fvd_batch_size", type=int, default=8)
    return parser.parse_args()


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _resolve_path(value: Any, base_dir: Path) -> Path:
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


def _sample_frames_for_fid(video_rgb: np.ndarray, frame_stride: int, max_frames: int) -> np.ndarray:
    frames = video_rgb[:: max(1, int(frame_stride))]
    if frames.shape[0] > max_frames:
        idx = np.linspace(0, frames.shape[0] - 1, max_frames).round().astype(np.int64)
        frames = frames[idx]
    return frames


def _sample_clip_for_fvd(video_rgb: np.ndarray, num_frames: int) -> np.ndarray:
    total = video_rgb.shape[0]
    if total == num_frames:
        return video_rgb
    idx = np.linspace(0, total - 1, num_frames).round().astype(np.int64)
    return video_rgb[idx]


def _resize_frames_tensor(frames: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(frames, size=(size, size), mode="bilinear", align_corners=False)


def _build_inception(device: torch.device) -> torch.nn.Module:
    model = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=False, transform_input=False)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    return model


def _build_r3d18(device: torch.device) -> torch.nn.Module:
    model = r3d_18(weights=R3D_18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    return model


def _extract_fid_features(
    model: torch.nn.Module,
    frames_rgb: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if frames_rgb.size == 0:
        return torch.empty((0, 2048), device=device)
    tensor = torch.from_numpy(frames_rgb).permute(0, 3, 1, 2).float() / 255.0
    tensor = _resize_frames_tensor(tensor, 299)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std

    feats = []
    with torch.no_grad():
        for start in range(0, tensor.shape[0], batch_size):
            batch = tensor[start:start + batch_size].to(device)
            feat = model(batch)
            if isinstance(feat, (tuple, list)):
                feat = feat[0]
            feats.append(feat.float())
    return torch.cat(feats, dim=0)


def _extract_fvd_feature(
    model: torch.nn.Module,
    clip_rgb: np.ndarray,
    size: int,
    device: torch.device,
) -> torch.Tensor:
    tensor = torch.from_numpy(clip_rgb).permute(3, 0, 1, 2).float() / 255.0
    tensor = tensor.unsqueeze(0)
    tensor = F.interpolate(tensor, size=(clip_rgb.shape[0], size, size), mode="trilinear", align_corners=False)
    mean = torch.tensor([0.43216, 0.394666, 0.37645], dtype=tensor.dtype).view(1, 3, 1, 1, 1)
    std = torch.tensor([0.22803, 0.22145, 0.216989], dtype=tensor.dtype).view(1, 3, 1, 1, 1)
    tensor = (tensor - mean) / std
    with torch.no_grad():
        feat = model(tensor.to(device))
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
    return feat.float().squeeze(0)


def _compute_stats(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if features.ndim != 2:
        raise RuntimeError(f"Expected 2D features, got shape {tuple(features.shape)}")
    mu = features.mean(dim=0)
    centered = features - mu
    cov = centered.T @ centered / max(features.shape[0] - 1, 1)
    return mu, cov


def _trace_sqrt_product(cov1: torch.Tensor, cov2: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dim = cov1.shape[0]
    eye = torch.eye(dim, device=cov1.device, dtype=cov1.dtype)
    cov1 = cov1 + eps * eye
    cov2 = cov2 + eps * eye

    evals1, evecs1 = torch.linalg.eigh(cov1)
    evals1 = torch.clamp(evals1, min=0)
    cov1_sqrt = (evecs1 * evals1.sqrt().unsqueeze(0)) @ evecs1.T
    middle = cov1_sqrt @ cov2 @ cov1_sqrt
    evals_mid = torch.linalg.eigvalsh(middle)
    evals_mid = torch.clamp(evals_mid, min=0)
    return evals_mid.sqrt().sum()


def _frechet_distance(feats1: torch.Tensor, feats2: torch.Tensor) -> float:
    mu1, cov1 = _compute_stats(feats1)
    mu2, cov2 = _compute_stats(feats2)
    diff = mu1 - mu2
    trace_sqrt = _trace_sqrt_product(cov1, cov2)
    value = diff.dot(diff) + torch.trace(cov1) + torch.trace(cov2) - 2.0 * trace_sqrt
    return float(torch.clamp(value, min=0).item())


def main() -> None:
    args = _parse_args()
    manifest_path = Path(args.manifest).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else manifest_path.parent / "fid_fvd_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_manifest(manifest_path)
    valid_rows = [r for r in rows if str(r.get("status", "")).lower() == "ok"]
    if args.limit > 0:
        valid_rows = valid_rows[: int(args.limit)]
    if not valid_rows:
        raise SystemExit("No valid manifest rows with status=ok to evaluate.")

    device = torch.device(args.device)
    fid_model = _build_inception(device)
    fvd_model = _build_r3d18(device)

    gen_fid_features = []
    gt_fid_features = []
    gen_fvd_features = []
    gt_fvd_features = []
    per_video_rows: list[dict[str, Any]] = []

    for row in valid_rows:
        sample_id = str(row.get("sample_id", "")).strip() or Path(str(row.get("generated_video", ""))).stem
        generated_video = _resolve_path(row.get("generated_video", ""), manifest_path.parent)
        source_video = _resolve_path(row.get("source_video", ""), manifest_path.parent)

        if not generated_video.exists():
            raise FileNotFoundError(f"Generated video not found: {generated_video}")
        if not source_video.exists():
            raise FileNotFoundError(f"Source video not found: {source_video}")

        gen_video = _load_video_rgb(generated_video)
        gt_video = _load_video_rgb(source_video)

        gen_frames = _sample_frames_for_fid(gen_video, args.frame_stride, args.max_frames_per_video_fid)
        gt_frames = _sample_frames_for_fid(gt_video, args.frame_stride, args.max_frames_per_video_fid)
        gen_fid_feat = _extract_fid_features(fid_model, gen_frames, args.fid_batch_size, device)
        gt_fid_feat = _extract_fid_features(fid_model, gt_frames, args.fid_batch_size, device)
        gen_fid_features.append(gen_fid_feat)
        gt_fid_features.append(gt_fid_feat)

        gen_clip = _sample_clip_for_fvd(gen_video, args.video_num_frames)
        gt_clip = _sample_clip_for_fvd(gt_video, args.video_num_frames)
        gen_fvd_feat = _extract_fvd_feature(fvd_model, gen_clip, args.video_size, device)
        gt_fvd_feat = _extract_fvd_feature(fvd_model, gt_clip, args.video_size, device)
        gen_fvd_features.append(gen_fvd_feat.unsqueeze(0))
        gt_fvd_features.append(gt_fvd_feat.unsqueeze(0))

        per_video_rows.append(
            {
                "sample_id": sample_id,
                "source_video": str(source_video),
                "generated_video": str(generated_video),
                "gen_total_frames": int(gen_video.shape[0]),
                "gt_total_frames": int(gt_video.shape[0]),
                "fid_frame_count_gen": int(gen_frames.shape[0]),
                "fid_frame_count_gt": int(gt_frames.shape[0]),
            }
        )
        print(
            f"[{sample_id}] frames(gen/gt)={gen_video.shape[0]}/{gt_video.shape[0]} | "
            f"fid_frames={gen_frames.shape[0]}/{gt_frames.shape[0]}"
        )

    gen_fid_all = torch.cat(gen_fid_features, dim=0)
    gt_fid_all = torch.cat(gt_fid_features, dim=0)
    gen_fvd_all = torch.cat(gen_fvd_features, dim=0)
    gt_fvd_all = torch.cat(gt_fvd_features, dim=0)

    fid_value = _frechet_distance(gt_fid_all, gen_fid_all)
    fvd_value = _frechet_distance(gt_fvd_all, gen_fvd_all)

    summary = {
        "manifest": str(manifest_path),
        "sample_count": len(valid_rows),
        "fid": fid_value,
        "fvd_r3d18": fvd_value,
        "fid_feature_count_gen": int(gen_fid_all.shape[0]),
        "fid_feature_count_gt": int(gt_fid_all.shape[0]),
        "fvd_feature_count_gen": int(gen_fvd_all.shape[0]),
        "fvd_feature_count_gt": int(gt_fvd_all.shape[0]),
        "args": vars(args),
        "notes": {
            "fid": "Frame-level FID computed with torchvision InceptionV3 pool features.",
            "fvd_r3d18": "Video Fréchet distance computed with torchvision r3d_18 features. This is not the canonical I3D-FVD implementation.",
        },
    }

    per_video_csv = out_dir / "per_video.csv"
    summary_json = out_dir / "summary.json"
    _write_csv(per_video_csv, per_video_rows)
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nSummary")
    print(f"sample_count : {summary['sample_count']}")
    print(f"fid          : {summary['fid']:.6f}")
    print(f"fvd_r3d18    : {summary['fvd_r3d18']:.6f}")
    print(f"per_video_csv: {per_video_csv}")
    print(f"summary_json : {summary_json}")


if __name__ == "__main__":
    main()
