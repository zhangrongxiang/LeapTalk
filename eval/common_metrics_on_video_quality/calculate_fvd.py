import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


def trans(x):
    # if greyscale images add channel
    if x.shape[-3] == 1:
        x = x.repeat(1, 1, 3, 1, 1)

    # permute BTCHW -> BCTHW
    x = x.permute(0, 2, 1, 3, 4) 

    return x

def calculate_fvd(videos1, videos2, device, method='styleganv', only_final=False):

    if method == 'styleganv':
        from fvd.styleganv.fvd import get_fvd_feats, frechet_distance, load_i3d_pretrained
    elif method == 'videogpt':
        from fvd.videogpt.fvd import load_i3d_pretrained, frechet_distance
        from fvd.videogpt.fvd import get_fvd_logits as get_fvd_feats

    print("calculate_fvd...")

    # videos [batch_size, timestamps, channel, h, w]
    
    assert videos1.ndim == 5 and videos2.ndim == 5
    assert videos1.shape[1:] == videos2.shape[1:], (
        "Video tensors must share [T, C, H, W] after sampling/resizing; "
        f"got {tuple(videos1.shape)} vs {tuple(videos2.shape)}"
    )

    i3d = load_i3d_pretrained(device=device)
    fvd_results = []

    # support grayscale input, if grayscale -> channel*3
    # BTCHW -> BCTHW
    # videos -> [batch_size, channel, timestamps, h, w]

    videos1 = trans(videos1)
    videos2 = trans(videos2)

    fvd_results = []

    if only_final:

        assert videos1.shape[2] >= 10, "for calculate FVD, each clip_timestamp must >= 10"

        # videos_clip [batch_size, channel, timestamps, h, w]
        videos_clip1 = videos1
        videos_clip2 = videos2

        # get FVD features
        feats1 = get_fvd_feats(videos_clip1, i3d=i3d, device=device)
        feats2 = get_fvd_feats(videos_clip2, i3d=i3d, device=device)

        # calculate FVD
        fvd_results.append(frechet_distance(feats1, feats2))
    
    else:

        # for calculate FVD, each clip_timestamp must >= 10
        for clip_timestamp in tqdm(range(10, videos1.shape[-3]+1)):
        
            # get a video clip
            # videos_clip [batch_size, channel, timestamps[:clip], h, w]
            videos_clip1 = videos1[:, :, : clip_timestamp]
            videos_clip2 = videos2[:, :, : clip_timestamp]

            # get FVD features
            feats1 = get_fvd_feats(videos_clip1, i3d=i3d, device=device)
            feats2 = get_fvd_feats(videos_clip2, i3d=i3d, device=device)
        
            # calculate FVD when timestamps[:clip]
            fvd_results.append(frechet_distance(feats1, feats2))

    result = {
        "value": fvd_results,
    }

    return result


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


def _sample_video_to_tchw(video_rgb: np.ndarray, num_frames: int, size: int) -> torch.Tensor:
    total = int(video_rgb.shape[0])
    if total <= 0:
        raise RuntimeError("Video has no frames")
    if total == num_frames:
        sample = video_rgb
    else:
        idx = np.linspace(0, total - 1, num_frames).round().astype(np.int64)
        sample = video_rgb[idx]

    tensor = torch.from_numpy(sample).permute(0, 3, 1, 2).float() / 255.0
    tensor = torch.nn.functional.interpolate(
        tensor,
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )
    return tensor


def load_videos(
    manifest_path,
    real_videos_dir,
    num_frames=16,
    size=224,
    limit=0,
):
    manifest_path = Path(manifest_path).resolve()
    real_videos_dir = Path(real_videos_dir).resolve()
    rows = _read_manifest(manifest_path)
    rows = [r for r in rows if str(r.get("status", "")).lower() == "ok"]
    if limit and int(limit) > 0:
        rows = rows[: int(limit)]
    if not rows:
        raise RuntimeError("No valid manifest rows with status=ok")

    real_videos = []
    generated_videos = []
    fake_meta = []

    real_paths = _list_video_files(real_videos_dir)

    for real_path in tqdm(real_paths, desc="loading real videos"):
        real_rgb = _load_video_rgb(real_path)
        real_videos.append(_sample_video_to_tchw(real_rgb, num_frames=num_frames, size=size))

    for row in tqdm(rows, desc="loading generated videos"):
        sample_id = str(row.get("sample_id", "")).strip() or Path(str(row.get("generated_video", ""))).stem
        generated_path = _resolve_path(row.get("generated_video", ""), manifest_path.parent)

        if not generated_path.exists():
            raise FileNotFoundError(f"Generated video not found: {generated_path}")

        gen_rgb = _load_video_rgb(generated_path)

        generated_videos.append(_sample_video_to_tchw(gen_rgb, num_frames=num_frames, size=size))
        fake_meta.append(
            {
                "sample_id": sample_id,
                "generated_video": str(generated_path),
                "generated_frames": int(gen_rgb.shape[0]),
            }
        )

    videos1 = torch.stack(real_videos, dim=0)
    videos2 = torch.stack(generated_videos, dim=0)
    return videos1, videos2, real_paths, fake_meta


def _parse_args():
    parser = argparse.ArgumentParser(description="Calculate FVD from a ViBT batch-inference manifest.")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--real_videos_dir", type=str, default="/mnt/public_2/liusonghua/rxcache/HDTF/videos")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--method", type=str, default="videogpt", choices=["styleganv", "videogpt"])
    parser.add_argument("--only_final", action="store_true", help="Only compute final FVD value instead of per-prefix values.")
    parser.add_argument("--num_frames", type=int, default=16, help="Uniformly sample this many frames from each video.")
    parser.add_argument("--size", type=int, default=224, help="Resize sampled frames to this square size before FVD.")
    parser.add_argument("--limit", type=int, default=0, help="0 means use all valid fake rows in the manifest.")
    parser.add_argument("--out_json", type=str, default=None)
    return parser.parse_args()


def _cli_main():
    args = _parse_args()
    device = torch.device(args.device)
    videos1, videos2, real_paths, fake_meta = load_videos(
        args.manifest,
        args.real_videos_dir,
        num_frames=args.num_frames,
        size=args.size,
        limit=args.limit,
    )
    result = calculate_fvd(
        videos1,
        videos2,
        device=device,
        method=args.method,
        only_final=bool(args.only_final),
    )

    payload = {
        "manifest": str(Path(args.manifest).resolve()),
        "real_videos_dir": str(Path(args.real_videos_dir).resolve()),
        "num_real_samples": int(videos1.shape[0]),
        "num_fake_samples": int(videos2.shape[0]),
        "num_frames": int(videos1.shape[1]),
        "size": int(args.size),
        "method": args.method,
        "only_final": bool(args.only_final),
        "fvd": result["value"],
        "real_videos": [str(p) for p in real_paths],
        "fake_meta": fake_meta,
        "notes": {
            "real_distribution": "All mp4 videos found recursively under real_videos_dir are used as the real distribution.",
            "fake_distribution": "Generated videos come from manifest rows with status=ok; --limit only truncates fake/manifest rows.",
            "sample_count_constraint": "Real/fake video counts may differ; FVD compares distributions, not paired samples.",
        },
    }

    if args.out_json:
        out_path = Path(args.out_json).resolve()
    else:
        out_path = Path(args.manifest).resolve().with_name(f"fvd_{args.method}.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"num_real_samples: {payload['num_real_samples']}")
    print(f"num_fake_samples: {payload['num_fake_samples']}")
    print(f"num_frames : {payload['num_frames']}")
    print(f"method     : {payload['method']}")
    print(f"fvd        : {payload['fvd']}")
    print(f"saved_json : {out_path}")

if __name__ == "__main__":
    _cli_main()
