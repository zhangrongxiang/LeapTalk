import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


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


def _sanitize_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in name)
    cleaned = cleaned.strip("._")
    return cleaned or "video"


def _reset_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _extract_frames_ffmpeg(video_path: Path, out_dir: Path, sample_id: str, max_frames: int = 0) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-start_number",
        "0",
    ]
    if max_frames and int(max_frames) > 0:
        cmd.extend(["-frames:v", str(int(max_frames))])
    cmd.append(str(out_dir / f"{sample_id}_%06d.png"))
    subprocess.run(cmd, check=True)
    return len(list(out_dir.glob(f"{sample_id}_*.png")))


def _symlink_video(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)


def _build_jobs_from_manifest(manifest_path: Path, limit: int):
    rows = _read_manifest(manifest_path)
    rows = [r for r in rows if str(r.get("status", "")).lower() == "ok"]
    if limit and int(limit) > 0:
        rows = rows[: int(limit)]
    if not rows:
        raise RuntimeError("No valid manifest rows with status=ok")

    real_jobs = []
    fake_jobs = []
    for row in rows:
        sample_id = str(row.get("sample_id", "")).strip() or Path(str(row.get("generated_video", ""))).stem
        sample_id = _sanitize_name(sample_id)
        source_video = _resolve_path(row.get("source_video", ""), manifest_path.parent)
        generated_video = _resolve_path(row.get("generated_video", ""), manifest_path.parent)
        if not source_video.exists():
            raise FileNotFoundError(f"Source video not found: {source_video}")
        if not generated_video.exists():
            raise FileNotFoundError(f"Generated video not found: {generated_video}")
        real_jobs.append({"sample_id": sample_id, "video_path": source_video})
        fake_jobs.append({"sample_id": sample_id, "video_path": generated_video})
    return real_jobs, fake_jobs


def _build_real_jobs_from_dir(real_videos_dir: Path, real_limit: int):
    jobs = []
    real_paths = _list_video_files(real_videos_dir)
    if real_limit and int(real_limit) > 0:
        real_paths = real_paths[: int(real_limit)]
    for video_path in real_paths:
        rel = video_path.relative_to(real_videos_dir)
        sample_id = _sanitize_name(str(rel.with_suffix("")).replace("/", "__"))
        jobs.append({"sample_id": sample_id, "video_path": video_path})
    return jobs


def _populate_frame_dir(jobs, out_dir: Path, image_limit: int):
    meta = []
    total_images = 0
    for job in jobs:
        remaining = 0
        if image_limit and int(image_limit) > 0:
            remaining = int(image_limit) - total_images
            if remaining <= 0:
                break
        extracted = _extract_frames_ffmpeg(job["video_path"], out_dir, job["sample_id"], max_frames=remaining)
        total_images += extracted
        meta.append(
            {
                "sample_id": job["sample_id"],
                "video_path": str(job["video_path"]),
                "frame_pattern": str(out_dir / f"{job['sample_id']}_%06d.png"),
                "num_frames": extracted,
            }
        )
    return meta, total_images


def _populate_video_dir(jobs, out_dir: Path):
    meta = []
    for job in jobs:
        dst = out_dir / f"{job['sample_id']}.mp4"
        _symlink_video(job["video_path"], dst)
        meta.append(
            {
                "sample_id": job["sample_id"],
                "video_path": str(job["video_path"]),
                "linked_video": str(dst),
            }
        )
    return meta


def _run_pytorch_fid(pred_frames_dir: Path, gt_frames_dir: Path, device: str, batch_size: int, num_workers: int):
    cmd = [
        sys.executable,
        "-m",
        "pytorch_fid",
        str(pred_frames_dir),
        str(gt_frames_dir),
        "--device",
        device,
        "--batch-size",
        str(int(batch_size)),
        "--num-workers",
        str(int(num_workers)),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    text = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        raise RuntimeError(
            "pytorch-fid failed.\n"
            f"command: {' '.join(cmd)}\n"
            f"output:\n{text}"
        )
    match = re.search(r"FID:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", text)
    fid_value = float(match.group(1)) if match else None
    return fid_value, text


def _compute_disco_video_metric_with_fallback(
    compute_fn,
    pred_videos: list[str],
    gt_videos: list[str],
    feat_model,
    mode: str,
    sample_duration: int,
    num_workers: int,
):
    try:
        return compute_fn(
            pred_videos,
            gt_videos,
            feat_model,
            mode=mode,
            sample_duration=sample_duration,
            batch_size=1,
            num_workers=num_workers,
        )
    except RuntimeError as exc:
        message = str(exc)
        if num_workers > 0 and ("DataLoader worker" in message or "worker" in message):
            print(
                f"[warn] {mode} failed with num_workers={num_workers}; "
                "retrying with num_workers=0"
            )
            return compute_fn(
                pred_videos,
                gt_videos,
                feat_model,
                mode=mode,
                sample_duration=sample_duration,
                batch_size=1,
                num_workers=0,
            )
        raise


def _run_disco_video_metrics(pred_videos_dir: Path, gt_videos_dir: Path, disco_root_dir: Path, fvd_root_dir: Path, device_name: str, sample_duration: int, num_workers: int):
    if str(disco_root_dir) not in sys.path:
        sys.path.insert(0, str(disco_root_dir))

    import torch
    from tool.metrics.features import build_feature_extractor
    from tool.metrics.metric_center import compute_fid_video_scores

    pred_videos = sorted(pred_videos_dir.glob("*.mp4"))
    gt_videos = sorted(gt_videos_dir.glob("*.mp4"))
    if not pred_videos:
        raise RuntimeError(f"No mp4 files found in {pred_videos_dir}")
    if not gt_videos:
        raise RuntimeError(f"No mp4 files found in {gt_videos_dir}")

    device = torch.device(device_name)
    fid_vid_model = build_feature_extractor(
        mode="FVD-3DRN50",
        root_dir=str(fvd_root_dir),
        device=device,
        sample_duration=sample_duration,
    )
    pred_videos_str = [str(p) for p in pred_videos]
    gt_videos_str = [str(p) for p in gt_videos]

    fid_vid_value = _compute_disco_video_metric_with_fallback(
        compute_fid_video_scores,
        pred_videos_str,
        gt_videos_str,
        fid_vid_model,
        mode="FVD-3DRN50",
        sample_duration=sample_duration,
        num_workers=num_workers,
    )

    fvd_model = build_feature_extractor(
        mode="FVD-3DInception",
        root_dir=str(fvd_root_dir),
        device=device,
        sample_duration=sample_duration,
    )
    fvd_value = _compute_disco_video_metric_with_fallback(
        compute_fid_video_scores,
        pred_videos_str,
        gt_videos_str,
        fvd_model,
        mode="FVD-3DInception",
        sample_duration=sample_duration,
        num_workers=num_workers,
    )

    return {
        "FVD-3DRN50": float(fid_vid_value),
        "FVD-3DInception": float(fvd_value),
        "num_pred_videos": len(pred_videos),
        "num_gt_videos": len(gt_videos),
        "sample_duration": int(sample_duration),
    }


def _parse_args():
    parser = argparse.ArgumentParser(description="Run DisCo-style FID/FVD evaluation on a ViBT batch output directory.")
    parser.add_argument("--output_dir", type=str, default="/mnt/public_2/liusonghua/rxcache/outputs/flashheadlite_vibt_hdtf_batch_infer")
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--real_mode", type=str, default="manifest", choices=["manifest", "dir"])
    parser.add_argument("--real_videos_dir", type=str, default="/mnt/public_2/liusonghua/rxcache/HDTF/videos")
    parser.add_argument("--real_limit", type=int, default=0, help="0 means use all eligible real videos.")
    parser.add_argument("--limit", type=int, default=0, help="0 means use all eligible fake videos from the manifest.")
    parser.add_argument("--real_image_limit", type=int, default=0, help="0 means no cap on real frame count for FID.")
    parser.add_argument("--fake_image_limit", type=int, default=0, help="0 means no cap on fake frame count for FID.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--fid_batch_size", type=int, default=1, help="Use 1 when extracted frame resolutions differ across videos.")
    parser.add_argument("--fid_num_workers", type=int, default=4)
    parser.add_argument("--disco_root_dir", type=str, default="/mnt/public_2/liusonghua/rx/DisCo")
    parser.add_argument("--fvd_root_dir", type=str, default="/mnt/public_2/liusonghua/rxcache/DisCo")
    parser.add_argument("--sample_duration", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0, help="Worker count for DisCo video metrics. 0 is safest for memory.")
    parser.add_argument("--work_dir", type=str, default=None)
    return parser.parse_args()


def _cli_main():
    args = _parse_args()
    output_dir = Path(args.output_dir).resolve()
    manifest_path = Path(args.manifest).resolve() if args.manifest else output_dir / "manifest.json"
    disco_root_dir = Path(args.disco_root_dir).resolve()
    fvd_root_dir = Path(args.fvd_root_dir).resolve()
    work_dir = Path(args.work_dir).resolve() if args.work_dir else output_dir / "disco_style_eval"

    if args.real_mode == "manifest":
        real_jobs, fake_jobs = _build_jobs_from_manifest(manifest_path, args.limit)
    else:
        _, fake_jobs = _build_jobs_from_manifest(manifest_path, args.limit)
        real_jobs = _build_real_jobs_from_dir(Path(args.real_videos_dir).resolve(), args.real_limit)

    gt_frames_dir = work_dir / "gt_frames"
    pred_frames_dir = work_dir / "pred_frames"
    gt_videos_dir = work_dir / "gt_videos"
    pred_videos_dir = work_dir / "pred_videos"
    _reset_dir(gt_frames_dir)
    _reset_dir(pred_frames_dir)
    _reset_dir(gt_videos_dir)
    _reset_dir(pred_videos_dir)

    real_frame_meta, num_real_images = _populate_frame_dir(real_jobs, gt_frames_dir, args.real_image_limit)
    fake_frame_meta, num_fake_images = _populate_frame_dir(fake_jobs, pred_frames_dir, args.fake_image_limit)
    real_video_meta = _populate_video_dir(real_jobs, gt_videos_dir)
    fake_video_meta = _populate_video_dir(fake_jobs, pred_videos_dir)

    fid_value, fid_output = _run_pytorch_fid(
        pred_frames_dir,
        gt_frames_dir,
        args.device,
        batch_size=args.fid_batch_size,
        num_workers=args.fid_num_workers,
    )
    print(f"fidvalue:{fid_value},\n fidoutput:{fid_output}")
    video_metrics = _run_disco_video_metrics(
        pred_videos_dir=pred_videos_dir,
        gt_videos_dir=gt_videos_dir,
        disco_root_dir=disco_root_dir,
        fvd_root_dir=fvd_root_dir,
        device_name=args.device,
        sample_duration=args.sample_duration,
        num_workers=args.num_workers,
    )

    payload = {
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "real_mode": args.real_mode,
        "real_videos_dir": str(Path(args.real_videos_dir).resolve()),
        "work_dir": str(work_dir),
        "num_real_videos": len(real_video_meta),
        "num_fake_videos": len(fake_video_meta),
        "num_real_images": int(num_real_images),
        "num_fake_images": int(num_fake_images),
        "real_image_limit": int(args.real_image_limit),
        "fake_image_limit": int(args.fake_image_limit),
        "fid": fid_value,
        "fid_stdout": fid_output,
        "fid_batch_size": int(args.fid_batch_size),
        "fid_num_workers": int(args.fid_num_workers),
        "video_metrics": video_metrics,
        "paths": {
            "gt_frames_dir": str(gt_frames_dir),
            "pred_frames_dir": str(pred_frames_dir),
            "gt_videos_dir": str(gt_videos_dir),
            "pred_videos_dir": str(pred_videos_dir),
        },
        "notes": {
            "fid": "Uses python -m pytorch_fid on flat frame folders, matching DisCo gen_eval.sh style.",
            "fid_vid": "Uses DisCo metric_center FVD-3DRN50 path on raw mp4 videos.",
            "fvd": "Uses DisCo metric_center FVD-3DInception path on raw mp4 videos.",
        },
    }

    summary_path = work_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"num_real_videos : {payload['num_real_videos']}")
    print(f"num_fake_videos : {payload['num_fake_videos']}")
    print(f"num_real_images : {payload['num_real_images']}")
    print(f"num_fake_images : {payload['num_fake_images']}")
    print(f"fid             : {payload['fid']}")
    print(f"fid_vid         : {payload['video_metrics']['FVD-3DRN50']}")
    print(f"fvd             : {payload['video_metrics']['FVD-3DInception']}")
    print(f"saved_json      : {summary_path}")


if __name__ == "__main__":
    _cli_main()
