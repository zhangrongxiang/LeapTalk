import argparse
import csv
import json
import subprocess
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


def _build_real_jobs(real_videos_dir: Path, real_limit: int):
    jobs = []
    real_paths = _list_video_files(real_videos_dir)
    if real_limit and int(real_limit) > 0:
        real_paths = real_paths[: int(real_limit)]
    for video_path in real_paths:
        rel = video_path.relative_to(real_videos_dir)
        stem = rel.with_suffix("")
        job_name = _sanitize_name(str(stem).replace("/", "__"))
        jobs.append(
            {
                "sample_id": job_name,
                "video_path": video_path,
            }
        )
    return jobs


def _build_fake_jobs(manifest_path: Path, limit: int):
    rows = _read_manifest(manifest_path)
    rows = [r for r in rows if str(r.get("status", "")).lower() == "ok"]
    if limit and int(limit) > 0:
        rows = rows[: int(limit)]
    if not rows:
        raise RuntimeError("No valid manifest rows with status=ok")

    jobs = []
    for row in rows:
        sample_id = str(row.get("sample_id", "")).strip() or Path(str(row.get("generated_video", ""))).stem
        generated_path = _resolve_path(row.get("generated_video", ""), manifest_path.parent)
        if not generated_path.exists():
            raise FileNotFoundError(f"Generated video not found: {generated_path}")
        jobs.append(
            {
                "sample_id": _sanitize_name(sample_id),
                "video_path": generated_path,
            }
        )
    return jobs


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Extract all frames for real videos and generated videos into two separate directories."
    )
    parser.add_argument("--manifest", type=str, default="/mnt/public_2/liusonghua/rxcache/outputs/flashheadpro_vibt_hdtf_batch_infer/manifest.json")
    parser.add_argument("--real_videos_dir", type=str, default="/mnt/public_2/liusonghua/rxcache/HDTF/videos")
    parser.add_argument("--out_root", type=str, default="/cache/exframes")
    parser.add_argument("--real_limit", type=int, default=80, help="0 means use all real videos under real_videos_dir.")
    parser.add_argument("--real_image_limit", type=int, default=10000, help="0 means no cap; otherwise keep at most this many real images in real_frames.")
    parser.add_argument("--limit", type=int, default=0, help="0 means use all valid fake rows in the manifest.")
    parser.add_argument("--fake_image_limit", type=int, default=10000, help="0 means no cap; otherwise keep at most this many fake images in fake_frames.")
    return parser.parse_args()


def _cli_main():
    args = _parse_args()
    manifest_path = Path(args.manifest).resolve()
    real_videos_dir = Path(args.real_videos_dir).resolve()
    out_root = Path(args.out_root).resolve()

    real_jobs = _build_real_jobs(real_videos_dir, args.real_limit)
    fake_jobs = _build_fake_jobs(manifest_path, args.limit)

    real_root = out_root / "real_frames"
    fake_root = out_root / "fake_frames"
    real_root.mkdir(parents=True, exist_ok=True)
    fake_root.mkdir(parents=True, exist_ok=True)

    real_meta = []
    real_total_images = 0
    for job in real_jobs:
        remaining = 0
        if args.real_image_limit and int(args.real_image_limit) > 0:
            remaining = int(args.real_image_limit) - real_total_images
            if remaining <= 0:
                break
        extracted_count = _extract_frames_ffmpeg(
            job["video_path"],
            real_root,
            job["sample_id"],
            max_frames=remaining,
        )
        real_total_images += extracted_count
        real_meta.append(
            {
                "sample_id": job["sample_id"],
                "video_path": str(job["video_path"]),
                "frame_pattern": str(real_root / f"{job['sample_id']}_%06d.png"),
                "num_frames": extracted_count,
            }
        )

    fake_meta = []
    fake_total_images = 0
    for job in fake_jobs:
        remaining = 0
        if args.fake_image_limit and int(args.fake_image_limit) > 0:
            remaining = int(args.fake_image_limit) - fake_total_images
            if remaining <= 0:
                break
        extracted_count = _extract_frames_ffmpeg(
            job["video_path"],
            fake_root,
            job["sample_id"],
            max_frames=remaining,
        )
        fake_total_images += extracted_count
        fake_meta.append(
            {
                "sample_id": job["sample_id"],
                "video_path": str(job["video_path"]),
                "frame_pattern": str(fake_root / f"{job['sample_id']}_%06d.png"),
                "num_frames": extracted_count,
            }
        )

    payload = {
        "manifest": str(manifest_path),
        "real_videos_dir": str(real_videos_dir),
        "out_root": str(out_root),
        "num_real_videos": len(real_meta),
        "num_fake_videos": len(fake_meta),
        "num_real_images": int(real_total_images),
        "num_fake_images": int(fake_total_images),
        "real_image_limit": int(args.real_image_limit),
        "fake_image_limit": int(args.fake_image_limit),
        "real_frames_root": str(real_root),
        "fake_frames_root": str(fake_root),
        "real_meta": real_meta,
        "fake_meta": fake_meta,
    }

    out_json = out_root / "extract_frames_manifest.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"num_real_videos : {payload['num_real_videos']}")
    print(f"num_fake_videos : {payload['num_fake_videos']}")
    print(f"num_real_images : {payload['num_real_images']}")
    print(f"num_fake_images : {payload['num_fake_images']}")
    print(f"real_frames_root: {payload['real_frames_root']}")
    print(f"fake_frames_root: {payload['fake_frames_root']}")
    print(f"saved_json      : {out_json}")


if __name__ == "__main__":
    _cli_main()
