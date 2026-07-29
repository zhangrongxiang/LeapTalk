from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

from syncnet_pipeline import SyncNetPipeline


SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_ROOT = SCRIPT_DIR.parent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SyncNet evaluation on generated results from ViBT batch inference manifest.",
    )
    parser.add_argument("--manifest", type=str, default="/mnt/public_2/liusonghua/rx/hallo3/batch_runscele/hdtf_batch_20260326_092810/manifest.json", help="Path to manifest.csv or manifest.json produced by batch_infer_flashhead_vibt_hdtf.py")
    parser.add_argument("--out_dir", type=str, default=None, help="Directory for evaluation outputs. Default: <manifest_dir>/lipsync_eval")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ffmpeg_bin", type=str, default="ffmpeg")
    parser.add_argument("--limit", type=int, default=0, help="0 means evaluate all valid rows")
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


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _mean(values: list[float]) -> float:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def _resolve_path(value: Any, base_dir: Path) -> Path:
    p = Path(str(value))
    if p.is_absolute():
        return p

    candidates = []
    candidates.append((base_dir / p).resolve())
    for parent in base_dir.parents:
        candidates.append((parent / p).resolve())
    candidates.append((Path.cwd() / p).resolve())

    seen = set()
    deduped = []
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            deduped.append(candidate)

    for candidate in deduped:
        if candidate.exists():
            return candidate

    return deduped[0]


def _make_syncnet(args: argparse.Namespace) -> SyncNetPipeline:
    return SyncNetPipeline(
        {
            "s3fd_weights": str(EVAL_ROOT / "weights" / "sfd_face.pth"),
            "syncnet_weights": str(EVAL_ROOT / "weights" / "syncnet_v2.model"),
            "ffmpeg_bin": args.ffmpeg_bin,
        },
        device=args.device,
    )


def main() -> None:
    args = _parse_args()
    manifest_path = Path(args.manifest).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else manifest_path.parent / "lipsync_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_manifest(manifest_path)
    valid_rows = [r for r in rows if str(r.get("status", "")).lower() == "ok"]
    if args.limit > 0:
        valid_rows = valid_rows[: int(args.limit)]
    if not valid_rows:
        raise SystemExit("No valid manifest rows with status=ok to evaluate.")

    pipe = _make_syncnet(args)
    results: list[dict[str, Any]] = []

    for row in valid_rows:
        sample_id = str(row.get("sample_id", "")).strip() or Path(str(row.get("generated_video", ""))).stem
        generated_video = _resolve_path(row.get("generated_video", ""), manifest_path.parent)
        audio_16k = _resolve_path(row.get("audio_16k", ""), manifest_path.parent)

        result_row: dict[str, Any] = {
            "index": row.get("index"),
            "sample_id": sample_id,
            "source_video": row.get("source_video"),
            "generated_video": str(generated_video),
            "audio_16k": str(audio_16k),
            "fps": _float_or_nan(row.get("fps")),
            "sync": float("nan"),
            "syncd": float("nan"),
            "has_face": False,
            "eval_runtime_s": float("nan"),
            "offsets": "[]",
            "confs": "[]",
            "dists": "[]",
            "status": "pending",
            "error": "",
        }

        try:
            if not generated_video.exists():
                raise FileNotFoundError(f"Generated video not found: {generated_video}")
            if not audio_16k.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_16k}")

            cache_dir = out_dir / "cache" / sample_id
            t0 = time.perf_counter()
            offsets, confs, dists, best_conf, min_dist, _, has_face = pipe.inference(
                video_path=str(generated_video),
                audio_path=str(audio_16k),
                cache_dir=str(cache_dir),
            )
            result_row.update(
                {
                    "sync": float(best_conf),
                    "syncd": float(min_dist),
                    "has_face": bool(has_face),
                    "eval_runtime_s": float(time.perf_counter() - t0),
                    "offsets": json.dumps([int(x) for x in offsets]),
                    "confs": json.dumps([float(x) for x in confs]),
                    "dists": json.dumps([float(x) for x in dists]),
                    "status": "ok",
                }
            )
            print(
                f"[{sample_id}] sync={result_row['sync']:.3f} | "
                f"syncd={result_row['syncd']:.3f} | fps={result_row['fps']:.2f} | "
                f"has_face={result_row['has_face']}"
            )
        except Exception as exc:
            result_row["status"] = "error"
            result_row["error"] = str(exc)
            print(f"[{sample_id}] ERROR: {exc}")

        results.append(result_row)

    ok_rows = [r for r in results if r["status"] == "ok"]
    face_rows = [r for r in ok_rows if r["has_face"]]
    summary = {
        "manifest": str(manifest_path),
        "selected_count": len(valid_rows),
        "success_count": len(ok_rows),
        "face_detected_count": len(face_rows),
        "mean_sync_face_detected": _mean([r["sync"] for r in face_rows]),
        "mean_syncd_face_detected": _mean([r["syncd"] for r in face_rows]),
        "mean_fps_success": _mean([r["fps"] for r in ok_rows]),
        "args": vars(args),
    }

    results_csv = out_dir / "results.csv"
    results_json = out_dir / "results.json"
    summary_json = out_dir / "summary.json"
    _write_csv(results_csv, results)
    with results_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nSummary")
    print(f"selected_count      : {summary['selected_count']}")
    print(f"success_count       : {summary['success_count']}")
    print(f"face_detected_count : {summary['face_detected_count']}")
    print(f"mean_sync           : {summary['mean_sync_face_detected']:.4f}")
    print(f"mean_syncd          : {summary['mean_syncd_face_detected']:.4f}")
    print(f"mean_fps            : {summary['mean_fps_success']:.4f}")
    print(f"results_csv         : {results_csv}")
    print(f"results_json        : {results_json}")
    print(f"summary_json        : {summary_json}")


if __name__ == "__main__":
    main()
