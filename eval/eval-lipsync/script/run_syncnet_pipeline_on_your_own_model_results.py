#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch SyncNet evaluation for MoChaBench.
"""

import json
import csv
from pathlib import Path
import time
from collections import defaultdict

import pandas as pd
from syncnet_pipeline import SyncNetPipeline

# ------------------------------------------------------------------ #
# 1)  paths & config                                                 #
# ------------------------------------------------------------------ #

# folder that contains *this* script
ROOT = Path(r"/full/path/to/MoChaBench") # <<< fill with your own repo path/MoChaBench

# ---- adapt these if you move folders around ---------------------- #
BASE_VIDEO = ROOT / "your model output videos"       # <<< fill with your own path e.g. …/MoChaBench/xxx-model-out
BASE_BENCHMARK = ROOT / "benchmark"                  # parent folder that contains 'speeches/'
CSV_FILE   = BASE_BENCHMARK / "benchmark.csv"
# ------------------------------------------------------------------ #

OUT_CSV  = ROOT / "eval-lipsync" / "your own model-eval-results" / "sync_scores.csv"  # <<< fill with your own path
OUT_JSON = ROOT / "eval-lipsync"/  "your own model-eval-results" / "sync_scores.json"  # <<< fill with your own path

pipe = SyncNetPipeline(
    {
        "s3fd_weights":  ROOT / "eval-lipsync" / "weights" / "sfd_face.pth",
        "syncnet_weights": ROOT / "eval-lipsync" /"weights" / "syncnet_v2.model",
    },
    device="cuda",
)

# category buckets for your extra means
ENGLISH_1P_CATEGORIES = {
    "1p_closeup_facingcamera",
    "1p_camera_movement",
    "1p_emotion",
    "1p_mediumshot_actioncontrol",
    "2p_1clip_1talk",
    "1p_protrait",
}
TURNTALK_CATEGORY = {"2p_2clip_2talk"}

# ------------------------------------------------------------------ #
# 2)  helper: run one sample                                         #
# ------------------------------------------------------------------ #
def run_sample(row):
    idx = int(row["idx_in_category"])
    cat = row["category"].strip()
    base_name = row["context_id"].strip()  # e.g. 1_man_bag_of_gold
    id = f"{cat}_{base_name}"

    video_fp = BASE_VIDEO / cat / f"{base_name}.mp4"
    audio_fp = BASE_BENCHMARK / "speeches" / cat / f"{base_name}_speech.wav"

    if not video_fp.exists():
        raise FileNotFoundError(f"Video not found: {video_fp}")
    if not audio_fp.exists():
        raise FileNotFoundError(f"Audio not found: {audio_fp}")

    t0 = time.time()
    off, confs, dists, best_conf, min_dist, _, has_face = pipe.inference(
        video_path=str(video_fp),
        audio_path=str(audio_fp),
        cache_dir= ROOT / "eval-lipsync"/  "mocha-eval-results" / "cache" / id
    )
    return {
        "idx": idx,
        "category": cat,
        "video": str(Path(cat) / f"{base_name}.mp4"),
        "audio": str(Path(cat) / f"{base_name}_speech.wav"),
        "offsets": [int(o) for o in off],
        "best_conf": float(best_conf),
        "min_dist": float(min_dist),
        "has_face": has_face,
        "runtime_s": round(time.time() - t0, 2),
    }


# ------------------------------------------------------------------ #
# 3)  main loop                                                      #
# ------------------------------------------------------------------ #
def main():
    df = pd.read_csv(CSV_FILE)
    results = []
    for _, row in df.iterrows():
        try:
            res = run_sample(row)
            results.append(res)
            print(
                f"[{res['idx']}] {res['category']}  "
                f"Δ={res['offsets']}  conf={res['best_conf']:.3f}  "
                f"dist={res['min_dist']:.3f}"
            )
        except Exception as e:
            print(f"[{row['idx']}] ERROR – {e}")

    # ------------------------------------------------------------------ #
    # 4)  save full table                                                #
    # ------------------------------------------------------------------ #
    pd.DataFrame(results).to_csv(OUT_CSV, index=False)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    # ------------------------------------------------------------------ #
    # 5)  category aggregates                                            #
    # ------------------------------------------------------------------ #
    # Separate stats for each category (only when face is detected)
    cat_dists = defaultdict(list)
    cat_confs = defaultdict(list)

    for r in results:
        # !!!! sometimes SyncNetPipeline fails to detect faces, we should not inlcude those
        if r["has_face"]:
            cat_dists[r["category"]].append(r["min_dist"])
            cat_confs[r["category"]].append(r["best_conf"])

    def mean(vals):
        return sum(vals) / len(vals) if vals else float("nan")

    print("\n=== per-category averages (only if face detected) ===")
    for cat in sorted(set(cat_dists.keys()) | set(cat_confs.keys())):
        avg_dist = mean(cat_dists[cat])
        avg_conf = mean(cat_confs[cat])
        print(f"{cat:30s}:  dist={avg_dist:.3f}   conf={avg_conf:.3f}   (n={len(cat_dists[cat])})")


    #  super-sets
    english_dists = [
        d for c, v in cat_dists.items() if c in ENGLISH_1P_CATEGORIES for d in v
    ]
    english_confs = [
        c for cat, v in cat_confs.items() if cat in ENGLISH_1P_CATEGORIES for c in v
    ]

    dialog_dists = [
        d for c, v in cat_dists.items() if c in TURNTALK_CATEGORY for d in v
    ]
    dialog_confs = [
        c for cat, v in cat_confs.items() if cat in TURNTALK_CATEGORY for c in v
    ]
    print("\n--- aggregate groups (only face-detected entries) ---")
    print(f"single-character English (1p*): dist={mean(english_dists):.3f}  conf={mean(english_confs):.3f}")
    print(f"turn-based dialogue English (2p_2clip): dist={mean(dialog_dists):.3f}  conf={mean(dialog_confs):.3f}")

    print(f"\nSaved detailed table → {OUT_CSV}")
    print(f"Saved JSON dump      → {OUT_JSON}")


if __name__ == "__main__":
    main()
