from syncnet_pipeline import SyncNetPipeline   # file you just saved
import logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG if you want more detail
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# 1.  Initialise the pipeline  (put your weight files in the same folder or give absolute paths)
pipe = SyncNetPipeline(
    {
        "s3fd_weights":  "../weights/sfd_face.pth",
        "syncnet_weights": "../weights/syncnet_v2.model",
    },
    device="cuda",          # or "cpu"
)

# 2.  Run inference on one clip
results = pipe.inference(
    video_path="../example/girl1step.mp4",   # RGB video
    audio_path="/cache/newszheng.wav",   # speech track (any ffmpeg-readable format)
    # cache_dir="../example/cache",    # optional; omit to auto-cleanup intermediates
)

# 3.  Inspect outputs
offsets, confs, dists, max_conf, min_dist, s3fd_json, has_face = results
print("best-confidence   :", max_conf)
print("lowest distance   :", min_dist)
print("per-crop offsets  :", offsets)
