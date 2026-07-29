import subprocess
import sys

def resample_ffmpeg(input_path, output_path):
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-ar", "16000",   # 目标采样率
        "-ac", "1",       # 可选：转单声道（推荐做ASR时）
        output_path
    ]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    resample_ffmpeg("/mnt/public_2/liusonghua/rxcache/3min.wav", "/mnt/public_2/liusonghua/rxcache/3mins.wav")