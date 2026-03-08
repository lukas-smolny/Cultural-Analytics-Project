#1. First Code. Extracts the Audio from the Videos

import os
import subprocess

PROJECT_ROOT = "Project_Files"

BASE_INPUT = os.path.join(PROJECT_ROOT, "1_input_movies")
BASE_AUDIO = os.path.join(PROJECT_ROOT, "2_audio")

os.makedirs(BASE_INPUT, exist_ok=True)
os.makedirs(BASE_AUDIO, exist_ok=True)

#Automatically detect all periods (folders inside 1_input_movies = periods)
PERIODS = [d for d in os.listdir(BASE_INPUT) if os.path.isdir(os.path.join(BASE_INPUT, d))]

for period in PERIODS:
    input_dir = os.path.join(BASE_INPUT, period)
    output_dir = os.path.join(BASE_AUDIO, period)

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith((".mp4", ".mkv", ".avi")):
            input_path = os.path.join(input_dir, filename)

            #Keep original quality
            output_filename = os.path.splitext(filename)[0] + ".wav"
            output_path = os.path.join(output_dir, output_filename)

            print(f"[{period.upper()}] Extracting HIGH-QUALITY audio from {filename}...")

            command = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-i", input_path,
                "-vn",

                #HighQuality output WAV
                "-acodec", "pcm_f32le",   # 32-bit float WAV
                "-y",
                output_path
            ]

            subprocess.run(command, check=False)

print("Audio extraction complete.")
