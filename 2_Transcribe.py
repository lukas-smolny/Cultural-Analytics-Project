#2nd Code. Transcribe the audio wrt speech, silence, music but only keep speech and make it exactly timestamped
import json
from pathlib import Path
import torch
import whisper_timestamped as whisper

import threading
import time


# =========================
# SETTINGS
# =========================
PROJECT_ROOT = Path("Project_Files")

BASE_AUDIO_DIR = PROJECT_ROOT / "2_audio"
BASE_OUT_DIR   = PROJECT_ROOT / "3_whisper_timestamped"

MODEL_NAME = "medium"
LANGUAGE   = "de"

EXTS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"}

#switch beetween period if u want to compile them separatelly
# 1 = nazi
# 2 = postwar
# 3 = both
PERIOD_MODE = 2


def main():

    BASE_OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEVICE] {device}")

    print(f"[LOAD] Loading model: {MODEL_NAME}")
    model = whisper.load_model(MODEL_NAME, device=device)

    #automatically detect all periods (folders inside 2_audio)
    periods = [p for p in BASE_AUDIO_DIR.iterdir() if p.is_dir()]

    # ---- period filter ----
    if PERIOD_MODE == 1:
        periods = [p for p in periods if p.name.lower() == "nazi"]
    elif PERIOD_MODE == 2:
        periods = [p for p in periods if p.name.lower() == "postwar"]
    elif PERIOD_MODE == 3:
        pass
    else:
        raise ValueError("PERIOD_MODE must be 1, 2, or 3")
    # -----------------------

    for period_dir in periods:

        AUDIO_DIR = period_dir
        OUT_DIR = BASE_OUT_DIR / period_dir.name

        OUT_DIR.mkdir(parents=True, exist_ok=True)

        files = [p for p in AUDIO_DIR.rglob("*") if p.suffix.lower() in EXTS]
        files.sort()

        print(f"\n[SCAN] Period: {period_dir.name}")
        print(f"[SCAN] Found {len(files)} audio files")

        for idx, audio_path in enumerate(files, start=1):

            print("\n" + "="*60)
            print(f"[{idx}/{len(files)}] Processing: {audio_path.name}")
            print("="*60)

            done_flag = {"done": False}

            #debug
            def heartbeat():
                while not done_flag["done"]:
                    print("[WORKING] still running (alignment/VAD/decoding)...")
                    time.sleep(10)

            t = threading.Thread(target=heartbeat, daemon=True)
            t.start()

            result = whisper.transcribe(
                model,
                str(audio_path),

                #
                language=LANGUAGE,
                task="transcribe",

                #vad
                vad="silero:v3.1",

                #accuracy
                beam_size=5,
                best_of=5,
                temperature=(0.0, 0.2, 0.4, 0.6, 0.8),

                #refine timestamps
                refine_whisper_precision=0.4,
                min_word_duration=0.01,
                trust_whisper_timestamps=False,

                #ignore
                #compute_word_confidence=True,

                #stability
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6,
                #condition_on_previous_text=True,

                fp16=(device == "cuda"),


            )

            done_flag["done"] = True


            #print
            print("\n--- TRANSCRIPT ---")
            print(result["text"])

            #count words
            word_count = sum(len(seg.get("words", [])) for seg in result["segments"])
            print(f"\n[INFO] Segments: {len(result['segments'])}")
            print(f"[INFO] Words: {word_count}")

            #save json
            out_path = OUT_DIR / f"{audio_path.stem}_whisper_timestamped.json"

            with out_path.open("w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            print(f"[SAVE] {out_path}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
