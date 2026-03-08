#4. Fourth Code. Speaker Diarization
import os
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import librosa
import torch

from pyannote.audio import Pipeline


# =========================
# SETTINGS
# =========================
PROJECT_ROOT = Path("Project_Files")

BASE_AUDIO_DIR   = PROJECT_ROOT / "2_audio"
BASE_DIALOG_DIR  = PROJECT_ROOT / "4_dialogs"
BASE_OUT_DIR     = PROJECT_ROOT / "5_diarization_pyannote"

SR_TARGET = 16000

PAD_START_S = 0.0
PAD_END_S   = 0.0

HF_TOKEN_ENV = "HF_TOKEN"
PYANNOTE_MODEL_ID = "pyannote/speaker-diarization-3.1"

PRINT_SEGMENTS = True

# -------------------------
#OPTIONAL Custom diarization params True = use, False = keep defaults; used for fine tuning when doing the project
# -------------------------
USE_CUSTOM_PARAMS = True

CLUSTERING_THRESHOLD = 0.85
CLUSTERING_MIN_CLUSTER_SIZE = 15
SEGMENTATION_MIN_DURATION_OFF = 0.30
# =========================


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def load_dialogs(path: str) -> List[Tuple[float, float]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dialogs = data.get("dialogs", [])
    if not dialogs:
        raise RuntimeError("No dialogs found")

    out = []
    for d in dialogs:
        s = float(d["start"]) - PAD_START_S
        e = float(d["end"]) + PAD_END_S
        if e > s:
            out.append((s, e))

    out.sort(key=lambda x: (x[0], x[1]))
    return out


def load_audio_mono(path: str, sr_target: int) -> Tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=sr_target, mono=True)
    return y.astype(np.float32), sr


def slice_audio(y: np.ndarray, sr: int, start_s: float, end_s: float) -> np.ndarray:
    a = int(round(start_s * sr))
    b = int(round(end_s * sr))
    a = clamp(a, 0, len(y))
    b = clamp(b, 0, len(y))
    if b <= a:
        return np.zeros((0,), dtype=np.float32)
    return y[a:b]


def build_pipeline(device: str, hf_token: str) -> Pipeline:
    print("=" * 90)
    print("Loading pyannote diarization pipeline...")
    print(f"model={PYANNOTE_MODEL_ID}")
    print("=" * 90)

    pipeline = Pipeline.from_pretrained(
        PYANNOTE_MODEL_ID,
        #i removed the token for safety so it doesnt get scraped
        token=""
    )


    if USE_CUSTOM_PARAMS:
        pipeline.instantiate({
            "clustering": {
                "threshold": float(CLUSTERING_THRESHOLD)#,
                #"min_cluster_size": int(CLUSTERING_MIN_CLUSTER_SIZE)
            }#,
           # "segmentation": {
                #"min_duration_off": float(SEGMENTATION_MIN_DURATION_OFF)
            #}
        })


    pipeline.to(torch.device(device))
    print(f"[INIT] Pipeline ready | device={device}")
    return pipeline


def extract_annotation(diar_out):

    #different versions, different style of output.
    if hasattr(diar_out, "itertracks"):
        return diar_out

    if isinstance(diar_out, dict) and "diarization" in diar_out:
        ann = diar_out["diarization"]
        if hasattr(ann, "itertracks"):
            return ann

    if hasattr(diar_out, "diarization"):
        ann = getattr(diar_out, "diarization")
        if hasattr(ann, "itertracks"):
            return ann

    try:
        ann = diar_out["diarization"]
        if hasattr(ann, "itertracks"):
            return ann
    except Exception:
        pass

    raise TypeError("cannot find Annotation with .itertracks().")


def diarize_in_memory(pipeline: Pipeline, y: np.ndarray, sr: int) -> List[Dict[str, Any]]:
    if y.size == 0:
        return []

    waveform = torch.from_numpy(y).unsqueeze(0)
    audio_dict = {"waveform": waveform, "sample_rate": sr}

    diar_out = pipeline(audio_dict)

    if hasattr(diar_out, "speaker_diarization"):
        diarization = diar_out.speaker_diarization
    else:
        diarization = diar_out

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": str(speaker)
        })

    segments.sort(key=lambda d: (d["start"], d["end"], d["speaker"]))
    return segments


def main():

    BASE_OUT_DIR.mkdir(parents=True, exist_ok=True)

    periods = [p for p in BASE_DIALOG_DIR.iterdir() if p.is_dir()]

    #token from env variable. redundant, had problems with env variable so i hardcoded it
    hf_token = ""
    if not hf_token:
        raise RuntimeError(
            f"Missing Hugging Face token."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 90)
    print("STARTED")
    print(f"device={device}")
    print("=" * 90)

    for period_dir in periods:

        dialog_dir = period_dir
        audio_dir  = BASE_AUDIO_DIR / period_dir.name
        out_dir    = BASE_OUT_DIR / period_dir.name

        out_dir.mkdir(parents=True, exist_ok=True)

        dialog_files = sorted(dialog_dir.glob("*_dialogs_simple.json"))

        #with debug messages
        for SEGMENTS_JSON in dialog_files:

            film_name = SEGMENTS_JSON.stem.replace("_dialogs_simple", "")
            AUDIO_PATH = audio_dir / f"{film_name}.wav"
            OUT_JSON   = out_dir / f"{film_name}_diarization_pyannote.json"

            if not AUDIO_PATH.exists():
                print("Missing audio:", AUDIO_PATH)
                continue

            print("=" * 90)
            print(f"FILE: {film_name}")
            print(f"AUDIO={AUDIO_PATH}")
            print(f"SEGMENTS={SEGMENTS_JSON}")
            print("=" * 90)

            dialogs = load_dialogs(str(SEGMENTS_JSON))
            print(f"dialogs: {len(dialogs)}")

            y, sr = load_audio_mono(str(AUDIO_PATH), SR_TARGET)
            print(f"audio loaded: sr={sr} duration={len(y)/sr:.2f}s")

            pipeline = build_pipeline(device, hf_token)

            out_dialogs = []

            for di, (ds, de) in enumerate(dialogs, start=1):
                print("\n" + "=" * 90)
                print(f"DIALOG {di}/{len(dialogs)}] {ds:.2f} - {de:.2f} (len={de-ds:.2f}s)")
                print("=" * 90)

                y_d = slice_audio(y, sr, ds, de)
                if y_d.size == 0:
                    print("DIALOG empty slice -> skip")
                    out_dialogs.append({
                        "dialog_index": di,
                        "start": float(ds),
                        "end": float(de),
                        "segments": [],
                        "speakers": []
                    })
                    continue

                segs_rel = diarize_in_memory(pipeline, y_d, sr)
                print(f"DIALOG diarization segments: {len(segs_rel)}")

                segs_abs = []
                for s in segs_rel:
                    segs_abs.append({
                        "start": float(ds + s["start"]),
                        "end": float(ds + s["end"]),
                        "speaker": s["speaker"]
                    })

                speakers = sorted(set(s["speaker"] for s in segs_abs))
                print(f"speakers detected: {len(speakers)} -> {speakers}")

                if PRINT_SEGMENTS:
                    for i, s in enumerate(segs_abs, start=1):
                        print(f"SEG {i:03d}] {s['start']:.2f}-{s['end']:.2f} {s['speaker']}")

                out_dialogs.append({
                    "dialog_index": di,
                    "start": float(ds),
                    "end": float(de),
                    "segments": segs_abs,
                    "speakers": speakers
                })

            out = {
                "audio": str(AUDIO_PATH),
                "segments_json": str(SEGMENTS_JSON),
                "sr": sr,
                "model": PYANNOTE_MODEL_ID,
                "dialogs": out_dialogs
            }

            with open(OUT_JSON, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)

            print("\n" + "=" * 90)
            print("DONE and Saved:", OUT_JSON)
            print("=" * 90)


if __name__ == "__main__":
    main()
