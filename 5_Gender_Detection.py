#5. Fifth Code. Detect gender.
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import librosa
import torch
from transformers import pipeline as hf_pipeline


# =========================
# SETTINGS
# =========================
PROJECT_ROOT = Path("Project_Files")

BASE_AUDIO_DIR = PROJECT_ROOT / "2_audio"
BASE_DIAR_DIR  = PROJECT_ROOT / "5_diarization_pyannote"
BASE_OUT_DIR   = PROJECT_ROOT / "6_gender"

SR_TARGET = 16000

DO_GENDER = True

# so far worked the best
GENDER_MODEL_ID = "prithivMLmods/Common-Voice-Gender-Detection"

PRINT_SUMMARY = True
PRINT_DEBUG_OUT = True
# =========================


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


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


def build_gender_model(device: str):
    if not DO_GENDER:
        return None

    print("=" * 90)
    print("Loading gender model...")
    print(f"model={GENDER_MODEL_ID}")
    print("=" * 90)

    gender_clf = hf_pipeline(
        task="audio-classification",
        model=GENDER_MODEL_ID,
        device=0 if (device == "cuda") else -1,
        top_k=None
    )

    print("Gender model ready")
    return gender_clf

#Predict gender once for the whole segment. No chunking, no votes, no thresholds. + Avoid crashes on small segments
def gender_predict_segment(gender_clf, y_seg: np.ndarray, sr: int) -> Dict[str, Any]:
    if y_seg is None:
        return {"gender": "Unknown", "score": None, "raw_label": None, "raw_out": None}

    y_seg = np.asarray(y_seg, dtype=np.float32).flatten()

    if y_seg.size < 2:
        return {"gender": "Unknown", "score": None, "raw_label": None, "raw_out": None}

    try:
        out = gender_clf({"array": y_seg, "sampling_rate": sr})
    except Exception:
        return {"gender": "Unknown", "score": None, "raw_label": None, "raw_out": None}

    if not out:
        return {"gender": "Unknown", "score": None, "raw_label": None, "raw_out": out}

    best = max(out, key=lambda x: x["score"])
    raw = str(best["label"]).strip().lower()
    score = float(best["score"])

    if raw == "female":
        gender = "Female"
    elif raw == "male":
        gender = "Male"
    else:
        gender = "Unknown"

    return {"gender": gender, "score": score, "raw_label": raw, "raw_out": out}


def pick_one_segment_for_speaker(segs_abs: List[Dict[str, Any]], speaker: str) -> Optional[Tuple[float, float]]:
    EPS = 1e-3

    spk_segs = []
    for s in segs_abs:
        if s.get("speaker") != speaker:
            continue
        st = float(s["start"])
        en = float(s["end"])
        if en > st:
            spk_segs.append((st, en))

    if not spk_segs:
        return None

    spk_segs.sort(key=lambda x: (x[0], x[1]))

    runs = []
    cur_st, cur_en = spk_segs[0]

    for st, en in spk_segs[1:]:
        if st <= cur_en + EPS:
            if en > cur_en:
                cur_en = en
        else:
            runs.append((cur_st, cur_en))
            cur_st, cur_en = st, en

    runs.append((cur_st, cur_en))

    best_st, best_en = max(runs, key=lambda x: (x[1] - x[0], x[0]))
    return best_st, best_en



def main():

    BASE_OUT_DIR.mkdir(parents=True, exist_ok=True)

    periods = [p for p in BASE_DIAR_DIR.iterdir() if p.is_dir()]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 90)
    print("Gender labeling from diarization startd")
    print(f"device={device}")
    print("=" * 90)

    for period_dir in periods:

        diar_dir = period_dir
        audio_dir = BASE_AUDIO_DIR / period_dir.name
        out_dir = BASE_OUT_DIR / period_dir.name

        out_dir.mkdir(parents=True, exist_ok=True)

        diar_files = sorted(diar_dir.glob("*_diarization_pyannote.json"))

        for DIARIZATION_JSON in diar_files:

            film_name = DIARIZATION_JSON.stem.replace("_diarization_pyannote", "")
            AUDIO_PATH = audio_dir / f"{film_name}.wav"
            OUT_JSON   = out_dir / f"{film_name}_diarization_gender.json"

            if not AUDIO_PATH.exists():
                print("Missing audio:", AUDIO_PATH)
                continue

            print("=" * 90)
            print(f"Filr: {film_name}")
            print(f"AUDIO={AUDIO_PATH}")
            print(f"DIAR_JSON={DIARIZATION_JSON}")
            print("=" * 90)

            with open(DIARIZATION_JSON, "r", encoding="utf-8") as f:
                diar = json.load(f)

            dialogs = diar.get("dialogs", [])
            if not dialogs:
                raise RuntimeError("No dialogs found in diarization JSON")

            y, sr = load_audio_mono(str(AUDIO_PATH), SR_TARGET)
            print(f"audio loaded: sr={sr} duration={len(y)/sr:.2f}s")

            gender_clf = build_gender_model(device)

            diar["gender"] = {
                "DO_GENDER": DO_GENDER,
                "model": GENDER_MODEL_ID if DO_GENDER else None
            }

            for di, d in enumerate(dialogs, start=1):
                ds = float(d.get("start", 0.0))
                de = float(d.get("end", 0.0))
                segs_abs = d.get("segments", [])

                speakers = d.get("speakers", None)
                if not speakers:
                    speakers = sorted(set(s.get("speaker") for s in segs_abs if "speaker" in s))

                if PRINT_SUMMARY:
                    print("\n" + "=" * 90)
                    print(f"DIALOG {di}/{len(dialogs)}] {ds:.2f}-{de:.2f} | speakers={len(speakers)}")
                    print("=" * 90)

                speaker_genders = []
                speaker_to_gender = {}

                for spk in speakers:
                    rep = pick_one_segment_for_speaker(segs_abs, spk)
                    if rep is None:
                        info = {
                            "speaker": spk,
                            "gender": "Unknown",
                            "score": None,
                            "raw_label": None,
                            "representative_segment": None
                        }
                        speaker_genders.append(info)
                        speaker_to_gender[spk] = "Unknown"
                        if PRINT_SUMMARY:
                            print(f"GENDER {spk} -> Unknown (no segment)")
                        continue

                    st, en = rep
                    y_seg = slice_audio(y, sr, st, en)

                    g = gender_predict_segment(gender_clf, y_seg, sr)

                    if PRINT_DEBUG_OUT:
                        print("DEBUG OUT", spk, g["raw_out"])

                    info = {
                        "speaker": spk,
                        "gender": g["gender"],
                        "score": g["score"],
                        "raw_label": g["raw_label"],
                        "representative_segment": {"start": float(st), "end": float(en)}
                    }
                    speaker_genders.append(info)
                    speaker_to_gender[spk] = g["gender"]

                    if PRINT_SUMMARY:
                        print(f"GENDER {spk} -> {g['gender']} score={g['score']} rep={st:.2f}-{en:.2f}")

                segs_abs_out = []
                for s in segs_abs:
                    spk = s.get("speaker", None)
                    s2 = dict(s)
                    s2["gender"] = speaker_to_gender.get(spk, "Unknown")
                    segs_abs_out.append(s2)

                d["speaker_genders"] = speaker_genders
                d["segments"] = segs_abs_out

            diar["dialogs"] = dialogs

            with open(OUT_JSON, "w", encoding="utf-8") as f:
                json.dump(diar, f, indent=2, ensure_ascii=False)

            print("\n" + "=" * 90)
            print("Done, Saved:", OUT_JSON)
            print("=" * 90)


if __name__ == "__main__":
    main()
