#7th Code. a lot based from 6th code
#Filters dialogues based on what we need. Here we need all dialogues.

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional


# =========================
# SETTINGS
# =========================
PROJECT_ROOT = Path("Project_Files")

BASE_IN_DIR  = PROJECT_ROOT / "6_gender"
BASE_OUT_DIR = PROJECT_ROOT / "7_all_dialogues"

PRINT_SUMMARY = True
# =========================

#Returns speaker label of the earliest segment (by start time).
def first_speaker_in_dialog(segments: List[Dict[str, Any]]) -> Optional[str]:
    best = None  # (start, speaker)
    for s in segments:
        if "start" not in s or "speaker" not in s:
            continue
        try:
            st = float(s["start"])
        except (TypeError, ValueError):
            continue
        spk = s["speaker"]
        if best is None or st < best[0]:
            best = (st, spk)
    return None if best is None else best[1]

#Normalizes possible gender labels to: Male / Female / Unknown
def normalize_gender(g: Any) -> str:
    if g is None:
        return "Unknown"

    gs = str(g).strip().lower()

    if gs == "male":
        return "Male"
    if gs == "female":
        return "Female"

    return "Unknown"


def composition_label_from_genders(genders: List[str]) -> str:
    """
    Labels for composition
      1 speaker: "1spk_male"
      2 speakers: "2spk_mf", "2spk_mm", "2spk_ff", "2spk_mu", ...
      3+ speakers: "3+spk_mixed", "3+spk_all_male", "3+spk_all_female", "3+spk_all_unknown"
    """
    n = len(genders)

    male_count = genders.count("Male")
    female_count = genders.count("Female")
    unknown_count = genders.count("Unknown")

    if n == 0:
        return "0spk_none"

    if n == 1:
        if male_count == 1:
            return "1spk_male"
        if female_count == 1:
            return "1spk_female"
        return "1spk_unknown"

    if n == 2:
        code = []
        for g in genders:
            if g == "Male":
                code.append("m")
            elif g == "Female":
                code.append("f")
            else:
                code.append("u")
        return "2spk_" + "".join(code)

    # 3+ speakers
    if male_count == n:
        return "3+spk_all_male"
    if female_count == n:
        return "3+spk_all_female"
    if unknown_count == n:
        return "3+spk_all_unknown"

    return "3+spk_mixed"


def speaker_count_bucket(n: int) -> str:
    if n <= 0:
        return "0"
    if n == 1:
        return "1"
    if n == 2:
        return "2"
    if n == 3:
        return "3"
    return "4+"


def inc_count(counter: Dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def main():

    BASE_OUT_DIR.mkdir(parents=True, exist_ok=True)

    periods = [p for p in BASE_IN_DIR.iterdir() if p.is_dir()]

    for period_dir in periods:

        in_dir  = period_dir
        out_dir = BASE_OUT_DIR / period_dir.name

        out_dir.mkdir(parents=True, exist_ok=True)

        json_files = sorted(in_dir.glob("*_diarization_gender.json"))

        for IN_JSON in json_files:

            film_name = IN_JSON.stem.replace("_diarization_gender", "")
            OUT_JSON = out_dir / f"{film_name}_all_dialogues.json"

            if not IN_JSON.exists():
                continue

            with open(IN_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)

            dialogs = data.get("dialogs", [])
            if not dialogs:
                raise RuntimeError("No dialogs found")

            dialog_records: List[Dict[str, Any]] = []

            starter_gender_counts = {"Male": 0, "Female": 0, "Unknown": 0}
            speaker_count_distribution: Dict[str, int] = {}
            exact_speaker_count_distribution: Dict[str, int] = {}
            composition_counts: Dict[str, int] = {}

            dialogs_with_segments = 0
            dialogs_with_any_speaker = 0
            dialogs_with_known_starter_gender = 0

            # robustness / comparison vs original strict logic
            strict_like_match_count = 0
            strict_like_starter_gender_counts = {"Male": 0, "Female": 0, "Unknown": 0}

            for d in dialogs:
                di = d.get("dialog_index", None)

                try:
                    ds = float(d.get("start", 0.0))
                except (TypeError, ValueError):
                    ds = 0.0

                try:
                    de = float(d.get("end", 0.0))
                except (TypeError, ValueError):
                    de = ds

                segments = d.get("segments", []) or []
                if segments:
                    dialogs_with_segments += 1

                speakers = d.get("speakers", None)
                if not speakers:
                    speakers = sorted(set(s.get("speaker") for s in segments if "speaker" in s and s.get("speaker") is not None))

                speakers = [s for s in speakers if s is not None]

                if speakers:
                    dialogs_with_any_speaker += 1

                #speaker -> gender mapping from speaker_genders list
                speaker_genders = d.get("speaker_genders", []) or []
                spk2g: Dict[str, str] = {}

                for sg in speaker_genders:
                    spk = sg.get("speaker", None)
                    if spk is None:
                        continue
                    if spk in speakers:
                        spk2g[spk] = normalize_gender(sg.get("gender", "Unknown"))

                #ensure all speakers exist in mapping (fallback Unknown)
                for spk in speakers:
                    if spk not in spk2g:
                        spk2g[spk] = "Unknown"

                #ordered genders aligned with speakers list
                genders_in_order = [spk2g[spk] for spk in speakers]

                #first speaker / starter
                first_spk = first_speaker_in_dialog(segments)
                starter_gender = spk2g.get(first_spk, "Unknown") if first_spk is not None else "Unknown"
                starter_gender = normalize_gender(starter_gender)

                starter_gender_counts[starter_gender] = starter_gender_counts.get(starter_gender, 0) + 1
                if starter_gender != "Unknown":
                    dialogs_with_known_starter_gender += 1

                #distributions
                n_speakers = len(speakers)
                inc_count(speaker_count_distribution, speaker_count_bucket(n_speakers))
                inc_count(exact_speaker_count_distribution, str(n_speakers))

                comp_label = composition_label_from_genders(genders_in_order)
                inc_count(composition_counts, comp_label)

                #strict-like check (same core criteria as original script) for comparison only
                is_strict_like = False
                if n_speakers == 2:
                    if genders_in_order.count("Male") == 1 and genders_in_order.count("Female") == 1:
                        is_strict_like = True
                        strict_like_match_count += 1
                        strict_like_starter_gender_counts[starter_gender] = strict_like_starter_gender_counts.get(starter_gender, 0) + 1

                #store full per-dialog record
                dialog_records.append({
                    "dialog_index": di,
                    "start": ds,
                    "end": de,
                    "duration": max(0.0, de - ds),

                    "num_speakers": n_speakers,
                    "speaker_count_bucket": speaker_count_bucket(n_speakers),

                    "starter_speaker": first_spk,
                    "starter_gender": starter_gender,

                    "composition_label": comp_label,

                    "is_strict_2speakers_1male_1female": is_strict_like,

                    "speakers": [
                        {"speaker": spk, "gender": spk2g.get(spk, "Unknown")}
                        for spk in speakers
                    ]
                })

            out = {
                "source_file": str(IN_JSON),
                "num_dialogs_total": len(dialogs),
                "criteria": "ALL dialogues (no 2-speaker / no 1M1F filter)",

                "summary": {
                    "dialogs_with_segments": dialogs_with_segments,
                    "dialogs_with_any_speaker": dialogs_with_any_speaker,
                    "dialogs_with_known_starter_gender": dialogs_with_known_starter_gender
                },

                "starter_gender_counts": {
                    "Male": int(starter_gender_counts.get("Male", 0)),
                    "Female": int(starter_gender_counts.get("Female", 0)),
                    "Unknown": int(starter_gender_counts.get("Unknown", 0)),
                },

                "speaker_count_distribution": speaker_count_distribution,
                "exact_speaker_count_distribution": exact_speaker_count_distribution,
                "composition_counts": composition_counts,

                "strict_like_counts": {
                    "criteria": "exactly 2 speakers AND exactly 1 Male + 1 Female (for comparison only; not filtered output)",
                    "num_matches": int(strict_like_match_count),
                    "starter_gender_counts": {
                        "Male": int(strict_like_starter_gender_counts.get("Male", 0)),
                        "Female": int(strict_like_starter_gender_counts.get("Female", 0)),
                        "Unknown": int(strict_like_starter_gender_counts.get("Unknown", 0)),
                    }
                },

                "dialogs": dialog_records
            }

            with open(OUT_JSON, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)

            if PRINT_SUMMARY:
                print("=" * 100)
                print("DONE - ALL DIALOGUES")
                print("Saved:", OUT_JSON)
                print(f"Total dialogs:                  {len(dialogs)}")
                print(f"Dialogs with any speaker:       {dialogs_with_any_speaker}")
                print(f"Known starter gender dialogs:   {dialogs_with_known_starter_gender}")
                print(f"Strict-like matches (compare):  {strict_like_match_count}")
                print("STARTER COUNTS - ALL:", out["starter_gender_counts"])
                print("SPEAKER COUNT DIST.:", out["speaker_count_distribution"])
                print("=" * 100)

                for rec in dialog_records[:20]:
                    speaker_str = " | ".join([f"{x['speaker']}={x['gender']}" for x in rec["speakers"]])
                    print(
                        f"DIALOG idx={rec['dialog_index']} {rec['start']:.2f}-{rec['end']:.2f} "
                        f"dur={rec['duration']:.2f}s | n_spk={rec['num_speakers']} ({rec['speaker_count_bucket']}) | "
                        f"starter={rec['starter_speaker']}({rec['starter_gender']}) | "
                        f"comp={rec['composition_label']} | strict_like={rec['is_strict_2speakers_1male_1female']} | "
                        f"{speaker_str}"
                    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())