#6th code. it is _All_Dialogues.py and _M_and_F.py put together, with added min duration filter


#Re-runs the final filtering logic from _M_and_F.py and _All_Dialogues.py
#with an additional minimum duration constraint.
#
#Output folders and file structure are identical to the originals.
#Only the output directory names differ:
#   7_two_speakers       -> 7_two_speakers_min5s
#   7_all_dialogues      -> 7_all_dialogues_min5s

# Input: Project_Files/6_gender/  (same as _All_Dialogues.py and _M_and_F.py)

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional


# =========================
# SETTINGS
# =========================
PROJECT_ROOT = Path("Project_Files")

BASE_IN_DIR  = PROJECT_ROOT / "6_gender"

BASE_OUT_6A  = PROJECT_ROOT / "7_two_speakers_min5s"
BASE_OUT_6B  = PROJECT_ROOT / "7_all_dialogues_min5s"

MIN_DURATION_S = 5.0

PRINT_SUMMARY = True
# =========================


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def first_speaker_in_dialog(segments: List[Dict[str, Any]]) -> Optional[str]:
    best = None
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
    n = len(genders)
    male_count    = genders.count("Male")
    female_count  = genders.count("Female")
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


# -----------------------------------------------------------------------------
# 6A  —  identical logic to _M_and_F.py + duration filter
# -----------------------------------------------------------------------------

def run_6a(IN_JSON: Path, OUT_JSON: Path, min_dur: float) -> None:
    with open(IN_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    dialogs = data.get("dialogs", [])
    if not dialogs:
        raise RuntimeError("No dialogs found (expected key 'dialogs').")

    matches: List[Dict[str, Any]] = []

    for d in dialogs:
        di = d.get("dialog_index", None)
        ds = float(d.get("start", 0.0))
        de = float(d.get("end", 0.0))

        # duration filter
        try:
            duration = float(d.get("duration", de - ds))
        except (TypeError, ValueError):
            duration = de - ds
        if duration < min_dur:
            continue

        segments = d.get("segments", [])

        speakers = d.get("speakers", None)
        if not speakers:
            speakers = sorted(set(s.get("speaker") for s in segments if "speaker" in s))

        if len(speakers) != 2:
            continue

        speaker_genders = d.get("speaker_genders", [])

        spk2g = {}
        for sg in speaker_genders:
            spk = sg.get("speaker", None)
            if spk in speakers:
                spk2g[spk] = sg.get("gender", "Unknown")

        if len(spk2g) != 2:
            continue

        genders = [spk2g[speakers[0]], spk2g[speakers[1]]]

        if genders.count("Male") != 1 or genders.count("Female") != 1:
            continue

        first_spk = first_speaker_in_dialog(segments)
        starter_gender = spk2g.get(first_spk, "Unknown") if first_spk is not None else "Unknown"

        matches.append({
            "dialog_index": di,
            "start": ds,
            "end": de,
            "duration": de - ds,

            "starter_speaker": first_spk,
            "starter_gender": starter_gender,

            "speakers": [
                {"speaker": speakers[0], "gender": spk2g[speakers[0]]},
                {"speaker": speakers[1], "gender": spk2g[speakers[1]]},
            ]
        })

    out = {
        "source_file": str(IN_JSON),
        "num_dialogs_total": len(dialogs),
        "num_matches": len(matches),
        "criteria": "exactly 2 speakers AND exactly 1 Male + 1 Female",

        "starter_gender_counts": {
            "Male":    sum(1 for m in matches if m.get("starter_gender") == "Male"),
            "Female":  sum(1 for m in matches if m.get("starter_gender") == "Female"),
            "Unknown": sum(1 for m in matches if m.get("starter_gender") == "Unknown"),
        },

        "matches": matches
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    if PRINT_SUMMARY:
        print("=" * 90)
        print("[DONE 6A]")
        print(f"Total dialogs: {len(dialogs)}")
        print(f"Matches:       {len(matches)}")
        print("Saved:", OUT_JSON)
        print("=" * 90)
        print("[STARTER COUNTS]", out["starter_gender_counts"])

        for m in matches[:20]:
            sp = m["speakers"]
            print(
                f"[MATCH] dialog={m['dialog_index']} {m['start']:.2f}-{m['end']:.2f} "
                f"starter={m['starter_speaker']}({m['starter_gender']}) | "
                f"{sp[0]['speaker']}={sp[0]['gender']} | {sp[1]['speaker']}={sp[1]['gender']}"
            )


# -----------------------------------------------------------------------------
# 6B  —  identical logic to _All_Dialogues.py + duration filter
# -----------------------------------------------------------------------------

def run_6b(IN_JSON: Path, OUT_JSON: Path, min_dur: float) -> None:
    with open(IN_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    dialogs = data.get("dialogs", [])
    if not dialogs:
        raise RuntimeError("No dialogs found (expected key 'dialogs').")

    dialog_records: List[Dict[str, Any]] = []

    starter_gender_counts = {"Male": 0, "Female": 0, "Unknown": 0}
    speaker_count_distribution: Dict[str, int] = {}
    exact_speaker_count_distribution: Dict[str, int] = {}
    composition_counts: Dict[str, int] = {}

    dialogs_with_segments = 0
    dialogs_with_any_speaker = 0
    dialogs_with_known_starter_gender = 0

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

        # duration filter
        try:
            duration = float(d.get("duration", de - ds))
        except (TypeError, ValueError):
            duration = de - ds
        if duration < min_dur:
            continue

        segments = d.get("segments", []) or []
        if segments:
            dialogs_with_segments += 1

        speakers = d.get("speakers", None)
        if not speakers:
            speakers = sorted(set(s.get("speaker") for s in segments if "speaker" in s and s.get("speaker") is not None))

        speakers = [s for s in speakers if s is not None]

        if speakers:
            dialogs_with_any_speaker += 1

        speaker_genders = d.get("speaker_genders", []) or []
        spk2g: Dict[str, str] = {}

        for sg in speaker_genders:
            spk = sg.get("speaker", None)
            if spk is None:
                continue
            if spk in speakers:
                spk2g[spk] = normalize_gender(sg.get("gender", "Unknown"))

        for spk in speakers:
            if spk not in spk2g:
                spk2g[spk] = "Unknown"

        genders_in_order = [spk2g[spk] for spk in speakers]

        first_spk = first_speaker_in_dialog(segments)
        starter_gender = spk2g.get(first_spk, "Unknown") if first_spk is not None else "Unknown"
        starter_gender = normalize_gender(starter_gender)

        starter_gender_counts[starter_gender] = starter_gender_counts.get(starter_gender, 0) + 1
        if starter_gender != "Unknown":
            dialogs_with_known_starter_gender += 1

        n_speakers = len(speakers)
        inc_count(speaker_count_distribution, speaker_count_bucket(n_speakers))
        inc_count(exact_speaker_count_distribution, str(n_speakers))

        comp_label = composition_label_from_genders(genders_in_order)
        inc_count(composition_counts, comp_label)

        is_strict_like = False
        if n_speakers == 2:
            if genders_in_order.count("Male") == 1 and genders_in_order.count("Female") == 1:
                is_strict_like = True
                strict_like_match_count += 1
                strict_like_starter_gender_counts[starter_gender] = strict_like_starter_gender_counts.get(starter_gender, 0) + 1

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
            "Male":    int(starter_gender_counts.get("Male", 0)),
            "Female":  int(starter_gender_counts.get("Female", 0)),
            "Unknown": int(starter_gender_counts.get("Unknown", 0)),
        },

        "speaker_count_distribution": speaker_count_distribution,
        "exact_speaker_count_distribution": exact_speaker_count_distribution,
        "composition_counts": composition_counts,

        "strict_like_counts": {
            "criteria": "exactly 2 speakers AND exactly 1 Male + 1 Female (for comparison only; not filtered output)",
            "num_matches": int(strict_like_match_count),
            "starter_gender_counts": {
                "Male":    int(strict_like_starter_gender_counts.get("Male", 0)),
                "Female":  int(strict_like_starter_gender_counts.get("Female", 0)),
                "Unknown": int(strict_like_starter_gender_counts.get("Unknown", 0)),
            }
        },

        "dialogs": dialog_records
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    if PRINT_SUMMARY:
        print("=" * 100)
        print("[DONE - ALL DIALOGUES]")
        print("Saved:", OUT_JSON)
        print(f"Total dialogs (after duration filter): {len(dialog_records)}")
        print(f"Dialogs with any speaker:              {dialogs_with_any_speaker}")
        print(f"Known starter gender dialogs:          {dialogs_with_known_starter_gender}")
        print(f"Strict-like matches (compare):         {strict_like_match_count}")
        print("[STARTER COUNTS - ALL]", out["starter_gender_counts"])
        print("[SPEAKER COUNT DIST.]", out["speaker_count_distribution"])
        print("=" * 100)

        for rec in dialog_records[:20]:
            speaker_str = " | ".join([f"{x['speaker']}={x['gender']}" for x in rec["speakers"]])
            print(
                f"[DIALOG] idx={rec['dialog_index']} {rec['start']:.2f}-{rec['end']:.2f} "
                f"dur={rec['duration']:.2f}s | n_spk={rec['num_speakers']} ({rec['speaker_count_bucket']}) | "
                f"starter={rec['starter_speaker']}({rec['starter_gender']}) | "
                f"comp={rec['composition_label']} | strict_like={rec['is_strict_2speakers_1male_1female']} | "
                f"{speaker_str}"
            )


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main() -> int:
    BASE_OUT_6A.mkdir(parents=True, exist_ok=True)
    BASE_OUT_6B.mkdir(parents=True, exist_ok=True)

    periods = [p for p in BASE_IN_DIR.iterdir() if p.is_dir()]

    print("=" * 90)
    print(f"[FilterMinDuration] MIN_DURATION_S = {MIN_DURATION_S}")
    print("=" * 90)

    for period_dir in periods:
        out_dir_6a = BASE_OUT_6A / period_dir.name
        out_dir_6b = BASE_OUT_6B / period_dir.name
        out_dir_6a.mkdir(parents=True, exist_ok=True)
        out_dir_6b.mkdir(parents=True, exist_ok=True)

        json_files = sorted(period_dir.glob("*_diarization_gender.json"))

        for IN_JSON in json_files:
            film_name = IN_JSON.stem.replace("_diarization_gender", "")

            OUT_6A = out_dir_6a / f"{film_name}_2speakers_1male_1female.json"
            OUT_6B = out_dir_6b / f"{film_name}_all_dialogues.json"

            print(f"\n[FILM] {film_name}  |  period: {period_dir.name}")

            run_6a(IN_JSON, OUT_6A, MIN_DURATION_S)
            run_6b(IN_JSON, OUT_6B, MIN_DURATION_S)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())