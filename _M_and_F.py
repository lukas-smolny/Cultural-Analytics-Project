#6. Sixth code. Filters dialogs with only 2 speakers male + female
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional


# =========================
# SETTINGS
# =========================
PROJECT_ROOT = Path("Project_Files")

BASE_IN_DIR  = PROJECT_ROOT / "6_gender"
BASE_OUT_DIR = PROJECT_ROOT / "7_two_speakers"

PRINT_SUMMARY = True
# =========================

#Returns speaker label of the earliest segment (by start time).
def first_speaker_in_dialog(segments: List[Dict[str, Any]]) -> Optional[str]:
    best = None  # (start, speaker)
    for s in segments:
        if "start" not in s or "speaker" not in s:
            continue
        st = float(s["start"])
        spk = s["speaker"]
        if best is None or st < best[0]:
            best = (st, spk)
    return None if best is None else best[1]


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
            OUT_JSON = out_dir / f"{film_name}_2speakers_1male_1female.json"

            if not IN_JSON.exists():
                continue

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
                    "Male": sum(1 for m in matches if m.get("starter_gender") == "Male"),
                    "Female": sum(1 for m in matches if m.get("starter_gender") == "Female"),
                    "Unknown": sum(1 for m in matches if m.get("starter_gender") == "Unknown"),
                },

                "matches": matches
            }

            with open(OUT_JSON, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)

            if PRINT_SUMMARY:
                print("=" * 90)
                print("DONE")
                print(f"Total dialogs: {len(dialogs)}")
                print(f"Matches:       {len(matches)}")
                print("Saved:", OUT_JSON)
                print("=" * 90)
                print("STARTER COUNTS", out["starter_gender_counts"])

                for m in matches[:20]:
                    sp = m["speakers"]
                    print(
                        f"MATCH dialog={m['dialog_index']} {m['start']:.2f}-{m['end']:.2f} "
                        f"starter={m['starter_speaker']}({m['starter_gender']}) | "
                        f"{sp[0]['speaker']}={sp[0]['gender']} | {sp[1]['speaker']}={sp[1]['gender']}"
                    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
