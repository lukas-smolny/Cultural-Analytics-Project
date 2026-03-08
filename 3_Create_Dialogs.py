#3. Third Code. Simply create dialog units based on Gaps
import json
from pathlib import Path


#=========================
#SETTINGS
#=========================
PROJECT_ROOT = Path("Project_Files")

BASE_INPUT_DIR = PROJECT_ROOT / "3_whisper_timestamped"
BASE_OUT_DIR   = PROJECT_ROOT / "4_dialogs"

MAX_GAP_S = 3.0
MAX_WORD_GAP_S = 5

#if False = do not split by word gap at all
USE_MAX_WORD_GAP = True
#=========================

#Split ONE whisper segment into multiple segments if there is a gap > max_word_gap_s between words
def split_segment_by_word_gap(seg, max_word_gap_s):
    words = seg.get("words", [])
    if not words or len(words) < 2:
        return [seg]

    chunks = []
    cur_words = [words[0]]

    for i in range(len(words) - 1):
        a = words[i]
        b = words[i + 1]

        a_end = float(a.get("end", seg["end"]))
        b_start = float(b.get("start", seg["start"]))
        gap = b_start - a_end

        if gap > max_word_gap_s:
            #close chunk at word i
            chunks.append(cur_words)
            cur_words = [b]
        else:
            cur_words.append(b)

    if cur_words:
        chunks.append(cur_words)

    #build new segments from chunks
    out = []
    for ci, wchunk in enumerate(chunks):
        start = float(wchunk[0].get("start", seg["start"]))
        end = float(wchunk[-1].get("end", seg["end"]))
        text = " ".join([w.get("text", "").strip() for w in wchunk]).strip()

        out.append({
            "start": start,
            "end": end,
            "text": text,
            "words": list(wchunk),
            "debug": {
                "split_from_original": True,
                "original_start": float(seg["start"]),
                "original_end": float(seg["end"]),
                "chunk_index": ci,
                "num_chunks": len(chunks),
            }
        })

    return out

#Split segments by word gap optional
def preprocess_segments(segments):
    out = []
    for seg in segments:
        if USE_MAX_WORD_GAP:
            out.extend(split_segment_by_word_gap(seg, MAX_WORD_GAP_S))
        else:
            out.append({
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "text": seg.get("text", "").strip(),
                "words": list(seg.get("words", [])),
                "debug": {
                    "split_from_original": False
                }
            })

    out.sort(key=lambda s: float(s["start"]))
    return out

#Merge to dialogs
def merge_to_dialogs(segments):
    dialogs = []
    cur = None

    for seg in segments:
        seg_start = float(seg["start"])
        seg_end   = float(seg["end"])
        seg_text  = seg.get("text", "").strip()
        seg_words = seg.get("words", [])

        if cur is None:
            cur = {
                "start": seg_start,
                "end": seg_end,
                "text": seg_text,
                "words": list(seg_words),
                "debug": {
                    "merge_checks": []
                }
            }
            continue

        segment_gap = seg_start - cur["end"]
        can_merge = (segment_gap <= MAX_GAP_S)

        cur["debug"]["merge_checks"].append({
            "next_seg_start": seg_start,
            "next_seg_end": seg_end,
            "segment_gap_s": segment_gap,
            "can_merge": can_merge
        })

        if can_merge:
            cur["end"] = seg_end
            if seg_text:
                if cur["text"]:
                    cur["text"] += " " + seg_text
                else:
                    cur["text"] = seg_text
            cur["words"].extend(seg_words)
        else:
            dialogs.append(cur)
            cur = {
                "start": seg_start,
                "end": seg_end,
                "text": seg_text,
                "words": list(seg_words),
                "debug": {
                    "merge_checks": []
                }
            }

    if cur is not None:
        dialogs.append(cur)

    return dialogs


def main():

    BASE_OUT_DIR.mkdir(parents=True, exist_ok=True)

    #detect all periods automatically from folder stucture
    periods = [p for p in BASE_INPUT_DIR.iterdir() if p.is_dir()]

    for period_dir in periods:

        INPUT_DIR = period_dir
        OUT_DIR = BASE_OUT_DIR / period_dir.name

        OUT_DIR.mkdir(parents=True, exist_ok=True)

        json_files = sorted(INPUT_DIR.glob("*_whisper_timestamped.json"))

        for INPUT_JSON in json_files:

            OUT_SIMPLE = OUT_DIR / (INPUT_JSON.stem.replace("_whisper_timestamped", "") + "_dialogs_simple.json")
            OUT_WORDS  = OUT_DIR / (INPUT_JSON.stem.replace("_whisper_timestamped", "") + "_dialogs_with_words.json")

            data = json.loads(INPUT_JSON.read_text(encoding="utf-8"))
            segments = data.get("segments", [])
            if not segments:
                raise RuntimeError("No segments found in whisper_timestamped JSON")

            #1) split by word gap (optional)
            processed = preprocess_segments(segments)

            #2) merge into dialogs by segment gap
            dialogs = merge_to_dialogs(processed)

            simple_out = {
                "source": str(INPUT_JSON),
                "max_gap_s": MAX_GAP_S,
                "max_word_gap_s": MAX_WORD_GAP_S,
                "use_max_word_gap": USE_MAX_WORD_GAP,
                "dialogs": [
                    {"start": d["start"], "end": d["end"], "text": d["text"]}
                    for d in dialogs
                ]
            }
            OUT_SIMPLE.write_text(json.dumps(simple_out, ensure_ascii=False, indent=2), encoding="utf-8")

            words_out = {
                "source": str(INPUT_JSON),
                "max_gap_s": MAX_GAP_S,
                "max_word_gap_s": MAX_WORD_GAP_S,
                "use_max_word_gap": USE_MAX_WORD_GAP,
                "dialogs": dialogs
            }
            OUT_WORDS.write_text(json.dumps(words_out, ensure_ascii=False, indent=2), encoding="utf-8")

            print("[DONE]")
            print(" -", OUT_SIMPLE)
            print(" -", OUT_WORDS)


if __name__ == "__main__":
    main()
