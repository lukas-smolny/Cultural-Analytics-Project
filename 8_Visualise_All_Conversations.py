#8 Visualize for all dialogs. CREATED BY CHATGPT****

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import matplotlib.pyplot as plt


# =========================
# SETTINGS
# =========================
PROJECT_ROOT = Path("Project_Files")

ALL_DIR = PROJECT_ROOT / "7_all_dialogues_min5s"
STRICT_DIR = PROJECT_ROOT / "7_two_speakers_min5s"  # optional (for comparison graph)
OUT_DIR = PROJECT_ROOT / "8_gender_visualizations_all_dialogues"

DPI = 200
EXPORT_PDF = False  # keep PNG only

# Simple labels (no jargon)
DATASET_LABEL = "ALL dialogues (no 2-speaker / no 1 Male + 1 Female filter)"
STARTER_LABEL = "Starter = speaker who talks first in the dialogue (earliest diarization segment)."
KNOWN_PCT_LABEL = "Known-starter percentages = Female/(Male+Female) and Male/(Male+Female)."
ALL_STARTER_PCT_LABEL = "All-dialogue starter split = Male/Female/Unknown as shares of all dialogues."

PERIOD_ORDER = ["nazi", "postwar"]
PERIOD_LABELS = {"nazi": "National Socialism", "postwar": "Postwar Era"}

FONT_FAMILY = "DejaVu Sans"
TITLE_SIZE = 13
AXIS_LABEL_SIZE = 10
TICK_SIZE = 9
LEGEND_SIZE = 9
NOTE_SIZE = 9
GRID_ALPHA = 0.25

FIG_WIDE = (17, 9)
FIG_WIDE_SHORT = (17, 8)
FIG_TALL = (17, 10)
FIG_TALL_LONG = (18, 10)
# =========================


# -------------------------
# HELPERS
# -------------------------
def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _pct(x: float) -> float:
    return 100.0 * x


def _period_sort_key(p: str) -> Tuple[int, str]:
    p_l = str(p).strip().lower()
    if p_l in PERIOD_ORDER:
        return (PERIOD_ORDER.index(p_l), p_l)
    return (999, p_l)


def _display_period(p: str) -> str:
    return PERIOD_LABELS.get(str(p).strip().lower(), str(p))


def _apply_style():
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": AXIS_LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "axes.prop_cycle": plt.cycler(color=[
            "#4d5b69",
            "#d6b8b8",
            "#5d735d",
            "#a69772",
        ])
    })


def _save_fig(fig, out_dir: Path, stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(bottom=0.25)
    fig.savefig(out_dir / f"{stem}.png", dpi=DPI, bbox_inches="tight")
    if EXPORT_PDF:
        fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")


def _add_note(fig, text: str):
    import textwrap
    wrapped = "\n".join(textwrap.wrap(text, 120))
    fig.text(0.5, 0.01, wrapped, ha="center", va="bottom", fontsize=9)


def _ecdf(values: List[float]) -> Tuple[List[float], List[float]]:
    vals = sorted(float(v) for v in values if v is not None)
    if not vals:
        return [0.0], [0.0]
    n = len(vals)
    y = [(i + 1) / n for i in range(n)]
    return vals, y


def _wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + (z ** 2) / n
    center = (p + (z ** 2) / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) / n) + (z ** 2) / (4 * n ** 2))) / denom
    low = max(0.0, center - half)
    high = min(1.0, center + half)
    return low, high


# -------------------------
# LOAD ALL-DIALOGUES DATA
# -------------------------
def load_all_dialogues_data(all_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Expects JSONs from MovFinal_AllDialogs.py in:
      Project_Files/7_all_dialogues/{period}/*_all_dialogues.json

    Returns:
      film_df  : one row per film
      dialog_df: one row per dialogue
    """
    film_rows: List[Dict] = []
    dialog_rows: List[Dict] = []

    if not all_dir.exists():
        raise FileNotFoundError(f"All-dialogues folder not found: {all_dir}")

    period_dirs = [p for p in all_dir.iterdir() if p.is_dir()]
    for period_dir in sorted(period_dirs, key=lambda p: _period_sort_key(p.name)):
        period = period_dir.name

        for fp in sorted(period_dir.glob("*_all_dialogues.json")):
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)

            film = fp.stem.replace("_all_dialogues", "")

            total = int(data.get("num_dialogs_total", 0))
            summary = data.get("summary", {}) or {}
            dialogs_with_known_starter_gender = int(summary.get("dialogs_with_known_starter_gender", 0))
            dialogs_with_any_speaker = int(summary.get("dialogs_with_any_speaker", 0))
            dialogs_with_segments = int(summary.get("dialogs_with_segments", 0))

            sgc = data.get("starter_gender_counts", {}) or {}
            male = int(sgc.get("Male", 0))
            female = int(sgc.get("Female", 0))
            unknown = int(sgc.get("Unknown", 0))
            known = male + female

            strict_like = data.get("strict_like_counts", {}) or {}
            strict_like_num = int(strict_like.get("num_matches", 0))
            strict_like_starter = strict_like.get("starter_gender_counts", {}) or {}
            strict_like_m = int(strict_like_starter.get("Male", 0))
            strict_like_f = int(strict_like_starter.get("Female", 0))
            strict_like_u = int(strict_like_starter.get("Unknown", 0))
            strict_like_known = strict_like_m + strict_like_f

            bucket_dist = data.get("speaker_count_distribution", {}) or {}
            exact_dist = data.get("exact_speaker_count_distribution", {}) or {}
            composition_counts = data.get("composition_counts", {}) or {}

            # speaker-count bucket shares
            b0 = int(bucket_dist.get("0", 0))
            b1 = int(bucket_dist.get("1", 0))
            b2 = int(bucket_dist.get("2", 0))
            b3 = int(bucket_dist.get("3", 0))
            b4p = int(bucket_dist.get("4+", 0))

            film_rows.append({
                "period": period,
                "film": film,
                "source_file": str(fp),

                "num_dialogs_total": total,
                "dialogs_with_segments": dialogs_with_segments,
                "dialogs_with_any_speaker": dialogs_with_any_speaker,
                "dialogs_with_known_starter_gender": dialogs_with_known_starter_gender,

                "male_starters": male,
                "female_starters": female,
                "unknown_starters": unknown,
                "known_starters": known,

                # all-dialogue split (M/F/U sum to ~100 if every dialogue gets one starter category)
                "male_share_all_dialogues": _safe_div(male, total),
                "female_share_all_dialogues": _safe_div(female, total),
                "unknown_share_all_dialogues": _safe_div(unknown, total),

                # known-only share (main gender comparison in broad dataset)
                "male_share_known": _safe_div(male, known),
                "female_share_known": _safe_div(female, known),

                "known_starter_rate": _safe_div(known, total),
                "unknown_starter_rate": _safe_div(unknown, total),

                # strict-like subset (computed inside all-dialogues file, for comparison)
                "strict_like_num_matches": strict_like_num,
                "strict_like_male_starters": strict_like_m,
                "strict_like_female_starters": strict_like_f,
                "strict_like_unknown_starters": strict_like_u,
                "strict_like_known_starters": strict_like_known,
                "strict_like_rate": _safe_div(strict_like_num, total),
                "strict_like_female_share_known": _safe_div(strict_like_f, strict_like_known),
                "strict_like_male_share_known": _safe_div(strict_like_m, strict_like_known),
                "strict_like_unknown_share": _safe_div(strict_like_u, strict_like_num),

                # speaker-count buckets
                "spk_bucket_0": b0,
                "spk_bucket_1": b1,
                "spk_bucket_2": b2,
                "spk_bucket_3": b3,
                "spk_bucket_4plus": b4p,

                "spk_bucket_0_share": _safe_div(b0, total),
                "spk_bucket_1_share": _safe_div(b1, total),
                "spk_bucket_2_share": _safe_div(b2, total),
                "spk_bucket_3_share": _safe_div(b3, total),
                "spk_bucket_4plus_share": _safe_div(b4p, total),

                # quick composition examples (safe defaults)
                "comp_2spk_mf": int(composition_counts.get("2spk_mf", 0)),
                "comp_2spk_mm": int(composition_counts.get("2spk_mm", 0)),
                "comp_2spk_ff": int(composition_counts.get("2spk_ff", 0)),
                "comp_3plus_mixed": int(composition_counts.get("3+spk_mixed", 0)),
            })

            for d in data.get("dialogs", []) or []:
                try:
                    dur = float(d.get("duration", 0.0))
                except (TypeError, ValueError):
                    dur = 0.0

                num_speakers = int(d.get("num_speakers", 0) or 0)
                speaker_bucket = str(d.get("speaker_count_bucket", "0"))
                starter_gender = str(d.get("starter_gender", "Unknown"))
                comp = str(d.get("composition_label", ""))
                is_strict = bool(d.get("is_strict_2speakers_1male_1female", False))

                dialog_rows.append({
                    "period": period,
                    "film": film,
                    "dialog_index": d.get("dialog_index"),
                    "duration": max(0.0, dur),
                    "starter_gender": starter_gender,
                    "num_speakers": num_speakers,
                    "speaker_count_bucket": speaker_bucket,
                    "composition_label": comp,
                    "is_strict_like": is_strict,
                })

    film_df = pd.DataFrame(film_rows)
    dialog_df = pd.DataFrame(dialog_rows)

    if film_df.empty:
        raise RuntimeError(f"No all-dialogues JSON files found in {all_dir}")

    film_df["period_sort"] = film_df["period"].map(lambda x: _period_sort_key(x)[0])
    film_df = film_df.sort_values(["period_sort", "film"]).drop(columns=["period_sort"]).reset_index(drop=True)
    film_df["film_id"] = [f"F{i+1:02d}" for i in range(len(film_df))]

    # Wilson CI for film-level female share among known starters (all dialogues)
    ci_low = []
    ci_high = []
    for row in film_df.itertuples(index=False):
        lo, hi = _wilson_interval(int(row.female_starters), int(row.known_starters))
        ci_low.append(lo)
        ci_high.append(hi)
    film_df["female_share_known_ci_low"] = ci_low
    film_df["female_share_known_ci_high"] = ci_high

    return film_df, dialog_df


# -------------------------
# OPTIONAL STRICT LOAD (for comparison graph)
# -------------------------
def load_strict_period_dialog_weighted(strict_dir: Path) -> pd.DataFrame:
    """
    Loads strict dataset and returns only period-level dialog-weighted female share among known starters.
    Optional helper for comparison graph.
    """
    if not strict_dir.exists():
        return pd.DataFrame()

    rows = []
    period_dirs = [p for p in strict_dir.iterdir() if p.is_dir()]
    for period_dir in sorted(period_dirs, key=lambda p: _period_sort_key(p.name)):
        period = period_dir.name
        male_sum = 0
        female_sum = 0
        unknown_sum = 0
        total_matches_sum = 0
        total_dialogs_sum = 0
        n_files = 0

        for fp in sorted(period_dir.glob("*_2speakers_1male_1female.json")):
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            c = data.get("starter_gender_counts", {}) or {}
            male_sum += int(c.get("Male", 0))
            female_sum += int(c.get("Female", 0))
            unknown_sum += int(c.get("Unknown", 0))
            total_matches_sum += int(data.get("num_matches", 0))
            total_dialogs_sum += int(data.get("num_dialogs_total", 0))
            n_files += 1

        if n_files > 0:
            known = male_sum + female_sum
            rows.append({
                "period": period,
                "n_films": n_files,
                "male_starters": male_sum,
                "female_starters": female_sum,
                "unknown_starters": unknown_sum,
                "known_starters": known,
                "num_matches": total_matches_sum,
                "num_dialogs_total": total_dialogs_sum,
                "male_share_known": _safe_div(male_sum, known),
                "female_share_known": _safe_div(female_sum, known),
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df.sort_values("period", key=lambda s: s.map(lambda x: _period_sort_key(x))).reset_index(drop=True)


# -------------------------
# SUMMARIES
# -------------------------
def build_period_pooled_summary_all(film_df: pd.DataFrame) -> pd.DataFrame:
    """
    Dialog-weighted by construction (sum counts across films within period).
    """
    rows = []
    for period, g in film_df.groupby("period"):
        total = int(g["num_dialogs_total"].sum())

        male = int(g["male_starters"].sum())
        female = int(g["female_starters"].sum())
        unknown = int(g["unknown_starters"].sum())
        known = male + female

        strict_like_num = int(g["strict_like_num_matches"].sum())
        strict_like_m = int(g["strict_like_male_starters"].sum())
        strict_like_f = int(g["strict_like_female_starters"].sum())
        strict_like_u = int(g["strict_like_unknown_starters"].sum())
        strict_like_known = strict_like_m + strict_like_f

        rows.append({
            "period": period,
            "n_films": int(g.shape[0]),

            "num_dialogs_total": total,
            "male_starters": male,
            "female_starters": female,
            "unknown_starters": unknown,
            "known_starters": known,

            # all-dialogue split (M/F/U)
            "male_share_all_dialogues": _safe_div(male, total),
            "female_share_all_dialogues": _safe_div(female, total),
            "unknown_share_all_dialogues": _safe_div(unknown, total),

            # known-only split
            "male_share_known": _safe_div(male, known),
            "female_share_known": _safe_div(female, known),

            "known_starter_rate": _safe_div(known, total),
            "unknown_starter_rate": _safe_div(unknown, total),

            # strict-like subset summary
            "strict_like_num_matches": strict_like_num,
            "strict_like_male_starters": strict_like_m,
            "strict_like_female_starters": strict_like_f,
            "strict_like_unknown_starters": strict_like_u,
            "strict_like_known_starters": strict_like_known,
            "strict_like_rate": _safe_div(strict_like_num, total),
            "strict_like_female_share_known": _safe_div(strict_like_f, strict_like_known),
            "strict_like_male_share_known": _safe_div(strict_like_m, strict_like_known),
            "strict_like_unknown_share": _safe_div(strict_like_u, strict_like_num),

            # speaker-count bucket shares pooled
            "spk_bucket_0_share": _safe_div(int(g["spk_bucket_0"].sum()), total),
            "spk_bucket_1_share": _safe_div(int(g["spk_bucket_1"].sum()), total),
            "spk_bucket_2_share": _safe_div(int(g["spk_bucket_2"].sum()), total),
            "spk_bucket_3_share": _safe_div(int(g["spk_bucket_3"].sum()), total),
            "spk_bucket_4plus_share": _safe_div(int(g["spk_bucket_4plus"].sum()), total),
        })

    df = pd.DataFrame(rows)
    return df.sort_values("period", key=lambda s: s.map(lambda x: _period_sort_key(x))).reset_index(drop=True)


def build_weight_method_summary_all(film_df: pd.DataFrame) -> pd.DataFrame:
    """
    Different weighting methods for the broad (all-dialogues) MAIN share:
      Female/(Male+Female), Male/(Male+Female)
    """
    method_rows = []

    for period, g in film_df.groupby("period"):
        n_films = int(g.shape[0])
        known_total = int(g["known_starters"].sum())

        # dialog-weighted from counts
        m_sum = int(g["male_starters"].sum())
        f_sum = int(g["female_starters"].sum())
        k_sum = m_sum + f_sum

        method_rows.append({
            "period": period,
            "method_key": "dialog_weighted",
            "method_label": "Dialog-weighted",
            "male_share_known": _safe_div(m_sum, k_sum),
            "female_share_known": _safe_div(f_sum, k_sum),
            "female_share_known_sd": 0.0,
            "n_films": n_films,
            "known_starters_total": known_total,
        })

        # film mean
        method_rows.append({
            "period": period,
            "method_key": "film_mean",
            "method_label": "Film-weighted mean (equal film)",
            "male_share_known": float(g["male_share_known"].mean()),
            "female_share_known": float(g["female_share_known"].mean()),
            "female_share_known_sd": float(g["female_share_known"].std(ddof=1)) if n_films > 1 else 0.0,
            "n_films": n_films,
            "known_starters_total": known_total,
        })

        # film median
        f_med = float(g["female_share_known"].median())
        method_rows.append({
            "period": period,
            "method_key": "film_median",
            "method_label": "Film-weighted median",
            "male_share_known": 1.0 - f_med,
            "female_share_known": f_med,
            "female_share_known_sd": 0.0,
            "n_films": n_films,
            "known_starters_total": known_total,
        })

        # sqrt(known N)
        w = g["known_starters"].astype(float).apply(lambda x: math.sqrt(max(0.0, x)))
        w_sum = float(w.sum())
        f_sqrt = float((g["female_share_known"] * w).sum() / w_sum) if w_sum else 0.0
        m_sqrt = float((g["male_share_known"] * w).sum() / w_sum) if w_sum else 0.0
        method_rows.append({
            "period": period,
            "method_key": "sqrt_known_weighted",
            "method_label": "Weighted by sqrt(known N)",
            "male_share_known": m_sqrt,
            "female_share_known": f_sqrt,
            "female_share_known_sd": 0.0,
            "n_films": n_films,
            "known_starters_total": known_total,
        })

        # weighted by all dialogues per film
        w2 = g["num_dialogs_total"].astype(float)
        w2_sum = float(w2.sum())
        f_w2 = float((g["female_share_known"] * w2).sum() / w2_sum) if w2_sum else 0.0
        m_w2 = float((g["male_share_known"] * w2).sum() / w2_sum) if w2_sum else 0.0
        method_rows.append({
            "period": period,
            "method_key": "total_dialogs_weighted",
            "method_label": "Weighted by total detected dialogues",
            "male_share_known": m_w2,
            "female_share_known": f_w2,
            "female_share_known_sd": 0.0,
            "n_films": n_films,
            "known_starters_total": known_total,
        })

    out = pd.DataFrame(method_rows)
    out["period_sort"] = out["period"].map(lambda x: _period_sort_key(x)[0])
    out = out.sort_values(["period_sort", "method_key"]).drop(columns=["period_sort"]).reset_index(drop=True)
    return out


def export_tables(film_df: pd.DataFrame, period_df: pd.DataFrame, weight_df: pd.DataFrame, dialog_df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    film_df.to_csv(out_dir / "all_dialogues_film_summary.csv", sep=";", index=False)
    period_df.to_csv(out_dir / "all_dialogues_period_summary_dialog_weighted.csv", sep=";", index=False)
    weight_df.to_csv(out_dir / "all_dialogues_period_summary_weight_methods.csv", sep=";", index=False)

    # small helper table: composition counts pooled by period
    if not dialog_df.empty:
        comp = (
            dialog_df.groupby(["period", "composition_label"])
            .size()
            .reset_index(name="count")
        )
        comp.to_csv(out_dir / "all_dialogues_composition_counts_long.csv", sep=";", index=False)


# -------------------------
# GENERIC MAIN PERCENT PLOT (M/F among known starters, 100% logic)
# -------------------------
def plot_period_gender_share_same_style(df: pd.DataFrame, out_dir: Path, title: str, filename: str, note: str):
    periods = df["period"].tolist()
    x = list(range(len(periods)))
    width = 0.36

    m_vals = [_pct(v) for v in df["male_share_known"].tolist()]
    f_vals = [_pct(v) for v in df["female_share_known"].tolist()]

    fig, ax = plt.subplots(figsize=FIG_WIDE_SHORT, dpi=DPI)
    ax.bar([i - width/2 for i in x], m_vals, width=width, label="Male starters (%)")
    ax.bar([i + width/2 for i in x], f_vals, width=width, label="Female starters (%)")

    ax.set_xticks(x)
    ax.set_xticklabels([_display_period(p) for p in periods])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share among known starters (%)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=GRID_ALPHA)
    ax.legend(loc="upper right")

    for i, row in df.reset_index(drop=True).iterrows():
        if "known_starters_total" in row.index:
            ax.text(i, 99, f"Known N={int(row['known_starters_total'])}", ha="center", va="top", fontsize=9)
        elif "known_starters" in row.index:
            ax.text(i, 99, f"Known N={int(row['known_starters'])}", ha="center", va="top", fontsize=9)
        if "n_films" in row.index:
            ax.text(i, 94, f"films={int(row['n_films'])}", ha="center", va="top", fontsize=9)

    _add_note(fig, note)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    _save_fig(fig, out_dir, filename)
    plt.close(fig)


# -------------------------
# FIGURES
# -------------------------
def plot_01_all_dialogue_starter_split_MFU(period_df: pd.DataFrame, out_dir: Path):
    """
    Male/Female/Unknown shares of ALL dialogues (sums to ~100%).
    """
    df = period_df.copy()
    periods = df["period"].tolist()
    x = list(range(len(periods)))

    m_vals = [_pct(v) for v in df["male_share_all_dialogues"].tolist()]
    f_vals = [_pct(v) for v in df["female_share_all_dialogues"].tolist()]
    u_vals = [_pct(v) for v in df["unknown_share_all_dialogues"].tolist()]

    fig, ax = plt.subplots(figsize=FIG_WIDE_SHORT, dpi=DPI)
    ax.bar(x, m_vals, label="Male starters (%)")
    ax.bar(x, f_vals, bottom=m_vals, label="Female starters (%)")
    ax.bar(x, u_vals, bottom=[m_vals[i] + f_vals[i] for i in range(len(x))], label="Unknown starters (%)")

    ax.set_xticks(x)
    ax.set_xticklabels([_display_period(p) for p in periods])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of all dialogues (%)")
    ax.set_title("Starter categories across ALL dialogues (Male / Female / Unknown)")
    ax.grid(axis="y", alpha=GRID_ALPHA)
    ax.legend(loc="upper right")

    for i, row in df.iterrows():
        ax.text(i, 99, f"Total N={int(row['num_dialogs_total'])}", ha="center", va="top", fontsize=9)

    _add_note(fig, f"Note. {DATASET_LABEL}. {STARTER_LABEL} {ALL_STARTER_PCT_LABEL}")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    _save_fig(fig, out_dir, "01_all_dialogue_starter_split_MFU_by_period")
    plt.close(fig)


def plot_02_dialog_weighted_known_share(period_df: pd.DataFrame, out_dir: Path):
    df = period_df[["period", "male_share_known", "female_share_known", "known_starters", "n_films"]].copy()
    plot_period_gender_share_same_style(
        df, out_dir,
        "Starter gender share by period (ALL dialogues, dialog-weighted, known starters only)",
        "02_all_dialogues_known_share_dialog_weighted",
        f"Note. {DATASET_LABEL}. {STARTER_LABEL} {KNOWN_PCT_LABEL} Dialog-weighted = sums all dialogues in the period."
    )


def plot_03_film_mean_known_share(weight_df: pd.DataFrame, out_dir: Path):
    df = weight_df[weight_df["method_key"] == "film_mean"][["period", "male_share_known", "female_share_known", "known_starters_total", "n_films"]].copy()
    plot_period_gender_share_same_style(
        df, out_dir,
        "Starter gender share by period (ALL dialogues, film-weighted mean)",
        "03_all_dialogues_known_share_film_mean",
        f"Note. {DATASET_LABEL}. {STARTER_LABEL} {KNOWN_PCT_LABEL} Film-weighted mean = average of per-film shares."
    )


def plot_04_compare_female_share_weights(weight_df: pd.DataFrame, out_dir: Path):
    order = ["dialog_weighted", "film_mean", "film_median", "sqrt_known_weighted", "total_dialogs_weighted"]
    label_map = {
        "dialog_weighted": "Dialog\nweighted",
        "film_mean": "Film mean\n(equal film)",
        "film_median": "Film\nmedian",
        "sqrt_known_weighted": "sqrt(N)\nweighted",
        "total_dialogs_weighted": "Total dialogs\nweighted",
    }

    df = weight_df.copy()
    df["method_order"] = df["method_key"].map({k: i for i, k in enumerate(order)})
    df = df.sort_values(["method_order", "period"]).reset_index(drop=True)

    x = list(range(len(order)))
    width = 0.36

    fig, ax = plt.subplots(figsize=FIG_TALL_LONG, dpi=DPI)
    for j, period in enumerate(PERIOD_ORDER):
        sub = df[df["period"] == period].sort_values("method_order")
        vals = [_pct(v) for v in sub["female_share_known"].tolist()]
        xpos = [i + (-width/2 if j == 0 else width/2) for i in x]
        ax.bar(xpos, vals, width=width, label=_display_period(period))

    ax.set_xticks(x)
    ax.set_xticklabels([label_map[m] for m in order])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Female starter share among known starters (%)")
    ax.set_title("ALL dialogues: female share under different weighting methods")
    ax.grid(axis="y", alpha=GRID_ALPHA)
    ax.legend(loc="upper right")

    _add_note(fig, f"Note. {DATASET_LABEL}. {STARTER_LABEL} {KNOWN_PCT_LABEL} Weighting sensitivity check.")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    _save_fig(fig, out_dir, "04_all_dialogues_compare_female_share_weight_methods")
    plt.close(fig)


def plot_05_film_female_share_dot(film_df: pd.DataFrame, out_dir: Path):
    df = film_df.copy()
    x_pos = {"nazi": 0, "postwar": 1}
    jit = [-0.14, -0.06, 0.06, 0.14, -0.10, 0.10, -0.02, 0.02]

    fig, ax = plt.subplots(figsize=FIG_WIDE, dpi=DPI)
    for period, g in df.groupby("period"):
        base = x_pos.get(period, 2)
        for j, row in enumerate(g.itertuples(index=False)):
            x = base + jit[j % len(jit)]
            y = _pct(float(row.female_share_known))
            ax.scatter(x, y, s=65)
            ax.text(x, y + 1.5, f"{row.film_id} (N={int(row.known_starters)})", ha="center", va="bottom", fontsize=8)
        mean_y = _pct(float(g["female_share_known"].mean()))
        ax.hlines(mean_y, base - 0.22, base + 0.22, linewidth=2)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Nazi", "Postwar"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Female starter share among known starters (%)")
    ax.set_title("ALL dialogues: per-film female share (known starters only)")
    ax.grid(axis="y", alpha=GRID_ALPHA)

    _add_note(fig, "Note. Film IDs map to filenames in all_dialogues_film_summary.csv.")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    _save_fig(fig, out_dir, "05_all_dialogues_film_female_share_dot")
    plt.close(fig)


def plot_06_film_female_share_with_ci(film_df: pd.DataFrame, out_dir: Path):
    df = film_df.copy()
    x_pos = {"nazi": 0, "postwar": 1}
    jit = [-0.14, -0.06, 0.06, 0.14, -0.10, 0.10, -0.02, 0.02]

    fig, ax = plt.subplots(figsize=FIG_WIDE, dpi=DPI)
    for period, g in df.groupby("period"):
        base = x_pos.get(period, 2)
        for j, row in enumerate(g.itertuples(index=False)):
            x = base + jit[j % len(jit)]
            y = _pct(float(row.female_share_known))
            lo = _pct(float(row.female_share_known_ci_low))
            hi = _pct(float(row.female_share_known_ci_high))
            ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt="o", capsize=4)
            ax.text(x, hi + 1.2, row.film_id, ha="center", va="bottom", fontsize=8)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Nazi", "Postwar"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Female starter share among known starters (%)")
    ax.set_title("ALL dialogues: per-film female share with 95% Wilson intervals")
    ax.grid(axis="y", alpha=GRID_ALPHA)

    _add_note(fig, "Note. Intervals reflect film-specific known starter sample size (Male+Female).")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    _save_fig(fig, out_dir, "06_all_dialogues_film_female_share_with_ci")
    plt.close(fig)


def plot_07_known_and_unknown_rates_per_film(film_df: pd.DataFrame, out_dir: Path):
    df = film_df.copy()
    df["period_sort"] = df["period"].map(lambda x: _period_sort_key(x)[0])
    df = df.sort_values(["period_sort", "film"]).drop(columns=["period_sort"]).reset_index(drop=True)

    labels = [f"{_display_period(p)}-{fid}" for p, fid in zip(df["period"], df["film_id"])]
    x = list(range(len(df)))

    known_vals = [_pct(v) for v in df["known_starter_rate"].tolist()]
    unknown_vals = [_pct(v) for v in df["unknown_starter_rate"].tolist()]

    fig, ax = plt.subplots(figsize=FIG_TALL, dpi=DPI)
    ax.bar(x, known_vals, label="Known starter rate (%)")
    ax.bar(x, unknown_vals, bottom=known_vals, label="Unknown starter rate (%)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Share of all dialogues (%)")
    ax.set_title("ALL dialogues: known vs unknown starter rate per film")
    ax.grid(axis="y", alpha=GRID_ALPHA)
    ax.legend(loc="upper right")

    for i, row in enumerate(df.itertuples(index=False)):
        ax.text(i, 99, f"N={int(row.num_dialogs_total)}", ha="center", va="top", fontsize=8)

    _add_note(fig, "Note. Bars show how many dialogues get a known starter (Male/Female) vs Unknown starter category.")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    _save_fig(fig, out_dir, "07_all_dialogues_known_vs_unknown_rate_per_film")
    plt.close(fig)


def plot_08_speaker_count_distribution_by_period(period_df: pd.DataFrame, out_dir: Path):
    df = period_df.copy()
    periods = df["period"].tolist()
    x = list(range(len(periods)))

    s1 = [_pct(v) for v in df["spk_bucket_1_share"].tolist()]
    s2 = [_pct(v) for v in df["spk_bucket_2_share"].tolist()]
    s3 = [_pct(v) for v in df["spk_bucket_3_share"].tolist()]
    s4 = [_pct(v) for v in df["spk_bucket_4plus_share"].tolist()]
    s0 = [_pct(v) for v in df["spk_bucket_0_share"].tolist()]

    fig, ax = plt.subplots(figsize=FIG_WIDE_SHORT, dpi=DPI)
    ax.bar(x, s1, label="1 speaker")
    ax.bar(x, s2, bottom=s1, label="2 speakers")
    ax.bar(x, s3, bottom=[s1[i] + s2[i] for i in range(len(x))], label="3 speakers")
    ax.bar(x, s4, bottom=[s1[i] + s2[i] + s3[i] for i in range(len(x))], label="4+ speakers")
    # 0-speaker bucket stacked on top only if present
    ax.bar(x, s0, bottom=[s1[i] + s2[i] + s3[i] + s4[i] for i in range(len(x))], label="0 speakers")

    ax.set_xticks(x)
    ax.set_xticklabels([_display_period(p) for p in periods])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of all dialogues (%)")
    ax.set_title("ALL dialogues: speaker-count distribution by period")
    ax.grid(axis="y", alpha=GRID_ALPHA)
    ax.legend(loc="upper right", ncols=3)

    _add_note(fig, "Note. This shows dialogue structure (1 speaker / 2 / 3 / 4+), not starter preference.")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    _save_fig(fig, out_dir, "08_all_dialogues_speaker_count_distribution_by_period")
    plt.close(fig)


def plot_09_female_share_by_speaker_count_bucket(dialog_df: pd.DataFrame, out_dir: Path):
    """
    Female share among known starters within each speaker-count bucket and period.
    """
    if dialog_df.empty:
        return

    df = dialog_df.copy()
    df = df[df["starter_gender"].isin(["Male", "Female"])].copy()
    if df.empty:
        return

    bucket_order = ["1", "2", "3", "4+"]
    rows = []
    for period, g1 in df.groupby("period"):
        for bucket, g2 in g1.groupby("speaker_count_bucket"):
            if bucket not in bucket_order:
                continue
            female = int((g2["starter_gender"] == "Female").sum())
            male = int((g2["starter_gender"] == "Male").sum())
            known = female + male
            rows.append({
                "period": period,
                "speaker_count_bucket": bucket,
                "female_share_known": _safe_div(female, known),
                "male_share_known": _safe_div(male, known),
                "known_n": known,
            })

    agg = pd.DataFrame(rows)
    if agg.empty:
        return

    x = list(range(len(bucket_order)))
    width = 0.36

    fig, ax = plt.subplots(figsize=FIG_TALL_LONG, dpi=DPI)
    for j, period in enumerate(PERIOD_ORDER):
        sub = agg[agg["period"] == period].copy()
        sub["bucket_order"] = sub["speaker_count_bucket"].map({b: i for i, b in enumerate(bucket_order)})
        sub = sub.sort_values("bucket_order")
        # align values by bucket
        vals = []
        ns = []
        for b in bucket_order:
            r = sub[sub["speaker_count_bucket"] == b]
            if r.empty:
                vals.append(0.0)
                ns.append(0)
            else:
                vals.append(_pct(float(r.iloc[0]["female_share_known"])))
                ns.append(int(r.iloc[0]["known_n"]))
        xpos = [i + (-width/2 if j == 0 else width/2) for i in x]
        ax.bar(xpos, vals, width=width, label=_display_period(period))
        for i in range(len(x)):
            ax.text(xpos[i], vals[i] + 1.2, f"N={ns[i]}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{b} speaker{'s' if b != '1' else ''}" if b != "4+" else "4+ speakers" for b in bucket_order])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Female starter share among known starters (%)")
    ax.set_title("ALL dialogues: female share by speaker-count bucket")
    ax.grid(axis="y", alpha=GRID_ALPHA)
    ax.legend(loc="upper right")

    _add_note(fig, f"Note. {KNOWN_PCT_LABEL} Computed separately within each speaker-count bucket.")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    _save_fig(fig, out_dir, "09_all_dialogues_female_share_by_speaker_count_bucket")
    plt.close(fig)


def plot_10_starter_split_in_two_speaker_dialogues_any_composition(dialog_df: pd.DataFrame, out_dir: Path):
    """
    Broad 2-speaker subset (ANY composition, not only 1M+1F).
    Shows M/F/U split across all 2-speaker dialogues.
    """
    if dialog_df.empty:
        return

    df = dialog_df.copy()
    df = df[df["speaker_count_bucket"] == "2"]
    if df.empty:
        return

    rows = []
    for period, g in df.groupby("period"):
        total = int(g.shape[0])
        m = int((g["starter_gender"] == "Male").sum())
        f = int((g["starter_gender"] == "Female").sum())
        u = int((g["starter_gender"] == "Unknown").sum())
        rows.append({
            "period": period,
            "total": total,
            "m_share": _safe_div(m, total),
            "f_share": _safe_div(f, total),
            "u_share": _safe_div(u, total),
        })

    agg = pd.DataFrame(rows)
    if agg.empty:
        return
    agg = agg.sort_values("period", key=lambda s: s.map(lambda x: _period_sort_key(x))).reset_index(drop=True)

    x = list(range(len(agg)))
    m_vals = [_pct(v) for v in agg["m_share"]]
    f_vals = [_pct(v) for v in agg["f_share"]]
    u_vals = [_pct(v) for v in agg["u_share"]]

    fig, ax = plt.subplots(figsize=FIG_WIDE_SHORT, dpi=DPI)
    ax.bar(x, m_vals, label="Male starters (%)")
    ax.bar(x, f_vals, bottom=m_vals, label="Female starters (%)")
    ax.bar(x, u_vals, bottom=[m_vals[i] + f_vals[i] for i in range(len(x))], label="Unknown starters (%)")

    ax.set_xticks(x)
    ax.set_xticklabels([_display_period(p) for p in agg["period"]])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of 2-speaker dialogues (%)")
    ax.set_title("ALL dialogues: starter split in 2-speaker dialogues (ANY gender composition)")
    ax.grid(axis="y", alpha=GRID_ALPHA)
    ax.legend(loc="upper right")

    for i, row in agg.iterrows():
        ax.text(i, 99, f"N={int(row['total'])}", ha="center", va="top", fontsize=9)

    _add_note(fig, "Note. This broad 2-speaker subset includes MM, FF, MF, and dialogues with Unknown speaker labels.")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    _save_fig(fig, out_dir, "10_all_dialogues_starter_split_two_speaker_any_composition")
    plt.close(fig)


def plot_11_top_composition_labels_by_period(dialog_df: pd.DataFrame, out_dir: Path, top_k: int = 8):
    """
    Top composition labels (e.g., 2spk_mf, 2spk_mm, 3+spk_mixed, ...)
    shown as shares of all dialogues within each period.
    """
    if dialog_df.empty:
        return

    d = dialog_df.copy()
    comp = d.groupby(["period", "composition_label"]).size().reset_index(name="count")
    if comp.empty:
        return

    # select top_k labels by total count across periods
    top_labels = (
        comp.groupby("composition_label")["count"]
        .sum()
        .sort_values(ascending=False)
        .head(top_k)
        .index.tolist()
    )

    totals = d.groupby("period").size().to_dict()

    rows = []
    for period in PERIOD_ORDER:
        for label in top_labels:
            c = int(comp[(comp["period"] == period) & (comp["composition_label"] == label)]["count"].sum())
            total = int(totals.get(period, 0))
            rows.append({
                "period": period,
                "composition_label": label,
                "share_of_all_dialogues": _safe_div(c, total),
                "count": c,
                "total_dialogues": total,
            })

    agg = pd.DataFrame(rows)
    if agg.empty:
        return

    x = list(range(len(top_labels)))
    width = 0.36

    fig, ax = plt.subplots(figsize=FIG_TALL_LONG, dpi=DPI)
    for j, period in enumerate(PERIOD_ORDER):
        sub = agg[agg["period"] == period].copy()
        sub["label_order"] = sub["composition_label"].map({lab: i for i, lab in enumerate(top_labels)})
        sub = sub.sort_values("label_order")
        vals = [_pct(v) for v in sub["share_of_all_dialogues"].tolist()]
        xpos = [i + (-width/2 if j == 0 else width/2) for i in x]
        ax.bar(xpos, vals, width=width, label=_display_period(period))

    ax.set_xticks(x)
    ax.set_xticklabels(top_labels, rotation=20, ha="right")
    ax.set_ylabel("Share of all dialogues (%)")
    ax.set_title("ALL dialogues: top composition labels by period")
    ax.grid(axis="y", alpha=GRID_ALPHA)
    ax.legend(loc="upper right")

    _add_note(fig, "Note. Composition labels describe speaker-count + gender composition categories from the all-dialogues export.")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    _save_fig(fig, out_dir, "11_all_dialogues_top_composition_labels_by_period")
    plt.close(fig)


def plot_12_strict_like_rate_per_film(film_df: pd.DataFrame, out_dir: Path):
    """
    strict-like subset inside all-dialogues outputs:
    exactly 2 speakers and exactly 1 Male + 1 Female
    """
    df = film_df.copy()
    df["period_sort"] = df["period"].map(lambda x: _period_sort_key(x)[0])
    df = df.sort_values(["period_sort", "strict_like_rate", "film"], ascending=[True, False, True]).drop(columns=["period_sort"]).reset_index(drop=True)

    labels = [f"{_display_period(p)}-{fid}" for p, fid in zip(df["period"], df["film_id"])]
    vals = [_pct(v) for v in df["strict_like_rate"].tolist()]

    fig, ax = plt.subplots(figsize=FIG_TALL, dpi=DPI)
    x = list(range(len(df)))
    ax.bar(x, vals)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Strict-like subset / all dialogues (%)")
    ax.set_title("ALL dialogues: how much of each film falls into the strict-like subset")
    ax.grid(axis="y", alpha=GRID_ALPHA)

    for i, row in enumerate(df.itertuples(index=False)):
        ax.text(i, vals[i] + 0.8, f"{int(row.strict_like_num_matches)}/{int(row.num_dialogs_total)}", ha="center", va="bottom", fontsize=8)

    _add_note(fig, "Note. Strict-like subset = exactly 2 speakers and exactly 1 Male + 1 Female (same core criteria as the strict analysis).")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    _save_fig(fig, out_dir, "12_all_dialogues_strict_like_rate_per_film")
    plt.close(fig)


def plot_13_strict_vs_all_female_share(period_all_df: pd.DataFrame, strict_period_df: pd.DataFrame, out_dir: Path):
    """
    Optional comparison graph:
    broad all-dialogues (known starters only) vs strict 2-speaker M/F subset
    """
    if strict_period_df is None or strict_period_df.empty:
        return

    a = period_all_df[["period", "female_share_known"]].copy().rename(columns={"female_share_known": "female_share_known_all"})
    s = strict_period_df[["period", "female_share_known"]].copy().rename(columns={"female_share_known": "female_share_known_strict"})

    m = pd.merge(a, s, on="period", how="inner")
    if m.empty:
        return
    m = m.sort_values("period", key=lambda x: x.map(lambda v: _period_sort_key(v))).reset_index(drop=True)

    x = list(range(len(m)))
    width = 0.36
    all_vals = [_pct(v) for v in m["female_share_known_all"]]
    strict_vals = [_pct(v) for v in m["female_share_known_strict"]]

    fig, ax = plt.subplots(figsize=FIG_WIDE_SHORT, dpi=DPI)
    ax.bar([i - width/2 for i in x], all_vals, width=width, label="ALL dialogues (known starters)")
    ax.bar([i + width/2 for i in x], strict_vals, width=width, label="STRICT 2-speaker 1M+1F")

    ax.set_xticks(x)
    ax.set_xticklabels([_display_period(p) for p in m["period"]])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Female starter share among known starters (%)")
    ax.set_title("Female share: ALL dialogues vs STRICT subset")
    ax.grid(axis="y", alpha=GRID_ALPHA)
    ax.legend(loc="upper right")

    _add_note(fig, "Note. This compares the broad all-dialogues result to the strict main analysis result using the same known-starter percentage definition.")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    _save_fig(fig, out_dir, "13_compare_strict_vs_all_female_share")
    plt.close(fig)


def plot_14_duration_boxplot(dialog_df: pd.DataFrame, out_dir: Path):
    if dialog_df.empty:
        return
    df = dialog_df[dialog_df["starter_gender"].isin(["Male", "Female"])].copy()
    if df.empty:
        return

    order = [("nazi", "Male"), ("nazi", "Female"), ("postwar", "Male"), ("postwar", "Female")]
    series, labels, ns = [], [], []
    for period, gender in order:
        vals = df[(df["period"] == period) & (df["starter_gender"] == gender)]["duration"].tolist()
        series.append(vals if vals else [0.0])
        labels.append(f"{_display_period(period)}\n{gender}")
        ns.append(int(df[(df["period"] == period) & (df["starter_gender"] == gender)].shape[0]))

    fig, ax = plt.subplots(figsize=FIG_WIDE, dpi=DPI)
    ax.boxplot(series, labels=labels, showfliers=True)
    ax.set_ylabel("Dialogue duration (seconds)")
    ax.set_title("ALL dialogues: duration by starter gender and period")
    ax.grid(axis="y", alpha=GRID_ALPHA)

    y0, y1 = ax.get_ylim()
    for i, n in enumerate(ns, start=1):
        ax.text(i, y0 + 0.01 * (y1 - y0), f"N={n}", ha="center", va="bottom", fontsize=8)

    _add_note(fig, f"Note. {DATASET_LABEL}. {STARTER_LABEL}")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    _save_fig(fig, out_dir, "14_all_dialogues_duration_boxplot")
    plt.close(fig)


def plot_15_duration_ecdf(dialog_df: pd.DataFrame, out_dir: Path):
    if dialog_df.empty:
        return
    df = dialog_df[dialog_df["starter_gender"].isin(["Male", "Female"])].copy()
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=FIG_WIDE, dpi=DPI)
    for period in PERIOD_ORDER:
        for gender in ["Male", "Female"]:
            vals = df[(df["period"] == period) & (df["starter_gender"] == gender)]["duration"].tolist()
            x, y = _ecdf(vals)
            ax.plot(x, y, label=f"{_display_period(period)} | {gender} starter")

    ax.set_xlabel("Dialogue duration (seconds)")
    ax.set_ylabel("Cumulative proportion (ECDF)")
    ax.set_title("ALL dialogues: duration distributions (ECDF)")
    ax.grid(axis="both", alpha=GRID_ALPHA)
    ax.legend(loc="lower right")

    _add_note(fig, "Note. ECDF = cumulative distribution of dialogue durations.")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    _save_fig(fig, out_dir, "15_all_dialogues_duration_ecdf")
    plt.close(fig)


def plot_16_unknown_rate_in_strict_like_subset(period_df: pd.DataFrame, out_dir: Path):
    """
    Tiny quality check for the strict-like subset inside all-dialogues summaries.
    """
    df = period_df.copy()
    periods = df["period"].tolist()
    vals = [_pct(v) for v in df["strict_like_unknown_share"].tolist()]

    fig, ax = plt.subplots(figsize=FIG_WIDE_SHORT, dpi=DPI)
    x = list(range(len(periods)))
    ax.bar(x, vals)

    ax.set_xticks(x)
    ax.set_xticklabels([_display_period(p) for p in periods])
    ax.set_ylabel("Unknown starter rate in strict-like subset (%)")
    ax.setTitle = ax.set_title("Quality check: unknown starter rate in strict-like subset")
    ax.grid(axis="y", alpha=GRID_ALPHA)

    for i, row in df.iterrows():
        ax.text(i, vals[i] + 0.1, f"{int(row['strict_like_unknown_starters'])}/{int(row['strict_like_num_matches'])}", ha="center", va="bottom", fontsize=9)

    _add_note(fig, "Note. Usually small. Included only as a quality check for the strict-like subset inside the broad export.")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    _save_fig(fig, out_dir, "16_quality_check_unknown_rate_in_strict_like_subset")
    plt.close(fig)


# -------------------------
# MAIN
# -------------------------
def main() -> int:
    _apply_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    film_df, dialog_df = load_all_dialogues_data(ALL_DIR)
    period_df = build_period_pooled_summary_all(film_df)
    weight_df = build_weight_method_summary_all(film_df)
    strict_period_df = load_strict_period_dialog_weighted(STRICT_DIR)  # optional

    export_tables(film_df, period_df, weight_df, dialog_df, OUT_DIR)

    # Core broad complementary graphs
    plot_01_all_dialogue_starter_split_MFU(period_df, OUT_DIR)
    plot_02_dialog_weighted_known_share(period_df, OUT_DIR)
    plot_03_film_mean_known_share(weight_df, OUT_DIR)
    plot_04_compare_female_share_weights(weight_df, OUT_DIR)
    plot_05_film_female_share_dot(film_df, OUT_DIR)
    plot_06_film_female_share_with_ci(film_df, OUT_DIR)
    plot_07_known_and_unknown_rates_per_film(film_df, OUT_DIR)
    plot_08_speaker_count_distribution_by_period(period_df, OUT_DIR)
    plot_09_female_share_by_speaker_count_bucket(dialog_df, OUT_DIR)
    plot_10_starter_split_in_two_speaker_dialogues_any_composition(dialog_df, OUT_DIR)
    plot_11_top_composition_labels_by_period(dialog_df, OUT_DIR, top_k=8)
    plot_12_strict_like_rate_per_film(film_df, OUT_DIR)
    plot_13_strict_vs_all_female_share(period_df, strict_period_df, OUT_DIR)  # optional
    plot_14_duration_boxplot(dialog_df, OUT_DIR)
    plot_15_duration_ecdf(dialog_df, OUT_DIR)
    plot_16_unknown_rate_in_strict_like_subset(period_df, OUT_DIR)

    print("Saved ALL-dialogues complementary visualizations to:", OUT_DIR)
    print("Films:", film_df.shape[0])
    print("Dialogs:", dialog_df.shape[0])
    print("Period rows:", period_df.shape[0])
    print("Weight-method rows:", weight_df.shape[0])
    if strict_period_df is not None and not strict_period_df.empty:
        print("Strict comparison available: yes")
    else:
        print("Strict comparison available: no (7_two_speakers not found or empty)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())