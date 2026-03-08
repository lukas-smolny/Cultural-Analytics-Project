#7. visualize dialogues 1+1 fm. as written in the course that we should indicate: this code was ***CREATED BY CHATGPT***

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
STRICT_DIR = PROJECT_ROOT / "7_two_speakers_min5s"
OUT_DIR = PROJECT_ROOT / "8_gender_visualizations_strict"


DPI = 200
EXPORT_PDF = False  # user does NOT want pdf

# Simple wording
STRICT_LABEL = "STRICT dataset = ONLY dialogues with 2 speakers: 1 Male + 1 Female"
STARTER_LABEL = "Starter = speaker who talks first in the dialogue (earliest diarization segment)."
MAIN_PCT_LABEL = "Main percentages = Female/(Male+Female) and Male/(Male+Female), i.e., known starters only."

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
    fig.subplots_adjust(bottom=0.25)  # ← přidej tento řádek
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
# DATA LOADING
# -------------------------
def load_strict_data(strict_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads strict JSON outputs from:
      Project_Files/7_two_speakers/{period}/*_2speakers_1male_1female.json

    Returns:
      film_df: one row per film
      match_df: one row per strict dialogue match
    """
    film_rows: List[Dict] = []
    match_rows: List[Dict] = []

    if not strict_dir.exists():
        raise FileNotFoundError(f"Strict folder not found: {strict_dir}")

    period_dirs = [p for p in strict_dir.iterdir() if p.is_dir()]

    for period_dir in sorted(period_dirs, key=lambda p: _period_sort_key(p.name)):
        period = period_dir.name

        for fp in sorted(period_dir.glob("*_2speakers_1male_1female.json")):
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)

            film = fp.stem.replace("_2speakers_1male_1female", "")

            c = data.get("starter_gender_counts", {}) or {}
            male = int(c.get("Male", 0))
            female = int(c.get("Female", 0))
            unknown = int(c.get("Unknown", 0))

            num_matches = int(data.get("num_matches", 0))
            num_dialogs_total = int(data.get("num_dialogs_total", 0))
            known = male + female

            film_rows.append({
                "period": period,
                "film": film,
                "source_file": str(fp),

                "num_dialogs_total": num_dialogs_total,
                "num_matches": num_matches,

                "male_starters": male,
                "female_starters": female,
                "unknown_starters": unknown,
                "known_starters": known,

                # MAIN shares (only sensible 100% split for M/F)
                "male_share_known": _safe_div(male, known),
                "female_share_known": _safe_div(female, known),

                # coverage / quality
                "coverage_strict_over_all_dialogs": _safe_div(num_matches, num_dialogs_total),
                "unknown_share_strict_matches": _safe_div(unknown, num_matches),

                # "rate" style metrics (not 100% split, kept explicit)
                "male_per_100_all_dialogs": 100.0 * _safe_div(male, num_dialogs_total),
                "female_per_100_all_dialogs": 100.0 * _safe_div(female, num_dialogs_total),
            })

            for m in data.get("matches", []) or []:
                starter_gender = str(m.get("starter_gender", "Unknown"))
                try:
                    dur = float(m.get("duration", 0.0))
                except (TypeError, ValueError):
                    dur = 0.0

                match_rows.append({
                    "period": period,
                    "film": film,
                    "dialog_index": m.get("dialog_index"),
                    "starter_gender": starter_gender,
                    "duration": max(0.0, dur),
                })

    film_df = pd.DataFrame(film_rows)
    match_df = pd.DataFrame(match_rows)

    if film_df.empty:
        raise RuntimeError(f"No strict JSON files found in {strict_dir}")

    film_df["period_sort"] = film_df["period"].map(lambda x: _period_sort_key(x)[0])
    film_df = film_df.sort_values(["period_sort", "film"]).drop(columns=["period_sort"]).reset_index(drop=True)
    film_df["film_id"] = [f"F{i+1:02d}" for i in range(len(film_df))]

    # Wilson CI for per-film female share among known starters
    ci_low = []
    ci_high = []
    for row in film_df.itertuples(index=False):
        lo, hi = _wilson_interval(int(row.female_starters), int(row.known_starters))
        ci_low.append(lo)
        ci_high.append(hi)
    film_df["female_share_known_ci_low"] = ci_low
    film_df["female_share_known_ci_high"] = ci_high

    return film_df, match_df


# -------------------------
# SUMMARIES
# -------------------------
def build_period_pooled_summary(film_df: pd.DataFrame) -> pd.DataFrame:
    """
    Dialog-weighted by construction (sums counts across films in each period).
    """
    rows = []
    for period, g in film_df.groupby("period"):
        male = int(g["male_starters"].sum())
        female = int(g["female_starters"].sum())
        unknown = int(g["unknown_starters"].sum())
        known = male + female
        matches = int(g["num_matches"].sum())
        dialogs = int(g["num_dialogs_total"].sum())

        rows.append({
            "period": period,
            "n_films": int(g.shape[0]),

            "male_starters": male,
            "female_starters": female,
            "unknown_starters": unknown,
            "known_starters": known,

            "num_matches": matches,
            "num_dialogs_total": dialogs,

            # MAIN shares (100% split)
            "male_share_known": _safe_div(male, known),
            "female_share_known": _safe_div(female, known),

            # quality / coverage
            "coverage_strict_over_all_dialogs": _safe_div(matches, dialogs),
            "unknown_share_strict_matches": _safe_div(unknown, matches),

            # rate style (NOT 100% split)
            "male_per_100_all_dialogs": 100.0 * _safe_div(male, dialogs),
            "female_per_100_all_dialogs": 100.0 * _safe_div(female, dialogs),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("period", key=lambda s: s.map(lambda x: _period_sort_key(x))).reset_index(drop=True)
    return df


def build_weight_method_summary(film_df: pd.DataFrame) -> pd.DataFrame:
    """
    Period-level summaries for multiple weighting strategies of MAIN percentage:
      Female/(Male+Female), Male/(Male+Female)
    """
    method_rows = []

    for period, g in film_df.groupby("period"):
        g = g.copy()
        n_films = int(g.shape[0])
        known_total = int(g["known_starters"].sum())

        # 1) dialog-weighted
        male_sum = int(g["male_starters"].sum())
        female_sum = int(g["female_starters"].sum())
        known_sum = male_sum + female_sum

        method_rows.append({
            "period": period,
            "method_key": "dialog_weighted",
            "method_label": "Dialog-weighted (sum of all dialogues)",
            "female_share_known": _safe_div(female_sum, known_sum),
            "male_share_known": _safe_div(male_sum, known_sum),
            "female_share_known_sd": 0.0,
            "n_films": n_films,
            "known_starters_total": known_total,
        })

        # 2) film mean
        method_rows.append({
            "period": period,
            "method_key": "film_mean",
            "method_label": "Film-weighted mean (each film equal)",
            "female_share_known": float(g["female_share_known"].mean()),
            "male_share_known": float(g["male_share_known"].mean()),
            "female_share_known_sd": float(g["female_share_known"].std(ddof=1)) if n_films > 1 else 0.0,
            "n_films": n_films,
            "known_starters_total": known_total,
        })

        # 3) film median
        f_med = float(g["female_share_known"].median())
        method_rows.append({
            "period": period,
            "method_key": "film_median",
            "method_label": "Film-weighted median",
            "female_share_known": f_med,
            "male_share_known": 1.0 - f_med,
            "female_share_known_sd": 0.0,
            "n_films": n_films,
            "known_starters_total": known_total,
        })

        # 4) sqrt(known N)-weighted
        w = g["known_starters"].astype(float).apply(lambda x: math.sqrt(max(0.0, x)))
        w_sum = float(w.sum())
        f_sqrt = float((g["female_share_known"] * w).sum() / w_sum) if w_sum else 0.0
        m_sqrt = float((g["male_share_known"] * w).sum() / w_sum) if w_sum else 0.0
        method_rows.append({
            "period": period,
            "method_key": "sqrt_known_weighted",
            "method_label": "Weighted by sqrt(known N) per film",
            "female_share_known": f_sqrt,
            "male_share_known": m_sqrt,
            "female_share_known_sd": 0.0,
            "n_films": n_films,
            "known_starters_total": known_total,
        })

        # 5) total-detected-dialogues-weighted
        w2 = g["num_dialogs_total"].astype(float)
        w2_sum = float(w2.sum())
        f_w2 = float((g["female_share_known"] * w2).sum() / w2_sum) if w2_sum else 0.0
        m_w2 = float((g["male_share_known"] * w2).sum() / w2_sum) if w2_sum else 0.0
        method_rows.append({
            "period": period,
            "method_key": "total_dialogs_weighted",
            "method_label": "Weighted by total detected dialogues per film",
            "female_share_known": f_w2,
            "male_share_known": m_w2,
            "female_share_known_sd": 0.0,
            "n_films": n_films,
            "known_starters_total": known_total,
        })

    out = pd.DataFrame(method_rows)
    out["period_sort"] = out["period"].map(lambda x: _period_sort_key(x)[0])
    out = out.sort_values(["period_sort", "method_key"]).drop(columns=["period_sort"]).reset_index(drop=True)
    return out


def export_tables(film_df: pd.DataFrame, pooled_df: pd.DataFrame, weight_df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    film_df.to_csv(out_dir / "strict_film_summary.csv", sep=";", index=False)
    pooled_df.to_csv(out_dir / "strict_period_summary_dialog_weighted.csv", sep=";", index=False)
    weight_df.to_csv(out_dir / "strict_period_summary_weight_methods.csv", sep=";", index=False)


# -------------------------
# GENERIC MAIN SHARE PLOT (100% split logic)
# -------------------------
def plot_period_gender_share_same_style(df: pd.DataFrame, out_dir: Path, title: str, filename: str, note: str):
    """
    Expects columns:
      period, male_share_known, female_share_known
    Optional:
      n_films, known_starters_total / known_starters
    """
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
# 01-05: SAME METRIC, DIFFERENT WEIGHTS
# -------------------------
def plot_01_dialog_weighted_percent(pooled_df: pd.DataFrame, out_dir: Path):
    df = pooled_df[["period", "male_share_known", "female_share_known", "known_starters", "n_films"]].copy()
    plot_period_gender_share_same_style(
        df, out_dir,
        "Starter gender share by period (Dialog-weighted)",
        "01_period_gender_share_dialog_weighted_strict",
        f"Note. {STRICT_LABEL}. {STARTER_LABEL} {MAIN_PCT_LABEL} Dialog-weighted = sums all strict dialogues in the period."
    )


def plot_02_film_mean_percent(weight_df: pd.DataFrame, out_dir: Path):
    df = weight_df[weight_df["method_key"] == "film_mean"][["period", "male_share_known", "female_share_known", "known_starters_total", "n_films"]].copy()
    plot_period_gender_share_same_style(
        df, out_dir,
        "Starter gender share by period (Film-weighted mean)",
        "02_period_gender_share_film_mean_strict",
        f"Note. {STRICT_LABEL}. {STARTER_LABEL} {MAIN_PCT_LABEL} Film-weighted mean = average of per-film shares (each film same weight)."
    )


def plot_03_film_median_percent(weight_df: pd.DataFrame, out_dir: Path):
    df = weight_df[weight_df["method_key"] == "film_median"][["period", "male_share_known", "female_share_known", "known_starters_total", "n_films"]].copy()
    plot_period_gender_share_same_style(
        df, out_dir,
        "Starter gender share by period (Film-weighted median)",
        "03_period_gender_share_film_median_strict",
        f"Note. {STRICT_LABEL}. {STARTER_LABEL} {MAIN_PCT_LABEL} Uses median of per-film shares (more robust to outlier films)."
    )


def plot_04_sqrtN_weighted_percent(weight_df: pd.DataFrame, out_dir: Path):
    df = weight_df[weight_df["method_key"] == "sqrt_known_weighted"][["period", "male_share_known", "female_share_known", "known_starters_total", "n_films"]].copy()
    plot_period_gender_share_same_style(
        df, out_dir,
        "Starter gender share by period (Weighted by sqrt(known N) per film)",
        "04_period_gender_share_sqrtN_weighted_strict",
        f"Note. {STRICT_LABEL}. {STARTER_LABEL} {MAIN_PCT_LABEL} Compromise between equal-film and dialog-weighted summaries."
    )


def plot_05_total_dialogs_weighted_percent(weight_df: pd.DataFrame, out_dir: Path):
    df = weight_df[weight_df["method_key"] == "total_dialogs_weighted"][["period", "male_share_known", "female_share_known", "known_starters_total", "n_films"]].copy()
    plot_period_gender_share_same_style(
        df, out_dir,
        "Starter gender share by period (Weighted by total detected dialogues per film)",
        "05_period_gender_share_total_dialogs_weighted_strict",
        f"Note. {STRICT_LABEL}. {STARTER_LABEL} {MAIN_PCT_LABEL} Sensitivity variant; weights reflect total detected dialogues per film."
    )


# -------------------------
# 06: Compare weight methods (female share only)
# -------------------------
def plot_06_compare_female_share_across_weight_methods(weight_df: pd.DataFrame, out_dir: Path):
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
    ax.set_title("Female share by period under different weighting methods")
    ax.grid(axis="y", alpha=GRID_ALPHA)
    ax.legend(loc="upper right")

    _add_note(fig, f"Note. {STRICT_LABEL}. {STARTER_LABEL} {MAIN_PCT_LABEL} This is a weighting sensitivity check.")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    _save_fig(fig, out_dir, "06_compare_female_share_across_weight_methods_strict")
    plt.close(fig)


# -------------------------
# 07: Same 100% split logic, two weights in one graph
# -------------------------
def plot_07_same_percent_logic_two_weights(weight_df: pd.DataFrame, out_dir: Path):
    sub = weight_df[weight_df["method_key"].isin(["dialog_weighted", "film_mean"])].copy()

    x_labels, male_vals, female_vals = [], [], []
    for period in PERIOD_ORDER:
        for method in ["dialog_weighted", "film_mean"]:
            row = sub[(sub["period"] == period) & (sub["method_key"] == method)]
            if row.empty:
                continue
            r = row.iloc[0]
            x_labels.append(f"{_display_period(period)}\n{'Dialog-weighted' if method=='dialog_weighted' else 'Film-weighted mean'}")
            male_vals.append(_pct(float(r["male_share_known"])))
            female_vals.append(_pct(float(r["female_share_known"])))

    fig, ax = plt.subplots(figsize=FIG_TALL, dpi=DPI)
    x = list(range(len(x_labels)))
    ax.bar(x, male_vals, label="Male starters (%)")
    ax.bar(x, female_vals, bottom=male_vals, label="Female starters (%)")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share among known starters (%)")
    ax.set_title("Same 100% split logic, shown for two weighting methods")
    ax.grid(axis="y", alpha=GRID_ALPHA)
    ax.legend(loc="upper right")

    _add_note(fig, f"Note. {STRICT_LABEL}. {STARTER_LABEL} {MAIN_PCT_LABEL}")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    _save_fig(fig, out_dir, "07_same_percent_logic_two_weights_strict")
    plt.close(fig)


# -------------------------
# 08-13: Film-level shares / counts / quality
# -------------------------
def plot_08_film_female_share_dot(film_df: pd.DataFrame, out_dir: Path):
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
    ax.set_xticklabels(["National Socialism", "Postwar Era"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Female starter share among known starters (%)")
    ax.set_title("Per-film female share (each point = one film)")
    ax.grid(axis="y", alpha=GRID_ALPHA)

    _add_note(fig, "Note. Film IDs map to filenames in strict_film_summary.csv.")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    _save_fig(fig, out_dir, "08_film_female_share_dot_strict")
    plt.close(fig)


def plot_09_film_female_share_with_ci(film_df: pd.DataFrame, out_dir: Path):
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
    ax.set_xticklabels(["National Socialism", "Postwar Era"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Female starter share among known starters (%)")
    ax.set_title("Per-film female share with 95% Wilson intervals")
    ax.grid(axis="y", alpha=GRID_ALPHA)

    _add_note(fig, "Note. Intervals reflect film-specific sample size (known starter N).")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    _save_fig(fig, out_dir, "09_film_female_share_with_ci_strict")
    plt.close(fig)


def plot_10_film_gender_split_100_stacked(film_df: pd.DataFrame, out_dir: Path):
    df = film_df.copy()
    df["period_sort"] = df["period"].map(lambda x: _period_sort_key(x)[0])
    df = df.sort_values(["period_sort", "female_share_known", "film"], ascending=[True, False, True]).drop(columns=["period_sort"]).reset_index(drop=True)

    labels = [f"{_display_period(p)}-{fid}" for p, fid in zip(df["period"], df["film_id"])]
    male = [_pct(v) for v in df["male_share_known"].tolist()]
    female = [_pct(v) for v in df["female_share_known"].tolist()]

    fig, ax = plt.subplots(figsize=FIG_TALL, dpi=DPI)
    x = list(range(len(df)))
    ax.bar(x, male, label="Male starters (%)")
    ax.bar(x, female, bottom=male, label="Female starters (%)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share among known starters (%)")
    ax.set_title("Male vs Female starter share per film (100% bars)")
    ax.grid(axis="y", alpha=GRID_ALPHA)
    ax.legend(loc="upper right")

    for i, row in enumerate(df.itertuples(index=False)):
        ax.text(i, 2.0, f"N={int(row.known_starters)}", ha="center", va="bottom", fontsize=8)

    _add_note(fig, f"Note. {STRICT_LABEL}. {MAIN_PCT_LABEL}")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    _save_fig(fig, out_dir, "10_film_gender_split_100_stacked_strict")
    plt.close(fig)


def plot_11_film_gender_counts_absolute(film_df: pd.DataFrame, out_dir: Path):
    df = film_df.copy()
    df["period_sort"] = df["period"].map(lambda x: _period_sort_key(x)[0])
    df = df.sort_values(["period_sort", "film"]).drop(columns=["period_sort"]).reset_index(drop=True)

    labels = [f"{_display_period(p)}\n{fid}" for p, fid in zip(df["period"], df["film_id"])]
    x = list(range(len(df)))
    width = 0.36

    fig, ax = plt.subplots(figsize=FIG_TALL, dpi=DPI)
    ax.bar([i - width/2 for i in x], df["male_starters"], width=width, label="Male starter count")
    ax.bar([i + width/2 for i in x], df["female_starters"], width=width, label="Female starter count")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Number of strict dialogues")
    ax.set_title("Absolute starter counts per film")
    ax.grid(axis="y", alpha=GRID_ALPHA)
    ax.legend(loc="upper right")

    _add_note(fig, "Note. Raw counts per film (not percentages). Use together with percentage and N graphs.")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    _save_fig(fig, out_dir, "11_film_gender_counts_absolute_strict")
    plt.close(fig)


def plot_12_coverage_per_film(film_df: pd.DataFrame, out_dir: Path):
    df = film_df.copy()
    df["period_sort"] = df["period"].map(lambda x: _period_sort_key(x)[0])
    df = df.sort_values(["period_sort", "coverage_strict_over_all_dialogs", "film"], ascending=[True, False, True]).drop(columns=["period_sort"]).reset_index(drop=True)

    labels = [f"{_display_period(p)}-{fid}" for p, fid in zip(df["period"], df["film_id"])]
    vals = [_pct(v) for v in df["coverage_strict_over_all_dialogs"].tolist()]

    fig, ax = plt.subplots(figsize=FIG_TALL, dpi=DPI)
    x = list(range(len(df)))
    ax.bar(x, vals)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Strict dialogues / all detected dialogues (%)")
    ax.set_title("Coverage per film (how much data remains after strict filter)")
    ax.grid(axis="y", alpha=GRID_ALPHA)

    for i, row in enumerate(df.itertuples(index=False)):
        ax.text(i, vals[i] + 0.8, f"{int(row.num_matches)}/{int(row.num_dialogs_total)}", ha="center", va="bottom", fontsize=8)

    _add_note(fig, "Note. Labels show kept/total dialogues for the strict filter.")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    _save_fig(fig, out_dir, "12_coverage_per_film_strict")
    plt.close(fig)


def plot_13_known_n_per_film(film_df: pd.DataFrame, out_dir: Path):
    df = film_df.copy()
    df["period_sort"] = df["period"].map(lambda x: _period_sort_key(x)[0])
    df = df.sort_values(["period_sort", "known_starters", "film"], ascending=[True, False, True]).drop(columns=["period_sort"]).reset_index(drop=True)

    labels = [f"{_display_period(p)}-{fid}" for p, fid in zip(df["period"], df["film_id"])]
    vals = df["known_starters"].tolist()

    fig, ax = plt.subplots(figsize=FIG_TALL, dpi=DPI)
    x = list(range(len(df)))
    ax.bar(x, vals)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Known starter N (Male+Female)")
    ax.set_title("Known starter sample size per film")
    ax.grid(axis="y", alpha=GRID_ALPHA)

    top_pad = max(1, int(max(vals) * 0.01)) if len(vals) else 1
    for i, v in enumerate(vals):
        ax.text(i, v + top_pad, f"N={int(v)}", ha="center", va="bottom", fontsize=8)

    _add_note(fig, "Note. This is the sample size behind the gender percentages per film.")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    _save_fig(fig, out_dir, "13_known_n_per_film_strict")
    plt.close(fig)


# -------------------------
# 14-16: Robustness / alternative summaries
# -------------------------
def plot_14_female_share_vs_coverage_scatter(film_df: pd.DataFrame, out_dir: Path):
    df = film_df.copy()

    fig, ax = plt.subplots(figsize=FIG_WIDE_SHORT, dpi=DPI)
    for period in PERIOD_ORDER:
        g = df[df["period"] == period]
        if g.empty:
            continue
        x = [_pct(v) for v in g["coverage_strict_over_all_dialogs"]]
        y = [_pct(v) for v in g["female_share_known"]]
        ax.scatter(x, y, s=70, label=_display_period(period))
        for row in g.itertuples(index=False):
            ax.text(_pct(row.coverage_strict_over_all_dialogs), _pct(row.female_share_known) + 1.2, row.film_id, ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Coverage: strict dialogues / all detected dialogues (%)")
    ax.set_ylabel("Female starter share among known starters (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Check for coverage effect (film-level)")
    ax.grid(axis="both", alpha=GRID_ALPHA)
    ax.legend(loc="upper right")

    _add_note(fig, "Note. Helps check whether coverage is strongly related to the film-level female share.")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    _save_fig(fig, out_dir, "14_female_share_vs_coverage_scatter_strict")
    plt.close(fig)


def plot_15_balance_index_per_film(film_df: pd.DataFrame, out_dir: Path):
    df = film_df.copy()
    df["balance_index"] = df.apply(
        lambda r: _safe_div((r["female_starters"] - r["male_starters"]), (r["female_starters"] + r["male_starters"])),
        axis=1
    )

    x_pos = {"nazi": 0, "postwar": 1}
    jit = [-0.14, -0.06, 0.06, 0.14, -0.10, 0.10, -0.02, 0.02]

    fig, ax = plt.subplots(figsize=FIG_WIDE_SHORT, dpi=DPI)

    for period, g in df.groupby("period"):
        base = x_pos.get(period, 2)
        for j, row in enumerate(g.itertuples(index=False)):
            x = base + jit[j % len(jit)]
            y = float(row.balance_index)
            ax.scatter(x, y, s=65)
            ax.text(x, y + 0.03, row.film_id, ha="center", va="bottom", fontsize=8)
        mean_y = float(g["balance_index"].mean())
        ax.hlines(mean_y, base - 0.22, base + 0.22, linewidth=2)

    ax.axhline(0.0, linewidth=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Nazi", "Postwar"])
    ax.set_ylabel("(Female - Male) / (Female + Male)")
    ax.set_title("Balance index per film (positive = more female starters)")
    ax.grid(axis="y", alpha=GRID_ALPHA)

    _add_note(fig, "Note. Alternative one-number summary for each film, based on known starters only.")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    _save_fig(fig, out_dir, "15_balance_index_per_film_strict")
    plt.close(fig)


def plot_16_unknown_rate_in_strict_matches(pooled_df: pd.DataFrame, out_dir: Path):
    """
    Tiny quality check for the thing that caused confusion.
    This is NOT a main result, just shows unknown starter rate in strict matches.
    """
    df = pooled_df.copy()
    periods = df["period"].tolist()
    vals = [_pct(v) for v in df["unknown_share_strict_matches"].tolist()]

    fig, ax = plt.subplots(figsize=FIG_WIDE_SHORT, dpi=DPI)
    x = list(range(len(periods)))
    ax.bar(x, vals)

    ax.set_xticks(x)
    ax.set_xticklabels([_display_period(p) for p in periods])
    ax.set_ylabel("Unknown starter rate within strict matches (%)")
    ax.set_title("Quality check: unknown starter rate in strict matches")
    ax.grid(axis="y", alpha=GRID_ALPHA)

    for i, row in df.iterrows():
        ax.text(i, vals[i] + 0.1, f"{int(row['unknown_starters'])}/{int(row['num_matches'])}", ha="center", va="bottom", fontsize=9)

    _add_note(fig, "Note. This explains why percentages based on strict matches can be slightly below 100% when using Male% + Female% with strict matches as denominator.")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    _save_fig(fig, out_dir, "16_unknown_rate_in_strict_matches_quality_check")
    plt.close(fig)


# -------------------------
# 17-19: DURATION
# -------------------------
def plot_17_duration_boxplot(match_df: pd.DataFrame, out_dir: Path):
    if match_df.empty:
        return
    df = match_df[match_df["starter_gender"].isin(["Male", "Female"])].copy()
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
    ax.set_title("Dialogue duration by starter gender and period")
    ax.grid(axis="y", alpha=GRID_ALPHA)

    y0, y1 = ax.get_ylim()
    for i, n in enumerate(ns, start=1):
        ax.text(i, y0 + 0.01 * (y1 - y0), f"N={n}", ha="center", va="bottom", fontsize=8)

    _add_note(fig, f"Note. {STRICT_LABEL}. {STARTER_LABEL}")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    _save_fig(fig, out_dir, "17_duration_boxplot_strict")
    plt.close(fig)


def plot_18_duration_ecdf(match_df: pd.DataFrame, out_dir: Path):
    if match_df.empty:
        return
    df = match_df[match_df["starter_gender"].isin(["Male", "Female"])].copy()
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
    ax.set_title("Dialogue duration distributions (ECDF)")
    ax.grid(axis="both", alpha=GRID_ALPHA)
    ax.legend(loc="lower right")

    _add_note(fig, "Note. ECDF = cumulative distribution of dialogue durations.")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    _save_fig(fig, out_dir, "18_duration_ecdf_strict")
    plt.close(fig)


def plot_19_duration_mean_per_film_by_starter(match_df: pd.DataFrame, out_dir: Path):
    """
    Film-level means of durations for Male-starter and Female-starter strict dialogues.
    This avoids one dialogue-rich film dominating as much as pooled dialogue-level plots.
    """
    if match_df.empty:
        return

    df = match_df[match_df["starter_gender"].isin(["Male", "Female"])].copy()
    if df.empty:
        return

    # mean duration per film x starter gender
    g = (
        df.groupby(["period", "film", "starter_gender"])["duration"]
        .mean()
        .reset_index()
    )

    # then aggregate across films per period
    rows = []
    for period, sub in g.groupby("period"):
        for gender in ["Male", "Female"]:
            vals = sub[sub["starter_gender"] == gender]["duration"].tolist()
            if len(vals) == 0:
                mean_v = 0.0
                sd_v = 0.0
                n_films = 0
            else:
                mean_v = float(pd.Series(vals).mean())
                sd_v = float(pd.Series(vals).std(ddof=1)) if len(vals) > 1 else 0.0
                n_films = len(vals)
            rows.append({
                "period": period,
                "starter_gender": gender,
                "mean_duration_per_film": mean_v,
                "sd_across_films": sd_v,
                "n_films_with_data": n_films,
            })

    agg = pd.DataFrame(rows)
    if agg.empty:
        return

    x = list(range(len(PERIOD_ORDER)))
    width = 0.36
    fig, ax = plt.subplots(figsize=FIG_WIDE_SHORT, dpi=DPI)

    for j, gender in enumerate(["Male", "Female"]):
        vals = []
        sds = []
        for period in PERIOD_ORDER:
            row = agg[(agg["period"] == period) & (agg["starter_gender"] == gender)]
            if row.empty:
                vals.append(0.0)
                sds.append(0.0)
            else:
                vals.append(float(row.iloc[0]["mean_duration_per_film"]))
                sds.append(float(row.iloc[0]["sd_across_films"]))
        xpos = [i + (-width/2 if gender == "Male" else width/2) for i in x]
        ax.bar(xpos, vals, width=width, yerr=sds, capsize=4, label=f"{gender} starter")

    ax.set_xticks(x)
    ax.set_xticklabels([_display_period(p) for p in PERIOD_ORDER])
    ax.set_ylabel("Mean dialogue duration per film (seconds)")
    ax.set_title("Average duration per film by starter gender (with SD across films)")
    ax.grid(axis="y", alpha=GRID_ALPHA)
    ax.legend(loc="upper right")

    _add_note(fig, "Note. This is a film-level duration summary (each film contributes one mean per starter gender, if available).")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    _save_fig(fig, out_dir, "19_duration_mean_per_film_by_starter_strict")
    plt.close(fig)


# -------------------------
# 20-22: COUNTS / RATES (clearly NOT 100% split)
# -------------------------
def plot_20_absolute_counts_by_period_dialog_weighted(pooled_df: pd.DataFrame, out_dir: Path):
    df = pooled_df.copy()
    periods = df["period"].tolist()
    x = list(range(len(periods)))
    width = 0.36

    male = df["male_starters"].tolist()
    female = df["female_starters"].tolist()

    fig, ax = plt.subplots(figsize=FIG_WIDE_SHORT, dpi=DPI)
    ax.bar([i - width/2 for i in x], male, width=width, label="Male starter count")
    ax.bar([i + width/2 for i in x], female, width=width, label="Female starter count")

    ax.set_xticks(x)
    ax.set_xticklabels([_display_period(p) for p in periods])
    ax.set_ylabel("Number of strict dialogues")
    ax.set_title("Absolute starter counts by period (dialog-weighted counts)")
    ax.grid(axis="y", alpha=GRID_ALPHA)
    ax.legend(loc="upper right")

    for i in x:
        ax.text(i - width/2, male[i] + 1, f"n={int(male[i])}", ha="center", va="bottom", fontsize=9)
        ax.text(i + width/2, female[i] + 1, f"n={int(female[i])}", ha="center", va="bottom", fontsize=9)

    _add_note(fig, "Note. Raw counts (not percentages). Useful as context, but main interpretation should use percentage graphs.")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    _save_fig(fig, out_dir, "20_absolute_counts_by_period_dialog_weighted_strict")
    plt.close(fig)


def plot_21_mean_counts_per_film_with_sd(film_df: pd.DataFrame, out_dir: Path):
    rows = []
    for period, g in film_df.groupby("period"):
        rows.append({
            "period": period,
            "male_mean": float(g["male_starters"].mean()),
            "female_mean": float(g["female_starters"].mean()),
            "male_sd": float(g["male_starters"].std(ddof=1)) if g.shape[0] > 1 else 0.0,
            "female_sd": float(g["female_starters"].std(ddof=1)) if g.shape[0] > 1 else 0.0,
            "n_films": int(g.shape[0]),
        })
    df = pd.DataFrame(rows).sort_values("period", key=lambda s: s.map(lambda x: _period_sort_key(x))).reset_index(drop=True)

    periods = df["period"].tolist()
    x = list(range(len(periods)))
    width = 0.36

    fig, ax = plt.subplots(figsize=FIG_WIDE_SHORT, dpi=DPI)
    ax.bar([i - width/2 for i in x], df["male_mean"], yerr=df["male_sd"], width=width, label="Male mean count per film", capsize=4)
    ax.bar([i + width/2 for i in x], df["female_mean"], yerr=df["female_sd"], width=width, label="Female mean count per film", capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels([_display_period(p) for p in periods])
    ax.set_ylabel("Mean count per film")
    ax.set_title("Average starter counts per film (with SD across films)")
    ax.grid(axis="y", alpha=GRID_ALPHA)
    ax.legend(loc="upper right")

    _add_note(fig, "Note. Film-level count summary (not percentage).")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    _save_fig(fig, out_dir, "21_mean_counts_per_film_with_sd_strict")
    plt.close(fig)


def plot_22_rates_per_100_all_detected_dialogues(pooled_df: pd.DataFrame, out_dir: Path):
    """
    Clear "rate" graph (NOT a 100% split). This is optional but can be useful.
    """
    df = pooled_df.copy()
    periods = df["period"].tolist()
    x = list(range(len(periods)))
    width = 0.36

    male_rates = df["male_per_100_all_dialogs"].tolist()
    female_rates = df["female_per_100_all_dialogs"].tolist()

    fig, ax = plt.subplots(figsize=FIG_WIDE_SHORT, dpi=DPI)
    ax.bar([i - width/2 for i in x], male_rates, width=width, label="Male starters per 100 detected dialogues")
    ax.bar([i + width/2 for i in x], female_rates, width=width, label="Female starters per 100 detected dialogues")

    ax.set_xticks(x)
    ax.set_xticklabels([_display_period(p) for p in periods])
    ax.set_ylabel("Count per 100 detected dialogues")
    ax.set_title("Starter rates per 100 detected dialogues (NOT a 100% split)")
    ax.grid(axis="y", alpha=GRID_ALPHA)
    ax.legend(loc="upper right")

    _add_note(fig, "Note. This is a rate metric with denominator = all detected dialogues. Male+Female do NOT need to sum to 100 here.")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    _save_fig(fig, out_dir, "22_rates_per_100_all_detected_dialogues_strict")
    plt.close(fig)


# -------------------------
# MAIN
# -------------------------
def main() -> int:
    _apply_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    film_df, match_df = load_strict_data(STRICT_DIR)
    pooled_df = build_period_pooled_summary(film_df)
    weight_df = build_weight_method_summary(film_df)
    export_tables(film_df, pooled_df, weight_df, OUT_DIR)

    # MAIN % graphs (same denominator logic = known starters only)
    plot_01_dialog_weighted_percent(pooled_df, OUT_DIR)
    plot_02_film_mean_percent(weight_df, OUT_DIR)
    plot_03_film_median_percent(weight_df, OUT_DIR)
    plot_04_sqrtN_weighted_percent(weight_df, OUT_DIR)
    plot_05_total_dialogs_weighted_percent(weight_df, OUT_DIR)
    plot_06_compare_female_share_across_weight_methods(weight_df, OUT_DIR)
    plot_07_same_percent_logic_two_weights(weight_df, OUT_DIR)

    # Film-level / quality / robustness
    plot_08_film_female_share_dot(film_df, OUT_DIR)
    plot_09_film_female_share_with_ci(film_df, OUT_DIR)
    plot_10_film_gender_split_100_stacked(film_df, OUT_DIR)
    plot_11_film_gender_counts_absolute(film_df, OUT_DIR)
    plot_12_coverage_per_film(film_df, OUT_DIR)
    plot_13_known_n_per_film(film_df, OUT_DIR)
    plot_14_female_share_vs_coverage_scatter(film_df, OUT_DIR)
    plot_15_balance_index_per_film(film_df, OUT_DIR)
    plot_16_unknown_rate_in_strict_matches(pooled_df, OUT_DIR)  # tiny quality check, optional but useful

    # Durations
    plot_17_duration_boxplot(match_df, OUT_DIR)
    plot_18_duration_ecdf(match_df, OUT_DIR)
    plot_19_duration_mean_per_film_by_starter(match_df, OUT_DIR)

    # Counts / rates (explicitly not 100% split)
    plot_20_absolute_counts_by_period_dialog_weighted(pooled_df, OUT_DIR)
    plot_21_mean_counts_per_film_with_sd(film_df, OUT_DIR)
    plot_22_rates_per_100_all_detected_dialogues(pooled_df, OUT_DIR)

    print("Saved strict visualizations to:", OUT_DIR)
    print("Films:", film_df.shape[0])
    print("Strict dialogues (matches):", match_df.shape[0])
    print("Period summary rows:", pooled_df.shape[0])
    print("Weight-method summary rows:", weight_df.shape[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())