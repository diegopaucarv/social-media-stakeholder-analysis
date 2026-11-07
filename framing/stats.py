"""Step 4b - statistics: chi-square tests (Pearson for RxC, Yates-corrected for 2x2), Cramer's V (plain and
bias-corrected, with a multinomial bootstrap CI), Fisher exact test for 2x2 tables and Holm correction.
The default hypothesis set is produced by default_tests() from the config (frames, focus source group, period
cut-off, declared genders); a study may override it with study["stats"]["tests"]."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import files as F
from .config import Config
from .schemas import EMERGENT
from .users import load_coded
from .utils import write_csv

warnings.filterwarnings("ignore")

# `compute_stats` returns this extra key next to the contingency tables so the
# text writer can report rows dropped from thread-level tests without treating
# the report as a table.
EXCLUDED_EMERGENT_KEY = "_excluded_emergent_threads"


def _pearson(tab: np.ndarray) -> tuple[float, float, np.ndarray]:
    n = tab.sum()
    expected = np.outer(tab.sum(1), tab.sum(0)) / n
    return float(((tab - expected) ** 2 / expected).sum()), float(n), expected


def cramers_v(tab: pd.DataFrame) -> tuple[float, float]:
    """Returns (V, bias-corrected V of Bergsma 2013)."""
    chi2, n, _ = _pearson(tab.values.astype(float))
    r, k = tab.shape
    v = np.sqrt(chi2 / (n * (min(r, k) - 1))) if min(r, k) > 1 else np.nan
    phi2 = chi2 / n
    phi2c = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rc, kc = r - ((r - 1) ** 2) / (n - 1), k - ((k - 1) ** 2) / (n - 1)
    vc = np.sqrt(phi2c / min(rc - 1, kc - 1)) if min(rc, kc) > 1 else np.nan
    return float(v), float(vc)


def bootstrap_ci(
    tab: pd.DataFrame, iterations: int, rng: np.random.Generator
) -> tuple[float, float]:
    if iterations <= 0:
        return (np.nan, np.nan)
    n = int(tab.values.sum())
    p = (tab.values / n).ravel()
    vals = []
    for _ in range(iterations):
        s = rng.multinomial(n, p).reshape(tab.shape).astype(float)
        s = s[s.sum(1) > 0][:, s.sum(0) > 0]
        if min(s.shape) < 2:
            continue
        chi2, nn, _ = _pearson(s)
        vals.append(np.sqrt(chi2 / (nn * (min(s.shape) - 1))))
    return tuple(np.percentile(vals, [2.5, 97.5])) if vals else (np.nan, np.nan)


def adjusted_residuals(tab: pd.DataFrame) -> pd.DataFrame:
    _, n, expected = _pearson(tab.values.astype(float))
    rp, cp = tab.sum(axis=1).values[:, None] / n, tab.sum(axis=0).values[None, :] / n
    return pd.DataFrame(
        (tab.values - expected) / np.sqrt(expected * (1 - rp) * (1 - cp)),
        index=tab.index,
        columns=tab.columns,
    )


def omnibus(
    test: dict, df: pd.DataFrame, row: str, col: str, cfg: Config, rng
) -> tuple[dict | None, pd.DataFrame | None]:
    tab = pd.crosstab(df[row], df[col])
    if min(tab.shape) < 2:
        return None, None
    chi2, n, expected = _pearson(tab.values.astype(float))
    gl = (tab.shape[0] - 1) * (tab.shape[1] - 1)
    v, vc = cramers_v(tab)
    warn = cfg.study["stats"]["min_expected_warning"]
    lo, hi = bootstrap_ci(tab, cfg.study["stats"]["bootstrap_iterations"], rng)
    return {
        "test_id": test["id"],
        "test": test["name"],
        "kind": "omnibus (Pearson chi2; Yates applies only to 2x2)",
        "variables": f"{row} x {col}",
        "population": test["pop"],
        "n": int(n),
        "df": gl,
        "chi2": round(chi2, 3),
        "p": float(sps.chi2.sf(chi2, gl)),
        "cramers_v": round(v, 3),
        "cramers_v_bias_corrected": round(vc, 3),
        "v_ci95_bootstrap": f"[{lo:.3f}, {hi:.3f}]",
        "pct_cells_expected_lt5": round(float((expected < warn).mean() * 100), 1),
        "min_expected": round(float(expected.min()), 2),
        "note": test.get("note", ""),
    }, tab


def two_by_two(
    test: dict, df: pd.DataFrame, flag: str, col: str, label: str, warn: float = 5.0
) -> dict | None:
    tab = pd.crosstab(df[flag], df[col])
    if tab.shape != (2, 2):
        return None
    tab = tab.reindex(index=sorted(tab.index), columns=sorted(tab.columns))
    chi2_y, p_y, _, expected = sps.chi2_contingency(tab.values, correction=True)
    chi2_n = sps.chi2_contingency(tab.values, correction=False)[0]
    n = int(tab.values.sum())
    odds, p_f = sps.fisher_exact(tab.values)
    return {
        "test_id": test["id"],
        "test": f"{label}",
        "kind": "2x2 (chi2 with Yates correction)",
        "variables": f"{flag} x {col}",
        "population": test["pop"],
        "n": n,
        "df": 1,
        "chi2": round(float(chi2_y), 3),
        "p": float(p_y),
        "cramers_v": round(float(np.sqrt(chi2_n / n)), 3),
        "cramers_v_bias_corrected": np.nan,
        "v_ci95_bootstrap": "",
        "pct_cells_expected_lt5": round(float((expected < warn).mean() * 100), 1),
        "min_expected": round(float(expected.min()), 2),
        "fisher_p": float(p_f),
        "odds_ratio": round(float(odds), 3) if np.isfinite(odds) else np.nan,
        "contingency": json.dumps(tab.values.tolist()),
        "note": test.get("note", ""),
    }


def default_tests(cfg: Config) -> list[dict]:
    st = cfg.study["stats"]
    tests = [
        {
            "id": "T1",
            "name": "frame x source group",
            "kind": "omnibus",
            "pop": "citizen",
            "row": "frame_id",
            "col": "source_group",
        }
    ]
    if st.get("focus_source_group"):
        tests += [
            {
                "id": "T1b",
                "name": "frame vs rest x focus source group",
                "kind": "one_vs_rest",
                "pop": "citizen",
                "col": "is_focus_source",
            },
            {
                "id": "T1c",
                "name": "any frame vs none x focus source group",
                "kind": "any_frame",
                "pop": "citizen",
                "col": "is_focus_source",
            },
        ]
    if cfg.study["period"]["cutoff_date"]:
        tests += [
            {
                "id": "T2",
                "name": "frame x period",
                "kind": "omnibus",
                "pop": "citizen",
                "row": "frame_id",
                "col": "period",
            },
            {
                "id": "T2b",
                "name": "frame vs rest x after-cut-off period",
                "kind": "one_vs_rest",
                "pop": "citizen",
                "col": "is_after",
            },
        ]
    # T5/T5b compare a frame against declared gender, so they only apply when the study declares gender values.
    t5 = (
        []
        if not st.get("gender_values")
        else [
            {
                "id": "T5",
                "name": "frame x declared gender",
                "kind": "omnibus",
                "pop": "citizen_gender",
                "row": "frame_id",
                "col": "gender2",
                "note": "Only comments by authors with a declared gender among the configured values.",
            },
            {
                "id": "T5b",
                "name": "frame vs rest x declared gender",
                "kind": "one_vs_rest",
                "pop": "citizen_gender",
                "col": "gender2",
            },
        ]
    )
    tests += [
        {
            "id": "T3",
            "name": "frame x comment type (root/reply)",
            "kind": "omnibus",
            "pop": "citizen",
            "row": "frame_id",
            "col": "is_reply",
        },
        {
            "id": "T3b",
            "name": "frame vs rest x reply",
            "kind": "one_vs_rest",
            "pop": "citizen",
            "col": "is_reply",
        },
        {
            "id": "T4",
            "name": "frame x high relevance (coded comments only)",
            "kind": "omnibus",
            "pop": "coded",
            "row": "frame_id",
            "col": "high_relevance",
            "note": "Only coded comments: selection by relevance would make the uncoded ones circular.",
        },
        {
            "id": "T4b",
            "name": "frame vs other frames x high relevance (coded only)",
            "kind": "one_vs_rest",
            "pop": "coded",
            "col": "high_relevance",
        },
        *t5,
        {
            "id": "T6",
            "name": "thread within frame x source group (exploratory)",
            "kind": "thread_within_frame",
            "pop": "coded",
            "col": "source_group",
        },
        {
            "id": "T7",
            "name": "frame x stance (coded comments)",
            "kind": "omnibus",
            "pop": "coded",
            "row": "frame_id",
            "col": "stance",
        },
        {
            "id": "T8",
            "name": "SENSITIVITY frame x source group (one comment per author and frame)",
            "kind": "omnibus",
            "pop": "coded_dedup",
            "row": "frame_id",
            "col": "source_group",
            "note": "Reduces pseudo-replication by author.",
        },
    ]
    return tests


def prepare_variables(c: pd.DataFrame, cfg: Config) -> dict:
    st = cfg.study["stats"]
    none = cfg.no_frame_id
    cit = c[(~c["is_institutional"]) & c["has_text"]].copy().reset_index(drop=True)
    cit["high_relevance"] = (cit["relevance_tier"] == "high").astype(int)
    cit["is_focus_source"] = (
        cit["source_group"] == st.get("focus_source_group")
    ).astype(int)
    cit["is_after"] = (cit["period"] == cfg.study["period"]["after_label"]).astype(int)
    cit["gender2"] = cit["gender"].where(cit["gender"].isin(st["gender_values"]))
    cit["any_frame"] = (cit["frame_id"] != none).astype(int)
    coded = cit[cit["frame_id"] != none].reset_index(drop=True)
    dedup = (
        coded.sort_values("relevance_score", ascending=False)
        .drop_duplicates(["user_id", "frame_id"])
        .reset_index(drop=True)
    )
    return {
        "citizen": cit,
        "coded": coded,
        "citizen_gender": cit.dropna(subset=["gender2"]).reset_index(drop=True),
        "coded_dedup": dedup,
    }


def compute_stats(
    c: pd.DataFrame, cfg: Config, tests: list[dict] | None = None
) -> tuple[pd.DataFrame, dict]:
    pops = prepare_variables(c, cfg)
    rng = np.random.default_rng(cfg.study["stats"]["seed"])
    results, tables = [], {}
    gap = cfg.study["gap_checks"]
    warn = cfg.study["stats"]["min_expected_warning"]
    for t in tests or cfg.study["stats"].get("tests") or default_tests(cfg):
        df = pops[t["pop"]]
        kind = t["kind"]
        if kind == "omnibus":
            res, tab = omnibus(t, df, t["row"], t["col"], cfg, rng)
            if res:
                results.append(res)
                tables[f"{t['id']}_{t['name']}"] = tab
        elif kind == "one_vs_rest":
            for fid in cfg.frame_ids:
                d = df.assign(_flag=(df["frame_id"] == fid).astype(int))
                res = two_by_two(t, d, "_flag", t["col"], f"{t['name']} [{fid}]", warn)
                if res:
                    res["variables"] = f"is_{fid} x {t['col']}"
                    results.append(res)
        elif kind == "any_frame":
            res = two_by_two(t, df, "any_frame", t["col"], t["name"], warn)
            if res:
                results.append(res)
        elif kind == "thread_within_frame":
            excluded_rows = []
            for fid in cfg.frame_ids:
                sub = df[df["frame_id"] == fid]
                # EMERGENT is a thread_id value offered to the model for new
                # proposals, not an analyst-defined thread. Keep those rows in
                # `coded` (frame-level tests legitimately use them) but drop
                # them here so the omnibus test does not manufacture a phantom
                # EMERGENT thread category.
                n_frame = len(sub)
                sub = sub[sub["thread_id"] != EMERGENT]
                n_excluded = n_frame - len(sub)
                if n_excluded:
                    excluded_rows.append(
                        {
                            "frame_id": fid,
                            "n_excluded": n_excluded,
                            "n_thread_test": len(sub),
                        }
                    )
                if len(sub) >= gap["min_frame_comments"]:
                    tt = dict(t, id=f"{t['id']}_{fid}", name=f"{t['name']} [{fid}]")
                    res, tab = omnibus(tt, sub, "thread_id", t["col"], cfg, rng)
                    if res:
                        results.append(res)
                        tables[f"{tt['id']}_{tt['name']}"] = tab
            if excluded_rows:
                tables[EXCLUDED_EMERGENT_KEY] = pd.DataFrame(excluded_rows)
    res = pd.DataFrame(results)
    if res.empty:
        return res, tables
    mask = res["kind"].str.startswith("2x2")
    if mask.any():
        pv = res.loc[mask, "p"].values
        order = np.argsort(pv)
        adj, running = np.empty(len(pv)), 0.0
        for rank, i in enumerate(order):
            running = max(running, (len(pv) - rank) * pv[i])
            adj[i] = min(1.0, running)
        res.loc[mask, "p_holm"] = adj
    res["significant_0_05"] = np.where(res["p"] < 0.05, "yes", "no")
    if "p_holm" in res:
        res.loc[mask, "significant_holm_0_05"] = np.where(
            res.loc[mask, "p_holm"] < 0.05, "yes", "no"
        )
    res["p"] = res["p"].map(lambda x: float(f"{x:.4g}"))
    return res, tables


def run_stats(cfg: Config, work_dir, out_dir=None) -> dict:
    out = Path(out_dir) if out_dir else Path(work_dir)
    c = load_coded(work_dir, cfg)
    res, tables = compute_stats(c, cfg)
    write_csv(res, out / F.STATS)
    n_tables = 0
    with open(out / F.TABLES, "w", encoding="utf-8") as fh:
        for name, tab in tables.items():
            if name == EXCLUDED_EMERGENT_KEY:
                fh.write(
                    f"== {name}\n{tab.to_string(index=False)}\n\n"
                    f"Rows with thread_id == {EMERGENT} are emergent proposals, not "
                    "analyst-defined threads: frame-level tests keep them, "
                    "thread-level tests exclude them.\n\n"
                )
                continue
            n_tables += 1
            fh.write(
                f"== {name}\n{tab.to_string()}\n\nadjusted standardised residuals:\n{adjusted_residuals(tab).round(2).to_string()}\n\n"
            )
    return {"tests": int(len(res)), "tables": n_tables}
