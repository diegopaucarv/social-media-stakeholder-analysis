"""Agreement between two codings of the same comments (e.g. human vs model, model vs model, run 1 vs run 2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import files as F
from .config import Config
from .users import load_coded
from .utils import write_json


def cohen_kappa(a, b) -> float | None:
    """Cohen's kappa between two labellings.

    Returns ``None`` when the coefficient is undefined: either there are no
    paired observations, or expected agreement is 1 because a marginal is
    degenerate (e.g. the same label everywhere, as when a study does not
    collect a field). An undefined coefficient must not be read as perfect
    agreement.
    """
    a, b = pd.Series(list(a)).astype(str), pd.Series(list(b)).astype(str)
    labels = sorted(set(a) | set(b))
    t = (
        pd.crosstab(a, b)
        .reindex(index=labels, columns=labels, fill_value=0)
        .values.astype(float)
    )
    n = t.sum()
    if n == 0:
        return None
    po = np.trace(t) / n
    pe = (t.sum(0) * t.sum(1)).sum() / n**2
    return float((po - pe) / (1 - pe)) if pe < 1 else None


def _rounded(value):
    """Round a coefficient for JSON, preserving ``None`` for undefined values."""
    return None if value is None else round(value, 4)


def compare_codings(cfg: Config, a: pd.DataFrame, b: pd.DataFrame) -> dict:
    m = a.merge(b, on="comment_id", suffixes=("_a", "_b"))
    m = m[(~m["is_institutional_a"]) & m["has_text_a"]]
    none = cfg.no_frame_id
    rep = {
        "n_compared": int(len(m)),
        "n_only_in_a": int((~a["comment_id"].isin(b["comment_id"])).sum()),
        "n_only_in_b": int((~b["comment_id"].isin(a["comment_id"])).sum()),
    }
    sel_a, sel_b = m["frame_id_a"] != none, m["frame_id_b"] != none
    rep["frame_all_categories"] = {
        "agreement": round(float((m["frame_id_a"] == m["frame_id_b"]).mean()), 4),
        "kappa": _rounded(cohen_kappa(m["frame_id_a"], m["frame_id_b"])),
    }
    rep["coded_vs_not"] = {
        "agreement": round(float((sel_a == sel_b).mean()), 4),
        "kappa": _rounded(cohen_kappa(sel_a, sel_b)),
        "only_a": int((sel_a & ~sel_b).sum()),
        "only_b": int((~sel_a & sel_b).sum()),
    }
    both = m[sel_a & sel_b]
    same = both[both["frame_id_a"] == both["frame_id_b"]]
    rep["both_coded"] = {
        "n": int(len(both)),
        "frame_agreement": round(
            float((both["frame_id_a"] == both["frame_id_b"]).mean()), 4
        )
        if len(both)
        else None,
    }
    for fld in ("thread_id", "stance", "evidence_level"):
        rep[f"{fld}_given_same_frame"] = {
            "n": int(len(same)),
            "agreement": round(
                float(
                    (same[f"{fld}_a"].fillna("") == same[f"{fld}_b"].fillna("")).mean()
                ),
                4,
            )
            if len(same)
            else None,
            "kappa": _rounded(
                cohen_kappa(same[f"{fld}_a"].fillna(""), same[f"{fld}_b"].fillna(""))
            )
            if len(same)
            else None,
        }
    per = {}
    for fid in cfg.frame_ids:
        x, y = m["frame_id_a"] == fid, m["frame_id_b"] == fid
        per[fid] = {
            "n_a": int(x.sum()),
            "n_b": int(y.sum()),
            "both": int((x & y).sum()),
            "kappa": _rounded(cohen_kappa(x, y)),
            "jaccard": round(float((x & y).sum() / max(1, (x | y).sum())), 4),
        }
    rep["per_frame"] = per
    rep["confusion_frame"] = pd.crosstab(m["frame_id_a"], m["frame_id_b"]).to_dict()
    return rep


def run_compare(cfg: Config, a_dir, b_dir, out_dir) -> dict:
    a, b = load_coded(a_dir, cfg), load_coded(b_dir, cfg)
    rep = compare_codings(cfg, a, b)
    rep["config_hash"] = cfg.hash
    write_json(Path(out_dir) / F.RELIABILITY, rep)
    return rep
