"""Step 4a - users: author profiles, contact candidates and a per-location community-relevance proxy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import files as F
from .config import Config
from .provenance import warn_if_stale
from .schemas import STANCES
from .utils import as_bool, write_csv


def load_coded(work_dir, cfg: Config | None = None) -> pd.DataFrame:
    """Read the coded corpus. When `cfg` is given, warn on provenance/config-hash mismatch."""
    if cfg is not None:
        warn_if_stale(cfg, work_dir)
    c = pd.read_csv(
        Path(work_dir) / F.ALL_CODED,
        dtype={"comment_id": str, "user_id": str},
        low_memory=False,
    )
    c["timestamp"] = pd.to_datetime(c["timestamp"], utc=True)
    for col in (
        "is_institutional",
        "selected",
        "is_post_author",
        "has_text",
        "needs_human_review",
        "actor_self_declared",
    ):
        if col in c:
            c[col] = as_bool(c[col])
    for col in (
        "frame_id",
        "thread_id",
        "stance",
        "evidence_level",
        "actor_category",
        "locations_mentioned",
        "actor_evidence",
        "secondary_frame_id",
        "source_type",
        "source_group",
        "period",
        "gender",
    ):
        if col in c:
            c[col] = c[col].fillna("")
    c["text"] = c["text"].fillna("")
    return c


def _join(values) -> str:
    return "; ".join(sorted({str(v) for v in values if str(v)}))


def _support_stance_keys(cfg: Config) -> set[str]:
    """Stance ids the study codebook defines as support.

    The vocabulary is configured, so the literal ``"supports"`` must not be hard-coded.
    ``stance_group`` normalises the configured English/legacy aliases to ``support``;
    it is imported lazily because ``network_export`` imports this module.
    """
    from .network_export import stance_group

    keys = list(cfg.codebook.get("stance_definitions") or {}) or list(STANCES)
    return {k for k in keys if stance_group(k) == "support"}


def build_users(c: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "selected" not in c:  # degrade gracefully on reduced tables
        c = c.assign(selected=False)
    g = c.groupby("user_id")
    users = pd.DataFrame(
        {
            "user_pseudonym": g["user_pseudonym"].first(),
            "user_name": g["user_name"].first(),
            "gender": g["gender"].first(),
            "verified": g["verified"].first() if "verified" in c else "",
            "profile_url": g["profile_url"].first() if "profile_url" in c else "",
            "is_institutional": g["is_institutional"].max(),
            "is_post_author": g["is_post_author"].max(),
            "n_comments": g.size(),
            "n_posts": g["post_id"].nunique(),
            "source_types": g["source_type"].agg(_join),
            "likes_received": g["likes"].sum(),
            "replies_received": g["reply_count"].sum(),
            "n_selected": g["selected"].sum(),
            "first_comment": g["timestamp"].min().dt.strftime("%Y-%m-%d"),
            "last_comment": g["timestamp"].max().dt.strftime("%Y-%m-%d"),
            "locations_mentioned": g["locations_mentioned"].agg(
                lambda s: _join(x for v in s for x in v.split("; "))
            ),
        }
    )
    support = _support_stance_keys(cfg)
    is_support = (
        c["stance"].fillna("").astype(str).isin(support)
        if "stance" in c
        else pd.Series(False, index=c.index)
    )
    sup = c[c["selected"] & is_support].groupby("user_id")
    users["frames_supported"] = sup["frame_id"].agg(_join)
    users["threads_supported"] = sup["thread_id"].agg(_join)
    act = c[c["actor_category"] != ""]
    counts = act.groupby(["user_id", "actor_category"]).size()
    selfd = (
        act[act["actor_self_declared"]].groupby(["user_id", "actor_category"]).size()
    )
    cats, tops, evid = {}, {}, {}
    for uid, sub in counts.groupby(level=0):
        items = sorted(
            sub.droplevel(0).items(),
            key=lambda kv: (-(selfd.get((uid, kv[0]), 0)), -kv[1], kv[0]),
        )
        cats[uid] = "; ".join(f"{k} ({v})" for k, v in items)
        tops[uid] = items[0][0]
        evid[uid] = (
            act[(act["user_id"] == uid) & (act["actor_category"] == items[0][0])][
                "actor_evidence"
            ]
            .replace("", np.nan)
            .dropna()
            .head(1)
            .tolist()
        )
    users["actor_categories"] = pd.Series(cats)
    users["top_actor_category"] = pd.Series(tops)
    users["actor_evidence"] = pd.Series(
        {k: (v[0] if v else "") for k, v in evid.items()}
    )
    # institutional authors take their category from the sources module
    for uid, name in users["user_name"].items():
        if name in cfg.institutional:
            users.loc[uid, "top_actor_category"] = cfg.institutional[name]
    users = users.fillna(
        {
            "frames_supported": "",
            "threads_supported": "",
            "actor_categories": "",
            "top_actor_category": "",
            "actor_evidence": "",
        }
    )
    users["ecosystem_level"] = (
        users["top_actor_category"].map(cfg.actor_level_id).fillna("")
    )
    users["activity_score"] = (
        users["n_comments"] + users["likes_received"] + 2 * users["replies_received"]
    )

    contact_cats = {
        k for k, v in cfg.actor_categories.items() if v["contact_candidate"]
    }
    min_c = cfg.study["contacts"]["min_comments_for_activity_candidate"]
    reasons = []
    for uid, r in users.iterrows():
        why = []
        own = {x.split(" (")[0] for x in r["actor_categories"].split("; ") if x}
        for cat in sorted(own & contact_cats):
            why.append(f"self-declared:{cat}")
        if r["n_comments"] >= min_c and r["n_selected"] >= 1:
            why.append("high activity")
        if r["is_post_author"]:
            why.append("authors a post in the corpus")
        reasons.append("; ".join(why) if not r["is_institutional"] else "")
    users["contact_reasons"] = reasons
    users["contact_candidate"] = users["contact_reasons"] != ""
    users = users.sort_values("activity_score", ascending=False).reset_index()
    contacts = users[users["contact_candidate"]][
        [
            "user_id",
            "user_pseudonym",
            "user_name",
            "profile_url",
            "contact_reasons",
            "actor_categories",
            "actor_evidence",
            "frames_supported",
            "n_comments",
            "n_selected",
            "locations_mentioned",
        ]
    ].copy()
    contacts["validation_status"] = (
        "to_validate (public data of private persons; no consent yet)"
    )
    return users, contacts


def build_community_relevance(c: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    base = c[
        (~c["is_institutional"]) & c["has_text"] & (c["locations_mentioned"] != "")
    ].copy()
    base["loc_list"] = base["locations_mentioned"].str.split("; ")
    rows = []
    for g in cfg.gazetteer:
        sub = base[base["loc_list"].map(lambda L: g["name"] in L)]
        row = {
            "location": g["name"],
            "level": g["level"],
            "in_study_area": g.get("in_study_area", True),
            "n_comments": len(sub),
            "n_users": sub["user_id"].nunique(),
            "n_selected": int(sub["selected"].sum()),
        }
        score = 0.0
        for ind in cfg.indicators:
            m = (
                sub["selected"]
                & (
                    sub["thread_id"].isin(ind.get("thread_ids", []))
                    | sub["frame_id"].isin(ind.get("frame_ids", []))
                )
            ) | (
                sub["actor_self_declared"]
                & sub["actor_category"].isin(ind.get("actor_categories", []))
            )
            row[f"ind_{ind['id']}"] = int(m.sum())
            score += ind["weight"] * int(m.sum())
        row["relevance_score"] = round(score, 2)
        rows.append(row)
    out = pd.DataFrame(rows)
    ranked = out[out["n_comments"] > 0].sort_values(
        ["relevance_score", "n_comments"], ascending=False
    )
    out["rank"] = np.nan
    out.loc[ranked.index, "rank"] = range(1, len(ranked) + 1)
    out["data_gaps_note"] = (cfg.locations.get("community_relevance") or {}).get(
        "data_gaps_note", ""
    )
    return out.sort_values(["rank", "location"], na_position="last").reset_index(
        drop=True
    )


def run_users(cfg: Config, work_dir, out_dir=None) -> dict:
    out = Path(out_dir) if out_dir else Path(work_dir)
    c = load_coded(work_dir, cfg)
    users, contacts = build_users(c, cfg)
    comm = build_community_relevance(c, cfg)
    write_csv(users, out / F.USERS)
    write_csv(contacts, out / F.CONTACTS)
    write_csv(comm, out / F.COMMUNITY)
    return {
        "users": len(users),
        "contact_candidates": len(contacts),
        "locations_with_mentions": int((comm["n_comments"] > 0).sum()),
    }
