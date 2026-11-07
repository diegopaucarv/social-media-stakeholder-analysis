"""Step 3 - merge: join the LLM classifications with the enriched comments by original comment id,
validate and normalise them, attach labels from the codebook and write the analysis tables."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import files as F
from .config import Config
from .institutional import combine
from .provenance import warn_if_stale, write_run
from .utils import as_bool, read_json, write_csv, write_json

#: Temporary column carrying the model-declared institutional flag through the merge.
_AI_INSTITUTIONAL = "_ai_institutional"

SELECTED_COLUMNS = [
    "comment_seq",
    "post_id",
    "source_type",
    "source_name",
    "post_topic",
    "post_frame_id",
    "post_author",
    "timestamp",
    "phase",
    "chronological_order",
    "relevance_tier",
    "relevance_score",
    "likes",
    "reply_count",
    "thread_depth",
    "is_reply",
    "frame_id",
    "frame_label",
    "secondary_frame_id",
    "thread_id",
    "thread_label",
    "stance",
    "evidence_level",
    "rationale",
    "text",
    "user_id",
    "user_pseudonym",
    "user_name",
    "gender",
    "verified",
    "profile_url",
    "is_post_author",
    "is_institutional",
    "locations_mentioned",
    "has_attachment",
    "comment_url",
    "post_key",
    "parent_author",
    "literal_excerpt",
    "confidence",
    "needs_human_review",
    "review_reason",
    "emergent_thread_proposal",
    "actor_category",
    "actor_self_declared",
    "actor_evidence",
]
ALL_CODED_COLUMNS = [
    "comment_seq",
    "comment_id",
    "post_id",
    "source_type",
    "source_group",
    "period",
    "timestamp",
    "phase",
    "relevance_tier",
    "relevance_score",
    "likes",
    "reply_count",
    "thread_depth",
    "is_reply",
    "frame_id",
    "secondary_frame_id",
    "thread_id",
    "emergent_thread_proposal",
    "stance",
    "evidence_level",
    "confidence",
    "needs_human_review",
    "actor_category",
    "actor_self_declared",
    "user_id",
    "user_pseudonym",
    "user_name",
    "gender",
    "verified",
    "profile_url",
    "source_name",
    "is_institutional",
    "is_post_author",
    "has_text",
    "post_author",
    "selected",
    "locations_mentioned",
    "actor_evidence",
    "text",
]


def load_classifications(patterns, cfg: Config) -> tuple[pd.DataFrame, dict]:
    """Read model answers: JSON files or the CSV returned by the chatbot branch.

    Kept as the name `merge` calls; the format handling lives in `framing.responses`."""
    from .responses import load_responses

    return load_responses(patterns, cfg)


def merge(
    cfg: Config,
    work_dir,
    responses,
    out_dir=None,
    post_responses=None,
    min_confidence: int | None = None,
    require_coverage: float | None = None,
) -> dict:
    """Join model answers with the corpus and write every analysis table.

    Comment answers are normalised by :func:`framing.responses.load_responses`; post
    answers (topic, primary frame, source type guess) by
    :func:`framing.responses.load_post_responses`, which accepts the chatbot CSV.

    When ``out_dir`` differs from ``work_dir`` the merged folder is made
    self-contained so ``analyze`` / ``export-network`` can run with ``--work <out>``:
    the enriched ``posts_table.csv`` is written **into ``out`` only** (never back into
    ``work``, where prepare's original must survive) and ``join_report.json`` is
    copied. The enriched post classification itself is always written to
    ``out / 03_posts_classified.csv``.
    """
    work = Path(work_dir)
    warn_if_stale(cfg, work)
    out = Path(out_dir) if out_dir else work
    out.mkdir(parents=True, exist_ok=True)
    min_conf = (
        min_confidence
        if min_confidence is not None
        else cfg.study["classification"]["min_confidence"]
    )

    c = pd.read_csv(
        work / F.COMMENTS_ENRICHED,
        dtype={"comment_id": str, "user_id": str, "reply_to_id": str},
        low_memory=False,
    )
    c["timestamp"] = pd.to_datetime(c["timestamp"], utc=True)
    for col in ("has_text", "is_institutional", "has_attachment"):
        c[col] = as_bool(c[col])
    c["text"] = c["text"].fillna("")
    posts = pd.read_csv(work / F.POSTS_TABLE, dtype=str).fillna("")
    posts["post_id"] = posts["post_id"].astype(int)
    for col in ("registered_source", "has_post_text"):
        posts[col] = as_bool(posts[col])
    posts["n_comments"] = posts["n_comments"].astype(int)

    cls, report = load_classifications(responses, cfg)
    report["classified_ids_found_in_comments"] = float(
        cls["comment_id"].isin(c["comment_id"]).mean()
    )
    unknown = cls[~cls["comment_id"].isin(c["comment_id"])]
    report["unknown_comment_ids"] = unknown["comment_id"].tolist()[:20]
    # Keep the model's institutional flag under a private name: the author-derived
    # mask already owns `is_institutional`, and the two must be OR-ed together
    # before coverage is computed.
    to_merge = cls.rename(columns={"is_institutional": _AI_INSTITUTIONAL})
    if _AI_INSTITUTIONAL not in to_merge.columns:
        to_merge[_AI_INSTITUTIONAL] = False
    c = c.merge(to_merge, on="comment_id", how="left")
    model_institutional = combine(None, c[_AI_INSTITUTIONAL])
    report["institutional_flagged_by_model"] = int(model_institutional.sum())
    c["is_institutional"] = combine(c["is_institutional"], c.pop(_AI_INSTITUTIONAL))
    report["institutional_total"] = int(c["is_institutional"].sum())
    report["unclassified_comments"] = int(c["frame_id"].isna().sum())
    report["unclassified_eligible"] = int(
        (c["frame_id"].isna() & c["has_text"] & ~c["is_institutional"]).sum()
    )
    eligible = int((c["has_text"] & ~c["is_institutional"]).sum())
    report["eligible_comments"] = eligible
    report["coverage"] = (
        1.0 if not eligible else 1.0 - report["unclassified_eligible"] / eligible
    )
    if report["invalid_records"]:
        print(
            f"WARNING: {report['invalid_records']} classification record(s) failed validation; "
            f"issue_examples are in {F.MERGE_REPORT}.",
            file=sys.stderr,
        )
    if report["coverage"] < 0.95:
        print(
            f"WARNING: only {report['coverage']:.1%} of eligible comments were classified "
            f"({report['unclassified_eligible']} missing); missing batches bias every downstream statistic.",
            file=sys.stderr,
        )
    if require_coverage is not None and report["coverage"] < require_coverage:
        raise ValueError(
            f"coverage {report['coverage']:.1%} is below the required {require_coverage:.0%} "
            f"({report['unclassified_eligible']} of {eligible} eligible comments unclassified)"
        )
    for col, default in cfg.record_defaults.items():
        if col in c and col not in ("comment_id", "is_institutional"):
            c[col] = c[col].where(c[col].notna(), default)
    c["frame_id"] = c["frame_id"].replace("", cfg.no_frame_id)
    c.loc[c["is_institutional"] | ~c["has_text"], "frame_id"] = cfg.no_frame_id
    for col in ("needs_human_review", "actor_self_declared"):
        c[col] = c[col].astype(bool)
    c["confidence"] = c["confidence"].astype(int)
    c.loc[
        c["frame_id"] == cfg.no_frame_id,
        ["secondary_frame_id", "thread_id", "stance", "evidence_level"],
    ] = ""
    c["frame_label"] = c["frame_id"].map(cfg.frame_label)
    c["thread_label"] = c["thread_id"].map(cfg.thread_label).fillna("")
    em = c["thread_id"] == cfg.emergent_thread_id
    c.loc[em, "thread_label"] = (
        f"{cfg.emergent_thread_id}: " + c.loc[em, "emergent_thread_proposal"]
    )
    c["selected"] = (c["frame_id"] != cfg.no_frame_id) & (c["confidence"] >= min_conf)

    # ---- post-level metadata and classification
    if post_responses:
        from .responses import load_post_responses

        pc, post_report = load_post_responses(post_responses, cfg)
        report["post_classification"] = post_report
        report["posts_classified_by_model"] = int(len(pc))
        if not pc.empty:
            pc = pc.set_index("post_key")
            topic = pc["topic"] if "topic" in pc.columns else None
            frame = pc["primary_frame_id"] if "primary_frame_id" in pc.columns else None
            guess = (
                pc["source_type_guess"] if "source_type_guess" in pc.columns else None
            )
            if topic is not None:
                fill_topic = posts["post_key"].map(topic).fillna("")
                empty = posts["post_topic"] == ""
                posts.loc[empty, "post_topic"] = fill_topic[empty]
            if frame is not None:
                fill_frame = posts["post_key"].map(frame).fillna("")
                empty = posts["post_frame_id"] == ""
                posts.loc[empty, "post_frame_id"] = fill_frame[empty]
            if guess is not None:
                unreg = (~posts["registered_source"]) & (
                    posts["source_type"] == "unclassified"
                )
                guess_vals = posts["post_key"].map(guess).fillna("unclassified")
                posts.loc[unreg, "source_type"] = guess_vals[unreg]
            posts["source_group"] = posts["source_type"].map(cfg.source_type_group)
    post_cols = [
        "post_id",
        "source_type",
        "source_group",
        "source_name",
        "post_topic",
        "post_frame_id",
        "post_author",
        "period",
    ]
    c = c.merge(posts[post_cols], on="post_id", how="left")
    c["is_post_author"] = (c["user_name"].fillna("") == c["post_author"].fillna("")) & (
        c["post_author"].fillna("") != ""
    )

    # ---- outputs
    sel_cols = [k for k in SELECTED_COLUMNS if k in c.columns]
    write_csv(
        c[c["selected"]][sel_cols].sort_values(["post_id", "timestamp"]),
        out / F.SELECTED,
    )
    write_csv(
        c[[k for k in ALL_CODED_COLUMNS if k in c.columns]].sort_values(
            ["post_id", "timestamp"]
        ),
        out / F.ALL_CODED,
    )
    counts = (
        c[c["selected"]]
        .groupby(["post_id", "frame_id"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=cfg.frame_ids, fill_value=0)
    )
    counts.columns = [f"n_selected_{x}" for x in counts.columns]
    ptab = posts.merge(counts, left_on="post_id", right_index=True, how="left")
    for col in counts.columns:
        ptab[col] = ptab[col].fillna(0).astype(int)
    ptab["n_selected_comments"] = ptab[list(counts.columns)].sum(axis=1)
    dom = (
        ptab[list(counts.columns)]
        .idxmax(axis=1)
        .str.replace("n_selected_", "", regex=False)
    )
    ptab["dominant_comment_frame_id"] = np.where(
        ptab["n_selected_comments"] > 0, dom, cfg.no_frame_id
    )
    write_csv(ptab, out / F.POSTS_CLASSIFIED)

    if out.resolve() != work.resolve():
        # Keep the merged folder self-contained for `analyze`/`export-network`
        # with `--work <out>`: ship the enriched post table in `out` (never touch
        # `work`'s original) and the join report prepare produced.
        enriched_posts = ptab[[col for col in posts.columns if col in ptab.columns]]
        write_csv(enriched_posts, out / F.POSTS_TABLE)
        join_report = work / F.JOIN_REPORT
        if join_report.exists():
            shutil.copy2(join_report, out / F.JOIN_REPORT)

    ph = []
    for (pid, phase), g in c.sort_values("timestamp").groupby(["post_id", "phase"]):
        high = g[(g["relevance_tier"] == "high") & (~g["is_institutional"])].head(1)
        first = g.iloc[0]
        anchor = (
            first
            if (phase == 1 and first["relevance_tier"] != "high") or high.empty
            else high.iloc[0]
        )
        cnt = g[g["selected"]]["frame_id"].value_counts()
        row = {
            "post_id": pid,
            "source_type": g["source_type"].iloc[0],
            "phase": phase,
            "start": g["timestamp"].min().strftime("%Y-%m-%d %H:%M"),
            "end": g["timestamp"].max().strftime("%Y-%m-%d %H:%M"),
            "n_comments": len(g),
            "n_institutional": int(g["is_institutional"].sum()),
            "anchor_comment_seq": int(anchor["comment_seq"]),
            "anchor_is_high_relevance": bool(
                phase > 1 or first["relevance_tier"] == "high"
            ),
            "total_likes": int(g["likes"].sum()),
            "dominant_frame_id": cnt.idxmax() if len(cnt) else cfg.no_frame_id,
        }
        row.update({f"n_{fid}": int(cnt.get(fid, 0)) for fid in cfg.frame_ids})
        ph.append(row)
    write_csv(pd.DataFrame(ph), out / F.PHASES)

    review = c[
        c["needs_human_review"]
        | ((c["frame_id"] != cfg.no_frame_id) & (c["confidence"] == 1))
    ]
    write_csv(
        review[
            [
                "comment_seq",
                "comment_id",
                "post_id",
                "frame_id",
                "thread_id",
                "stance",
                "evidence_level",
                "confidence",
                "review_reason",
                "literal_excerpt",
                "text",
            ]
        ],
        out / F.REVIEW,
    )
    report.update(
        {
            "config_hash": cfg.hash,
            "comments": int(len(c)),
            "selected": int(c["selected"].sum()),
            "selected_by_frame": c[c["selected"]]["frame_id"].value_counts().to_dict(),
            "phases": int(len(ph)),
            "review_queue": int(len(review)),
            "min_confidence": min_conf,
        }
    )
    write_json(out / F.MERGE_REPORT, report)
    write_run(
        cfg, work, "merge", response_files=report["files"], coverage=report["coverage"]
    )
    return report
