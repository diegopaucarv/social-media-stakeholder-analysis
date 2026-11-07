"""Step 4c - gaps: what is still missing according to the study's own configuration and methodology."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from . import files as F
from .config import Config
from .schemas import EMERGENT, EVIDENCE
from .users import load_coded
from .utils import read_json, write_csv, write_json

# Evidence-level reporting is descriptive only: no hypothesis test, no p-values,
# no Holm correction and no new output file. Thresholds stay module constants so
# study.json needs no new key (its gap_checks schema is strict); g.get(...) lets
# a study override them later without breaking existing configurations.
EMPTY_EVIDENCE_SHARE_WARNING = 0.6
WEAKEST_EVIDENCE_LEVEL = EVIDENCE[-1]
WEAKEST_EVIDENCE_SHARE_WARNING = 0.6


def _account(post_key: str) -> str:
    parts = [p for p in urlparse(post_key).path.split("/") if p]
    if not parts:
        return post_key
    return "/".join(parts[:2]) if parts[0] == "groups" and len(parts) > 1 else parts[0]


def source_coverage(posts: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []
    for e in cfg.sources["registry"]:
        sub = posts[posts["source_key"] == e["key"]]
        rows.append(
            {
                "source_key": e["key"],
                "kind": e["kind"],
                "name": e.get("name", ""),
                "type": e["type"],
                "type_verified": e.get("type_verified", False),
                "registered": True,
                "n_posts": len(sub),
                "n_comments": int(sub["n_comments"].sum()),
                "status": "observed" if len(sub) else "missing",
            }
        )
    unreg = posts[~posts["registered_source"]].copy()
    unreg["account"] = [
        n if n else _account(k) for n, k in zip(unreg["source_name"], unreg["post_key"])
    ]
    for acc, sub in unreg.groupby("account"):
        rows.append(
            {
                "source_key": acc,
                "kind": "",
                "name": ", ".join(sorted({s for s in sub["source_name"] if s})),
                "type": ", ".join(sorted(set(sub["source_type"]))),
                "type_verified": False,
                "registered": False,
                "n_posts": len(sub),
                "n_comments": int(sub["n_comments"].sum()),
                "status": "observed but not in registry",
            }
        )
    return pd.DataFrame(rows)


def build_gap_report(cfg: Config, work_dir, out_dir=None) -> dict:
    work = Path(work_dir)
    out = Path(out_dir) if out_dir else work
    c = load_coded(work, cfg)
    posts = pd.read_csv(work / F.POSTS_TABLE, dtype=str).fillna("")
    posts["n_comments"] = posts["n_comments"].astype(int)
    posts["registered_source"] = posts["registered_source"].str.lower() == "true"
    join = read_json(work / F.JOIN_REPORT) if (work / F.JOIN_REPORT).exists() else {}
    g = cfg.study["gap_checks"]
    cit = c[(~c["is_institutional"]) & c["has_text"]]
    coded = cit[cit["frame_id"] != cfg.no_frame_id]
    gaps: list[dict] = []

    def add(area, severity, finding, action):
        gaps.append(
            {"area": area, "severity": severity, "finding": finding, "action": action}
        )

    # 1 join quality
    if join:
        share = join.get("unmatched_comments", 0) / max(1, join.get("n_comments", 1))
        if share > g["join_warning_share"]:
            add(
                "join",
                "medium",
                f"{share:.0%} of comments could not be linked to a row of the posts file.",
                "Check the post URL/id columns or scrape the missing posts.",
            )
    # 2 sources
    cov = source_coverage(posts, cfg)
    write_csv(cov, out / F.COVERAGE)
    reg = cov[cov["registered"]]
    missing = reg[reg["status"] == "missing"]
    unreg = cov[~cov["registered"]]
    if len(missing):
        add(
            "sources",
            "high",
            f"{len(missing)} of {len(reg)} registered sources have no posts in the corpus.",
            "Scrape or join these sources: "
            + ", ".join(missing["source_key"].head(15))
            + (" ..." if len(missing) > 15 else ""),
        )
    if len(unreg):
        add(
            "sources",
            "low",
            f"{len(unreg)} observed accounts are not in the registry.",
            "Add them to sources.json (with a verified type) if relevant.",
        )
    if (reg["type_verified"] == False).any():  # noqa: E712
        add(
            "sources",
            "low",
            f"{int((reg['type_verified'] == False).sum())} registry entries have an unverified source type.",
            "Verify types by hand; unclassified sources fall into the 'other' stats group.",
        )  # noqa: E712
    # 3 frames and threads
    for fid in cfg.frame_ids:
        n = int((coded["frame_id"] == fid).sum())
        if n < g["min_frame_comments"]:
            add(
                "frames",
                "high" if n < g["severe_frame_comments"] else "medium",
                f"Frame {fid} ({cfg.frame_label[fid]}) has only {n} coded comments (< {g['min_frame_comments']}).",
                "Not analysable with chi-square; collect more material for this frame (targeted sources and searches) or merge it.",
            )
    for tid in cfg.thread_ids:
        n = int((coded["thread_id"] == tid).sum())
        if n < g["min_thread_comments"]:
            add(
                "threads",
                "low",
                f"Thread {tid} ({cfg.thread_label[tid]}) has {n} comments (< {g['min_thread_comments']}).",
                "Merge, retire or look for more evidence.",
            )
    em = coded[coded["thread_id"] == EMERGENT]
    if len(em):
        add(
            "threads",
            "medium",
            f"{len(em)} comments sit in {EMERGENT} threads.",
            "Run another discovery round to turn proposals into threads.",
        )
    # 4 actors
    declared = c[(c["actor_category"] != "") & c["actor_self_declared"]]
    for cid, meta in cfg.actor_categories.items():
        n = int((declared["actor_category"] == cid).sum())
        if n == 0 and meta["coverage_check"]:
            add(
                "actors",
                "medium" if meta["contact_candidate"] else "low",
                f"No self-declared '{cid}' ({meta['label']}) in the corpus.",
                "Search for these actors in other groups/pages; the corpus gives no way to contact them."
                if meta["contact_candidate"]
                else "Check whether this actor should be sought elsewhere.",
            )
    # 5 expected terms
    text = cit["text"].astype(str)
    term_counts = {}
    for t in g["expected_terms"]:
        n = int(text.str.contains(t["pattern"], case=False, regex=True).sum())
        term_counts[t["label"]] = n
        if n == 0:
            add(
                "evidence",
                "medium",
                f"Expected term group '{t['label']}' never appears in citizen comments.",
                "Check whether the topic is absent from the discourse or missing from the sampled sources.",
            )
    # 5b evidence level (descriptive only: summarised per frame, not tested)
    empty_warn = float(
        g.get("empty_evidence_share_warning", EMPTY_EVIDENCE_SHARE_WARNING)
    )
    weak_warn = float(
        g.get("weakest_evidence_share_warning", WEAKEST_EVIDENCE_SHARE_WARNING)
    )
    evidence_by_frame: dict[str, dict[str, int]] = {}
    for fid in cfg.frame_ids:
        sub = coded[coded["frame_id"] == fid]
        counts = sub["evidence_level"].fillna("").astype(str).value_counts()
        dist = {str(k): int(v) for k, v in counts.items()}
        dist["n"] = int(len(sub))
        evidence_by_frame[fid] = dist
        if not dist["n"]:
            continue
        empty_share = dist.get("", 0) / dist["n"]
        weak_share = dist.get(WEAKEST_EVIDENCE_LEVEL, 0) / dist["n"]
        if empty_share >= empty_warn:
            add(
                "evidence",
                "high" if empty_share >= 0.8 else "medium",
                f"{empty_share:.0%} of the {dist['n']} coded comments in frame {fid} "
                f"({cfg.frame_label[fid]}) carry no evidence_level.",
                "Re-code or review these comments: a frame claim needs an explicit evidence level.",
            )
        elif weak_share >= weak_warn:
            add(
                "evidence",
                "medium",
                f"{weak_share:.0%} of the {dist['n']} coded comments in frame {fid} "
                f"({cfg.frame_label[fid]}) rest on '{WEAKEST_EVIDENCE_LEVEL}' evidence.",
                "Look for explicit evidence (figures, concrete charges) or qualify the frame's support.",
            )
    # 6 temporal coverage
    years = cit["timestamp"].dt.year.value_counts().sort_index()
    span = (
        range(int(years.index.min()), int(years.index.max()) + 1) if len(years) else []
    )
    thin = [y for y in span if years.get(y, 0) < g["min_comments_per_year"]]
    if thin:
        add(
            "time",
            "medium",
            "Thin or empty years (fewer than %d comments): %s."
            % (g["min_comments_per_year"], ", ".join(map(str, thin))),
            "Extend the search window or state the limitation.",
        )
    # 7 locations
    mentions = c["locations_mentioned"].str.split("; ").explode()
    audit_level = (cfg.locations.get("coverage_audit") or {}).get("level")
    for loc in cfg.gazetteer:
        if (
            audit_level
            and loc["level"] == audit_level
            and loc.get("in_study_area", True)
            and not (mentions == loc["name"]).any()
        ):
            add(
                "locations",
                "low",
                f"Study-area location '{loc['name']}' (level: {audit_level}) is never mentioned.",
                "Look for local groups/pages for this location.",
            )
    # 8 statistical design
    n_posts = int(posts["post_id"].nunique())
    if n_posts < g["min_posts_for_pooled_tests"]:
        add(
            "statistics",
            "medium",
            f"Only {n_posts} posts (< {g['min_posts_for_pooled_tests']}): comments are clustered in posts and users repeat.",
            "Report chi-square results as descriptive; use a multilevel model (comment within post) once more posts are available.",
        )
    sparse = [
        f
        for f in cfg.frame_ids
        if (coded["frame_id"] == f).sum() < g["sparse_frame_comments"]
    ]
    if sparse:
        add(
            "statistics",
            "medium",
            f"Frames with fewer than {g['sparse_frame_comments']} coded comments: {', '.join(sparse)}.",
            "Use Fisher's exact test or drop them from RxC tests.",
        )
    # 9 review queue and reliability
    review = (
        c[
            c["needs_human_review"]
            | ((c["frame_id"] != cfg.no_frame_id) & (c["confidence"] == 1))
        ]
        if "confidence" in c
        else c.iloc[0:0]
    )
    if len(review):
        add(
            "coding",
            "medium",
            f"{len(review)} comments are in the human-review queue.",
            f"Review {F.REVIEW} and correct the codes.",
        )
    reliability = out / F.RELIABILITY
    if not reliability.exists():
        add(
            "coding",
            "high",
            "Inter-coder reliability has not been measured.",
            "Code a 20-25% sample by hand (or with a second model) and run the `compare` command.",
        )
    ethics = cfg.study.get("ethics") or {}
    if ethics.get("enabled", True):
        add(
            "ethics",
            "high",
            ethics.get("text")
            or "The corpus contains public data about private persons (ids, names, profile links) collected without consent.",
            "Use pseudonyms in any shared report; apply the ethics and data-protection protocol before contacting anyone.",
        )
    # 10 deliverables
    deliverables = []
    for d in cfg.deliverables:
        have = [a for a in d["artifacts"] if (out / a).exists() or (work / a).exists()]
        status = (
            "produced"
            if len(have) == len(d["artifacts"])
            else "partial"
            if have
            else "missing"
        )
        deliverables.append(
            {
                "id": d["id"],
                "task": d.get("task", ""),
                "description": d["description"],
                "status": status,
                "artifacts_found": have,
                "manual_followup": d.get("manual_followup", ""),
            }
        )
        if d.get("manual_followup"):
            add(
                "deliverables",
                "medium" if status != "missing" else "high",
                f"[{d['id']}] {d['manual_followup']}",
                f"Deliverable status: {status}.",
            )
    report = {
        "config_hash": cfg.hash,
        "study_id": cfg.study["id"],
        "n_posts": n_posts,
        "n_comments": int(len(c)),
        "frame_counts": {f: int((coded["frame_id"] == f).sum()) for f in cfg.frame_ids},
        "evidence_level_by_frame": evidence_by_frame,
        "thread_counts": {
            t: int((coded["thread_id"] == t).sum()) for t in cfg.thread_ids
        },
        "expected_term_counts": term_counts,
        "comments_per_year": {int(k): int(v) for k, v in years.items()},
        "deliverables": deliverables,
        "gaps": gaps,
    }
    write_json(out / F.GAPS_JSON, report)
    (out / F.GAPS_MD).write_text(render_markdown(report, cfg), encoding="utf-8")
    return {"gaps": len(gaps), "high": sum(1 for x in gaps if x["severity"] == "high")}


def render_markdown(r: dict, cfg: Config) -> str:
    order = {"high": 0, "medium": 1, "low": 2}
    lines = [
        f"# Gap report - {cfg.study['title']}",
        "",
        f"Config hash `{r['config_hash']}` | posts: {r['n_posts']} | comments: {r['n_comments']}",
        "",
        "## Deliverables",
        "",
        "| id | task | status | artifacts found |",
        "|---|---|---|---|",
    ]
    for d in r["deliverables"]:
        lines.append(
            f"| {d['id']} | {d['task']} | {d['status']} | {', '.join(d['artifacts_found'])} |"
        )
    lines += ["", "## Gaps (most severe first)", ""]
    for g in sorted(r["gaps"], key=lambda x: (order[x["severity"]], x["area"])):
        lines.append(
            f"- **[{g['severity']}] {g['area']}** - {g['finding']} _Action:_ {g['action']}"
        )
    lines += ["", "## Coded comments per frame", ""] + [
        f"- {cfg.frame_label[f]} ({f}): {n}" for f, n in r["frame_counts"].items()
    ]
    evidence = r.get("evidence_level_by_frame") or {}
    if evidence:
        levels = sorted(
            {lvl for dist in evidence.values() for lvl in dist if lvl != "n"}
        ) or list(EVIDENCE)
        lines += [
            "",
            "## Evidence level per frame (descriptive)",
            "",
            "Descriptive only: no hypothesis test, no p-values and no multiplicity correction.",
            "",
            "| frame | n | " + " | ".join(lvl or "(none)" for lvl in levels) + " |",
            "|---|---|" + "---|" * len(levels),
        ]
        for f, dist in evidence.items():
            n = dist.get("n", 0)
            cells = [
                f"{dist.get(lvl, 0)} ({dist.get(lvl, 0) / n:.0%})"
                if n
                else str(dist.get(lvl, 0))
                for lvl in levels
            ]
            lines.append(
                f"| {cfg.frame_label[f]} ({f}) | {n} | " + " | ".join(cells) + " |"
            )
    return "\n".join(lines) + "\n"
