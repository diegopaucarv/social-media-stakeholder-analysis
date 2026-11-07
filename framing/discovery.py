"""Iterative codebook discovery: build the input of each LLM round and apply the (validated) response.

Workflow: research_brief.json -> seed_codebook -> [discovery_input -> LLM -> apply_discovery] x N rounds -> codebook.json.
The LLM only proposes the `codebook` module; every other study module stays under human control."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import files as F
from .config import ConfigError, config_from_modules, load_modules, schema_errors
from .prompt_builder import (
    PROMPT_DIR,
    discovery_response_schema,
    discovery_user_message,
)
from .utils import as_bool, json_safe, parse_json_text, read_json, write_json

DEFAULT_STANCES = {
    "supports": "The comment sustains or reproduces the argument of the frame/thread.",
    "opposes": "The comment contradicts or rebuts the argument of the thread.",
    "qualifies": "The comment accepts the argument only partly, adds conditions or doubts.",
}
DEFAULT_EVIDENCE = {
    "core": "The comment reproduces almost literally the target argument of the frame.",
    "adjacent": "The comment gives evidence for the same frame without stating the whole argument.",
}


def seed_codebook(
    brief: dict, *, no_frame_id: str = "NONE", id_prefix: str = "F"
) -> dict:
    """Codebook with the researcher's seed frames (no threads, no rules, no examples yet).

    `id_prefix` and `no_frame_id` let the seed follow the id convention of the target study;
    the defaults keep the historical "F<i>" / "NONE" ids."""
    errs = schema_errors(brief, "research_brief")
    if errs:
        raise ConfigError("\n".join(errs))
    frames = [
        {
            "id": f"{id_prefix}{i}",
            "label": f["label"],
            "argument": f["argument"],
            "threads": [],
        }
        for i, f in enumerate(brief["seed_frames"], 1)
    ]
    return {
        "no_frame": {
            "id": no_frame_id,
            "label": "No frame",
            "description": "Comments that sustain none of the frames: greetings, praise or support without argument, generic insults without a reason, unrelated topics, tags, stickers or missing text.",
        },
        "frames": frames,
        "stance_definitions": DEFAULT_STANCES,
        "evidence_definitions": DEFAULT_EVIDENCE,
        "decision_rules": [],
        "examples": [],
    }


def _brief_for_prompt(cfg, brief: dict) -> dict:
    st = cfg.study
    return {
        "study": {
            "title": st["title"],
            "objective": st["objective"],
            "context": st["context"],
            "corpus_language": st["corpus_language"],
            "analysis_language": st.get("analysis_language", "en"),
            "language_notes": st.get("language_notes", ""),
        },
        "seed_frames": brief["seed_frames"],
        "researcher_instructions": brief.get("researcher_instructions", ""),
        "actor_taxonomy": [
            {
                "level": lvl["id"],
                "categories": [
                    {
                        "id": c["id"],
                        "label": c["label"],
                        "description": c["description"],
                    }
                    for c in lvl["categories"]
                ],
            }
            for lvl in cfg.actors["levels"]
        ],
        "source_types": {k: v["label"] for k, v in cfg.source_types.items()},
        "deliverables": [
            {"id": d["id"], "description": d["description"]} for d in cfg.deliverables
        ],
        "institutional_authors": list(cfg.institutional),
    }


def draw_sample(
    c: pd.DataFrame,
    posts: pd.DataFrame,
    used: set,
    size: int,
    seed: int,
    text_cap: int = 300,
) -> list[dict]:
    """Round-robin over posts; inside each post the most relevant comments first, then random ones."""
    rng = np.random.default_rng(seed)
    ctx = posts.set_index("post_id")
    text_by_id = c.set_index("comment_id")["text"].to_dict()
    pool = c[(~c["is_institutional"]) & c["has_text"] & (~c["comment_id"].isin(used))]
    queues = []
    for pid, g in pool.groupby("post_id"):
        g = g.assign(_r=rng.random(len(g)))
        high = g[g["relevance_tier"] == "high"].sort_values(
            "relevance_score", ascending=False
        )
        rest = g[g["relevance_tier"] != "high"].sort_values("_r")
        queues.append(list(pd.concat([high, rest]).itertuples(index=False)))
    rng.shuffle(queues)
    picked = []
    while len(picked) < size and any(queues):
        for q in queues:
            if q and len(picked) < size:
                picked.append(q.pop(0))
    out = []
    for r in picked:
        p = ctx.loc[r.post_id]
        par = r.reply_to_id
        out.append(
            {
                "comment_id": r.comment_id,
                "post_key": p["post_key"],
                "source_type": p["source_type"],
                "post_text": str(p["post_text"])[:text_cap],
                "thread_depth": int(r.thread_depth),
                "likes": int(r.likes),
                "reply_count": int(r.reply_count),
                "parent_text": ""
                if pd.isna(par)
                else str(text_by_id.get(par) or "")[:text_cap],
                "text": r.text,
            }
        )
    return out


def build_discovery_input(
    manifest,
    brief_path,
    work_dir,
    out_dir,
    round_no: int,
    sample_size: int = 80,
    seed: int = 1,
) -> dict:
    """Write the inputs of discovery round `round_no`."""
    modules, man, base = load_modules(manifest)
    cfg = config_from_modules(modules, man, base, indicator_refs=False)
    brief = read_json(brief_path)
    work, out = Path(work_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    c = pd.read_csv(
        work / F.COMMENTS_ENRICHED,
        dtype={"comment_id": str, "reply_to_id": str},
        low_memory=False,
    )
    for col in ("is_institutional", "has_text"):
        c[col] = as_bool(c[col])
    c["text"] = c["text"].fillna("")
    posts = pd.read_csv(work / F.POSTS_TABLE, dtype=str).fillna("")
    posts["post_id"] = posts["post_id"].astype(int)
    state_path = out / "discovery_state.json"
    state = (
        read_json(state_path)
        if state_path.exists()
        else {"used_comment_ids": [], "rounds": []}
    )
    text_cap = int(cfg.study["classification"]["parent_text_max_chars"])
    sample = draw_sample(
        c, posts, set(state["used_comment_ids"]), sample_size, seed + round_no, text_cap
    )
    if not sample:
        raise ValueError("no unused comments left to sample")
    state["used_comment_ids"] += [s["comment_id"] for s in sample]
    state["rounds"].append(
        {
            "round": round_no,
            "sample_size": len(sample),
            "built_from_codebook_hash": cfg.hash,
        }
    )
    write_json(state_path, state)
    codebook = cfg.codebook
    payload = {
        "round": round_no,
        "research_brief": _brief_for_prompt(cfg, brief),
        "current_codebook": codebook,
        "sample": sample,
    }
    write_json(out / f"discovery_round_{round_no:02d}_input.json", payload)
    message = discovery_user_message(
        round_no, payload["research_brief"], codebook, sample
    )
    (out / f"discovery_round_{round_no:02d}_user_message.md").write_text(
        message, encoding="utf-8"
    )
    (out / "discovery_system.md").write_text(
        (PROMPT_DIR / "discovery_system.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    write_json(out / "discovery_response_schema.json", discovery_response_schema())
    return {
        "round": round_no,
        "sample": len(sample),
        "posts_covered": len({s["post_key"] for s in sample}),
    }


def _ids(codebook: dict) -> set:
    ids = {f["id"] for f in codebook["frames"]}
    ids |= {t["id"] for f in codebook["frames"] for t in f["threads"]}
    return ids


def apply_discovery(manifest, response_path, out_codebook, history_dir=None) -> dict:
    """Validate an LLM discovery response and write the new codebook module. Raises ConfigError if invalid."""
    modules, man, base = load_modules(manifest)
    data = parse_json_text(Path(response_path).read_text(encoding="utf-8"))
    if len(data) != 1 or not isinstance(data[0], dict):
        raise ConfigError("the response must be exactly one JSON object")
    resp = data[0]
    errs = schema_errors(resp, "discovery_output")
    if errs:
        raise ConfigError(
            "response does not match the discovery schema:\n" + "\n".join(errs)
        )
    old_ids = _ids(modules["codebook"])
    new_modules = dict(modules, codebook=resp["codebook"])
    cfg = config_from_modules(
        new_modules, man, base, indicator_refs=False
    )  # schema + cross-module semantics (examples, actors...)
    logged = " ".join(
        f"{c['target']} {c['reason']}"
        for c in resp["changelog"]
        if c["op"] in ("retire_thread", "merge_threads", "split_thread", "edit_frame")
    )
    silent = sorted(i for i in old_ids - _ids(resp["codebook"]) if i not in logged)
    if silent:
        raise ConfigError(
            f"ids removed without a retire/merge/split entry in the changelog: {silent}"
        )
    out = Path(out_codebook)
    write_json(out, resp["codebook"])
    hist = Path(history_dir) if history_dir else out.parent
    hist.mkdir(parents=True, exist_ok=True)
    write_json(hist / f"codebook.round{resp['round']:02d}.json", resp["codebook"])
    with open(hist / "codebook_history.jsonl", "a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                json_safe(
                    {
                        "round": resp["round"],
                        "config_hash": cfg.hash,
                        "changelog": resp["changelog"],
                        "saturation": resp["saturation"],
                        "open_questions": resp.get("open_questions", []),
                    }
                ),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )
    return {
        "round": resp["round"],
        "frames": len(cfg.frames),
        "threads": len(cfg.thread_ids),
        "examples": len(cfg.codebook["examples"]),
        "saturated": resp["saturation"]["is_saturated"],
        "changes": len(resp["changelog"]),
        "open_questions": len(resp.get("open_questions", [])),
        "indicator_reference_warnings": cfg.broken_indicator_refs(),
    }
