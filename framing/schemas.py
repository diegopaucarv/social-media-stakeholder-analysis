"""JSON response schemas for the LLM steps (generated from the study config) and record validation."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .config import Config

STANCES = ["supports", "opposes", "qualifies"]
EVIDENCE = ["core", "adjacent"]
EMERGENT = "EMERGENT"

RECORD_DEFAULTS = {
    "comment_id": "", "is_institutional": False, "frame_id": "", "secondary_frame_id": "", "thread_id": "",
    "emergent_thread_proposal": "", "stance": "", "evidence_level": "", "literal_excerpt": "", "rationale": "",
    "confidence": 0, "needs_human_review": False, "review_reason": "", "actor_category": "",
    "actor_self_declared": False, "actor_evidence": "",
}
STRING_FIELDS = [k for k, v in RECORD_DEFAULTS.items() if isinstance(v, str)]
BOOL_FIELDS = [k for k, v in RECORD_DEFAULTS.items() if isinstance(v, bool)]


def classification_item_schema(cfg: "Config") -> dict:
    props = {
        "comment_id": {"type": "string", "description": "Original comment id, copied verbatim from the input."},
        "is_institutional": {"type": "boolean"},
        "frame_id": {"type": "string", "enum": cfg.all_frame_ids},
        "secondary_frame_id": {"type": "string", "enum": [""] + cfg.frame_ids},
        "thread_id": {"type": "string", "enum": [""] + cfg.thread_ids + [EMERGENT]},
        "emergent_thread_proposal": {"type": "string"},
        "stance": {"type": "string", "enum": [""] + STANCES},
        "evidence_level": {"type": "string", "enum": [""] + EVIDENCE},
        "literal_excerpt": {"type": "string"},
        "rationale": {"type": "string"},
        "confidence": {"type": "integer", "enum": [1, 2, 3]},
        "needs_human_review": {"type": "boolean"},
        "review_reason": {"type": "string"},
        "actor_category": {"type": "string", "enum": [""] + cfg.actor_ids},
        "actor_self_declared": {"type": "boolean"},
        "actor_evidence": {"type": "string"},
    }
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def classification_schema(cfg: "Config") -> dict:
    return {"type": "array", "items": classification_item_schema(cfg)}


def post_classification_schema(cfg: "Config") -> dict:
    item = {
        "type": "object",
        "properties": {
            "post_key": {"type": "string", "description": "Post key, copied verbatim from the input."},
            "topic": {"type": "string", "description": "Short topic label (<= 12 words) in the analysis language."},
            "primary_frame_id": {"type": "string", "enum": cfg.all_frame_ids},
            "related_frame_ids": {"type": "array", "items": {"type": "string", "enum": cfg.frame_ids}},
            "source_type_guess": {"type": "string", "enum": cfg.source_type_ids},
            "rationale": {"type": "string"},
            "confidence": {"type": "integer", "enum": [1, 2, 3]},
        },
        "required": ["post_key", "topic", "primary_frame_id", "related_frame_ids", "source_type_guess", "rationale", "confidence"],
        "additionalProperties": False,
    }
    return {"type": "array", "items": item}


def _match(value, valid, allow_empty=False):
    """Exact or case-insensitive match of an id; returns the canonical id or None."""
    v = "" if value is None else str(value).strip()
    if v == "" and allow_empty:
        return ""
    if v in valid:
        return v
    low = {x.lower(): x for x in valid if x}
    return low.get(v.lower())


def _to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "1", "yes")


def check_classification(rec: dict, cfg: "Config") -> list[str]:
    """Semantic checks on one classification record. Returns a list of problems (empty = valid)."""
    issues: list[str] = []
    fid = rec.get("frame_id")
    if fid not in cfg.all_frame_ids:
        return [f"invalid frame_id {fid!r}"]
    none = fid == cfg.no_frame_id
    sec, tid = rec.get("secondary_frame_id", ""), rec.get("thread_id", "")
    if sec and (sec not in cfg.frame_ids or sec == fid):
        issues.append(f"invalid secondary_frame_id {sec!r}")
    if none:
        if tid or rec.get("stance") or rec.get("evidence_level") or sec:
            issues.append("no-frame record must leave thread/stance/evidence/secondary empty")
    else:
        if not tid:
            issues.append("missing thread_id")
        elif tid != EMERGENT and cfg.thread_frame.get(tid) != fid:
            issues.append(f"thread {tid!r} does not belong to frame {fid!r}")
        if tid == EMERGENT and not rec.get("emergent_thread_proposal"):
            issues.append("EMERGENT thread without emergent_thread_proposal")
        if rec.get("stance") not in STANCES:
            issues.append(f"invalid stance {rec.get('stance')!r}")
        if rec.get("evidence_level") not in EVIDENCE:
            issues.append(f"invalid evidence_level {rec.get('evidence_level')!r}")
    if rec.get("confidence") not in (1, 2, 3):
        issues.append(f"invalid confidence {rec.get('confidence')!r}")
    if rec.get("actor_category", "") not in [""] + cfg.actor_ids:
        issues.append(f"invalid actor_category {rec.get('actor_category')!r}")
    if rec.get("is_institutional") and not none:
        issues.append("institutional comments must be coded as no-frame")
    return issues


def normalise_classification(rec: dict, cfg: "Config") -> tuple[dict, list[str]]:
    """Coerce a raw model record into a clean record. Returns (record, issues found before coercion)."""
    out = {k: rec.get(k, d) for k, d in RECORD_DEFAULTS.items()}
    for k in STRING_FIELDS:
        out[k] = "" if out[k] is None else str(out[k]).strip()
    for k in BOOL_FIELDS:
        out[k] = _to_bool(out[k])
    try:
        out["confidence"] = int(out["confidence"])
    except (TypeError, ValueError):
        out["confidence"] = 0
    f = _match(out["frame_id"], cfg.all_frame_ids, allow_empty=True)
    out["frame_id"] = cfg.no_frame_id if f in (None, "") and out["frame_id"] == "" else f if f else out["frame_id"]
    out["secondary_frame_id"] = _match(out["secondary_frame_id"], cfg.frame_ids, True) or out["secondary_frame_id"]
    t = _match(out["thread_id"], cfg.thread_ids + [EMERGENT], True)
    out["thread_id"] = t if t is not None else out["thread_id"]
    out["stance"] = out["stance"].lower()
    out["evidence_level"] = out["evidence_level"].lower()
    issues = check_classification(out, cfg)
    # ---- coercion to a safe state (the issues stay recorded for review)
    if out["frame_id"] not in cfg.all_frame_ids:
        out["frame_id"] = cfg.no_frame_id
    if out["frame_id"] == cfg.no_frame_id or out["is_institutional"]:
        out["frame_id"] = cfg.no_frame_id
        for k in ("secondary_frame_id", "thread_id", "stance", "evidence_level", "emergent_thread_proposal"):
            out[k] = ""
    if out["secondary_frame_id"] not in [""] + cfg.frame_ids or out["secondary_frame_id"] == out["frame_id"]:
        out["secondary_frame_id"] = ""
    if out["stance"] not in [""] + STANCES:
        out["stance"] = ""
    if out["evidence_level"] not in [""] + EVIDENCE:
        out["evidence_level"] = ""
    if out["actor_category"] not in [""] + cfg.actor_ids:
        out["actor_category"] = ""
        out["actor_self_declared"] = False
    if out["confidence"] not in (1, 2, 3):
        out["confidence"] = 1
    if issues:
        out["needs_human_review"] = True
        out["review_reason"] = "; ".join(filter(None, [out["review_reason"], "invalid model output: " + " | ".join(issues)]))
    return out, issues
