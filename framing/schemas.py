"""JSON response schemas for the LLM steps (generated from the study config) and record validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import fields as _fields
from . import post_fields as _post_fields

if TYPE_CHECKING:  # pragma: no cover
    from .config import Config

#: Built-in fallbacks, used only when a study's ``codebook.json`` omits
#: ``stance_definitions`` / ``evidence_definitions``. The study codebook is the
#: authority: ``Config.stance_ids`` / ``Config.evidence_ids`` read it, in declaration
#: order, and the prompt, the JSON schema and the record checks all derive from those.
STANCES = ["supports", "opposes", "qualifies"]
EVIDENCE = ["core", "adjacent"]

#: Default id with which the model proposes a thread that does not exist yet.
#: A study may rename it in ``study.json`` -> ``classification.emergent_thread_id``;
#: ``Config.emergent_thread_id`` is the value the pipeline actually uses.
EMERGENT = "EMERGENT"

#: Canonical record shape (every field, selected or not). Derived from the registry
#: so the record order, the JSON-schema order and the prompt order stay identical.
RECORD_DEFAULTS = {f.name: f.default for f in _fields.FIELDS}
STRING_FIELDS = [k for k, v in RECORD_DEFAULTS.items() if isinstance(v, str)]
BOOL_FIELDS = [k for k, v in RECORD_DEFAULTS.items() if isinstance(v, bool)]


def _property(spec: _fields.Field, cfg: "Config") -> dict:
    """The JSON-schema fragment of one answer field."""
    prop = spec.prop
    if prop == "comment_id":
        return {"type": "string", "description": spec.description}
    if prop == "bool":
        return {"type": "boolean"}
    if prop == "frame":
        return {"type": "string", "enum": cfg.all_frame_ids}
    if prop == "frame_or_empty":
        return {"type": "string", "enum": [""] + cfg.frame_ids}
    if prop == "thread":
        enum = [""] + cfg.thread_ids
        # The emergent id is only offerable when the model can also explain the
        # proposal it stands for.
        if "emergent_thread_proposal" in cfg.output_field_set:
            enum = enum + [cfg.emergent_thread_id]
        return {"type": "string", "enum": enum}
    if prop == "stance":
        return {"type": "string", "enum": [""] + cfg.stance_ids}
    if prop == "evidence":
        return {"type": "string", "enum": [""] + cfg.evidence_ids}
    if prop == "confidence":
        return {"type": "integer", "enum": [1, 2, 3]}
    if prop == "actor":
        return {"type": "string", "enum": [""] + cfg.actor_ids}
    return {"type": "string"}


def classification_item_schema(cfg: "Config") -> dict:
    """The response schema for one comment, restricted to the study's answer fields."""
    props = {spec.name: _property(spec, cfg) for spec in cfg.field_specs}
    return {
        "type": "object",
        "properties": props,
        "required": list(props),
        "additionalProperties": False,
    }


def classification_schema(cfg: "Config") -> dict:
    return {"type": "array", "items": classification_item_schema(cfg)}


def _post_property(spec: _post_fields.Field, cfg: "Config") -> dict:
    """The JSON-schema fragment of one post answer field."""
    prop = spec.prop
    if prop == "post_key_desc":
        return {"type": "string", "description": spec.description}
    if prop == "text_desc":
        return {"type": "string", "description": spec.description}
    if prop == "frame":
        return {"type": "string", "enum": cfg.all_frame_ids}
    if prop == "frame_list":
        return {"type": "array", "items": {"type": "string", "enum": cfg.frame_ids}}
    if prop == "source_type":
        return {"type": "string", "enum": cfg.source_type_ids}
    if prop == "confidence":
        return {"type": "integer", "enum": [1, 2, 3]}
    return {"type": "string"}


def post_classification_schema(cfg: "Config") -> dict:
    """The response schema for one post, restricted to the study's post fields."""
    props = {spec.name: _post_property(spec, cfg) for spec in cfg.post_field_specs}
    item = {
        "type": "object",
        "properties": props,
        "required": list(props),
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
    active = cfg.output_field_set
    fid = rec.get("frame_id")
    if fid not in cfg.all_frame_ids:
        return [f"invalid frame_id {fid!r}"]
    none = fid == cfg.no_frame_id
    # A field the study does not ask for is never graded: its absence is expected,
    # not a model error, so it must not push records into the human-review queue.
    sec = rec.get("secondary_frame_id", "") if "secondary_frame_id" in active else ""
    tid = rec.get("thread_id", "") if "thread_id" in active else ""
    stance = rec.get("stance", "") if "stance" in active else ""
    evidence = rec.get("evidence_level", "") if "evidence_level" in active else ""
    if sec and (sec not in cfg.frame_ids or sec == fid):
        issues.append(f"invalid secondary_frame_id {sec!r}")
    if none:
        if tid or stance or evidence or sec:
            issues.append(
                "no-frame record must leave thread/stance/evidence/secondary empty"
            )
    else:
        if "thread_id" in active:
            if not tid:
                issues.append("missing thread_id")
            elif tid != cfg.emergent_thread_id and cfg.thread_frame.get(tid) != fid:
                issues.append(f"thread {tid!r} does not belong to frame {fid!r}")
            if (
                tid == cfg.emergent_thread_id
                and "emergent_thread_proposal" in active
                and not rec.get("emergent_thread_proposal")
            ):
                issues.append(
                    f"{cfg.emergent_thread_id} thread without emergent_thread_proposal"
                )
        if "stance" in active and stance not in cfg.stance_ids:
            issues.append(f"invalid stance {stance!r}")
        if "evidence_level" in active and evidence not in cfg.evidence_ids:
            issues.append(f"invalid evidence_level {evidence!r}")
    if "confidence" in active and rec.get("confidence") not in (1, 2, 3):
        issues.append(f"invalid confidence {rec.get('confidence')!r}")
    if (
        "actor_category" in active
        and rec.get("actor_category", "") not in [""] + cfg.actor_ids
    ):
        issues.append(f"invalid actor_category {rec.get('actor_category')!r}")
    if "is_institutional" in active and rec.get("is_institutional") and not none:
        issues.append("institutional comments must be coded as no-frame")
    return issues


def normalise_classification(rec: dict, cfg: "Config") -> tuple[dict, list[str]]:
    """Coerce a raw model record into a clean record. Returns (record, issues found before coercion)."""
    out = {k: rec.get(k, d) for k, d in cfg.record_defaults.items()}
    for k in STRING_FIELDS:
        out[k] = "" if out[k] is None else str(out[k]).strip()
    for k in BOOL_FIELDS:
        out[k] = _to_bool(out[k])
    try:
        out["confidence"] = int(out["confidence"])
    except (TypeError, ValueError):
        out["confidence"] = 0
    f = _match(out["frame_id"], cfg.all_frame_ids, allow_empty=True)
    out["frame_id"] = (
        cfg.no_frame_id
        if f in (None, "") and out["frame_id"] == ""
        else f
        if f
        else out["frame_id"]
    )
    out["secondary_frame_id"] = (
        _match(out["secondary_frame_id"], cfg.frame_ids, True)
        or out["secondary_frame_id"]
    )
    t = _match(out["thread_id"], cfg.thread_ids + [cfg.emergent_thread_id], True)
    out["thread_id"] = t if t is not None else out["thread_id"]
    # Canonicalise against the study vocabulary instead of lower-casing blindly: a
    # codebook may declare ids that are not lower-case, and an unmatched value is kept
    # verbatim so ``check_classification`` still reports it before it is blanked.
    stance = _match(out["stance"], cfg.stance_ids, allow_empty=True)
    out["stance"] = out["stance"].lower() if stance is None else stance
    evidence = _match(out["evidence_level"], cfg.evidence_ids, allow_empty=True)
    out["evidence_level"] = (
        out["evidence_level"].lower() if evidence is None else evidence
    )
    issues = check_classification(out, cfg)
    # ---- coercion to a safe state (the issues stay recorded for review)
    if out["frame_id"] not in cfg.all_frame_ids:
        out["frame_id"] = cfg.no_frame_id
    if out["frame_id"] == cfg.no_frame_id or out["is_institutional"]:
        out["frame_id"] = cfg.no_frame_id
        for k in (
            "secondary_frame_id",
            "thread_id",
            "stance",
            "evidence_level",
            "emergent_thread_proposal",
        ):
            out[k] = ""
    if (
        out["secondary_frame_id"] not in [""] + cfg.frame_ids
        or out["secondary_frame_id"] == out["frame_id"]
    ):
        out["secondary_frame_id"] = ""
    if out["stance"] not in [""] + cfg.stance_ids:
        out["stance"] = ""
    if out["evidence_level"] not in [""] + cfg.evidence_ids:
        out["evidence_level"] = ""
    if out["actor_category"] not in [""] + cfg.actor_ids:
        out["actor_category"] = ""
        out["actor_self_declared"] = False
    if out["confidence"] not in (1, 2, 3):
        out["confidence"] = 1
    if issues:
        out["needs_human_review"] = True
        out["review_reason"] = "; ".join(
            filter(
                None,
                [out["review_reason"], "invalid model output: " + " | ".join(issues)],
            )
        )
    return out, issues
