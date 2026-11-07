"""Renders the LLM prompts and JSON response schemas from the unified study config and the templates."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

from . import fields
from .config import PKG_ROOT, Config
from .schemas import classification_schema, post_classification_schema
from .utils import read_json, write_json

PROMPT_DIR = PKG_ROOT / "prompts"

GENERIC_RULES = [
    "Evaluate each comment IN CONTEXT: use post_context, parent_text (the parent comment, when present) and the earlier comments of the batch. A bare agreement ('yes', 'exactly') is coded only if the parent shows what it agrees with; it then inherits the parent's frame as evidence_level 'adjacent', confidence 1 and needs_human_review true.",
    "Code the proposition the author sustains, not the topic they mention. Naming a source, an institution or a place is not enough: there must be an argument, a belief, a complaint with an explicit reason or a stance.",
    "The primary frame is the dominant argument. Add a secondary frame only if the comment clearly sustains a second argument (at most one). A comment is coded once.",
    "Irony and sarcasm: code the pragmatic meaning, not the literal words, and set needs_human_review true with review_reason 'irony' unless the intent is unmistakable.",
    "Do not translate the literal_excerpt: copy it verbatim from the comment, in the corpus language (at most 25 words).",
    "Do not invent. If a field cannot be determined from the available text, leave it empty ('') and prefer the no-frame id over a forced assignment.",
    "Comments written by institutional authors (listed in <institutional_authors>) get is_institutional true and the no-frame id.",
    "Conservative calibration: when in doubt between '{evidence_first}' and '{evidence_last}' choose '{evidence_last}'; between '{stance_first}' and '{stance_last}' choose '{stance_last}'; between a frame and no-frame choose no-frame.",
    "Support without argument is no-frame: praise, applause, encouragement, bare agreement and generic insults without a stated reason sustain no frame, even when the post is about the study topic. Exception: an endorsement counts as 'adjacent' evidence only if its own text names the subject matter of a frame (the study-specific rules below say what counts). If the endorsement is understood only through the parent comment, apply the first rule.",
    "Weak adjacent evidence: a comment of level '{evidence_last}' that (a) only voices general discontent without an explicit reason, or (b) is linked to the frame by inference only, gets confidence 1, needs_human_review true and review_reason 'weak adjacent'. Code it once: a single frame, no secondary frame, the most probable thread. An adjacent comment with an explicit reason (a figure, a concrete charge, a comparison) is firm and keeps confidence 2 or 3.",
    "Actor signals: set actor_category only when the author self-declares or the text unambiguously evidences their own role (e.g. stating their job or role, saying they run the organisation, or saying they live in the place under discussion). Never infer it from tone, ideology or identity. Otherwise actor_category is '', actor_self_declared false and actor_evidence ''. actor_evidence is a verbatim excerpt.",
]

#: Per-field one-liners for the rendered prompt, derived from the field registry so
#: there is exactly one place to change when a field is added, dropped or reworded.
#: A study may override any wording through ``study.json`` -> ``output_field_help``.
FIELD_HELP = {spec.name: spec.help for spec in fields.FIELDS}

#: Generic rules that only make sense while their field is still asked for. Rules
#: about the wider coding philosophy are always kept: dropping a field should not
#: silently delete the reasoning that surrounds it.
RULE_REQUIRES = {
    4: frozenset({"literal_excerpt"}),
    6: frozenset({"is_institutional"}),
    10: frozenset({"actor_category", "actor_self_declared", "actor_evidence"}),
}


def _attr(s: str) -> str:
    return quoteattr(s)


def codebook_xml(cfg: Config) -> str:
    parts = []
    for f in cfg.frames:
        threads = (
            "\n".join(
                f"<thread id={_attr(t['id'])} label={_attr(t['label'])}>{escape(t['description'])}</thread>"
                for t in f["threads"]
            )
            or f"<!-- no threads yet: propose {cfg.emergent_thread_id} threads -->"
        )
        parts.append(
            f"<frame id={_attr(f['id'])} label={_attr(f['label'])}>\n<argument>{escape(f['argument'])}</argument>\n"
            f"<threads>\n{threads}\n</threads>\n</frame>"
        )
    nf = cfg.codebook["no_frame"]
    parts.append(
        f"<no_frame id={_attr(nf['id'])} label={_attr(nf['label'])}>{escape(nf['description'])}</no_frame>"
    )
    return "\n".join(parts)


def stance_keys(cfg: Config) -> list[str]:
    """Stance ids declared by the study codebook, falling back to the built-in vocabulary."""
    return list(cfg.stance_ids)


def evidence_keys(cfg: Config) -> list[str]:
    """Evidence levels declared by the study codebook, falling back to the built-in vocabulary."""
    return list(cfg.evidence_ids)


def stances_xml(cfg: Config) -> str:
    return "\n".join(
        f'<value id="{k}">{escape(cfg.codebook["stance_definitions"][k])}</value>'
        for k in stance_keys(cfg)
    )


def evidence_xml(cfg: Config) -> str:
    return "\n".join(
        f'<value id="{k}">{escape(cfg.codebook["evidence_definitions"][k])}</value>'
        for k in evidence_keys(cfg)
    )


def actors_xml(cfg: Config) -> str:
    out = []
    for lvl in cfg.actors["levels"]:
        cats = "\n".join(
            f"<category id={_attr(c['id'])} label={_attr(c['label'])}>{escape(c['description'])}</category>"
            for c in lvl["categories"]
        )
        out.append(
            f"<level id={_attr(lvl['id'])} label={_attr(lvl['label'])}>\n{cats}\n</level>"
        )
    return "\n".join(out)


def institutional_text(cfg: Config) -> str:
    if not cfg.institutional:
        return "(none)"
    return "\n".join(
        f'- "{n}" (actor category: {c})' for n, c in cfg.institutional.items()
    )


def rules_text(cfg: Config) -> str:
    """The decision rules, with the study's own vocabulary where a rule names ids.

    A study declares its levels strongest first and its stances in the usual order
    (support, opposition, qualification), so ``[0]`` is the firm end of each scale and
    ``[-1]`` the cautious end. With the built-in vocabulary this renders exactly the
    wording it always had.
    """
    fmt = {
        "stance_first": cfg.stance_ids[0],
        "stance_last": cfg.stance_ids[-1],
        "evidence_first": cfg.evidence_ids[0],
        "evidence_last": cfg.evidence_ids[-1],
    }
    kept = [
        rule
        for i, rule in enumerate(GENERIC_RULES)
        if RULE_REQUIRES.get(i, frozenset()) <= cfg.output_field_set
    ]
    rules = [rule.format(**fmt) if "{" in rule else rule for rule in kept]
    rules += list(cfg.codebook["decision_rules"])
    return "\n".join(f"{i}. {r}" for i, r in enumerate(rules, 1))


def examples_xml(cfg: Config) -> str:
    out = []
    for ex in cfg.codebook["examples"]:
        note = f"\n<note>{escape(ex['note'])}</note>" if ex.get("note") else ""
        out.append(
            f"<example id={_attr(ex['input']['comment_id'])}>\n<input>{json.dumps(ex['input'], ensure_ascii=False)}</input>\n"
            f"<output>{json.dumps(ex['output'], ensure_ascii=False)}</output>{note}\n</example>"
        )
    return "\n\n".join(out) if out else "(no examples yet)"


def output_schema_text(cfg: Config) -> str:
    """The "exactly these fields" block, restricted to the study's answer fields."""
    fmt = {
        "frames": json.dumps(cfg.all_frame_ids),
        "frames_or_empty": json.dumps([""] + cfg.frame_ids),
        "actors_or_empty": json.dumps([""] + cfg.actor_ids),
        "stances": json.dumps(stance_keys(cfg)),
        "evidence_levels": json.dumps(evidence_keys(cfg)),
        "emergent": cfg.emergent_thread_id,
    }
    lines = []
    for spec in cfg.field_specs:
        text = cfg.field_help.get(spec.name, spec.help)
        lines.append(f"- {spec.name}: {text.format(**fmt) if '{' in text else text}")

    clauses = []
    blanks = [n for n in fields.PROMPT_NO_FRAME_BLANKS if n in cfg.output_field_set]
    if blanks:
        clauses.append(f'{fields.prose(blanks)} are ""')
    graded = [
        phrase
        for name, phrase in (
            ("confidence", "confidence is 3"),
            ("needs_human_review", "needs_human_review false"),
        )
        if name in cfg.output_field_set
    ]
    if graded:
        clauses.append(fields.prose(graded) + " unless there is a real doubt")
    note = "\nIn a no-frame record: " + "; ".join(clauses) + "." if clauses else ""
    head = (
        "Return ONLY a valid JSON array (no markdown, no extra text) with one object "
        "per comment, in the order of the input. Every object has exactly these fields:\n"
        + "\n".join(lines)
    )
    return (
        head
        + note
        + (" " if note else "\n")
        + "No trailing commas, no comments inside the JSON."
    )


def post_output_schema_text(cfg: Config) -> str:
    """The post-answer contract, rendered from the study's selected post fields."""
    fmt = {
        "frames": json.dumps(cfg.all_frame_ids),
        "frames_list": json.dumps(cfg.frame_ids),
        "source_types": json.dumps(cfg.source_type_ids),
    }
    lines = []
    for spec in cfg.post_field_specs:
        text = cfg.post_field_help.get(spec.name, spec.help)
        for key, value in fmt.items():
            text = text.replace("{" + key + "}", value)
        lines.append(f"- {spec.name}: {text}")
    return (
        "Return ONLY a valid JSON array (no markdown, no extra text), one object "
        "per post, in input order, with exactly these fields:\n" + "\n".join(lines)
    )


def source_types_text(cfg: Config) -> str:
    return "\n".join(f"- {k}: {v['label']}" for k, v in cfg.source_types.items())


def _fill(template: str, mapping: dict, check: bool = True) -> str:
    for k, v in mapping.items():
        template = template.replace("{{" + k + "}}", v)
    if check:
        left = sorted(set(re.findall(r"\{\{[A-Z_]+\}\}", template)))
        if left:
            raise ValueError(f"unfilled placeholders: {left}")
    return template


def _read(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8")


def study_mapping(cfg: Config) -> dict:
    st = cfg.study
    return {
        "STUDY_TITLE": st["title"],
        "STUDY_OBJECTIVE": st["objective"],
        "STUDY_CONTEXT": st["context"],
        "CORPUS_LANGUAGE": st["corpus_language"],
        "LANGUAGE_NOTES": st.get("language_notes", ""),
        "ANALYSIS_LANGUAGE": st.get("analysis_language") or "en",
        "NONE_ID": cfg.no_frame_id,
        "CODEBOOK": codebook_xml(cfg),
    }


def classifier_system_prompt(cfg: Config) -> str:
    m = study_mapping(cfg)
    m.update(
        {
            "STANCES": stances_xml(cfg),
            "EVIDENCE": evidence_xml(cfg),
            "EMERGENT": cfg.emergent_thread_id,
            "ACTOR_TAXONOMY": actors_xml(cfg),
            "INSTITUTIONAL_AUTHORS": institutional_text(cfg),
            "DECISION_RULES": rules_text(cfg),
            "EXAMPLES": examples_xml(cfg),
            "OUTPUT_SCHEMA": output_schema_text(cfg),
        }
    )
    return _fill(_read("classifier_system.template.md"), m)


def post_classifier_system_prompt(cfg: Config) -> str:
    m = study_mapping(cfg)
    m.update(
        {
            "SOURCE_TYPES": source_types_text(cfg),
            "OUTPUT_SCHEMA": post_output_schema_text(cfg),
        }
    )
    return _fill(_read("post_classifier_system.template.md"), m)


def classifier_user_message(batch: dict) -> str:
    return _fill(
        _read("classifier_user.template.md"),
        {
            "POST_CONTEXT": json.dumps(
                batch["post_context"], ensure_ascii=False, indent=1
            ),
            "COMMENTS_BATCH": json.dumps(
                batch["comments_batch"], ensure_ascii=False, indent=1
            ),
        },
        check=False,
    )


def post_classifier_user_message(posts: list[dict]) -> str:
    return _fill(
        _read("post_classifier_user.template.md"),
        {"POSTS": json.dumps(posts, ensure_ascii=False, indent=1)},
        check=False,
    )


def _inline_refs(node, defs):
    if isinstance(node, dict):
        if "$ref" in node:
            return _inline_refs(defs[node["$ref"].split("/")[-1]], defs)
        return {k: _inline_refs(v, defs) for k, v in node.items() if k != "$defs"}
    if isinstance(node, list):
        return [_inline_refs(v, defs) for v in node]
    return node


def discovery_response_schema() -> dict:
    schema = read_json(PKG_ROOT / "schemas" / "modules.schema.json")
    out = _inline_refs(schema["$defs"]["discovery_output"], schema["$defs"])
    out["$schema"] = schema["$schema"]
    return out


def discovery_user_message(
    round_no: int, brief: dict, codebook: dict, sample: list[dict]
) -> str:
    return _fill(
        _read("discovery_user.template.md"),
        {
            "ROUND": str(round_no),
            "BRIEF": json.dumps(brief, ensure_ascii=False, indent=1),
            "CODEBOOK": json.dumps(codebook, ensure_ascii=False, indent=1),
            "SAMPLE": json.dumps(sample, ensure_ascii=False, indent=1),
        },
        check=False,
    )


def render_all(cfg: Config, out_dir) -> dict:
    """Write every rendered prompt and response schema to `out_dir`. Returns a manifest dict."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "classifier_system.md": classifier_system_prompt(cfg),
        "classifier_user.template.md": _read("classifier_user.template.md"),
        "post_classifier_system.md": post_classifier_system_prompt(cfg),
        "post_classifier_user.template.md": _read("post_classifier_user.template.md"),
        "discovery_system.md": _read("discovery_system.md"),
        "discovery_user.template.md": _read("discovery_user.template.md"),
    }
    for name, text in files.items():
        (out / name).write_text(text, encoding="utf-8")
    write_json(out / "classifier_response_schema.json", classification_schema(cfg))
    write_json(
        out / "post_classifier_response_schema.json", post_classification_schema(cfg)
    )
    write_json(out / "discovery_response_schema.json", discovery_response_schema())
    manifest = {
        "config_hash": cfg.hash,
        "study_id": cfg.study["id"],
        "files": {
            n: hashlib.sha256(t.encode("utf-8")).hexdigest()[:12]
            for n, t in files.items()
        },
    }
    write_json(out / "prompt_manifest.json", manifest)
    return manifest
