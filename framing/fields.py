"""The answer-field vocabulary of the comment classifier.

Five places used to spell out the same list of answer fields: the record defaults,
the JSON response schema, the rendered prompt, the CSV header the chatbot branch
asks for and the CSV aliases accepted on the way back. This module is the single
declaration they all derive from, so the vocabulary can be trimmed, reordered and
re-documented from a study folder instead of from code.

A study configures it in ``study.json``::

    {
      "output_fields": ["comment_id", "frame_id", "thread_id", "stance", "confidence"],
      "output_field_help": {"stance": "one of {stances}, or \"\" for no-frame"}
    }

``comment_id`` and ``frame_id`` are *structural*: the join key and the core of the
analysis, so they can never be dropped. Every other field is optional. Dropping one
does not break the pipeline: the record keeps the field with a safe default, the
analysis tables keep the (empty) column and the gap report surfaces the absence
instead of inventing values. In particular, dropping ``confidence`` makes every
record count as "probable" (2) rather than "doubtful" (1), so the confidence filter
keeps working.

Only the fields listed here can be selected. Adding a brand-new concept still needs
a consumer in the analysis code; this registry reorders and trims, it does not
invent new columns.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Tiers, from "you cannot remove this" to "the pipeline already copes without it".
STRUCTURAL = "structural"
CORE = "core"
OPTIONAL = "optional"


@dataclass(frozen=True)
class Field:
    """One answer field: how the model is asked for it and what a missing value means."""

    name: str
    default: object
    prop: str
    help: str
    tier: str
    csv: bool = True
    aliases: tuple[str, ...] | None = None
    description: str = ""
    requires: tuple[str, ...] = ()

    def __post_init__(self):
        if self.aliases is None:
            object.__setattr__(self, "aliases", (self.name,))


#: Fields in record / JSON-schema / prompt order. This order is part of the
#: contract: it is the order of ``RECORD_DEFAULTS`` and of the rendered schema.
FIELDS: tuple[Field, ...] = (
    Field(
        "comment_id",
        "",
        "comment_id",
        "string, copied verbatim from the input",
        STRUCTURAL,
        aliases=("comment_id", "commentid", "id"),
        description="Original comment id, copied verbatim from the input.",
    ),
    Field(
        "is_institutional",
        False,
        "bool",
        "boolean",
        CORE,
        aliases=("is_institutional", "institutional"),
    ),
    Field(
        "frame_id",
        "",
        "frame",
        "one of {frames}",
        STRUCTURAL,
        aliases=("frame_id", "frame", "marco"),
    ),
    Field(
        "secondary_frame_id",
        "",
        "frame_or_empty",
        "one of {frames_or_empty}",
        OPTIONAL,
    ),
    Field(
        "thread_id",
        "",
        "thread",
        'a thread id of the chosen frame, "{emergent}", or "" when the frame is the no-frame id',
        CORE,
    ),
    Field(
        "emergent_thread_proposal",
        "",
        "text",
        'string; only when thread_id is "{emergent}", otherwise ""',
        OPTIONAL,
        requires=("thread_id",),
    ),
    Field(
        "stance",
        "",
        "stance",
        'one of {stances}, or "" for no-frame',
        CORE,
        aliases=("stance", "stance_id"),
    ),
    Field(
        "evidence_level",
        "",
        "evidence",
        'one of {evidence_levels}, or "" for no-frame',
        CORE,
    ),
    Field(
        "literal_excerpt",
        "",
        "text",
        'verbatim quote of at most 25 words, or "" for no-frame',
        OPTIONAL,
    ),
    Field(
        "rationale",
        "",
        "text",
        "at most 25 words, in the analysis language",
        OPTIONAL,
    ),
    Field(
        "confidence",
        0,
        "confidence",
        "integer: 3 unambiguous, 2 probable, 1 doubtful (1 forces needs_human_review true)",
        CORE,
    ),
    Field(
        "needs_human_review",
        False,
        "bool",
        "boolean",
        CORE,
    ),
    Field(
        "review_reason",
        "",
        "text",
        'string, e.g. irony, depends on parent comment, weak adjacent; "" if none',
        OPTIONAL,
        csv=False,
    ),
    Field(
        "actor_category",
        "",
        "actor",
        "one of {actors_or_empty}",
        OPTIONAL,
    ),
    Field(
        "actor_self_declared",
        False,
        "bool",
        "boolean",
        OPTIONAL,
        requires=("actor_category",),
    ),
    Field(
        "actor_evidence",
        "",
        "text",
        'verbatim excerpt that shows the role, or ""',
        OPTIONAL,
        requires=("actor_category",),
    ),
)

BY_NAME: dict[str, Field] = {f.name: f for f in FIELDS}
FIELD_NAMES: tuple[str, ...] = tuple(f.name for f in FIELDS)

#: Fields that cannot be dropped: the join key and the frame the analysis pivots on.
STRUCTURAL_FIELDS: tuple[str, ...] = ("comment_id", "frame_id")

#: CSV-column order. It is *not* the record order (it predates it and is part of
#: the published chatbot contract), hence its own tuple.
CSV_ORDER: tuple[str, ...] = (
    "comment_id",
    "is_institutional",
    "frame_id",
    "secondary_frame_id",
    "thread_id",
    "stance",
    "evidence_level",
    "confidence",
    "rationale",
    "literal_excerpt",
    "emergent_thread_proposal",
    "actor_category",
    "actor_self_declared",
    "actor_evidence",
    "needs_human_review",
)

#: Fields the *prompt* tells the model to leave empty on a no-frame record, in
#: record order (the sentence built in ``output_schema_text``).
PROMPT_NO_FRAME_BLANKS: tuple[str, ...] = (
    "secondary_frame_id",
    "thread_id",
    "stance",
    "evidence_level",
    "literal_excerpt",
)

#: Fields the *chatbot CSV instructions* tell the model to leave empty, in CSV
#: order (a different sentence, kept as it was).
CSV_NO_FRAME_BLANKS: tuple[str, ...] = (
    "stance",
    "evidence_level",
    "literal_excerpt",
    "secondary_frame_id",
)

if sorted(CSV_ORDER) != sorted(f.name for f in FIELDS if f.csv):  # pragma: no cover
    raise AssertionError("CSV_ORDER and the Field(csv=...) flags disagree")


def active(names) -> tuple[Field, ...]:
    """The selected fields, in the caller-supplied order (unknown names dropped)."""
    return tuple(BY_NAME[n] for n in names if n in BY_NAME)


def csv_fields(names) -> tuple[Field, ...]:
    """The selected CSV fields, in CSV order."""
    selected = set(names)
    return tuple(BY_NAME[n] for n in CSV_ORDER if n in selected)


def record_defaults(names) -> dict:
    """Defaults for every canonical field, whether selected or not.

    A field that is not asked for must still appear in the merged record (the
    analysis tables are built from a fixed-width frame). ``confidence`` is the one
    field where the "absent" default cannot be the generic one: 0 is not a legal
    confidence and would silently drop every record from the selected set, so an
    unasked confidence counts as 2 ("probable") instead.
    """
    selected = set(names)
    out = {f.name: f.default for f in FIELDS}
    if "confidence" not in selected:
        out["confidence"] = 2
    return out


def prose(names) -> str:
    """``a``, ``a and b``, ``a, b and c`` - the house style of the prompt sentences."""
    items = [str(n) for n in names]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def validate_selection(names) -> list[str]:
    """Problems with a study's ``output_fields``; an empty list means it is usable."""
    errors: list[str] = []
    seen: set[str] = set()
    duplicates: set[str] = set()
    for name in names:
        (duplicates if name in seen else seen).add(name)
    for name in sorted(duplicates):
        errors.append(f"output_fields: duplicate field {name!r}")
    for name in sorted(seen - set(BY_NAME)):
        errors.append(
            f"output_fields: unknown field {name!r} (valid fields: {', '.join(FIELD_NAMES)})"
        )
    for name in STRUCTURAL_FIELDS:
        if name not in seen:
            errors.append(
                f"output_fields: {name!r} is structural and cannot be dropped"
            )
    for spec in active(list(seen)):
        missing = [dep for dep in spec.requires if dep not in seen]
        if missing:
            errors.append(
                f"output_fields: {spec.name!r} only means something together with "
                + ", ".join(repr(dep) for dep in missing)
            )
    return errors


def is_default_selection(names) -> bool:
    """True when a selection is exactly the full default vocabulary, in order."""
    return tuple(names) == FIELD_NAMES


def normalize_help(overrides: dict, by_name: dict, selected) -> dict:
    """The help overrides that actually change rendered wording.

    An override for a field that is not selected, or one that merely restates the
    built-in text, has no effect on any prompt or schema. Forgetting those keeps a
    study's config hash equal to the same study without the keys, so a no-op edit
    never looks like a config change (which would force a re-classification).
    Unknown field names are kept on purpose: schema validation reports the typo.
    """
    selected = set(selected)
    kept = {}
    for name, value in overrides.items():
        spec = by_name.get(name)
        if spec is None or (name in selected and str(value) != spec.help):
            kept[name] = value
    return kept
