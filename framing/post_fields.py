"""The answer-field vocabulary of the post classifier.

This is the single declaration of what the model is asked to return for a post:
the record defaults, the JSON response schema, the rendered prompt and the CSV
header the chatbot branch asks for all derive from it, exactly as
``framing.fields`` does for comments.

A study configures it in ``study.json``::

    {
      "output_post_fields": ["post_key", "topic", "primary_frame_id", "source_type_guess"],
      "output_post_field_help": {"topic": "short label, at most 8 words"}
    }

Only ``post_key`` is *structural*: it is the join key back to the input post, so it
can never be dropped. Every other field is optional.

The JSON surface asks for seven fields, while the chatbot CSV surface carries only
four; ``csv=False`` marks the JSON-only ones (``related_frame_ids``, ``rationale``
and ``confidence``), which therefore never appear in the CSV contract.

Dropping a field does not break the pipeline: the merged record keeps the field
with a safe default, the analysis tables keep the (empty) column and the analysis
falls back to its documented behaviour instead of inventing values. This mirrors
``framing.fields``.
"""

from __future__ import annotations

from .fields import CORE, OPTIONAL, STRUCTURAL, Field
from .fields import normalize_help as _normalize_help

#: Fields in record / JSON-schema / prompt order. This order is part of the
#: contract: it is the order of the rendered schema and of the prompt.
POST_FIELDS: tuple[Field, ...] = (
    Field(
        "post_key",
        "",
        "post_key_desc",
        "string, copied verbatim from the input",
        STRUCTURAL,
        aliases=(
            "post_key",
            "post_id",
            "url",
            "post_url",
            "post_url_key",
            "postkey",
            "posturl",
        ),
        description="Post key, copied verbatim from the input.",
    ),
    Field(
        "topic",
        "",
        "text_desc",
        "short label, at most 12 words",
        CORE,
        aliases=("topic", "post_topic", "title"),
        description="Short topic label (<= 12 words) in the analysis language.",
    ),
    Field(
        "primary_frame_id",
        "",
        "frame",
        "one of {frames}",
        CORE,
        aliases=("primary_frame_id", "post_frame_id", "frame_id", "frame"),
    ),
    Field(
        "related_frame_ids",
        (),
        "frame_list",
        "array with ids from {frames_list} (may be empty)",
        OPTIONAL,
        csv=False,
    ),
    Field(
        "source_type_guess",
        "",
        "source_type",
        "one of {source_types}",
        CORE,
        aliases=("source_type_guess", "source_type", "source_type_hint"),
    ),
    Field(
        "rationale",
        "",
        "text",
        "one sentence",
        OPTIONAL,
        csv=False,
    ),
    Field(
        "confidence",
        0,
        "confidence",
        "integer 1, 2 or 3",
        OPTIONAL,
        csv=False,
    ),
)

POST_BY_NAME: dict[str, Field] = {f.name: f for f in POST_FIELDS}
POST_FIELD_NAMES: tuple[str, ...] = tuple(f.name for f in POST_FIELDS)

#: Only the join key cannot be dropped.
STRUCTURAL_POST_FIELDS: tuple[str, ...] = ("post_key",)

#: CSV/chatbot column order; a subsequence of ``POST_FIELDS`` (unlike comments).
POST_CSV_ORDER: tuple[str, ...] = tuple(f.name for f in POST_FIELDS if f.csv)

if sorted(POST_CSV_ORDER) != sorted(
    f.name for f in POST_FIELDS if f.csv
):  # pragma: no cover
    raise AssertionError("POST_CSV_ORDER and the Field(csv=...) flags disagree")


def active(names) -> tuple[Field, ...]:
    """The selected post fields, in the caller-supplied order (unknown names dropped)."""
    return tuple(POST_BY_NAME[n] for n in names if n in POST_BY_NAME)


def csv_fields(names) -> tuple[Field, ...]:
    """The selected CSV post fields, in CSV order."""
    selected = set(names)
    return tuple(POST_BY_NAME[n] for n in POST_CSV_ORDER if n in selected)


def record_defaults(names) -> dict:
    """Defaults for every post field, whether selected or not.

    Unlike comments there is no ``confidence`` special case: the post confidence is
    JSON-only and never gates selection, so its generic default applies.
    """
    return {f.name: f.default for f in POST_FIELDS}


def validate_selection(names) -> list[str]:
    """Problems with a study's ``output_post_fields``; empty list means usable."""
    errors: list[str] = []
    seen: set[str] = set()
    duplicates: set[str] = set()
    for name in names:
        (duplicates if name in seen else seen).add(name)
    for name in sorted(duplicates):
        errors.append(f"output_post_fields: duplicate field {name!r}")
    for name in sorted(seen - set(POST_BY_NAME)):
        errors.append(
            f"output_post_fields: unknown field {name!r} (valid fields: {', '.join(POST_FIELD_NAMES)})"
        )
    for name in STRUCTURAL_POST_FIELDS:
        if name not in seen:
            errors.append(
                f"output_post_fields: {name!r} is structural and cannot be dropped"
            )
    for spec in active(list(seen)):
        missing = [dep for dep in spec.requires if dep not in seen]
        if missing:
            errors.append(
                f"output_post_fields: {spec.name!r} only means something together with "
                + ", ".join(repr(dep) for dep in missing)
            )
    return errors


def is_default_selection(names) -> bool:
    """True when a selection is exactly the full default vocabulary, in order."""
    return tuple(names) == POST_FIELD_NAMES


def normalize_help(overrides: dict, selected) -> dict:
    """Post-field counterpart of :func:`framing.fields.normalize_help`."""
    return _normalize_help(overrides, POST_BY_NAME, selected)
