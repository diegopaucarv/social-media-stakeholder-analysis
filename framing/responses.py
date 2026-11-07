"""Ingestion of model response files for the classification step.

The orchestrator can collect responses from a raw JSON API or from a chatbot that
returns CSV. This module accepts both, detects the format from the file extension
and normalises every record with :func:`framing.schemas.normalise_classification`,
so the merge step sees exactly the record shape defined by
:data:`framing.schemas.RECORD_DEFAULTS` regardless of the source format.

Malformed rows never abort the load: they are coerced to a safe state by the
normaliser and reported back to the caller.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from . import fields, post_fields
from .progress import progress
from .schemas import RECORD_DEFAULTS, normalise_classification
from .utils import expand_globs, parse_json_text

if TYPE_CHECKING:  # pragma: no cover
    from .config import Config

__all__ = [
    "load_post_responses",
    "load_responses",
    "response_coverage",
    "validate_response_frame",
]


def post_columns(cfg) -> list[str]:
    """The post-answer CSV columns this study asks for, in the published CSV order."""
    return [spec.name for spec in post_fields.csv_fields(cfg.output_post_fields)]


# Canonical record field -> accepted CSV header names (matched case-insensitively
# after trimming whitespace).
#: Canonical record field -> accepted CSV header names, derived from the field
#: registry. Only fields that take part in the CSV contract are mapped here, so a
#: field such as ``review_reason`` stays JSON-only exactly as before.
_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    spec.name: tuple(spec.aliases) for spec in fields.csv_fields(fields.FIELD_NAMES)
}
#: The default post-answer columns, in output order (see ``framing.post_fields``).
POST_COLUMNS = [
    spec.name for spec in post_fields.csv_fields(post_fields.POST_FIELD_NAMES)
]
# Post-level answers: canonical field -> accepted header/JSON keys (case/space-insensitive).
# All fields are mapped, selected or not, so an answer produced before a study trimmed
# its vocabulary is still accepted and the extra keys are simply ignored.
_POST_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    spec.name: tuple(spec.aliases) for spec in post_fields.POST_FIELDS
}
MAX_ISSUE_EXAMPLES = 20
MAX_UNKNOWN_IDS = 20


def _text(value: Any) -> str:
    """Return a trimmed string for a scalar, mapping ``None``/``NaN`` to ``""``."""
    if value is None:
        return ""
    if not isinstance(value, str):
        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass
    return str(value).strip()


def _normalise_header(value: Any) -> str:
    """Normalise a CSV header or JSON key: lower-case, whitespace runs -> underscore."""
    return re.sub(r"\s+", "_", str(value).strip().lower())


def _detect_format(path: str) -> str:
    """Return ``"csv"``, ``"jsonl"`` or ``"json"`` from the file extension."""
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in (".jsonl", ".ndjson"):
        return "jsonl"
    return "json"


def _csv_column_map(columns: Iterable) -> dict[str, Any]:
    """Map canonical record fields to the actual CSV columns of a file.

    Header names are compared case-insensitively after trimming whitespace; the
    first column that matches a canonical field wins.
    """
    lookup = {
        alias: canonical
        for canonical, aliases in _COLUMN_ALIASES.items()
        for alias in aliases
    }
    mapped: dict[str, Any] = {}
    for column in columns:
        canonical = lookup.get(str(column).strip().lower())
        if canonical is None or canonical in mapped:
            continue
        mapped[canonical] = column
    return mapped


def _post_column_map(columns: Iterable) -> dict[str, Any]:
    """Map canonical post fields to the actual CSV columns of a post-answer file."""
    lookup = {
        alias: canonical
        for canonical, aliases in _POST_COLUMN_ALIASES.items()
        for alias in aliases
    }
    mapped: dict[str, Any] = {}
    for column in columns:
        canonical = lookup.get(_normalise_header(column))
        if canonical is None or canonical in mapped:
            continue
        mapped[canonical] = column
    return mapped


def _canonical_post_record(raw: dict) -> dict[str, Any]:
    """Re-key a post-answer object from its aliases to the canonical field names."""
    mapped = _post_column_map(raw.keys())
    return {canonical: raw[column] for canonical, column in mapped.items()}


def _read_post_csv(path: str) -> tuple[list[dict], list[str]]:
    """Read a post-answer CSV file. Returns ``(records, file_issues)``."""
    try:
        frame = pd.read_csv(
            path, encoding="utf-8-sig", dtype=str, keep_default_na=False
        )
    except (OSError, ValueError) as exc:
        return [], [f"could not read CSV: {exc}"]
    if frame.empty and len(frame.columns) == 0:
        return [], ["empty CSV file (no header row)"]
    mapped = _post_column_map(frame.columns)
    issues: list[str] = []
    if "post_key" not in mapped:
        issues.append(
            "CSV header has no post_key column (accepted: post_key, post_id, url, "
            "post_url, post_url_key)"
        )
    records = [
        {canonical: row[column] for canonical, column in mapped.items()}
        for _, row in frame.iterrows()
    ]
    return records, issues


def _read_post_json(path: str) -> tuple[list, list[str]]:
    """Read a post-answer JSON/JSON-lines file, unwrapping ``{"posts": [...]}``."""
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except OSError as exc:
        return [], [f"could not read JSON: {exc}"]
    try:
        data = parse_json_text(text)
    except (TypeError, ValueError) as exc:
        return [], [f"could not parse JSON: {exc}"]
    if (
        len(data) == 1
        and isinstance(data[0], dict)
        and isinstance(data[0].get("posts"), list)
    ):
        data = data[0]["posts"]
    return data, []


def _read_csv(path: str) -> tuple[list[dict], list[str]]:
    """Read a chatbot CSV file. Returns ``(records, file_issues)``.

    Values are read as strings with ``keep_default_na=False`` so that empty cells
    stay empty strings instead of becoming ``NaN``.
    """
    try:
        frame = pd.read_csv(
            path, encoding="utf-8-sig", dtype=str, keep_default_na=False
        )
    except (OSError, ValueError) as exc:
        return [], [f"could not read CSV: {exc}"]
    if frame.empty and len(frame.columns) == 0:
        return [], ["empty CSV file (no header row)"]
    mapped = _csv_column_map(frame.columns)
    issues: list[str] = []
    if "comment_id" not in mapped:
        issues.append(
            "CSV header has no comment_id column (accepted: comment_id, commentid, id)"
        )
    records = [
        {canonical: row[column] for canonical, column in mapped.items()}
        for _, row in frame.iterrows()
    ]
    return records, issues


def _read_json(path: str) -> tuple[list, list[str]]:
    """Read a JSON/JSON-lines response file using the merge-step parsing rules."""
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except OSError as exc:
        return [], [f"could not read JSON: {exc}"]
    try:
        data = parse_json_text(text)
    except (TypeError, ValueError) as exc:
        return [], [f"could not parse JSON: {exc}"]
    if (
        len(data) == 1
        and isinstance(data[0], dict)
        and isinstance(data[0].get("classifications"), list)
    ):
        data = data[0]["classifications"]
    return data, []


def _add_issue_example(
    report: dict, path: str, comment_id: str, issues: list[str]
) -> None:
    """Append one issue example to the report, up to the global cap."""
    if len(report["issue_examples"]) < MAX_ISSUE_EXAMPLES:
        report["issue_examples"].append(
            {"file": path, "comment_id": comment_id, "issues": issues}
        )


def _add_post_issue_example(
    report: dict, path: str, post_key: str, issues: list[str]
) -> None:
    """Append one post-answer issue example to the report, up to the global cap."""
    if len(report["issue_examples"]) < MAX_ISSUE_EXAMPLES:
        report["issue_examples"].append(
            {"file": path, "post_key": post_key, "issues": issues}
        )


def load_responses(patterns, cfg: Config) -> tuple[pd.DataFrame, dict]:
    """Load, normalise and de-duplicate model response files.

    Each file is parsed according to its extension: ``.csv`` files are treated as
    a chatbot export with a header row (column names are matched case-insensitively
    with whitespace trimmed, using the aliases in ``_COLUMN_ALIASES``); ``.json``
    and ``.jsonl`` files follow the same rules as
    :func:`framing.merge.load_classifications` (array, single object,
    ``{"classifications": [...]}`` or JSON lines).

    Every record is normalised with
    :func:`framing.schemas.normalise_classification`. Records that fail validation
    are coerced to a safe state and counted in the report instead of aborting the
    load. Response files are de-duplicated on ``comment_id`` keeping the last
    occurrence.

    Args:
        patterns: Glob pattern or iterable of glob patterns of response files.
        cfg: The study configuration used to normalise the records.

    Returns:
        A ``(responses, report)`` tuple. ``responses`` is a DataFrame with exactly
        the columns of :data:`framing.schemas.RECORD_DEFAULTS`. ``report`` has the
        keys ``files``, ``by_format`` (extension -> file count), ``records``,
        ``invalid_records``, ``dropped_without_id``, ``issue_examples`` (max 20)
        and ``duplicates``. Records with an empty ``comment_id`` are kept out of
        the returned frame: they cannot be joined to the corpus.

    Raises:
        FileNotFoundError: If no file matches ``patterns``.
    """
    files = expand_globs(patterns)
    if not files:
        raise FileNotFoundError(f"no response files match {patterns}")

    report: dict[str, Any] = {
        "files": len(files),
        "by_format": {},
        "records": 0,
        "invalid_records": 0,
        "issue_examples": [],
        "duplicates": 0,
    }
    clean: list[dict] = []
    for path in progress(files, total=len(files), desc="loading responses"):
        extension = Path(path).suffix.lower() or "<none>"
        report["by_format"][extension] = report["by_format"].get(extension, 0) + 1

        if _detect_format(path) == "csv":
            items, file_issues = _read_csv(path)
        else:
            items, file_issues = _read_json(path)

        if not items:
            # Unreadable/empty file: report it once instead of silently dropping it.
            if file_issues:
                report["invalid_records"] += 1
                _add_issue_example(report, path, "", file_issues)
            continue

        for raw in items:
            report["records"] += 1
            if not isinstance(raw, dict):
                report["invalid_records"] += 1
                _add_issue_example(
                    report, path, "", file_issues + ["record is not a JSON object"]
                )
                continue
            record, issues = normalise_classification(raw, cfg)
            issues = file_issues + issues
            if issues:
                report["invalid_records"] += 1
                _add_issue_example(report, path, record["comment_id"], issues)
            clean.append(record)

    frame = pd.DataFrame(clean, columns=list(RECORD_DEFAULTS))
    # Records without an id cannot be joined to the corpus and would otherwise
    # surface as one phantom unknown comment; keep them out of the usable frame.
    without_id = frame["comment_id"].astype(str).str.strip() == ""
    report["dropped_without_id"] = int(without_id.sum())
    frame = frame[~without_id].reset_index(drop=True)
    report["duplicates"] = int(frame.duplicated("comment_id").sum())
    frame = frame.drop_duplicates("comment_id", keep="last").reset_index(drop=True)
    return frame, report


def load_post_responses(patterns, cfg: Config) -> tuple[pd.DataFrame, dict]:
    """Load post-level answers (topic, primary frame, source type guess).

    Accepts CSV, JSON and JSON-lines files, including single objects wrapped as
    ``{"posts": [...]}``. Headers and JSON keys are matched case- and
    space-insensitively against ``_POST_COLUMN_ALIASES`` (for example ``Post Key``
    and ``post_key`` are equivalent).

    Rows without a ``post_key`` are dropped and counted as invalid; the rest is
    de-duplicated on ``post_key`` keeping the last occurrence. The result always
    has exactly the study's selected post columns (default: :data:`POST_COLUMNS`).

    Args:
        patterns: Glob pattern or iterable of glob patterns of post-answer files.
        cfg: The study configuration (kept for symmetry with :func:`load_responses`).

    Returns:
        A ``(posts, report)`` tuple. ``report`` has the keys ``files``,
        ``by_format`` (extension -> file count), ``records``, ``posts``,
        ``invalid_records`` and ``issue_examples`` (max 20).

    Raises:
        FileNotFoundError: If no file matches ``patterns``.
    """
    files = expand_globs(patterns)
    if not files:
        raise FileNotFoundError(f"no post response files match {patterns}")

    report: dict[str, Any] = {
        "files": len(files),
        "by_format": {},
        "records": 0,
        "posts": 0,
        "invalid_records": 0,
        "issue_examples": [],
    }
    columns = post_columns(cfg)
    rows: list[dict[str, str]] = []
    for path in progress(files, total=len(files), desc="loading post responses"):
        extension = Path(path).suffix.lower() or "<none>"
        report["by_format"][extension] = report["by_format"].get(extension, 0) + 1

        if _detect_format(path) == "csv":
            items, file_issues = _read_post_csv(path)
        else:
            items, file_issues = _read_post_json(path)

        if not items:
            if file_issues:
                report["invalid_records"] += 1
                _add_post_issue_example(report, path, "", file_issues)
            continue

        for raw in items:
            report["records"] += 1
            if not isinstance(raw, dict):
                report["invalid_records"] += 1
                _add_post_issue_example(
                    report, path, "", file_issues + ["record is not a JSON object"]
                )
                continue
            record = _canonical_post_record(raw)
            post_key = _text(record.get("post_key"))
            if not post_key:
                report["invalid_records"] += 1
                _add_post_issue_example(
                    report, path, "", file_issues + ["missing post_key"]
                )
                continue
            row = {"post_key": post_key}
            for column in columns[1:]:
                row[column] = _text(record.get(column))
            rows.append(row)

    frame = pd.DataFrame(rows, columns=columns)
    if not frame.empty:
        frame = frame.drop_duplicates("post_key", keep="last")
    frame = frame.fillna("").reset_index(drop=True)
    report["posts"] = int(len(frame))
    return frame, report


def response_coverage(responses: pd.DataFrame, comment_ids: Iterable) -> dict:
    """Measure how many valid corpus comments received a model response.

    ``matched`` counts the corpus ids that appear in the response frame,
    ``missing`` is the complement within the corpus and ``coverage`` is
    ``matched / total`` (``1.0`` for an empty corpus, mirroring ``merge``).
    ``unknown_ids`` lists response ids that are absent from the corpus (first 20).

    Args:
        responses: A response frame, normally the output of :func:`load_responses`.
        comment_ids: Iterable of valid comment ids from the prepared corpus.

    Returns:
        A dict with ``matched``, ``coverage``, ``missing`` and ``unknown_ids``.
    """
    corpus: list[str] = []
    corpus_set: set[str] = set()
    for value in comment_ids:
        cid = _text(value)
        if cid and cid not in corpus_set:
            corpus_set.add(cid)
            corpus.append(cid)

    responded: list[str] = []
    if "comment_id" in responses.columns:
        for value in responses["comment_id"]:
            cid = _text(value)
            if cid:
                responded.append(cid)

    matched = len(corpus_set & set(responded))
    total = len(corpus)
    unknown = [cid for cid in dict.fromkeys(responded) if cid not in corpus_set]
    return {
        "matched": matched,
        "coverage": 1.0 if total == 0 else matched / total,
        "missing": total - matched,
        "unknown_ids": unknown[:MAX_UNKNOWN_IDS],
    }


def validate_response_frame(responses: pd.DataFrame, cfg: Config) -> list[str]:
    """Check a response frame for structural problems.

    The frame is valid when it contains all
    :data:`framing.schemas.RECORD_DEFAULTS` columns, has at least one row, has a
    non-empty ``comment_id`` in at least one row and every ``frame_id`` is a
    configured frame id, the configured no-frame id or the study's
    :attr:`framing.config.Config.emergent_thread_id`.

    Args:
        responses: Response frame to validate.
        cfg: The study configuration providing the valid frame ids.

    Returns:
        Human-readable problem descriptions; an empty list means the frame is valid.
    """
    problems: list[str] = []

    missing = [column for column in RECORD_DEFAULTS if column not in responses.columns]
    if missing:
        problems.append("missing required columns: " + ", ".join(missing))

    if len(responses) == 0:
        problems.append("response frame has no rows")
    elif "comment_id" in responses.columns and not any(
        _text(value) for value in responses["comment_id"]
    ):
        problems.append("all comment_id values are empty")

    if "frame_id" in responses.columns:
        valid = {str(frame_id) for frame_id in cfg.all_frame_ids} | {
            cfg.emergent_thread_id
        }
        bad = sorted({_text(value) for value in responses["frame_id"]} - valid)
        if bad:
            problems.append(
                "invalid frame_id values: " + ", ".join(repr(value) for value in bad)
            )

    return problems
