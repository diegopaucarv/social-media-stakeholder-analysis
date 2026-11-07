"""Single source of truth for the ``is_institutional`` mask.

Institutional status is inferred from two independent signals:

``author``
    The comment's ``user_name`` appears in the study config
    (``sources.institutional_authors``, exposed as ``cfg.institutional``). This is
    deterministic and config-driven: ``prepare`` applies it before any model runs.
``model``
    The classifier set the per-record ``is_institutional`` boolean, usually because
    the author self-identifies in the text as an official body.

The two signals are combined with a logical OR. Either one is sufficient to keep a
comment out of the frame-coding universe, so the *union* -- never either field on
its own -- is the mask downstream code must read. OR rather than AND because the
signals are independent: the config list is authoritative when it matches (an
official account is institutional even if the model missed it), while the model can
recognise official voices the config never listed. A false positive merely removes
a comment from the statistics; a false negative understates the mask, inflates
``coverage`` and lets non-public voices bias the analysis.

``combine()`` is NaN-safe: unset or unmatched values (``None``, ``NaN``, ``pd.NA``)
count as ``False``.
"""

from __future__ import annotations

import pandas as pd

__all__ = ["SIGNALS", "combine", "describe"]

#: Names of the two contributing signals, in a stable order.
SIGNALS = ("author", "model")


def _is_true(value) -> bool:
    """Truthiness for a scalar cell, mapping every missing value to ``False``."""
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in ("true", "1", "1.0", "yes")


def _truthy(mask) -> pd.Series:
    """Coerce a mask-like object to a boolean Series (missing values become ``False``)."""
    if mask is None:
        return pd.Series(dtype=bool)
    series = mask if isinstance(mask, pd.Series) else pd.Series(mask)
    if series.dtype == bool:
        return series
    return series.map(_is_true).astype(bool)


def _aligned(mask, index) -> pd.Series:
    """Return ``mask`` as a boolean Series on ``index``, treating gaps as ``False``."""
    if mask is None:
        return pd.Series(False, index=index)
    if isinstance(mask, pd.Series):
        if not mask.index.equals(index):
            mask = mask.reindex(index)
        return _truthy(mask)
    return pd.Series(_is_true(mask), index=index)


def combine(author_mask=None, model_mask=None, index=None) -> pd.Series:
    """Return the boolean institutional mask: ``author_mask`` OR ``model_mask``.

    Either argument may be a boolean ``Series``, a scalar, or ``None`` (treated as
    an all-``False`` signal). Series are aligned on ``index`` when one is given,
    otherwise on the author mask's index (falling back to the model mask's index).
    Values that do not line up count as ``False``, so a model record missing for a
    comment never turns into a spurious flag.
    """
    author_is_series = isinstance(author_mask, pd.Series)
    model_is_series = isinstance(model_mask, pd.Series)
    if index is None:
        if author_is_series:
            index = author_mask.index
        elif model_is_series:
            index = model_mask.index
        else:
            index = pd.RangeIndex(0)
    author = _aligned(author_mask, index)
    model = _aligned(model_mask, index)
    return (author | model).astype(bool)


def describe(author_mask=None, model_mask=None, combined=None) -> dict[str, int]:
    """Count the truthy rows of each supplied mask, keyed by :data:`SIGNALS`.

    Omitted masks are left out, so the result is a small, JSON-friendly summary of
    how much each signal contributes.
    """
    counts: dict[str, int] = {}
    if author_mask is not None:
        counts["author"] = int(_truthy(author_mask).sum())
    if model_mask is not None:
        counts["model"] = int(_truthy(model_mask).sum())
    if combined is not None:
        counts["combined"] = int(_truthy(combined).sum())
    return counts
