"""Small shared helpers (I/O, JSON parsing, pseudonyms)."""

from __future__ import annotations

import datetime as _dt
import glob
import json
import math
import re
from pathlib import Path

import pandas as pd

_LAZY_MODULES = {}


def _numpy():
    """Import numpy lazily so this module still loads when it is absent."""
    if "numpy" not in _LAZY_MODULES:
        try:
            import numpy
        except ImportError:  # pragma: no cover - numpy is a soft dependency here
            numpy = None
        _LAZY_MODULES["numpy"] = numpy
    return _LAZY_MODULES["numpy"]


def json_safe(obj):
    """Return *obj* reduced to strict RFC-8259 JSON-compatible values.

    Non-finite floats become ``None``, numpy/pandas scalars are converted to
    native Python types, datetimes to ISO-8601 strings and mapping keys to
    ``str``. Unrecognised objects are left untouched so ``json.dump`` raises
    instead of silently stringifying them.
    """
    return _sanitize(obj, _numpy())


def _sanitize(obj, np):
    if obj is None:
        return None

    if np is not None:
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            value = float(obj)
            return value if math.isfinite(value) else None
        if isinstance(obj, np.datetime64):
            return None if np.isnat(obj) else pd.Timestamp(obj).isoformat()
        if isinstance(obj, np.timedelta64):
            return None if np.isnat(obj) else str(obj)
        if isinstance(obj, np.ndarray):
            return [_sanitize(value, np) for value in obj.tolist()]
        if isinstance(obj, np.generic):
            return _sanitize(obj.item(), np)

    if isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None

    # pandas sentinels first: NaT subclasses datetime, so order matters.
    if obj is pd.NaT or obj is pd.NA:
        return None
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, _dt.datetime):
        return obj.isoformat()
    if isinstance(obj, _dt.date):
        return obj.isoformat()
    if isinstance(obj, _dt.time):
        return obj.isoformat()

    if isinstance(obj, dict):
        return {_json_key(key, np): _sanitize(value, np) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(value, np) for value in obj]
    if isinstance(obj, (set, frozenset)):
        try:
            items = sorted(obj)
        except TypeError:
            items = sorted(obj, key=lambda value: (type(value).__name__, repr(value)))
        return [_sanitize(value, np) for value in items]

    return obj


def _json_key(key, np):
    if isinstance(key, str):
        return key
    return str(_sanitize(key, np))


def read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(json_safe(obj), fh, ensure_ascii=False, indent=2, allow_nan=False)


def write_csv(df: pd.DataFrame, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def parse_json_text(text: str) -> list:
    """Parse a model response. Strips markdown fences; accepts an array, an object or JSON lines."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.I)
    t = re.sub(r"\s*```$", "", t)
    try:
        data = json.loads(t)
    except json.JSONDecodeError:
        data = [
            json.loads(line) for line in t.splitlines() if line.strip().startswith("{")
        ]
        if not data:
            raise
    return data if isinstance(data, list) else [data]


def expand_globs(patterns) -> list[str]:
    if isinstance(patterns, str):
        patterns = [patterns]
    files = sorted({f for p in patterns for f in glob.glob(p)})
    return files


def pseudonym_map(user_ids) -> dict:
    """Deterministic pseudonyms U001, U002... following the sorted order of user ids."""
    ids = sorted({str(u) for u in user_ids if pd.notna(u)})
    width = max(3, len(str(len(ids))))
    return {u: "U" + str(i + 1).zfill(width) for i, u in enumerate(ids)}


def as_bool(series: pd.Series) -> pd.Series:
    """Robust boolean parsing for CSV round trips ('True'/'False'/1/0/NaN)."""
    if series.dtype == bool:
        return series
    return series.map(
        lambda v: str(v).strip().lower() in ("true", "1", "1.0", "yes")
    ).astype(bool)
