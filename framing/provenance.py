"""Run provenance: record which study configuration produced a work folder, and warn when it drifts.

The stages are separate commands that share one `--work` folder, and no artifact binds that
folder to a study config. Editing a study and re-running `analyze` over an old folder would
otherwise mix two codebooks silently. This module is the single place that records the link
(`run.json`) and checks it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from . import files as F
from .utils import read_json, write_json

if (
    TYPE_CHECKING
):  # pragma: no cover - import for typing only, avoids a cycle at runtime
    from .config import Config


def write_run(cfg: "Config", work_dir, stage: str, **extra) -> dict:
    """Record (or update) `run.json` in `work_dir`.

    Stages are appended, never replaced, so the file carries the whole history of the folder."""
    path = Path(work_dir) / F.RUN
    run = read_json(path) if path.exists() else {}
    run.update({"study_id": cfg.study["id"], "config_hash": cfg.hash})
    stages = run.setdefault("stages", [])
    if stage not in stages:
        stages.append(stage)
    run.update(extra)
    write_json(path, run)
    return run


def config_warnings(cfg: "Config", work_dir) -> list[str]:
    """Reasons to distrust `work_dir` for the current study config (empty when it matches)."""
    path = Path(work_dir) / F.RUN
    if not path.exists():
        return [
            f"{path} not found: cannot verify that this work folder belongs to study "
            f"'{cfg.study['id']}' (config hash {cfg.hash})."
        ]
    recorded = read_json(path).get("config_hash")
    if recorded != cfg.hash:
        return [
            f"{path} was built with config hash {recorded} but the study now hashes to "
            f"{cfg.hash}; re-run `prepare` and `merge` into a fresh --work folder."
        ]
    return []


def warn_if_stale(cfg: "Config", work_dir) -> list[str]:
    """Print any provenance mismatch to stderr and return it.

    Analysis stages warn instead of aborting: the report is still useful, but the numbers may
    mix two codebooks and the user must know."""
    found = config_warnings(cfg, work_dir)
    for msg in found:
        print(f"WARNING: {msg}", file=sys.stderr)
    return found
