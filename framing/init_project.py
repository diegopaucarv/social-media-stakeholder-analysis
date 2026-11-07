"""Scaffold a new, topic-neutral study folder and validate it immediately.

``scaffold`` writes a manifest plus the five module files every command expects
(``study``, ``sources``, ``actors``, ``locations``, ``codebook``) using neutral
placeholders, then runs the same ``load_study`` validation as ``validate-config``.
Nothing topic-specific is generated: the result is a starting point meant to be
edited by the researcher, not a finished study.
"""

from __future__ import annotations

from pathlib import Path

from .config import MODULES, Config, load_study
from .utils import write_json

# Module filenames are fixed by the study convention (`studies/<id>/<module>.json`).
MODULE_FILES = {name: f"{name}.json" for name in MODULES}

MANIFEST_FILE = "manifest.json"


def _study_module(study_id: str, title: str, language: str) -> dict:
    return {
        "id": study_id,
        "title": title,
        "objective": "To be defined: state what this study should extract from the comments.",
        "context": "To be defined: describe the corpus, the debate and the actors involved.",
        "corpus_language": language,
        "analysis_language": language,
        "period": {
            "cutoff_date": None,
            "before_label": "before",
            "after_label": "after",
        },
        "classification": {
            "batch_size": 30,
            "post_batch_size": 20,
            "min_confidence": 1,
            "parent_text_max_chars": 400,
        },
        "stats": {
            "focus_source_group": None,
            "bootstrap_iterations": 1000,
            "seed": 1,
            "gender_values": [],
        },
        "gap_checks": {
            "min_frame_comments": 30,
            "min_thread_comments": 5,
            "min_comments_per_year": 20,
            "expected_terms": [],
        },
    }


def _sources_module() -> dict:
    return {
        "source_types": [
            {"id": "general", "label": "General source", "stats_group": "general"}
        ],
        "registry": [
            {
                "key": "example_source",
                "kind": "page",
                "name": "Example source",
                "type": "general",
                "type_verified": False,
            }
        ],
        "institutional_authors": [],
    }


def _actors_module() -> dict:
    return {
        "levels": [
            {
                "id": "general",
                "label": "General level",
                "categories": [
                    {
                        "id": "general",
                        "label": "General participant",
                        "description": "To be defined: refine the actor categories for this study.",
                        "contact_candidate": False,
                    }
                ],
            }
        ]
    }


def _locations_module() -> dict:
    return {
        "gazetteer": [
            {
                "name": "Study Area",
                "level": "area",
                "aliases": [],
                "in_study_area": True,
            }
        ]
    }


def _codebook_module() -> dict:
    return {
        "no_frame": {
            "id": "NO_FRAME",
            "label": "No frame",
            "description": "Comments that do not sustain any frame (greetings, tags, unrelated topics).",
        },
        "frames": [
            {
                "id": "F1",
                "label": "To be defined",
                "argument": "To be defined: state the argument this frame captures.",
                "threads": [
                    {
                        "id": "T1",
                        "label": "To be defined",
                        "description": "To be defined: describe the sub-argument inside this frame.",
                    }
                ],
            }
        ],
        "stance_definitions": {
            "supports": "Sustains the thread.",
            "opposes": "Rebuts the thread.",
            "qualifies": "Partly agrees.",
        },
        "evidence_definitions": {
            "core": "States the target argument.",
            "adjacent": "Related but not the core argument.",
        },
        "decision_rules": [],
        "examples": [],
    }


def _modules(study_id: str, title: str, language: str) -> dict:
    return {
        "study": _study_module(study_id, title, language),
        "sources": _sources_module(),
        "actors": _actors_module(),
        "locations": _locations_module(),
        "codebook": _codebook_module(),
    }


def scaffold(
    study_id: str,
    title: str,
    language: str = "en",
    out_dir=None,
    force: bool = False,
) -> Config:
    """Write a neutral study folder and return its validated :class:`Config`.

    Args:
        study_id: Study id, also used as the ``study.id`` and default folder name.
        title: Human-readable study title.
        language: Corpus/analysis language code stored in the study module.
        out_dir: Folder to write into; defaults to ``studies/<study_id>``.
        force: Replace existing non-empty files instead of refusing.

    Raises:
        ValueError: The id/title is empty, or non-empty target files already exist
            and ``force`` is false.
        ConfigError: The generated folder does not pass the normal study validation.
    """
    study_id = str(study_id).strip()
    title = str(title).strip()
    if not study_id:
        raise ValueError("study id must not be empty")
    if not title:
        raise ValueError("study title must not be empty")

    out = Path(out_dir) if out_dir is not None else Path("studies") / study_id
    manifest_path = out / MANIFEST_FILE
    modules = _modules(study_id, title, language)
    manifest = {
        "schema_version": "1.0",
        "study_id": study_id,
        "modules": dict(MODULE_FILES),
    }

    targets = [manifest_path] + [out / name for name in MODULE_FILES.values()]
    existing = [p for p in targets if p.exists() and p.stat().st_size > 0]
    if existing and not force:
        raise ValueError(
            f"refusing to overwrite {existing[0]}: the study folder is not empty; "
            "pass --force to replace it"
        )

    write_json(manifest_path, manifest)
    for name, filename in MODULE_FILES.items():
        write_json(out / filename, modules[name])

    # Same validation path as `python -m framing validate-config`.
    return load_study(manifest_path)
