"""Study configuration: loads the module files listed in a manifest and unifies them into one object.

A study is a folder of JSON modules (study, sources, actors, locations, codebook) plus a manifest.
Every topic-specific string, id or threshold lives in those modules; the code never hard-codes them.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

from . import fields, jsonschema_lite, post_fields
from .schemas import EMERGENT, EVIDENCE, STANCES
from .utils import read_json, write_json

PKG_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_FILE = PKG_ROOT / "schemas" / "modules.schema.json"
MODULES = ("study", "sources", "actors", "locations", "codebook")

# Column mapping of the default scraper (Apify "facebook-comments-scraper" / "facebook-posts-scraper").
# Keys are canonical names used everywhere in the pipeline; values are raw column names in the CSV.
DEFAULT_INPUT_SCHEMA = {
    "comments": {
        "comment_id": "id",
        "post_url": "inputUrl",
        "post_numeric_id": "facebookId",
        "post_title": "postTitle",
        "timestamp": "date",
        "thread_depth": "threadingDepth",
        "likes": "likesCount",
        "reply_count": "commentsCount",
        "reply_to_id": "replyToCommentId",
        "user_id": "author/id",
        "user_name": "author/name",
        "user_gender": "author/gender",
        "user_verified": "author/is_verified",
        "user_url": "author/url",
        "text": "text",
        "comment_url": "commentUrl",
        "attachment_type": "attachments/0/__typename",
        "parent_author": "parentComment/author/name",
        "parent_author_alt": "parentReply/author/name",
    },
    "posts": {
        "post_id": "postId",
        "url": "url",
        "text": "text",
        "timestamp": "time",
        "author": "user/name",
        "page_name": "pageName",
        "shares": "sharesCount",
        "reactions": "topReactionsCount",
        "error": "error",
        "text_fallbacks": ["creation_story/message/text", "sharedPost/text"],
    },
}
REQUIRED_COMMENT_COLUMNS = ("comment_id", "post_url", "timestamp", "text")

DEFAULTS = {
    # Neutral, topic-free fallbacks: a study that omits a key gets a documented baseline and
    # must opt in to anything study-specific. `relevance` therefore uses equal weights and the
    # minimal distinct tiers (>= 2 high, >= 1 medium) instead of any one study's calibration,
    # and `stats.gender_values` is empty so gender tests are only produced when a study lists
    # the values its own corpus carries.
    "relevance": {
        "likes_weight": 1,
        "reply_weight": 1,
        "high_threshold": 2,
        "medium_threshold": 1,
    },
    "period": {"cutoff_date": None, "before_label": "before", "after_label": "after"},
    "classification": {
        "batch_size": 30,
        "post_batch_size": 20,
        "min_confidence": 1,
        "parent_text_max_chars": 400,
    },
    "contacts": {"min_comments_for_activity_candidate": 5},
    "stats": {
        "focus_source_group": None,
        "min_expected_warning": 5,
        "bootstrap_iterations": 1000,
        "seed": 1,
        "gender_values": [],
        "tests": None,
    },
    "gap_checks": {
        "min_frame_comments": 30,
        "min_thread_comments": 5,
        "min_posts_for_pooled_tests": 30,
        "min_comments_per_year": 20,
        "expected_terms": [],
        "join_warning_share": 0.05,
        "sparse_frame_comments": 5,
        "severe_frame_comments": 10,
    },
}


class ConfigError(Exception):
    """Raised when a study configuration is structurally or semantically invalid."""


def schema_errors(instance, def_name: str, force_lite: bool = False) -> list[str]:
    schema = read_json(SCHEMA_FILE)
    wrapper = {
        "$schema": schema["$schema"],
        "$defs": schema["$defs"],
        "$ref": f"#/$defs/{def_name}",
    }
    if not force_lite:
        try:
            import jsonschema  # type: ignore

            validator = jsonschema.Draft202012Validator(wrapper)
            return [
                f"{'.'.join(str(p) for p in e.absolute_path) or '$'}: {e.message}"
                for e in validator.iter_errors(instance)
            ]
        except ImportError:
            pass
    return jsonschema_lite.validate(instance, wrapper)


class Config:
    """Unified, validated view of all study modules."""

    def __init__(
        self, modules: dict, manifest: dict | None = None, base_dir: Path | None = None
    ):
        self.manifest = manifest or {}
        self.base_dir = base_dir
        self.modules = copy.deepcopy(modules)
        study = copy.deepcopy(modules["study"])
        for key, default in DEFAULTS.items():
            merged = copy.deepcopy(default)
            merged.update(study.get(key, {}))
            study[key] = merged
        schema = copy.deepcopy(DEFAULT_INPUT_SCHEMA)
        for section, mapping in (study.get("input_schema") or {}).items():
            schema.setdefault(section, {}).update(mapping)
        study["input_schema"] = schema
        self.study = study
        self.sources = modules["sources"]
        self.actors = modules["actors"]
        self.locations = modules["locations"]
        self.codebook = modules["codebook"]

        # ---- answer-field vocabulary (framing.fields): what the model is asked for
        # Resolved outside the `self.study` defaults on purpose: a study that does not
        # mention output_fields must keep the exact config hash it had before.
        selected = self.study.get("output_fields")
        self.output_fields = (
            tuple(selected) if selected is not None else fields.FIELD_NAMES
        )
        self.output_field_set = frozenset(self.output_fields)
        self.field_specs = fields.active(self.output_fields)
        overrides = self.study.get("output_field_help") or {}
        self.field_help = {
            spec.name: str(overrides.get(spec.name, spec.help))
            for spec in self.field_specs
        }
        self.record_defaults = fields.record_defaults(self.output_fields)

        # A study that spells out the default vocabulary, or wording identical to the
        # built-in text, must hash like a study that omits the keys: otherwise a no-op
        # edit looks like a config change and would force a pointless re-classification.
        if fields.is_default_selection(self.output_fields):
            self.study.pop("output_fields", None)
        kept = fields.normalize_help(overrides, fields.BY_NAME, self.output_field_set)
        if kept:
            self.study["output_field_help"] = kept
        else:
            self.study.pop("output_field_help", None)

        # ---- post answer vocabulary (framing.post_fields)
        pselected = self.study.get("output_post_fields")
        self.output_post_fields = (
            tuple(pselected) if pselected is not None else post_fields.POST_FIELD_NAMES
        )
        self.output_post_field_set = frozenset(self.output_post_fields)
        self.post_field_specs = post_fields.active(self.output_post_fields)
        poverrides = self.study.get("output_post_field_help") or {}
        self.post_field_help = {
            spec.name: str(poverrides.get(spec.name, spec.help))
            for spec in self.post_field_specs
        }
        self.post_record_defaults = post_fields.record_defaults(self.output_post_fields)
        if post_fields.is_default_selection(self.output_post_fields):
            self.study.pop("output_post_fields", None)
        pkept = post_fields.normalize_help(poverrides, self.output_post_field_set)
        if pkept:
            self.study["output_post_field_help"] = pkept
        else:
            self.study.pop("output_post_field_help", None)

        # ---- codebook lookups
        self.no_frame_id = self.codebook["no_frame"]["id"]
        self.frames = self.codebook["frames"]
        self.frame_ids = [f["id"] for f in self.frames]
        self.all_frame_ids = self.frame_ids + [self.no_frame_id]
        self.frame_label = {f["id"]: f["label"] for f in self.frames}
        self.frame_label[self.no_frame_id] = self.codebook["no_frame"]["label"]
        self.thread_label, self.thread_frame = {}, {}
        for f in self.frames:
            for t in f["threads"]:
                self.thread_label[t["id"]] = t["label"]
                self.thread_frame[t["id"]] = f["id"]
        self.thread_ids = list(self.thread_label)
        # The codebook is the authority on the vocabulary and on its order: the prompt,
        # the JSON schema and the record checks render the ids in declaration order, and
        # the *last* evidence level is the weakest one in the gap report.
        self.stance_ids = list(self.codebook.get("stance_definitions") or {}) or list(
            STANCES
        )
        self.evidence_ids = list(
            self.codebook.get("evidence_definitions") or {}
        ) or list(EVIDENCE)
        # ---- classification protocol sentinel (thread proposed by the model)
        self.emergent_thread_id = str(
            self.study["classification"].get("emergent_thread_id") or EMERGENT
        )
        # Restating the default must not change the hash: otherwise an explicit but
        # identical key would look like a config change and force a re-classification.
        if self.emergent_thread_id == EMERGENT:
            self.study["classification"].pop("emergent_thread_id", None)
        # ---- actors
        self.actor_categories = {}
        for lvl in self.actors["levels"]:
            for cat in lvl["categories"]:
                self.actor_categories[cat["id"]] = {
                    "label": cat["label"],
                    "level_id": lvl["id"],
                    "level_label": lvl["label"],
                    "description": cat["description"],
                    "contact_candidate": bool(cat.get("contact_candidate", False)),
                    "coverage_check": bool(lvl.get("coverage_check", True)),
                }
        self.actor_ids = list(self.actor_categories)
        # ---- sources
        self.source_types = {s["id"]: s for s in self.sources["source_types"]}
        self.source_type_ids = list(self.source_types)
        self.registry = sorted(
            self.sources["registry"], key=lambda e: len(e["key"]), reverse=True
        )
        self.institutional = {
            a["name"]: a["actor_category"]
            for a in self.sources["institutional_authors"]
        }
        # ---- misc
        self.gazetteer = self.locations["gazetteer"]
        self.indicators = (self.locations.get("community_relevance") or {}).get(
            "indicators", []
        )
        self.deliverables = study.get("deliverables", [])
        self.hash = hashlib.sha256(
            json.dumps(self.unified(), sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()[:12]

    # ------------------------------------------------------------------ helpers
    def broken_indicator_refs(self) -> list[str]:
        out = []
        for ind in self.indicators:
            out += [
                f"{ind['id']}: thread {t}"
                for t in ind.get("thread_ids", [])
                if t not in self.thread_label
            ]
            out += [
                f"{ind['id']}: frame {f}"
                for f in ind.get("frame_ids", [])
                if f not in self.frame_ids
            ]
        return out

    def unified(self) -> dict:
        """One dict with every module (post-defaults for study); useful for logging and prompts."""
        return {
            "study": self.study,
            "sources": self.sources,
            "actors": self.actors,
            "locations": self.locations,
            "codebook": self.codebook,
        }

    def source_type_group(self, type_id: str) -> str:
        return self.source_types.get(type_id, {}).get("stats_group", "other")

    def actor_level_id(self, category_id: str) -> str:
        return self.actor_categories.get(category_id, {}).get("level_id", "")

    def save_unified(self, path) -> None:
        write_json(path, {"config_hash": self.hash, **self.unified()})

    # ------------------------------------------------------------------ validation
    def semantic_errors(self, indicator_refs: bool = True) -> list[str]:
        from .schemas import (  # lazy: avoids import cycle
            check_classification,
            classification_item_schema,
        )

        errs: list[str] = []
        errs += fields.validate_selection(self.output_fields)
        errs += post_fields.validate_selection(self.output_post_fields)

        def dupes(values, what):
            seen, bad = set(), set()
            for v in values:
                (bad if v in seen else seen).add(v)
            return [f"duplicate {what}: {v!r}" for v in sorted(bad)]

        errs += dupes([f["id"] for f in self.frames], "frame id")
        errs += dupes([t["id"] for f in self.frames for t in f["threads"]], "thread id")
        errs += dupes(
            [
                c
                for lvl in self.actors["levels"]
                for c in [x["id"] for x in lvl["categories"]]
            ],
            "actor category id",
        )
        errs += dupes([s["id"] for s in self.sources["source_types"]], "source type id")
        errs += dupes(
            [e["key"].lower() for e in self.sources["registry"]], "registry key"
        )
        errs += dupes([g["name"] for g in self.gazetteer], "gazetteer name")
        errs += dupes([d["id"] for d in self.deliverables], "deliverable id")
        errs += dupes([i["id"] for i in self.indicators], "indicator id")
        if self.no_frame_id in self.frame_ids:
            errs.append(f"no_frame id {self.no_frame_id!r} collides with a frame id")
        if (
            self.emergent_thread_id in self.thread_ids
            or self.emergent_thread_id in self.frame_ids
        ):
            errs.append(f"{self.emergent_thread_id!r} is a reserved id")
        if self.emergent_thread_id == self.no_frame_id:
            errs.append(
                "classification.emergent_thread_id "
                f"{self.emergent_thread_id!r} collides with the no_frame id"
            )
        rel = self.study["relevance"]
        if rel["medium_threshold"] > rel["high_threshold"]:
            errs.append("relevance.medium_threshold must be <= high_threshold")
        for e in self.sources["registry"]:
            if e["type"] not in self.source_types:
                errs.append(
                    f"registry entry {e['key']!r} has unknown source type {e['type']!r}"
                )
        for a in self.sources["institutional_authors"]:
            if a["actor_category"] not in self.actor_categories:
                errs.append(
                    f"institutional author {a['name']!r}: unknown actor category {a['actor_category']!r}"
                )
        for ind in self.indicators:
            for a in ind.get("actor_categories", []):
                if a not in self.actor_categories:
                    errs.append(
                        f"indicator {ind['id']!r}: unknown actor category {a!r}"
                    )
            if not indicator_refs:  # discovery rounds: the codebook is still changing, the researcher updates indicators afterwards
                continue
            for t in ind.get("thread_ids", []):
                if t not in self.thread_label:
                    errs.append(f"indicator {ind['id']!r}: unknown thread {t!r}")
            for f in ind.get("frame_ids", []):
                if f not in self.frame_ids:
                    errs.append(f"indicator {ind['id']!r}: unknown frame {f!r}")
        focus = self.study["stats"].get("focus_source_group")
        if focus and focus not in {
            s["stats_group"] for s in self.sources["source_types"]
        }:
            errs.append(
                f"stats.focus_source_group {focus!r} is not a stats_group of any source type"
            )
        for pat in [t["pattern"] for t in self.study["gap_checks"]["expected_terms"]]:
            try:
                re.compile(pat)
            except re.error as exc:
                errs.append(
                    f"invalid regex in gap_checks.expected_terms: {pat!r} ({exc})"
                )
        item_schema = classification_item_schema(self)
        for i, ex in enumerate(self.codebook["examples"]):
            out = ex["output"]
            if out.get("comment_id") != ex["input"]["comment_id"]:
                errs.append(
                    f"example {i}: output.comment_id differs from input.comment_id"
                )
            # Fields the study no longer asks for are outside the contract, so they
            # are ignored here instead of forcing every stored example to be edited.
            graded = {
                k: v
                for k, v in out.items()
                if k in self.output_field_set or k not in fields.BY_NAME
            }
            wrapper = {"$defs": {"item": item_schema}, "$ref": "#/$defs/item"}
            errs += [
                f"example {i}: {m}" for m in jsonschema_lite.validate(graded, wrapper)
            ]
            errs += [f"example {i}: {m}" for m in check_classification(graded, self)]
        return errs


def load_modules(manifest_path) -> tuple[dict, dict, Path]:
    """Read the manifest and its module files WITHOUT validation (used by tools that replace one module)."""
    manifest_path = Path(manifest_path)
    manifest = read_json(manifest_path)
    base = manifest_path.parent
    return (
        {name: read_json(base / manifest["modules"][name]) for name in MODULES},
        manifest,
        base,
    )


def config_from_modules(
    modules: dict,
    manifest: dict | None = None,
    base_dir: Path | None = None,
    indicator_refs: bool = True,
) -> Config:
    """Validate (schema + semantics) and unify an in-memory set of modules."""
    errs = []
    for name in MODULES:
        errs += [f"{name}: {m}" for m in schema_errors(modules[name], name)]
    if errs:
        raise ConfigError("\n".join(errs))
    cfg = Config(modules, manifest, base_dir)
    errs = cfg.semantic_errors(indicator_refs)
    if errs:
        raise ConfigError("\n".join(errs))
    return cfg


def load_study(manifest_path, validate: bool = True) -> Config:
    """Load and unify the modules listed in `manifest_path`."""
    manifest_path = Path(manifest_path)
    manifest = read_json(manifest_path)
    errs = [f"manifest: {m}" for m in schema_errors(manifest, "manifest")]
    if errs:
        raise ConfigError("\n".join(errs))
    base = manifest_path.parent
    modules = {}
    for name in MODULES:
        path = base / manifest["modules"][name]
        if not path.exists():
            errs.append(f"{name}: file not found: {path}")
            continue
        modules[name] = read_json(path)
        errs += [f"{name}: {m}" for m in schema_errors(modules[name], name)]
    if errs:
        raise ConfigError("\n".join(errs))
    cfg = Config(modules, manifest, base)
    if validate:
        errs = cfg.semantic_errors()
        if errs:
            raise ConfigError("\n".join(errs))
    return cfg
