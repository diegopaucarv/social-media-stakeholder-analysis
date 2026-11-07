"""Tests for the declarative post answer-field vocabulary (``framing.post_fields``).

Two properties matter: the default study renders byte-identically to the old
hard-coded post contract, and a trimmed post field set stays consistent across the
prompt, the JSON schema, the CSV header, the ingestion and the merge step.
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
from test_pipeline import fake_classification, make_toy_corpus  # noqa: E402

from framing import files as F  # noqa: E402
from framing import post_fields  # noqa: E402
from framing.api import _normalise_post_records  # noqa: E402
from framing.config import ConfigError, load_study  # noqa: E402
from framing.export_md import (  # noqa: E402
    POST_RESPONSE_CSV_HEADER,
    post_response_csv_header,
)
from framing.merge import merge  # noqa: E402
from framing.prepare import prepare  # noqa: E402
from framing.prompt_builder import post_classifier_system_prompt  # noqa: E402
from framing.responses import POST_COLUMNS, load_post_responses  # noqa: E402
from framing.schemas import post_classification_schema  # noqa: E402

TOY = ROOT / "studies" / "toy_bikelanes" / "manifest.json"
ALL = list(post_fields.POST_FIELD_NAMES)
CSV_DEFAULTS = ["post_key", "topic", "primary_frame_id", "source_type_guess"]
JSON_ONLY = ("related_frame_ids", "rationale", "confidence")


def variant_study(tmp, **changes):
    """Copy the toy study, override keys in study.json and return its manifest path."""
    dst = Path(tmp) / "variant"
    shutil.copytree(TOY.parent, dst)
    data = json.loads((dst / "study.json").read_text(encoding="utf-8"))
    data.update(changes)
    (dst / "study.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return dst / "manifest.json"


class ContractTests(unittest.TestCase):
    """The registry must reproduce the published post contract exactly."""

    def test_registry_reproduces_published_post_contract(self):
        self.assertEqual(
            list(post_fields.POST_FIELD_NAMES),
            [
                "post_key",
                "topic",
                "primary_frame_id",
                "related_frame_ids",
                "source_type_guess",
                "rationale",
                "confidence",
            ],
        )
        self.assertEqual(list(post_fields.POST_CSV_ORDER), CSV_DEFAULTS)
        self.assertEqual(POST_RESPONSE_CSV_HEADER, CSV_DEFAULTS)
        self.assertEqual(POST_COLUMNS, CSV_DEFAULTS)
        self.assertEqual(post_fields.STRUCTURAL_POST_FIELDS, ("post_key",))
        self.assertEqual(
            {
                name: post_fields.POST_BY_NAME[name].csv
                for name in post_fields.POST_FIELD_NAMES
            },
            {
                "post_key": True,
                "topic": True,
                "primary_frame_id": True,
                "related_frame_ids": False,
                "source_type_guess": True,
                "rationale": False,
                "confidence": False,
            },
        )

    def test_full_selection_renders_identically_to_default(self):
        default = load_study(TOY)
        with tempfile.TemporaryDirectory() as tmp:
            explicit = load_study(variant_study(tmp, output_post_fields=ALL))
        self.assertEqual(
            post_classifier_system_prompt(explicit),
            post_classifier_system_prompt(default),
        )
        self.assertEqual(
            post_classification_schema(explicit),
            post_classification_schema(default),
        )
        self.assertEqual(
            post_response_csv_header(explicit), post_response_csv_header(default)
        )

    def test_explicit_full_selection_keeps_config_hash(self):
        default = load_study(TOY)
        with tempfile.TemporaryDirectory() as tmp:
            explicit = load_study(variant_study(tmp, output_post_fields=ALL))
        self.assertEqual(explicit.hash, default.hash)

    def test_dropped_field_is_absent_from_csv_header_schema_and_prompt(self):
        kept = ["post_key", "primary_frame_id"]
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_study(variant_study(tmp, output_post_fields=kept))
        item = post_classification_schema(cfg)["items"]
        prompt = post_classifier_system_prompt(cfg)
        self.assertEqual(post_response_csv_header(cfg), kept)
        for name in (
            "topic",
            "source_type_guess",
            "related_frame_ids",
            "rationale",
            "confidence",
        ):
            self.assertNotIn(name, post_response_csv_header(cfg))
            self.assertNotIn(name, item["properties"])
            self.assertNotIn(name, item["required"])
            self.assertNotIn(f"- {name}:", prompt)
        for name in kept:
            self.assertIn(name, item["properties"])
            self.assertIn(name, item["required"])
            self.assertIn(f"- {name}:", prompt)

    def test_primary_frame_id_enum_is_all_frame_ids(self):
        default = load_study(TOY)
        with tempfile.TemporaryDirectory() as tmp:
            trimmed = load_study(
                variant_study(tmp, output_post_fields=["post_key", "primary_frame_id"])
            )
        for cfg in (default, trimmed):
            props = post_classification_schema(cfg)["items"]["properties"]
            self.assertEqual(props["primary_frame_id"]["enum"], cfg.all_frame_ids)

    def test_json_only_fields_never_reach_the_csv(self):
        default = load_study(TOY)
        props = post_classification_schema(default)["items"]["properties"]
        header = post_response_csv_header(default)
        for name in JSON_ONLY:
            self.assertIn(name, props)
            self.assertNotIn(name, POST_RESPONSE_CSV_HEADER)
            self.assertNotIn(name, header)
        self.assertEqual(header, CSV_DEFAULTS)

    def test_post_field_help_override_reaches_the_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_study(
                variant_study(
                    tmp,
                    output_post_field_help={
                        "primary_frame_id": "CUSTOM POST FRAME WORDING"
                    },
                )
            )
        self.assertIn(
            "- primary_frame_id: CUSTOM POST FRAME WORDING",
            post_classifier_system_prompt(cfg),
        )


class ValidationTests(unittest.TestCase):
    """A study cannot select a broken post vocabulary."""

    def _errors(self, selection):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ConfigError) as ctx:
                load_study(variant_study(tmp, output_post_fields=selection))
        return str(ctx.exception)

    def test_structural_field_cannot_be_dropped(self):
        self.assertIn("post_key", self._errors(["topic", "primary_frame_id"]))

    def test_empty_selection_is_rejected(self):
        # The study schema requires at least one field, so an empty list is rejected
        # before the structural check can explain why post_key is mandatory.
        self.assertIn("non-empty", self._errors([]))

    def test_unknown_post_field_is_rejected(self):
        self.assertIn("sentiment", self._errors(["post_key", "topic", "sentiment"]))

    def test_duplicate_field_is_rejected(self):
        self.assertIn("duplicate", self._errors(["post_key", "topic", "topic"]))


class IngestionTests(unittest.TestCase):
    """The ingestion step follows the selected vocabulary, not stale CSV columns."""

    def test_ingestion_accepts_and_ignores_stale_extra_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_study(
                variant_study(tmp, output_post_fields=["post_key", "primary_frame_id"])
            )
            path = Path(tmp) / "posts.csv"
            path.write_text(
                "post_key,post_topic,source_type\nP1,Stale topic,forum\n",
                encoding="utf-8",
            )
            frame, report = load_post_responses([str(path)], cfg)
        self.assertEqual(list(frame.columns), ["post_key", "primary_frame_id"])
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame["primary_frame_id"].tolist(), [""])
        self.assertEqual(report["invalid_records"], 0)

    def test_api_normaliser_emits_only_selected_fields(self):
        record = {
            "postKey": "K1",
            "topic": "T",
            "primary_frame_id": "A",
            "source_type_guess": "forum",
        }
        default = load_study(TOY)
        self.assertEqual(
            _normalise_post_records([record], default),
            [
                {
                    "post_key": "K1",
                    "topic": "T",
                    "primary_frame_id": "A",
                    "source_type_guess": "forum",
                }
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            trimmed = load_study(variant_study(tmp, output_post_fields=["post_key"]))
        self.assertEqual(
            _normalise_post_records([record], trimmed), [{"post_key": "K1"}]
        )


class TrimmedMergeTests(unittest.TestCase):
    """Dropping a post field must not break the merge or invent values."""

    def test_dropping_topic_keeps_post_topic_empty_without_raising(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        cfg = load_study(
            variant_study(tmp, output_post_fields=["post_key", "primary_frame_id"])
        )
        comments, posts = make_toy_corpus(tmp)
        work = tmp / "work"
        prepare(cfg, comments, work, posts)

        answer = tmp / "answers.json"
        answer.write_text(json.dumps(fake_classification(work, cfg)), encoding="utf-8")
        keys = pd.read_csv(work / F.POSTS_TABLE, dtype=str)["post_key"].tolist()
        frame_id = cfg.frame_ids[0]
        post_answer = tmp / "post_answers.csv"
        post_answer.write_text(
            "post_key,primary_frame_id\n"
            + "".join(f"{key},{frame_id}\n" for key in keys),
            encoding="utf-8",
        )

        out = tmp / "out"
        report = merge(
            cfg,
            work,
            [str(answer)],
            out_dir=out,
            post_responses=[str(post_answer)],
        )

        table = pd.read_csv(out / F.POSTS_TABLE, dtype=str).fillna("")
        self.assertEqual(table["post_topic"].tolist(), [""] * len(keys))
        self.assertEqual(
            table["post_frame_id"].tolist(), [cfg.frame_ids[0]] * len(keys)
        )
        self.assertGreater(report["posts_classified_by_model"], 0)
