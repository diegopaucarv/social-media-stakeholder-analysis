"""Tests for the declarative answer-field vocabulary (``framing.fields``).

Two properties matter: the default study renders byte-identically to the old
hard-coded lists, and a trimmed field set stays consistent across the prompt, the
JSON schema, the CSV header and the merge step.
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

from framing import fields  # noqa: E402
from framing.config import ConfigError, load_study  # noqa: E402
from framing.export_md import RESPONSE_CSV_HEADER, response_csv_header  # noqa: E402
from framing.merge import merge  # noqa: E402
from framing.prepare import prepare  # noqa: E402
from framing.prompt_builder import classifier_system_prompt  # noqa: E402
from framing.schemas import (  # noqa: E402
    RECORD_DEFAULTS,
    classification_schema,
    normalise_classification,
)

from test_pipeline import fake_classification, make_toy_corpus  # noqa: E402

TOY = ROOT / "studies" / "toy_bikelanes" / "manifest.json"
ALL = list(fields.FIELD_NAMES)


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
    """The registry must reproduce the published contract exactly."""

    def test_registry_reproduces_the_published_contract(self):
        self.assertEqual(list(RECORD_DEFAULTS), ALL)
        self.assertEqual(RESPONSE_CSV_HEADER, list(fields.CSV_ORDER))
        self.assertNotIn("review_reason", RESPONSE_CSV_HEADER)
        self.assertEqual(fields.STRUCTURAL_FIELDS, ("comment_id", "frame_id"))

    def test_full_selection_renders_identically_to_the_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            explicit = load_study(variant_study(tmp, output_fields=ALL))
        default = load_study(TOY)
        self.assertEqual(
            classifier_system_prompt(explicit), classifier_system_prompt(default)
        )
        self.assertEqual(classification_schema(explicit), classification_schema(default))
        self.assertEqual(response_csv_header(explicit), response_csv_header(default))


class ValidationTests(unittest.TestCase):
    """A study cannot select a broken vocabulary."""

    def _errors(self, selection):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ConfigError) as ctx:
                load_study(variant_study(tmp, output_fields=selection))
        return str(ctx.exception)

    def test_structural_fields_cannot_be_dropped(self):
        self.assertIn("comment_id", self._errors(["frame_id", "stance"]))
        self.assertIn("frame_id", self._errors(["comment_id", "stance"]))

    def test_unknown_field_is_rejected(self):
        self.assertIn("sentiment", self._errors(["comment_id", "frame_id", "sentiment"]))

    def test_duplicate_field_is_rejected(self):
        self.assertIn("duplicate", self._errors(["comment_id", "frame_id", "frame_id"]))


class TrimmedStudyTests(unittest.TestCase):
    """A trimmed vocabulary must reach every consumer."""

    KEPT = [
        "comment_id",
        "frame_id",
        "thread_id",
        "stance",
        "confidence",
        "literal_excerpt",
    ]

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.cfg = load_study(variant_study(cls.tmp.name, output_fields=cls.KEPT))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_schema_and_csv_follow_the_selection(self):
        schema = classification_schema(self.cfg)
        self.assertEqual(list(schema["items"]["properties"]), self.KEPT)
        self.assertEqual(schema["items"]["required"], self.KEPT)
        self.assertEqual(
            response_csv_header(self.cfg),
            [name for name in fields.CSV_ORDER if name in self.KEPT],
        )

    def test_thread_enum_drops_emergent_when_the_proposal_is_not_asked_for(self):
        props = classification_schema(self.cfg)["items"]["properties"]
        self.assertNotIn("EMERGENT", props["thread_id"]["enum"])

    def test_prompt_lists_only_the_selected_fields(self):
        prompt = classifier_system_prompt(self.cfg)
        for name in self.KEPT:
            self.assertIn(f"- {name}:", prompt)
        for name in ("actor_category", "rationale", "needs_human_review", "review_reason"):
            self.assertNotIn(f"- {name}:", prompt)
        self.assertNotIn("Actor signals", prompt)

    def test_missing_optional_fields_do_not_raise_review_flags(self):
        record, issues = normalise_classification(
            {
                "comment_id": "a1",
                "frame_id": self.cfg.frame_ids[0],
                "thread_id": self.cfg.frames[0]["threads"][0]["id"],
                "stance": self.cfg.stance_ids[0],
                "confidence": 3,
            },
            self.cfg,
        )
        self.assertEqual(issues, [])
        self.assertFalse(record["needs_human_review"])
        self.assertEqual(record["actor_category"], "")

    def test_field_help_override_reaches_the_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_study(
                variant_study(
                    tmp,
                    output_fields=self.KEPT,
                    output_field_help={"stance": "CUSTOM STANCE WORDING"},
                )
            )
        self.assertIn("- stance: CUSTOM STANCE WORDING", classifier_system_prompt(cfg))


class TrimmedMergeTests(unittest.TestCase):
    """Trimming the vocabulary must not change the analysis outcome."""

    def _merge(self, changes, only_asked=False):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        cfg = load_study(variant_study(tmp, **changes))
        comments, posts = make_toy_corpus(tmp)
        work = tmp / "work"
        prepare(cfg, comments, work, posts)
        records = fake_classification(work, cfg)
        if only_asked:
            records = [
                {k: v for k, v in rec.items() if k in cfg.output_field_set}
                for rec in records
            ]
        answer = tmp / "answers.json"
        answer.write_text(json.dumps(records), encoding="utf-8")
        return merge(cfg, work, [str(answer)])

    def test_trimmed_study_merges_exactly_like_the_full_one(self):
        reference = self._merge({})
        kept = ["comment_id", "frame_id", "thread_id", "stance", "confidence"]
        trimmed = self._merge({"output_fields": kept}, only_asked=True)
        self.assertEqual(trimmed["selected"], reference["selected"])
        self.assertEqual(trimmed["invalid_records"], reference["invalid_records"])
        self.assertEqual(trimmed["coverage"], reference["coverage"])
        self.assertGreater(trimmed["selected"], 0)

    def test_dropping_confidence_keeps_records_selectable(self):
        kept = ["comment_id", "frame_id", "thread_id", "stance"]
        report = self._merge({"output_fields": kept}, only_asked=True)
        self.assertGreater(report["selected"], 0)
        self.assertEqual(
            sum(report["selected_by_frame"].values()), report["selected"]
        )

    def test_full_answers_merge_into_a_trimmed_study(self):
        kept = ["comment_id", "frame_id", "stance"]
        report = self._merge({"output_fields": kept})
        self.assertGreater(report["selected"], 0)
