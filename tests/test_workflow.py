"""Workflow-contract tests: response ingestion, network export, markdown bundle, API gating, progress.

Run:  python -m unittest tests.test_workflow -v   (from the package root).
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# Allow `python -m unittest tests.test_workflow` (repo root on sys.path) to reuse
# the sibling fixtures module, not only discovery from inside `tests/`.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_pipeline as T  # noqa: E402

from framing import api  # noqa: E402
from framing import files as F  # noqa: E402
from framing.config import load_study  # noqa: E402
from framing.export_md import (  # noqa: E402
    POST_RESPONSE_CSV_HEADER,
    RESPONSE_CSV_HEADER,
    build_classification_markdown,
)
from framing.merge import merge  # noqa: E402
from framing.network_export import (  # noqa: E402
    NETWORK_COLUMNS,
    export_network,
    stance_group,
)
from framing.prepare import prepare  # noqa: E402
from framing.progress import done, next_steps, progress  # noqa: E402
from framing.responses import (  # noqa: E402
    POST_COLUMNS,
    load_post_responses,
    load_responses,
)
from framing.schemas import RECORD_DEFAULTS  # noqa: E402

TOY = ROOT / "studies" / "toy_bikelanes" / "manifest.json"
PUBLISHED_AT = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
STANCE_GROUPS = {"support", "against", "qualify"}


def _make_temp_dir(case: unittest.TestCase) -> Path:
    """Create a temp dir that is removed when the test (or class) finishes."""
    path = Path(tempfile.mkdtemp())
    case.addCleanup(shutil.rmtree, path, ignore_errors=True)
    return path


def _class_temp_dir(case: type[unittest.TestCase]) -> Path:
    """Create a temp dir that is removed when the test class finishes."""
    path = Path(tempfile.mkdtemp())
    case.addClassCleanup(shutil.rmtree, path, ignore_errors=True)
    return path


class ResponseCSVTests(unittest.TestCase):
    """The chatbot comment-answer CSV round trip."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = load_study(TOY)

    def setUp(self):
        self.tmp = _make_temp_dir(self)

    def _write_csv(self, rows: list[dict], name: str = "responses.csv") -> Path:
        path = self.tmp / name
        pd.DataFrame(rows).reindex(columns=RESPONSE_CSV_HEADER).to_csv(
            path, index=False, encoding="utf-8"
        )
        return path

    def test_round_trip_dedup_and_quoted_rationale(self):
        frame_a = self.cfg.frames[0]["id"]
        thread_a1 = self.cfg.frames[0]["threads"][0]["id"]
        path = self._write_csv(
            [
                {
                    "comment_id": "a1",
                    "frame_id": frame_a,
                    "thread_id": thread_a1,
                    "stance": "supports",
                    "evidence_level": "core",
                    "confidence": 3,
                    "rationale": "first, wrong",
                },
                # all optional fields empty: survives as a no-frame record
                {"comment_id": "a2", "confidence": 2},
                # duplicate id, written last: must win over the first a1 row
                {"comment_id": "a1", "confidence": 2, "rationale": "second, kept"},
            ]
        )
        raw = path.read_text(encoding="utf-8")
        self.assertIn('"second, kept"', raw)  # the fixture really quotes the comma

        frame, report = load_responses([str(path)], self.cfg)

        self.assertEqual(list(frame.columns), list(RECORD_DEFAULTS))
        self.assertEqual(len(frame), 2)
        self.assertEqual(report["duplicates"], 1)
        self.assertEqual(report["invalid_records"], 0)
        by_id = frame.set_index("comment_id")
        self.assertEqual(by_id.loc["a1", "rationale"], "second, kept")
        self.assertEqual(by_id.loc["a2", "frame_id"], self.cfg.no_frame_id)

    def test_missing_comment_id_header_never_raises_and_has_no_usable_rows(self):
        path = self.tmp / "no_comment_id.csv"
        path.write_text(
            f"frame_id,stance\n{self.cfg.frames[0]['id']},supports\n", encoding="utf-8"
        )

        frame, report = load_responses([str(path)], self.cfg)

        # The row is kept in a coerced state but is unidentifiable, so no row is
        # usable by merge; the header problem is counted as invalid instead.
        usable = frame["comment_id"].astype(str).str.strip()
        self.assertEqual(int((usable != "").sum()), 0)
        self.assertGreater(report["invalid_records"], 0)
        self.assertIn(
            "comment_id",
            " ".join(
                issue
                for example in report["issue_examples"]
                for issue in example["issues"]
            ),
        )


class PostResponseTests(unittest.TestCase):
    """Post-level answers: aliases, JSON wrapper, blank ids, de-duplication."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = load_study(TOY)

    def setUp(self):
        self.tmp = _make_temp_dir(self)

    def test_aliased_headers_are_normalised_to_canonical_columns(self):
        path = self.tmp / "posts.csv"
        pd.DataFrame(
            [
                {
                    "post_url": "https://example.test/posts/1",
                    "post_topic": "Protected lanes",
                    "post_frame_id": "A",
                }
            ]
        ).to_csv(path, index=False, encoding="utf-8")

        frame, report = load_post_responses([str(path)], self.cfg)

        self.assertEqual(list(frame.columns), POST_COLUMNS)
        self.assertEqual(len(frame), 1)
        row = frame.iloc[0]
        self.assertEqual(row["post_key"], "https://example.test/posts/1")
        self.assertEqual(row["topic"], "Protected lanes")
        self.assertEqual(row["primary_frame_id"], "A")
        self.assertEqual(row["source_type_guess"], "")
        self.assertEqual(report["posts"], 1)

    def test_json_wrapper_posts_is_accepted(self):
        path = self.tmp / "posts.json"
        path.write_text(
            json.dumps(
                {
                    "posts": [
                        {
                            "post_key": "https://example.test/posts/3",
                            "topic": "Forum debate",
                            "primary_frame_id": "B",
                            "source_type_guess": "forum",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        frame, report = load_post_responses([str(path)], self.cfg)

        self.assertEqual(list(frame.columns), POST_COLUMNS)
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.loc[0, "post_key"], "https://example.test/posts/3")
        self.assertEqual(frame.loc[0, "topic"], "Forum debate")
        self.assertEqual(report["posts"], 1)

    def test_blank_post_key_is_dropped_and_duplicates_keep_the_last(self):
        path = self.tmp / "posts.csv"
        rows = [
            {
                "post_key": "https://example.test/posts/1",
                "topic": "First topic",
                "primary_frame_id": "A",
                "source_type_guess": "official",
            },
            {
                "post_key": "https://example.test/posts/1",
                "topic": "Last topic",
                "primary_frame_id": "A",
                "source_type_guess": "official",
            },
            {
                "post_key": "https://example.test/posts/2",
                "topic": "",
                "primary_frame_id": "B",
                "source_type_guess": "",
            },
            {
                "post_key": "",
                "topic": "Orphan",
                "primary_frame_id": "A",
                "source_type_guess": "official",
            },
        ]
        pd.DataFrame(rows).reindex(columns=POST_RESPONSE_CSV_HEADER).to_csv(
            path, index=False, encoding="utf-8"
        )

        frame, report = load_post_responses([str(path)], self.cfg)

        self.assertEqual(len(frame), 2)
        by_key = frame.set_index("post_key")
        self.assertEqual(
            by_key.loc["https://example.test/posts/1", "topic"], "Last topic"
        )
        self.assertNotIn("", by_key.index)
        self.assertEqual(report["records"], 4)
        self.assertEqual(report["invalid_records"], 1)

    def test_missing_files_raise_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_post_responses([str(self.tmp / "nothing*.csv")], self.cfg)


class NetworkExportTests(unittest.TestCase):
    """The flat CSV contract consumed by network.html."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = _class_temp_dir(cls)
        cls.cfg = load_study(TOY)
        comments, posts_csv = T.make_toy_corpus(cls.tmpdir)

        # With post answers: topics come from the model answers.
        cls.work_post = cls.tmpdir / "work_post"
        prepare(cls.cfg, comments, cls.work_post, posts_csv)
        recs = T.fake_classification(cls.work_post, cls.cfg)
        resp_post = cls.tmpdir / "responses_post"
        resp_post.mkdir()
        (resp_post / "r.json").write_text(json.dumps(recs), encoding="utf-8")
        post_answers = cls.tmpdir / "ai_posts.csv"
        pd.DataFrame(
            [
                {
                    "post_key": "https://www.facebook.com/citybikes/posts/111",
                    "topic": "Protected lanes downtown",
                    "primary_frame_id": "A",
                    "source_type_guess": "official",
                },
                {
                    "post_key": "https://www.facebook.com/groups/residentsforum/posts/222",
                    "topic": "Residents debate lanes",
                    "primary_frame_id": "B",
                    "source_type_guess": "forum",
                },
            ]
        ).reindex(columns=POST_RESPONSE_CSV_HEADER).to_csv(
            post_answers, index=False, encoding="utf-8"
        )
        merge(
            cls.cfg,
            cls.work_post,
            str(resp_post / "*.json"),
            post_responses=str(post_answers),
        )
        cls.post_path = cls.tmpdir / "post_network.csv"
        export_network(cls.cfg, cls.work_post, cls.post_path)
        cls.post_network = pd.read_csv(cls.post_path, dtype=str, keep_default_na=False)

        # Without post answers: post_topic is blank upstream, the export must fall back.
        cls.work_plain = cls.tmpdir / "work_plain"
        prepare(cls.cfg, comments, cls.work_plain, posts_csv)
        recs_plain = T.fake_classification(cls.work_plain, cls.cfg)
        resp_plain = cls.tmpdir / "responses_plain"
        resp_plain.mkdir()
        (resp_plain / "r.json").write_text(json.dumps(recs_plain), encoding="utf-8")
        merge(cls.cfg, cls.work_plain, str(resp_plain / "*.json"))
        cls.plain_path = cls.tmpdir / "plain_network.csv"
        export_network(cls.cfg, cls.work_plain, cls.plain_path)
        cls.plain = pd.read_csv(cls.plain_path, dtype=str, keep_default_na=False)

    def test_header_order_and_non_empty_rows(self):
        self.assertEqual(list(self.post_network.columns), NETWORK_COLUMNS)
        self.assertEqual(list(self.plain.columns), NETWORK_COLUMNS)
        self.assertGreater(len(self.post_network), 0)
        self.assertGreater(len(self.plain), 0)
        self.assertEqual(
            set(self.post_network["post_topic"].astype(str)),
            {"Protected lanes downtown", "Residents debate lanes"},
        )

    def test_empty_topic_falls_back_to_post_text(self):
        posts = pd.read_csv(
            self.work_plain / F.POSTS_CLASSIFIED, dtype=str, keep_default_na=False
        )
        self.assertTrue((posts["post_topic"].astype(str).str.strip() == "").all())

        self.assertGreater(len(self.plain), 0)
        self.assertTrue((self.plain["post_topic"].astype(str).str.strip() != "").all())
        self.assertTrue((self.plain["post_url"].astype(str).str.strip() != "").all())

    def test_stance_groups_are_normalised(self):
        for frame in (self.plain, self.post_network):
            self.assertTrue(set(frame["stance_group"]) <= STANCE_GROUPS)
        self.assertEqual(stance_group(None), "qualify")
        self.assertEqual(stance_group(""), "qualify")
        self.assertEqual(stance_group("nonsense"), "qualify")

    def test_published_at_is_utc_iso(self):
        for frame in (self.plain, self.post_network):
            for value in frame["published_at"]:
                if value:
                    self.assertRegex(value, PUBLISHED_AT)


class MarkdownBundleTests(unittest.TestCase):
    """The single-file chatbot bundle."""

    MANIFEST_ARG = "studies/toy_bikelanes/manifest.json"

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = _class_temp_dir(cls)
        cls.cfg = load_study(TOY)
        comments, posts_csv = T.make_toy_corpus(cls.tmpdir)
        cls.work = cls.tmpdir / "work"
        prepare(cls.cfg, comments, cls.work, posts_csv)

    def test_bundle_contract(self):
        markdown, info = build_classification_markdown(
            self.cfg, self.work, manifest_path=self.MANIFEST_ARG
        )
        work = str(self.work)

        self.assertIn(",".join(RESPONSE_CSV_HEADER), markdown)
        self.assertIn(",".join(POST_RESPONSE_CSV_HEADER), markdown)
        self.assertIn(f"{work}/ai_output.csv", markdown)
        self.assertIn(f"{work}/ai_posts.csv", markdown)
        self.assertIn("--post-ai-responses", markdown)
        self.assertIn(self.MANIFEST_ARG, markdown)
        self.assertNotIn("<manifest.json>", markdown)
        self.assertGreaterEqual(info["post_batches"], 1)

    def test_bundle_is_deterministic(self):
        first, _ = build_classification_markdown(
            self.cfg, self.work, manifest_path=self.MANIFEST_ARG
        )
        second, _ = build_classification_markdown(
            self.cfg, self.work, manifest_path=self.MANIFEST_ARG
        )
        self.assertEqual(first.encode("utf-8"), second.encode("utf-8"))


class ApiDisabledTests(unittest.TestCase):
    """API use is off by default and refused before touching any file."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = load_study(TOY)

    def setUp(self):
        self.tmp = _make_temp_dir(self)
        self.missing = self.tmp / "does_not_exist"

    def _assert_refused_before_io(self, func) -> str:
        with self.assertRaises(RuntimeError) as ctx:
            func(self.cfg, self.missing)
        message = str(ctx.exception)
        self.assertIn("--enable-api", message)
        lowered = message.lower()
        self.assertNotIn("batch", lowered)
        self.assertNotIn("could not read", lowered)
        self.assertNotIn("no such file", lowered)
        return message

    def test_comment_batches_refused_by_default(self):
        self._assert_refused_before_io(api.classify_batches)

    def test_post_batches_refused_by_default(self):
        self._assert_refused_before_io(api.classify_post_batches)

    def test_api_is_disabled_at_module_level(self):
        self.assertFalse(api.ENABLED_BY_DEFAULT)

    def test_api_configured_tracks_only_the_environment(self):
        original = os.environ.get(api.API_KEY_ENV)
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(api.api_configured())
        with mock.patch.dict(os.environ, {api.API_KEY_ENV: ""}, clear=True):
            self.assertFalse(api.api_configured())
        with mock.patch.dict(
            os.environ, {api.API_KEY_ENV: "dummy-test-key"}, clear=True
        ):
            self.assertTrue(api.api_configured())
        # patch.dict restores the process environment after each block.
        self.assertEqual(os.environ.get(api.API_KEY_ENV), original)


class ProgressContractTests(unittest.TestCase):
    """Console helpers keep stdout readable and never swallow items."""

    def test_done_writes_a_done_block(self):
        out = io.StringIO()
        with redirect_stdout(out):
            done("x", {"a": 1})
        text = out.getvalue()
        self.assertIn("DONE", text)
        self.assertIn("a: 1", text)

    def test_next_steps_lists_every_command(self):
        out = io.StringIO()
        with redirect_stdout(out):
            next_steps(["a", "b"])
        text = out.getvalue()
        self.assertIn("NEXT STEPS", text)
        self.assertIn("1. a", text)
        self.assertIn("2. b", text)

    def test_progress_yields_every_item(self):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            items = list(progress(range(3), total=3, desc="t"))
        self.assertEqual(items, [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
