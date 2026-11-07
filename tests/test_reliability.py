"""Reliability and wizard-header regressions: undefined kappa must not read as agreement."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from framing import files as F  # noqa: E402
from framing.config import load_study  # noqa: E402
from framing.export_md import RESPONSE_CSV_HEADER  # noqa: E402
from framing.provenance import write_run  # noqa: E402
from framing.reliability import (  # noqa: E402
    cohen_kappa,
    compare_codings,
    run_compare,
)
from framing.utils import write_csv, write_json  # noqa: E402
from framing.wizard import TEMPLATE_CSV, explain_chatbot_branch  # noqa: E402

TOY = ROOT / "studies" / "toy_bikelanes" / "manifest.json"


def _coded_table(frames):
    """One coder's rows; `frames` assigns a frame id to each comment."""
    n = len(frames)
    return pd.DataFrame(
        {
            "comment_id": [f"c{i}" for i in range(n)],
            "timestamp": ["2024-01-01T00:00:00Z"] * n,
            "text": ["bike lane comment"] * n,
            "is_institutional": [False] * n,
            "has_text": [True] * n,
            "frame_id": list(frames),
            "thread_id": ["A_1"] * n,
            "stance": [""] * n,
            "evidence_level": ["core"] * n,
        }
    )


class CohenKappaTests(unittest.TestCase):
    def test_perfect_agreement_with_variation(self):
        self.assertEqual(cohen_kappa(list("aabb"), list("aabb")), 1.0)

    def test_chance_level_is_zero(self):
        self.assertAlmostEqual(cohen_kappa(list("aabb"), list("abab")), 0.0)

    def test_constant_both_sides_is_undefined(self):
        # Regression: a constant column used to be reported as a perfect 1.0.
        self.assertIsNone(cohen_kappa([""] * 4, [""] * 4))
        self.assertIsNone(cohen_kappa(["x"] * 4, ["x"] * 4))

    def test_empty_input_is_none_not_nan(self):
        # Used to be float("nan"), which write_json(allow_nan=False) cannot serialise.
        self.assertIsNone(cohen_kappa([], []))

    def test_constant_on_one_side_is_still_defined(self):
        self.assertEqual(cohen_kappa(["x"] * 4, ["x", "x", "y", "y"]), 0.0)


class CompareCodingsTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_study(TOY)
        f0, f1 = self.cfg.frame_ids[:2]
        nf = self.cfg.no_frame_id
        self.a = _coded_table([f0, f0, f1, f1, f0, f1, nf, nf])
        self.b = _coded_table([f0, f1, f1, f1, f0, f0, nf, nf])
        self.rep = compare_codings(self.cfg, self.a, self.b)

    def test_frame_all_categories_is_defined(self):
        allcat = self.rep["frame_all_categories"]
        self.assertIsInstance(allcat["agreement"], float)
        self.assertIsNotNone(allcat["kappa"])

    def test_uncollected_stance_agrees_but_kappa_is_undefined(self):
        stance = self.rep["stance_given_same_frame"]
        self.assertGreater(stance["n"], 0)  # the metric was actually computed
        self.assertEqual(stance["agreement"], 1.0)
        self.assertIsNone(stance["kappa"])

    def test_per_frame_covers_every_study_frame(self):
        per = self.rep["per_frame"]
        self.assertEqual(set(per), set(self.cfg.frame_ids))
        expected = {"n_a", "n_b", "both", "kappa", "jaccard"}
        for fid, entry in per.items():
            self.assertEqual(set(entry), expected, fid)

    def test_report_is_strict_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / F.RELIABILITY
            write_json(path, self.rep)
            raw = path.read_text(encoding="utf-8")
        self.assertNotIn("NaN", raw)
        loaded = json.loads(raw)
        self.assertIsNone(loaded["stance_given_same_frame"]["kappa"])
        self.assertEqual(
            loaded["stance_given_same_frame"]["agreement"],
            self.rep["stance_given_same_frame"]["agreement"],
        )


class RunCompareTests(unittest.TestCase):
    def test_writes_strict_loadable_report(self):
        cfg = load_study(TOY)
        f0, f1 = cfg.frame_ids[:2]
        same = _coded_table([f0, f0, f1, f1, cfg.no_frame_id, cfg.no_frame_id])
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            a_dir, b_dir, out_dir = base / "a", base / "b", base / "out"
            for d in (a_dir, b_dir):
                d.mkdir()
                write_csv(same, d / F.ALL_CODED)
                write_run(cfg, d, "merge")
            rep = run_compare(cfg, a_dir, b_dir, out_dir)
            raw = (out_dir / F.RELIABILITY).read_text(encoding="utf-8")
        loaded = json.loads(raw)
        self.assertEqual(rep["n_compared"], 6)
        self.assertEqual(loaded["config_hash"], cfg.hash)
        self.assertEqual(loaded["frame_all_categories"]["kappa"], 1.0)
        self.assertNotIn("NaN", raw)


class ChatbotHeaderTests(unittest.TestCase):
    @staticmethod
    def _printed_header(work):
        lines = []
        explain_chatbot_branch(work, out=lines.append)
        return [line for line in lines if line.startswith("     ") and "," in line]

    def test_missing_template_uses_default_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            matches = self._printed_header(Path(tmp))
        self.assertEqual(len(matches), 1, matches)
        self.assertEqual(matches[0], "     " + ",".join(RESPONSE_CSV_HEADER))

    def test_template_with_default_header_is_byte_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / TEMPLATE_CSV).write_text(
                ",".join(RESPONSE_CSV_HEADER) + "\n", encoding="utf-8"
            )
            matches = self._printed_header(work)
        self.assertEqual(len(matches), 1, matches)
        self.assertEqual(matches[0], "     " + ",".join(RESPONSE_CSV_HEADER))

    def test_trimmed_template_wins_over_the_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / TEMPLATE_CSV).write_text(
                "comment_id,frame_id,stance\n", encoding="utf-8"
            )
            matches = self._printed_header(work)
        self.assertEqual(len(matches), 1, matches)
        self.assertEqual(matches[0], "     comment_id,frame_id,stance")

    def test_empty_template_falls_back_to_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / TEMPLATE_CSV).write_text("\n", encoding="utf-8")
            lines = []
            explain_chatbot_branch(work, out=lines.append)  # must not raise
        matches = [ln for ln in lines if ln.startswith("     ") and "," in ln]
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0], "     " + ",".join(RESPONSE_CSV_HEADER))


if __name__ == "__main__":
    unittest.main()
