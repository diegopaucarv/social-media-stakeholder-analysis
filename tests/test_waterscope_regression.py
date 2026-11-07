"""Opt-in regression against the original waterscope corpus.

Runs only when the raw scraper CSVs are provided and the study folder is present:
    FRAMING_REAL_COMMENTS=... FRAMING_REAL_POSTS=... python -m unittest discover -s tests
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from framing.config import load_study  # noqa: E402
from framing.prepare import prepare  # noqa: E402

WATER = ROOT / "studies" / "waterscope_viana" / "manifest.json"


@unittest.skipUnless(
    WATER.exists()
    and os.environ.get("FRAMING_REAL_COMMENTS")
    and os.environ.get("FRAMING_REAL_POSTS"),
    "set FRAMING_REAL_COMMENTS / FRAMING_REAL_POSTS and keep studies/waterscope_viana to run",
)
class RealDataRegression(unittest.TestCase):
    def test_waterscope_prepare_matches_the_first_manual_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = prepare(
                load_study(WATER),
                os.environ["FRAMING_REAL_COMMENTS"],
                tmp,
                os.environ["FRAMING_REAL_POSTS"],
                batches=False,
            )
            self.assertEqual((s["comments"], s["posts"], s["phases"]), (694, 29, 92))
            self.assertEqual(s["relevance"], {"low": 549, "high": 76, "medium": 69})
