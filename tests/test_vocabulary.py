"""Tests for the study-declared classification vocabulary.

Two of the three values that used to be module constants in ``framing.schemas`` are
already a study setting: the stance ids and the evidence levels live in
``codebook.json`` (``stance_definitions`` / ``evidence_definitions``). The third,
the ``EMERGENT`` sentinel, is now ``study.json`` ->
``classification.emergent_thread_id``.

Two properties matter and are asserted here:

* a study that declares nothing renders exactly as before (same config hash, same
  prompt, same JSON schema), so nothing has to be re-classified;
* a study that declares its own vocabulary is honoured by the prompt, the schema,
  the record checks and the analysis steps, and a value outside the vocabulary is
  flagged for human review instead of being silently accepted.
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
from framing.config import ConfigError, load_study  # noqa: E402
from framing.gaps import weakest_evidence_level  # noqa: E402
from framing.merge import merge  # noqa: E402
from framing.prepare import prepare  # noqa: E402
from framing.prompt_builder import (  # noqa: E402
    classifier_system_prompt,
    output_schema_text,
)
from framing.schemas import (  # noqa: E402
    check_classification,
    classification_item_schema,
    normalise_classification,
)

TOY = ROOT / "studies" / "toy_bikelanes" / "manifest.json"

#: Config hashes of the two shipped studies. They are the cascade guard: if either
#: changes, every existing study would have to be re-classified, which is exactly
#: what this design avoids.
BASELINE_HASHES = {
    "toy_bikelanes": "bf158570d23f",
    "waterscope_viana": "a3ebb67adf1c",
}

SPANISH_STANCES = {
    "a_favor": "Sustains the argument of the thread.",
    "en_contra": "Rebuts the argument of the thread.",
    "matiza": "Agrees only partly, adds conditions.",
}
EXTENDED_EVIDENCE = {
    "core": "States the target argument.",
    "adjacent": "Related evidence.",
    "anecdotal": "A single personal anecdote, the weakest support.",
}


def toy_codebook(**sections):
    """The toy codebook, with the given sections replaced."""
    data = json.loads((TOY.parent / "codebook.json").read_text(encoding="utf-8"))
    data.update(sections)
    return data


def variant_study(tmp, codebook=None, sentinel=None):
    """Copy the toy study, optionally overriding study.json / codebook.json."""
    dst = Path(tmp) / "variant"
    shutil.copytree(TOY.parent, dst)
    if sentinel is not None:
        path = dst / "study.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data.setdefault("classification", {})["emergent_thread_id"] = sentinel
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    if codebook is not None:
        path = dst / "codebook.json"
        path.write_text(
            json.dumps(codebook, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return dst / "manifest.json"


def retarget_examples(codebook, stance_map=None, evidence_map=None):
    """Rewrite the stored examples so they stay valid under a renamed vocabulary."""
    stance_map, evidence_map = stance_map or {}, evidence_map or {}
    for example in codebook.get("examples", []):
        out = example["output"]
        if out.get("stance") in stance_map:
            out["stance"] = stance_map[out["stance"]]
        if out.get("evidence_level") in evidence_map:
            out["evidence_level"] = evidence_map[out["evidence_level"]]
    return codebook


def framed_record(cfg, **overrides):
    """A minimal record the study considers valid, plus the caller's overrides."""
    record = {
        "comment_id": "c1",
        "frame_id": cfg.frame_ids[0],
        "thread_id": cfg.frames[0]["threads"][0]["id"],
        "stance": cfg.stance_ids[0],
        "evidence_level": cfg.evidence_ids[0],
        "confidence": 3,
    }
    record.update(overrides)
    return record


class ContractTests(unittest.TestCase):
    """A study that declares nothing must behave exactly as before."""

    def test_default_study_uses_the_builtin_vocabulary(self):
        cfg = load_study(TOY)
        self.assertEqual(cfg.stance_ids, ["supports", "opposes", "qualifies"])
        self.assertEqual(cfg.evidence_ids, ["core", "adjacent"])
        self.assertEqual(cfg.emergent_thread_id, "EMERGENT")

    def test_default_rendering_keeps_the_recorded_config_hash(self):
        for study_id, expected in BASELINE_HASHES.items():
            with self.subTest(study=study_id):
                cfg = load_study(ROOT / "studies" / study_id / "manifest.json")
                self.assertEqual(cfg.hash, expected)

    def test_default_sentinel_is_rendered_as_before(self):
        cfg = load_study(TOY)
        prompt = classifier_system_prompt(cfg)
        self.assertIn('use thread_id "EMERGENT"', prompt)
        self.assertIn("does it need an EMERGENT thread", prompt)
        self.assertIn('thread_id: a thread id of the chosen frame, "EMERGENT"', prompt)
        self.assertNotIn("{{EMERGENT}}", prompt)
        self.assertIn(
            "when in doubt between 'core' and 'adjacent' choose 'adjacent'; "
            "between 'supports' and 'qualifies' choose 'qualifies'",
            prompt,
        )
        enum = classification_item_schema(cfg)["properties"]["thread_id"]["enum"]
        self.assertEqual(enum[-1], "EMERGENT")

    def test_restating_the_default_sentinel_does_not_change_the_hash(self):
        default = load_study(TOY)
        with tempfile.TemporaryDirectory() as tmp:
            explicit = load_study(variant_study(tmp, sentinel="EMERGENT"))
        self.assertEqual(explicit.hash, default.hash)
        self.assertNotIn("emergent_thread_id", default.study["classification"])
        self.assertNotIn("emergent_thread_id", explicit.study["classification"])


class CustomStanceTests(unittest.TestCase):
    """A codebook that declares its own stance ids is the authority."""

    def _cfg(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        codebook = toy_codebook(stance_definitions=SPANISH_STANCES)
        retarget_examples(codebook, stance_map={"supports": "a_favor"})
        return load_study(variant_study(tmp.name, codebook=codebook))

    def test_declaration_order_drives_the_vocabulary(self):
        cfg = self._cfg()
        self.assertEqual(cfg.stance_ids, list(SPANISH_STANCES))

    def test_prompt_and_schema_follow_the_codebook(self):
        cfg = self._cfg()
        prompt = classifier_system_prompt(cfg)
        for stance, description in SPANISH_STANCES.items():
            self.assertIn(f'<value id="{stance}">{description}</value>', prompt)
        self.assertIn(json.dumps(list(SPANISH_STANCES)), output_schema_text(cfg))
        enum = classification_item_schema(cfg)["properties"]["stance"]["enum"]
        self.assertEqual(enum, [""] + list(SPANISH_STANCES))
        self.assertNotIn("supports", enum)
        # The generic calibration rule names the ends of the scale, not the defaults.
        self.assertIn("between 'a_favor' and 'matiza' choose 'matiza'", prompt)

    def test_custom_stance_survives_normalisation(self):
        """Regression: the record check used to blank anything outside the built-in list."""
        cfg = self._cfg()
        record, issues = normalise_classification(
            framed_record(cfg, stance="matiza"), cfg
        )
        self.assertEqual(issues, [])
        self.assertEqual(record["stance"], "matiza")
        self.assertFalse(record["needs_human_review"])

    def test_stance_outside_the_vocabulary_is_flagged_and_blanked(self):
        cfg = load_study(TOY)
        record, issues = normalise_classification(
            framed_record(cfg, stance="matiza"), cfg
        )
        self.assertTrue(any("invalid stance" in issue for issue in issues))
        self.assertEqual(record["stance"], "")
        self.assertTrue(record["needs_human_review"])


class CustomEvidenceTests(unittest.TestCase):
    """The evidence vocabulary is the codebook's, and its order ranks the levels."""

    def _cfg(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        codebook = toy_codebook(evidence_definitions=EXTENDED_EVIDENCE)
        retarget_examples(codebook, evidence_map={"core": "core"})
        return load_study(variant_study(tmp.name, codebook=codebook))

    def test_last_declared_level_is_the_weakest(self):
        cfg = self._cfg()
        self.assertEqual(cfg.evidence_ids, list(EXTENDED_EVIDENCE))
        self.assertEqual(weakest_evidence_level(cfg), "anecdotal")
        self.assertEqual(weakest_evidence_level(load_study(TOY)), "adjacent")

    def test_prompt_and_schema_follow_the_codebook(self):
        cfg = self._cfg()
        prompt = classifier_system_prompt(cfg)
        self.assertIn('<value id="anecdotal">', prompt)
        self.assertIn(json.dumps(list(EXTENDED_EVIDENCE)), output_schema_text(cfg))
        enum = classification_item_schema(cfg)["properties"]["evidence_level"]["enum"]
        self.assertEqual(enum, [""] + list(EXTENDED_EVIDENCE))
        # The weakest level of the *declared* order is the one the rules push down to.
        self.assertIn("a comment of level 'anecdotal' that", prompt)
        self.assertIn(
            "when in doubt between 'core' and 'anecdotal' choose 'anecdotal'", prompt
        )

    def test_custom_level_survives_normalisation(self):
        cfg = self._cfg()
        record, issues = normalise_classification(
            framed_record(cfg, evidence_level="anecdotal"), cfg
        )
        self.assertEqual(issues, [])
        self.assertEqual(record["evidence_level"], "anecdotal")


class EmergentSentinelTests(unittest.TestCase):
    """The thread-proposal sentinel is a study setting, not a module constant."""

    def _cfg(self, sentinel):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return load_study(variant_study(tmp.name, sentinel=sentinel))

    def test_custom_sentinel_reaches_the_schema_and_the_prompt(self):
        cfg = self._cfg("NUEVO")
        prompt = classifier_system_prompt(cfg)
        self.assertIn("does it need an NUEVO thread", prompt)
        self.assertIn('use thread_id "NUEVO"', prompt)
        self.assertNotIn('use thread_id "EMERGENT"', prompt)
        enum = classification_item_schema(cfg)["properties"]["thread_id"]["enum"]
        self.assertEqual(enum[-1], "NUEVO")

    def test_custom_sentinel_is_accepted_by_the_record_checks(self):
        cfg = self._cfg("NUEVO")
        record = framed_record(
            cfg, thread_id="NUEVO", emergent_thread_proposal="a new argument"
        )
        self.assertEqual(check_classification(record, cfg), [])
        normalised, issues = normalise_classification(record, cfg)
        self.assertEqual(issues, [])
        self.assertEqual(normalised["thread_id"], "NUEVO")

    def test_default_token_is_rejected_once_a_sentinel_is_configured(self):
        cfg = self._cfg("NUEVO")
        record = framed_record(
            cfg, thread_id="EMERGENT", emergent_thread_proposal="a new argument"
        )
        self.assertTrue(check_classification(record, cfg))

    def test_merge_labels_the_configured_token(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        cfg = load_study(variant_study(tmp, sentinel="NUEVO"))
        comments, posts = make_toy_corpus(tmp)
        work = tmp / "work"
        prepare(cfg, comments, work, posts)
        records = fake_classification(work, cfg)
        target = next(r for r in records if r["frame_id"] != cfg.no_frame_id)
        target["thread_id"] = "NUEVO"
        target["emergent_thread_proposal"] = "a brand new thread"
        answer = tmp / "answers.json"
        answer.write_text(json.dumps(records), encoding="utf-8")
        merge(cfg, work, [str(answer)])
        coded = pd.read_csv(work / F.ALL_CODED, dtype=str).fillna("")
        row = coded[coded["comment_id"] == target["comment_id"]].iloc[0]
        self.assertEqual(row["thread_id"], "NUEVO")
        self.assertEqual(row["emergent_thread_proposal"], "a brand new thread")
        self.assertEqual(row["frame_id"], target["frame_id"])
        # The label is written to the selected-comments table, which carries no
        # comment_id; the proposal text identifies the row unambiguously.
        selected = pd.read_csv(work / F.SELECTED, dtype=str).fillna("")
        self.assertIn("NUEVO: a brand new thread", set(selected["thread_label"]))

    def test_sentinel_colliding_with_a_thread_is_rejected(self):
        cfg = load_study(TOY)
        taken = cfg.frames[0]["threads"][0]["id"]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ConfigError) as caught:
                load_study(variant_study(tmp, sentinel=taken))
        self.assertIn("reserved id", str(caught.exception))

    def test_sentinel_equal_to_the_no_frame_id_is_rejected(self):
        no_frame = json.loads(
            (TOY.parent / "codebook.json").read_text(encoding="utf-8")
        )["no_frame"]["id"]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ConfigError) as caught:
                load_study(variant_study(tmp, sentinel=no_frame))
        self.assertIn("no_frame id", str(caught.exception))


class VocabularyValidationTests(unittest.TestCase):
    """Bad vocabulary declarations fail loudly instead of silently degrading."""

    def test_empty_vocabulary_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ConfigError):
                load_study(
                    variant_study(tmp, codebook=toy_codebook(stance_definitions={}))
                )

    def test_non_string_definition_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ConfigError):
                load_study(
                    variant_study(
                        tmp, codebook=toy_codebook(evidence_definitions={"core": 5})
                    )
                )

    def test_empty_sentinel_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ConfigError):
                load_study(variant_study(tmp, sentinel=""))


if __name__ == "__main__":
    unittest.main()
