"""Unit and end-to-end tests. Run:  python -m unittest discover -s tests -v   (from the package root)."""

from __future__ import annotations

import copy
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from framing import files as F  # noqa: E402
from framing.cli import build_parser  # noqa: E402
from framing.cli import main as cli_main  # noqa: E402
from framing.config import (  # noqa: E402
    Config,
    ConfigError,
    config_from_modules,
    load_modules,
    load_study,
    schema_errors,
)
from framing.discovery import apply_discovery, build_discovery_input, seed_codebook  # noqa: E402
from framing.gaps import build_gap_report  # noqa: E402
from framing.merge import merge  # noqa: E402
from framing.network_export import NETWORK_COLUMNS  # noqa: E402
from framing.prepare import POST_META_COLUMNS, detect_corpus, prepare  # noqa: E402
from framing.prompt_builder import (  # noqa: E402
    classifier_system_prompt,
    discovery_response_schema,
    post_classifier_system_prompt,
    render_all,
)
from framing.reliability import cohen_kappa, run_compare  # noqa: E402
from framing.schemas import (  # noqa: E402
    classification_schema,
    normalise_classification,
    post_classification_schema,
)
from framing.stats import compute_stats, cramers_v, two_by_two  # noqa: E402
from framing.users import load_coded, run_users  # noqa: E402

# Dot- and underscore-prefixed folders are disabled or temporarily moved copies, not studies.
STUDY_MANIFESTS = sorted(
    m
    for m in (ROOT / "studies").glob("*/manifest.json")
    if not m.parent.name.startswith((".", "_"))
)

# The synthetic study used by the end-to-end fixtures, pinned explicitly so the suite does not
# depend on how many studies exist or on the order of the glob.
TOY = ROOT / "studies" / "toy_bikelanes" / "manifest.json"


# ----------------------------------------------------------------------------- synthetic corpus for the toy study
def make_toy_corpus(tmp: Path) -> tuple[Path, Path]:
    posts = pd.DataFrame(
        [
            {
                "pid": "111",
                "link": "https://www.facebook.com/citybikes/posts/111",
                "body": "The city plans protected bike lanes downtown.",
                "ts": "2024-05-01T10:00:00Z",
                "by": "City Bikes Office",
                "page": "citybikes",
            },
            {
                "pid": "222",
                "link": "https://www.facebook.com/groups/residentsforum/posts/222",
                "body": "Residents forum: what do you think about the lanes?",
                "ts": "2025-03-01T10:00:00Z",
                "by": "Ana Admin",
                "page": "residentsforum",
            },
        ]
    )
    likes_a = [0, 1, 0, 5, 0, 0, 2, 0, 0, 0, 1, 0]
    texts = [
        "I was almost hit by a car in Old Town",
        "Lanes remove parking and kill shops",
        "Great idea for the kids",
        "Accidents happen every week",
        "Thanks!",
        "No parking left for customers",
        "Children need a safe route",
        "Where is the money coming from?",
        "Riverside has no lane at all",
        "Agreed",
        "I cycle daily and feel unsafe",
        "Altstadt is dangerous",
    ]
    rows = []
    for i, (t, lk) in enumerate(zip(texts, likes_a), 1):
        rows.append(
            {
                "cid": f"a{i}",
                "post": posts.link[0],
                "post_num": "111",
                "post_title": posts.body[0],
                "ts": f"2024-05-01T{10 + i:02d}:00:00Z",
                "depth": 0,
                "likes": lk,
                "replies": 0,
                "parent": np.nan,
                "uid": f"u{i % 5}",
                "uname": f"User {i % 5}",
                "body": t,
            }
        )
    rows[9].update(depth=1, parent="a4")  # "Agreed" replies to the 5-like comment
    rows[4].update(depth=1, parent="a3")
    rows.append(
        {
            "cid": "a13",
            "post": posts.link[0],
            "post_num": "111",
            "post_title": posts.body[0],
            "ts": "2024-05-02T09:00:00Z",
            "depth": 0,
            "likes": 9,
            "replies": 0,
            "parent": np.nan,
            "uid": "city",
            "uname": "City Hall",
            "body": "Thank you for your comments.",
        }
    )
    for j in range(1, 8):
        rows.append(
            {
                "cid": f"b{j}",
                "post": posts.link[1],
                "post_num": "222",
                "post_title": posts.body[1],
                "ts": f"2025-03-01T{10 + j:02d}:30:00Z",
                "depth": 0,
                "likes": j % 4,
                "replies": 0,
                "parent": np.nan,
                "uid": f"u{j}",
                "uname": f"User {j}",
                "body": f"Comment {j} about the lanes and accident risk",
            }
        )
    rows.append(
        {
            "cid": "b8",
            "post": posts.link[1],
            "post_num": "222",
            "post_title": posts.body[1],
            "ts": "2025-03-02T10:30:00Z",
            "depth": 0,
            "likes": 0,
            "replies": 0,
            "parent": np.nan,
            "uid": "u9",
            "uname": "User 9",
            "body": np.nan,
        }
    )  # no text (sticker)
    c_path, p_path = tmp / "toy_comments.csv", tmp / "toy_posts.csv"
    pd.DataFrame(rows).to_csv(c_path, index=False)
    posts.to_csv(p_path, index=False)
    return c_path, p_path


def fake_classification(work: Path, cfg: Config) -> list[dict]:
    enr = pd.read_csv(work / F.COMMENTS_ENRICHED, dtype={"comment_id": str})
    eligible = enr[(~enr["is_institutional"]) & enr["has_text"]]
    base = {
        "is_institutional": False,
        "secondary_frame_id": "",
        "emergent_thread_proposal": "",
        "literal_excerpt": "",
        "rationale": "t",
        "confidence": 3,
        "needs_human_review": False,
        "review_reason": "",
        "actor_category": "",
        "actor_self_declared": False,
        "actor_evidence": "",
    }
    # Fixture ids come from the loaded study, not from literals.
    frame_a, frame_b = cfg.frames[0]["id"], cfg.frames[1]["id"]
    thread_a1, thread_a2 = (t["id"] for t in cfg.frames[0]["threads"][:2])
    thread_b1 = cfg.frames[1]["threads"][0]["id"]
    recs = []
    for k, cid in enumerate(eligible["comment_id"]):
        r = dict(
            base,
            comment_id=cid,
            frame_id=cfg.no_frame_id,
            thread_id="",
            stance="",
            evidence_level="",
        )
        m = k % 4
        if m == 1:
            r.update(
                frame_id=frame_a,
                thread_id=thread_a1,
                stance="supports",
                evidence_level="core",
            )
            if k % 8 == 1:
                r.update(
                    actor_category="daily_cyclist",
                    actor_self_declared=True,
                    actor_evidence="I cycle daily",
                )
        elif m == 2:
            r.update(
                frame_id=frame_b,
                thread_id=thread_b1,
                stance="supports",
                evidence_level="adjacent",
            )
        elif m == 3:
            r.update(
                frame_id=frame_a,
                thread_id=thread_a2,
                stance="opposes",
                evidence_level="adjacent",
                confidence=1,
            )
        recs.append(r)
    # invalid: a frame-A record given a frame-B thread -> flagged
    recs[1]["thread_id"] = thread_b1
    recs[5]["frame_id"] = frame_a.lower()  # case-insensitive match -> coerced silently
    recs.pop()  # last eligible comment is left unclassified
    return recs


# ----------------------------------------------------------------------------- tests
class ConfigTests(unittest.TestCase):
    def test_all_studies_load(self):
        # Fixture-specific expectations (frame counts, sentinel ids) live with the fixtures,
        # in DiscoveryTests, so this only asserts the generic contract of every study.
        self.assertTrue(STUDY_MANIFESTS, "no study manifests found")
        cfgs = [load_study(m) for m in STUDY_MANIFESTS]
        for cfg in cfgs:
            self.assertGreaterEqual(len(cfg.frames), 1)
        self.assertEqual(len({c.hash for c in cfgs}), len(cfgs))

    def test_unified_config_has_every_module(self):
        self.assertEqual(
            set(load_study(TOY).unified()),
            {"study", "sources", "actors", "locations", "codebook"},
        )

    def test_semantic_errors_are_detected(self):
        modules, man, base = load_modules(TOY)
        cases = {
            "duplicate thread": lambda m: m["codebook"]["frames"][1]["threads"].append(
                dict(m["codebook"]["frames"][0]["threads"][0])
            ),
            "unknown actor": lambda m: m["sources"][
                "institutional_authors"
            ].__setitem__(0, {"name": "X", "actor_category": "nope"}),
            "unknown indicator thread": lambda m: m["locations"]["community_relevance"][
                "indicators"
            ][0]["thread_ids"].append("Z_9"),
            "bad example": lambda m: m["codebook"]["examples"][0]["output"].__setitem__(
                "thread_id", "B_1"
            ),
            "bad threshold": lambda m: m["study"]["relevance"].update(
                medium_threshold=99
            ),
            "unknown source type": lambda m: m["sources"]["registry"][0].__setitem__(
                "type", "ghost"
            ),
            "bad regex": lambda m: m["study"]["gap_checks"]["expected_terms"].append(
                {"label": "x", "pattern": "("}
            ),
        }
        for name, mutate in cases.items():
            with self.subTest(name):
                broken = copy.deepcopy(modules)
                mutate(broken)
                with self.assertRaises(ConfigError):
                    config_from_modules(broken, man, base)

    def test_schema_errors_are_detected(self):
        modules, _, _ = load_modules(TOY)
        broken = copy.deepcopy(modules["codebook"])
        del broken["frames"][0]["argument"]
        self.assertTrue(schema_errors(broken, "codebook"))
        self.assertTrue(schema_errors({"schema_version": "9"}, "manifest"))

    def test_lite_validator_agrees_with_jsonschema(self):
        for manifest in STUDY_MANIFESTS:
            modules, _, _ = load_modules(manifest)
            for name, mod in modules.items():
                self.assertEqual(
                    schema_errors(mod, name, force_lite=True),
                    [],
                    f"{manifest.parent.name}:{name}",
                )
                self.assertEqual(
                    schema_errors(mod, name), [], f"{manifest.parent.name}:{name}"
                )
        modules, _, _ = load_modules(TOY)
        bad = copy.deepcopy(modules["codebook"])
        bad["frames"][0]["threads"][0]["id"] = 5
        self.assertTrue(schema_errors(bad, "codebook", force_lite=True))
        self.assertTrue(schema_errors(bad, "codebook"))


class SchemaTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_study(TOY)

    def test_normalise_coerces_and_flags(self):
        rec, issues = normalise_classification(
            {
                "comment_id": 7,
                "frame_id": self.cfg.frame_ids[0].lower(),
                "thread_id": self.cfg.thread_ids[0].lower(),
                "stance": "SUPPORTS",
                "evidence_level": "Core",
                "confidence": "2",
            },
            self.cfg,
        )
        self.assertEqual(
            (
                rec["comment_id"],
                rec["frame_id"],
                rec["thread_id"],
                rec["stance"],
                rec["evidence_level"],
                rec["confidence"],
            ),
            (
                "7",
                self.cfg.frame_ids[0],
                self.cfg.thread_ids[0],
                "supports",
                "core",
                2,
            ),
        )
        self.assertEqual(issues, [])
        rec, issues = normalise_classification(
            {
                "comment_id": "x",
                "frame_id": self.cfg.frame_ids[1],
                "thread_id": self.cfg.thread_ids[0],
                "stance": "supports",
                "evidence_level": "core",
                "confidence": 3,
            },
            self.cfg,
        )
        self.assertTrue(issues and rec["needs_human_review"])
        rec, issues = normalise_classification(
            {"comment_id": "y", "frame_id": "ZZZ", "confidence": 9}, self.cfg
        )
        self.assertEqual(rec["frame_id"], self.cfg.no_frame_id)
        self.assertEqual(rec["confidence"], 1)
        rec, _ = normalise_classification(
            {
                "comment_id": "z",
                "frame_id": self.cfg.frame_ids[0],
                "thread_id": self.cfg.thread_ids[0],
                "stance": "supports",
                "evidence_level": "core",
                "confidence": 3,
                "is_institutional": True,
            },
            self.cfg,
        )
        self.assertEqual(
            (rec["frame_id"], rec["thread_id"]), (self.cfg.no_frame_id, "")
        )

    def test_response_schemas_are_valid_json_schema(self):
        import jsonschema

        for schema in (
            classification_schema(self.cfg),
            post_classification_schema(self.cfg),
            discovery_response_schema(),
        ):
            jsonschema.Draft202012Validator.check_schema(schema)
        ex = self.cfg.codebook["examples"][0]["output"]
        jsonschema.validate([ex], classification_schema(self.cfg))
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(
                [dict(ex, frame_id="Q")], classification_schema(self.cfg)
            )


SIGNATURE_MIN_CHARS = 8


# Sentinels every study shares; matching them says nothing about topical leakage. Each study's own
# no-frame id is added from its config in `_signatures`, so no study-specific id is hard-coded here.
GENERIC_SENTINELS = {"no frame", "no frames", "no-frame", "none", "other"}


def _signatures(cfg):
    """Multi-word labels/names that identify a study; single generic words are skipped on purpose."""
    sentinels = GENERIC_SENTINELS | {cfg.no_frame_id.lower()}
    sigs = {cfg.codebook["no_frame"]["label"]}
    for f in cfg.frames:
        sigs |= {f["label"], f["argument"][:60]}
        sigs |= {t["label"] for t in f["threads"]}
    sigs |= {g["name"] for g in cfg.gazetteer}
    sigs |= {a["name"] for a in cfg.sources["institutional_authors"]}
    sigs |= {e["name"] for e in cfg.sources["registry"] if e.get("name")}
    return {
        s.strip()
        for s in sigs
        if len(s.strip()) >= SIGNATURE_MIN_CHARS
        and len(s.split()) >= 2
        and s.strip().lower() not in sentinels
    }


def _system_prompts(cfg):
    """The three system prompts that must stay topic-neutral for every study."""
    return {
        "classifier_system": classifier_system_prompt(cfg),
        "post_classifier_system": post_classifier_system_prompt(cfg),
        "discovery_system": (ROOT / "prompts" / "discovery_system.md").read_text(
            encoding="utf-8"
        ),
    }


class PromptTests(unittest.TestCase):
    def test_prompts_render_for_every_study_without_leftovers(self):
        for manifest in STUDY_MANIFESTS:
            cfg = load_study(manifest)
            with tempfile.TemporaryDirectory() as tmp:
                render_all(cfg, tmp)
                text = (Path(tmp) / "classifier_system.md").read_text(encoding="utf-8")
                self.assertNotIn("{{", text)
                for fid in cfg.all_frame_ids + cfg.thread_ids + cfg.actor_ids:
                    self.assertIn(fid, text, fid)
                self.assertIn(cfg.study["title"], text)
                for ex in cfg.codebook["examples"]:
                    self.assertIn(json.dumps(ex["output"], ensure_ascii=False), text)
                post = (Path(tmp) / "post_classifier_system.md").read_text(
                    encoding="utf-8"
                )
                self.assertNotIn("{{", post)
                for name in (
                    "classifier_response_schema.json",
                    "post_classifier_response_schema.json",
                    "discovery_response_schema.json",
                    "prompt_manifest.json",
                ):
                    json.loads((Path(tmp) / name).read_text(encoding="utf-8"))

    def test_prompts_contain_no_topic_words_of_other_studies(self):
        cfgs = {m.parent.name: load_study(m) for m in STUDY_MANIFESTS}
        # 1. no study's prompt may contain another study's identifying phrases
        for name, cfg in cfgs.items():
            for other_name, other in cfgs.items():
                if other == cfg:
                    continue
                foreign = _signatures(other)
                for pname, text in _system_prompts(cfg).items():
                    for phrase in foreign:
                        with self.subTest(
                            study=name, prompt=pname, foreign=other_name, phrase=phrase
                        ):
                            self.assertNotIn(phrase.lower(), text.lower())


class StatsMathTests(unittest.TestCase):
    def test_cramers_v_and_yates(self):
        tab = pd.DataFrame([[10, 20], [20, 10]])
        v, _ = cramers_v(tab)
        self.assertAlmostEqual(v, np.sqrt(20 / 3 / 60), places=6)
        rows = [
            {"flag": f, "col": c}
            for f, c, n in ((0, 0, 10), (0, 1, 20), (1, 0, 20), (1, 1, 10))
            for _ in range(n)
        ]
        res = two_by_two(
            {"id": "X", "pop": "p"}, pd.DataFrame(rows), "flag", "col", "x"
        )
        self.assertAlmostEqual(res["chi2"], 5.4, places=3)  # Yates-corrected
        self.assertAlmostEqual(res["p"], sps.chi2.sf(5.4, 1), places=6)
        self.assertAlmostEqual(
            res["cramers_v"], round(float(np.sqrt(20 / 3 / 60)), 3), places=3
        )

    def test_kappa(self):
        self.assertEqual(cohen_kappa(list("aabb"), list("aabb")), 1.0)
        self.assertAlmostEqual(cohen_kappa(list("aabb"), list("abab")), 0.0)


class ToyPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        tmp = Path(cls.tmpdir.name)
        cls.cfg = load_study(TOY)
        cls.comments, cls.posts = make_toy_corpus(tmp)
        cls.work = tmp / "work"
        cls.summary = prepare(cls.cfg, cls.comments, cls.work, cls.posts)
        cls.responses = tmp / "responses"
        cls.responses.mkdir()
        recs = fake_classification(cls.work, cls.cfg)
        (cls.responses / "r1.json").write_text(json.dumps(recs[:10]), encoding="utf-8")
        (cls.responses / "r2.json").write_text(
            "```json\n" + json.dumps(recs[10:]) + "\n```", encoding="utf-8"
        )
        cls.recs = recs
        cls.report = merge(cls.cfg, cls.work, str(cls.responses / "*.json"))

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def test_prepare_counts_and_join(self):
        s = self.summary
        self.assertEqual(
            (s["comments"], s["posts"], s["unmatched_comments"]), (21, 2, 0)
        )
        self.assertEqual(s["comments_to_classify"], 19)  # 21 - City Hall - empty text
        rep = json.loads((self.work / F.JOIN_REPORT).read_text())
        self.assertEqual(rep["strategies"]["numeric_id"]["matched_comments"], 21)

    def test_post_meta_template_is_generated_and_not_overwritten(self):
        path = self.work / F.POST_META_TEMPLATE
        self.assertTrue(path.exists())
        template = pd.read_csv(path, dtype=str).fillna("")
        self.assertEqual(list(template.columns), list(POST_META_COLUMNS))
        self.assertEqual(len(template), len(pd.read_csv(self.work / F.POSTS_TABLE)))
        self.assertTrue(template["post_key"].str.strip().ne("").all())
        self.assertTrue(template["post_topic"].eq("").all())
        # A hand edit must survive another `prepare` run.
        template.loc[0, "post_topic"] = "labelled by hand"
        template.to_csv(path, index=False, encoding="utf-8-sig")
        prepare(self.cfg, self.comments, self.work, self.posts)
        again = pd.read_csv(path, dtype=str).fillna("")
        self.assertEqual(again.loc[0, "post_topic"], "labelled by hand")

    def test_merge_finds_ai_posts_automatically(self):
        tmp = Path(self.tmpdir.name)
        work = tmp / "auto_posts_work"
        prepare(self.cfg, self.comments, work, self.posts, batches=False)
        key = pd.read_csv(work / F.POSTS_TABLE, dtype=str)["post_key"].iloc[0]
        (work / "ai_posts.csv").write_text(
            "post_key,topic,primary_frame_id,source_type_guess\n"
            f"{key},Automatic topic,{self.cfg.frame_ids[0]},\n",
            encoding="utf-8",
        )
        self.assertEqual(
            cli_main(
                [
                    "merge",
                    "--study",
                    str(TOY),
                    "--work",
                    str(work),
                    "--responses",
                    str(self.responses / "*.json"),
                ]
            ),
            0,
        )
        posts = pd.read_csv(work / F.POSTS_CLASSIFIED, dtype=str).fillna("")
        row = posts[posts.post_key == key]
        self.assertTrue((row["post_topic"] == "Automatic topic").any())

    def test_sources_period_and_batches(self):
        posts = pd.read_csv(self.work / F.POSTS_TABLE)
        self.assertEqual(
            posts.set_index("post_id").source_type.to_dict(),
            {1: "official", 2: "forum"},
        )
        self.assertEqual(
            posts.set_index("post_id").period.to_dict(), {1: "early", 2: "late"}
        )
        batches = sorted((self.work / F.BATCH_DIR).glob("*.json"))
        sizes = [len(json.loads(p.read_text())["comments_batch"]) for p in batches]
        self.assertTrue(all(n <= 5 for n in sizes))
        self.assertEqual(sum(sizes), 19)
        ids = {
            c["comment_id"]
            for p in batches
            for c in json.loads(p.read_text())["comments_batch"]
        }
        self.assertNotIn("a13", ids)  # institutional author is never sent
        self.assertNotIn("b8", ids)  # no text
        reply = next(
            c
            for p in batches
            for c in json.loads(p.read_text())["comments_batch"]
            if c["comment_id"] == "a10"
        )
        self.assertEqual(reply["parent_text"], "Accidents happen every week")
        self.assertTrue(list((self.work / F.BATCH_DIR).glob("*.user_message.txt")))

    def test_relevance_and_phases(self):
        enr = pd.read_csv(
            self.work / F.COMMENTS_ENRICHED, dtype={"comment_id": str}
        ).set_index("comment_id")
        self.assertEqual(
            enr.loc["a4", "relevance_tier"], "high"
        )  # 5 likes + 1 reply * 2 = 7 >= 4
        self.assertEqual(enr.loc["a4", "reply_count"], 1)
        self.assertEqual(
            enr.loc["a7", "relevance_tier"], "medium"
        )  # 2 likes: score 2 (>= 2, < 4)
        self.assertEqual(enr.loc["a1", "relevance_tier"], "low")
        self.assertEqual(enr.loc["a3", "relevance_score"], 2)  # 0 likes + 1 reply * 2
        first = enr[enr.post_id == 1].sort_values("chronological_order")
        self.assertEqual(first.phase.tolist()[:3], [1, 1, 1])
        self.assertEqual(
            enr.loc["a4", "phase"], 2
        )  # first high-relevance comment opens phase 2
        self.assertEqual(
            enr.loc["a13", "phase"], enr.loc["a12", "phase"]
        )  # institutional comment never opens a phase
        self.assertEqual(enr[enr.post_id == 2].phase.max(), 1)

    def test_pseudonyms_are_deterministic(self):
        enr = pd.read_csv(self.work / F.COMMENTS_ENRICHED)
        self.assertTrue(enr.user_pseudonym.str.match(r"^U\d{3}$").all())
        again = prepare(
            self.cfg,
            self.comments,
            Path(self.tmpdir.name) / "work2",
            self.posts,
            pseudonymize=True,
        )
        b = next((Path(self.tmpdir.name) / "work2" / F.BATCH_DIR).glob("*.json"))
        self.assertTrue(
            json.loads(b.read_text())["comments_batch"][0]["author"].startswith("U")
        )
        self.assertEqual(again["comments"], 21)

    def test_merge_validation_and_selection(self):
        r = self.report
        self.assertEqual(r["files"], 2)
        self.assertEqual(
            r["invalid_records"], 1
        )  # the thread/frame mismatch (the lower-case 'a' is coerced silently)
        self.assertEqual(r["unclassified_eligible"], 1)
        self.assertEqual(r["classified_ids_found_in_comments"], 1.0)
        coded = pd.read_csv(self.work / F.ALL_CODED, dtype={"comment_id": str})
        self.assertEqual(int(coded.selected.sum()), r["selected"])
        self.assertTrue(
            coded[coded.is_institutional].frame_id.eq(self.cfg.no_frame_id).all()
        )
        self.assertTrue(coded[~coded.has_text].frame_id.eq(self.cfg.no_frame_id).all())
        review = pd.read_csv(self.work / F.REVIEW)
        self.assertTrue(
            (review.review_reason.fillna("").str.contains("invalid model output")).any()
        )
        sel = pd.read_csv(self.work / F.SELECTED)
        self.assertTrue(
            {
                "frame_label",
                "thread_label",
                "post_topic",
                "stance",
                "evidence_level",
                "user_pseudonym",
            }
            <= set(sel.columns)
        )
        self.assertTrue((sel.frame_id != self.cfg.no_frame_id).all())

    def test_posts_table_and_phases_file(self):
        posts = pd.read_csv(self.work / F.POSTS_CLASSIFIED)
        self.assertEqual(posts.n_selected_comments.sum(), self.report["selected"])
        self.assertIn("dominant_comment_frame_id", posts.columns)
        phases = pd.read_csv(self.work / F.PHASES)
        self.assertEqual(phases.n_comments.sum(), 21)

    def test_users_contacts_and_communities(self):
        out = run_users(self.cfg, self.work)
        users = pd.read_csv(self.work / F.USERS)
        self.assertEqual(len(users), users.user_id.nunique())
        contacts = pd.read_csv(self.work / F.CONTACTS)
        self.assertTrue(
            contacts.contact_reasons.str.contains("self-declared:daily_cyclist").any()
        )
        self.assertNotIn("City Hall", contacts.user_name.tolist())
        city = users[users.user_name == "City Hall"].iloc[0]
        self.assertEqual(
            (city.top_actor_category, city.ecosystem_level),
            ("city_government", "authority"),
        )
        comm = pd.read_csv(self.work / F.COMMUNITY)
        old_town = comm[comm.location == "Old Town"].iloc[0]
        self.assertGreaterEqual(
            old_town.n_comments, 2
        )  # 'Old Town' and the alias 'Altstadt'
        self.assertEqual(set(comm.location), {"Old Town", "Riverside", "Springfield"})
        self.assertIn("ind_safety_concern", comm.columns)
        self.assertEqual(out["users"], len(users))

    def test_stats_gaps_and_compare(self):
        coded = load_coded(self.work)
        res, tables = compute_stats(coded, self.cfg)
        self.assertIn("T1", set(res.test_id))
        t1 = res[res.test_id == "T1"].iloc[0]
        self.assertEqual(t1.n, int(((~coded.is_institutional) & coded.has_text).sum()))
        two = res[res.kind.str.startswith("2x2")]
        self.assertTrue((two.p_holm >= two.p - 1e-12).all())
        self.assertTrue(two.p_holm.le(1).all())
        build_gap_report(self.cfg, self.work)
        gaps = json.loads((self.work / F.GAPS_JSON).read_text())
        self.assertTrue(any("neverseen" in g["action"] for g in gaps["gaps"]))
        self.assertEqual(gaps["deliverables"][0]["status"], "produced")
        self.assertTrue(
            any(
                g["area"] == "coding" and "reliability" in g["finding"]
                for g in gaps["gaps"]
            )
        )
        rep = run_compare(self.cfg, self.work, self.work, self.work)
        self.assertEqual(rep["frame_all_categories"]["kappa"], 1.0)

    def test_cli_end_to_end(self):
        tmp = Path(self.tmpdir.name)
        work = tmp / "cli_work"
        self.assertEqual(
            cli_main(
                [
                    "prepare",
                    "--study",
                    str(TOY),
                    "--comments",
                    str(self.comments),
                    "--posts",
                    str(self.posts),
                    "--out",
                    str(work),
                    "--no-messages",
                ]
            ),
            0,
        )
        self.assertTrue((work / "prompts" / "classifier_system.md").exists())
        self.assertEqual(
            cli_main(
                [
                    "merge",
                    "--study",
                    str(TOY),
                    "--work",
                    str(work),
                    "--responses",
                    str(self.responses / "*.json"),
                ]
            ),
            0,
        )
        self.assertEqual(
            cli_main(["analyze", "--study", str(TOY), "--work", str(work)]), 0
        )
        self.assertEqual(
            cli_main(
                [
                    "merge",
                    "--study",
                    str(TOY),
                    "--work",
                    str(work),
                    "--responses",
                    str(tmp / "nothing*.json"),
                ]
            ),
            2,
        )


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        self.cfg = load_study(TOY)
        c, p = make_toy_corpus(t)
        self.work = t / "work"
        prepare(self.cfg, c, self.work, p, batches=False)
        self.brief = t / "brief.json"
        self.brief.write_text(
            json.dumps(
                {
                    "seed_frames": [
                        {"label": "Safety", "argument": "Lanes are safer."},
                        {"label": "Cost", "argument": "Lanes cost too much."},
                    ]
                }
            )
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_seed_codebook_is_valid_and_loadable(self):
        seed = seed_codebook(json.loads(self.brief.read_text()))
        self.assertEqual([f["id"] for f in seed["frames"]], ["F1", "F2"])
        self.assertEqual(schema_errors(seed, "codebook"), [])
        modules, man, base = load_modules(TOY)
        modules["codebook"] = seed
        modules["locations"]["community_relevance"]["indicators"] = []
        cfg = config_from_modules(modules, man, base)
        self.assertEqual(cfg.thread_ids, [])
        self.assertIn("(no examples yet)", classifier_system_prompt(cfg))

    def test_round_input_samples_without_reuse_and_apply_validates(self):
        out = Path(self.tmp.name) / "disc"
        r1 = build_discovery_input(
            TOY, self.brief, self.work, out, 1, sample_size=8, seed=1
        )
        self.assertEqual(r1["sample"], 8)
        payload = json.loads((out / "discovery_round_01_input.json").read_text())
        ids1 = {s["comment_id"] for s in payload["sample"]}
        self.assertNotIn("a13", ids1)
        self.assertEqual(payload["research_brief"]["seed_frames"][0]["label"], "Safety")
        build_discovery_input(TOY, self.brief, self.work, out, 2, sample_size=8, seed=1)
        ids2 = {
            s["comment_id"]
            for s in json.loads((out / "discovery_round_02_input.json").read_text())[
                "sample"
            ]
        }
        self.assertFalse(ids1 & ids2)
        self.assertTrue(
            (out / "discovery_round_02_user_message.md")
            .read_text()
            .startswith("<round>2</round>")
        )
        # a valid response adds one thread to the first configured frame
        new_thread = f"{self.cfg.frames[0]['id']}_3"
        cb = copy.deepcopy(self.cfg.codebook)
        cb["frames"][0]["threads"].append(
            {
                "id": new_thread,
                "label": "Night riding",
                "description": "Lanes need lighting.",
            }
        )
        resp = {
            "round": 1,
            "codebook": cb,
            "changelog": [
                {
                    "op": "add_thread",
                    "target": new_thread,
                    "reason": "new argument",
                    "evidence_comment_ids": ["a1"],
                }
            ],
            "saturation": {
                "new_threads_this_round": 1,
                "sample_coverage_share": 0.9,
                "is_saturated": False,
            },
        }
        rp = Path(self.tmp.name) / "resp.json"
        rp.write_text(json.dumps(resp))
        summary = apply_discovery(
            TOY, rp, Path(self.tmp.name) / "codebook.new.json", out
        )
        self.assertEqual(summary["threads"], len(self.cfg.thread_ids) + 1)
        self.assertTrue((out / "codebook_history.jsonl").exists())
        # a silent deletion is rejected
        cb2 = copy.deepcopy(self.cfg.codebook)
        cb2["frames"][0]["threads"].pop(0)
        resp2 = dict(
            resp,
            codebook=cb2,
            changelog=[{"op": "no_change", "target": "-", "reason": "none"}],
        )
        rp.write_text(json.dumps(resp2))
        with self.assertRaises(ConfigError):
            apply_discovery(TOY, rp, Path(self.tmp.name) / "codebook.bad.json", out)
        # a malformed response is rejected
        rp.write_text(json.dumps({"round": 1}))
        with self.assertRaises(ConfigError):
            apply_discovery(TOY, rp, Path(self.tmp.name) / "codebook.bad2.json", out)


class CorpusDetectionTests(unittest.TestCase):
    """`--corpus-dir` finds the input CSVs from their header, not their filename."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls.tmpdir.name)
        cls.cfg = load_study(TOY)

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def _corpus(self, name: str) -> tuple[Path, Path, Path]:
        folder = self.tmp / name
        folder.mkdir()
        comments, posts = make_toy_corpus(folder)
        return folder, comments, posts

    def test_detects_both_files_from_header(self):
        folder, comments, posts = self._corpus("both")
        renamed = folder / "dataset_facebook-posts-scraper_2026-09-29.csv"
        posts.rename(renamed)
        # A previous run's deliverable must never be mistaken for an input.
        pd.read_csv(comments).to_csv(folder / F.COMMENTS_ENRICHED, index=False)
        found_comments, found_posts = detect_corpus(folder, self.cfg)
        self.assertEqual(found_comments.resolve(), comments.resolve())
        self.assertEqual(found_posts.resolve(), renamed.resolve())

    def test_posts_are_optional(self):
        folder, comments, posts = self._corpus("comments_only")
        posts.unlink()
        found_comments, found_posts = detect_corpus(folder, self.cfg)
        self.assertEqual(found_comments.resolve(), comments.resolve())
        self.assertIsNone(found_posts)

    def test_ambiguity_and_missing_folders_raise(self):
        folder, comments, _ = self._corpus("two_comments")
        pd.read_csv(comments).to_csv(folder / "duplicate_comments.csv", index=False)
        with self.assertRaises(ValueError):
            detect_corpus(folder, self.cfg)
        empty = self.tmp / "empty"
        empty.mkdir()
        with self.assertRaises(ValueError):
            detect_corpus(empty, self.cfg)
        with self.assertRaises(FileNotFoundError):
            detect_corpus(self.tmp / "missing", self.cfg)

    def test_cli_prepare_with_corpus_dir(self):
        folder, _, _ = self._corpus("cli")
        work = self.tmp / "cli_work"
        code = cli_main(
            [
                "prepare",
                "--study",
                str(TOY),
                "--corpus-dir",
                str(folder),
                "--out",
                str(work),
                "--no-batches",
                "--no-messages",
            ]
        )
        self.assertEqual(code, 0)
        self.assertTrue((work / F.COMMENTS_ENRICHED).exists())
        rep = json.loads((work / F.JOIN_REPORT).read_text())
        self.assertEqual(rep["strategies"]["numeric_id"]["matched_comments"], 21)


class NetworkViewerTests(unittest.TestCase):
    """`network.html` must accept exactly the columns `export-network` writes."""

    @classmethod
    def setUpClass(cls):
        cls.html = (ROOT / "network.html").read_text(encoding="utf-8")

    def test_required_columns_match_exporter(self):
        block = self.html.split("const REQUIRED_COLUMNS = [", 1)[1].split("];", 1)[0]
        columns = re.findall(r'"([a-z_]+)"', block)
        self.assertEqual(columns, list(NETWORK_COLUMNS))

    def test_error_messages_name_a_real_command(self):
        self.assertIn("python -m framing export-network", self.html)
        group = next(a for a in build_parser()._actions if a.dest == "command")
        self.assertIn("export-network", group.choices)


if __name__ == "__main__":
    unittest.main()
