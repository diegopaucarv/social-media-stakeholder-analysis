"""Tests for the guided mode (`python -m framing` with no arguments).

Offline only: no network access, no study edits, and every prompt is answered by
a scripted ``input_fn`` so nothing ever blocks on a real terminal.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from framing import files as F  # noqa: E402
from framing.cli import main as cli_main  # noqa: E402
from framing.wizard import (  # noqa: E402
    WORKFLOW,
    analyze_argv,
    classify_api_argv,
    discover_studies,
    explain_chatbot_branch,
    export_md_argv,
    find_csvs,
    init_argv,
    merge_argv,
    network_argv,
    next_stage,
    prepare_argv,
    render_status,
    run_wizard,
    stage_status,
)

TOY = ROOT / "studies" / "toy_bikelanes" / "manifest.json"


class Scripted:
    """Stand-in for ``input()`` that replays a list and then stops the wizard."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if not self.answers:
            raise EOFError
        return self.answers.pop(0)


class FakeRunner:
    """Collect the argv lists the wizard would pass to ``cli.main``."""

    def __init__(self, code=0):
        self.calls = []
        self.code = code

    def __call__(self, argv):
        self.calls.append(list(argv))
        return self.code


def collect():
    lines = []
    return lines, lines.append


# ----------------------------------------------------------------------------- argv builders


class TestArgvBuilders(unittest.TestCase):
    def test_init(self):
        self.assertEqual(
            init_argv("s1", "My study", "en", "studies/s1"),
            [
                "init",
                "--id",
                "s1",
                "--title",
                "My study",
                "--language",
                "en",
                "--dir",
                "studies/s1",
            ],
        )
        self.assertEqual(init_argv("s1", "T", force=True)[-1], "--force")

    def test_prepare_minimal_and_full(self):
        self.assertEqual(
            prepare_argv("m.json", "c.csv", "work/x"),
            ["prepare", "--study", "m.json", "--comments", "c.csv", "--out", "work/x"],
        )
        full = prepare_argv(
            "m.json",
            "c.csv",
            "work/x",
            posts="p.csv",
            post_meta="pm.csv",
            pseudonymize_authors=True,
            no_batches=True,
            no_messages=True,
        )
        self.assertIn("--posts", full)
        self.assertIn("p.csv", full)
        self.assertIn("--pseudonymize-authors", full)
        self.assertIn("--no-batches", full)
        self.assertIn("--no-messages", full)

    def test_prepare_with_corpus_dir(self):
        self.assertEqual(
            prepare_argv("m.json", None, "work/x", corpus_dir="studies/s"),
            [
                "prepare",
                "--study",
                "m.json",
                "--corpus-dir",
                "studies/s",
                "--out",
                "work/x",
            ],
        )

    def test_classify_branches(self):
        self.assertEqual(
            export_md_argv("m.json", "work/x", "work/x/classification.md"),
            [
                "export-md",
                "--study",
                "m.json",
                "--work",
                "work/x",
                "--out",
                "work/x/classification.md",
            ],
        )
        api = classify_api_argv("m.json", "work/x", "work/x/responses", model="m")
        self.assertIn("--enable-api", api)
        self.assertIn("--model", api)
        self.assertNotIn("--api-key", api)

    def test_merge_includes_posts_and_accepts_one_path(self):
        argv = merge_argv("m.json", "work/x", "a.csv", "p.csv", out="work/x")
        self.assertEqual(argv.count("--ai-responses"), 1)
        self.assertIn("a.csv", argv)
        self.assertIn("--post-ai-responses", argv)
        self.assertIn("p.csv", argv)
        argv = merge_argv("m.json", "work/x", ["a.csv", "b.csv"])
        self.assertEqual(argv.index("a.csv") + 1, argv.index("b.csv"))

    def test_analyze_and_network(self):
        self.assertEqual(
            analyze_argv("m.json", "work/x", "work/x"),
            ["analyze", "--study", "m.json", "--work", "work/x", "--out", "work/x"],
        )
        self.assertEqual(
            network_argv("m.json", "work/x", "work/x/11_network.csv"),
            [
                "export-network",
                "--study",
                "m.json",
                "--work",
                "work/x",
                "--out",
                "work/x/11_network.csv",
            ],
        )


# ----------------------------------------------------------------------------- status


class TestStageStatus(unittest.TestCase):
    def test_fresh_folder_is_all_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            statuses = stage_status(tmp)
            self.assertEqual(set(statuses), {key for key, _l, _d in WORKFLOW})
            self.assertTrue(all(value == "missing" for value in statuses.values()))
            self.assertEqual(next_stage(tmp), "prepare")
            self.assertIn("TODO", render_status(tmp))

    def test_stage_progression(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / F.COMMENTS_ENRICHED).write_text("id\n", encoding="utf-8")
            self.assertEqual(stage_status(tmp)["prepare"], "done")
            self.assertEqual(next_stage(tmp), "classify")

            (work / "classification.md").write_text("# bundle\n", encoding="utf-8")
            self.assertEqual(stage_status(tmp)["classify"], "waiting")
            self.assertEqual(next_stage(tmp), "classify")

            (work / "ai_output.csv").write_text("comment_id\n", encoding="utf-8")
            self.assertEqual(stage_status(tmp)["classify"], "done")

            (work / F.ALL_CODED).write_text("id\n", encoding="utf-8")
            (work / F.STATS).write_text("x\n", encoding="utf-8")
            (work / F.NETWORK).write_text("x\n", encoding="utf-8")
            self.assertIsNone(next_stage(tmp))
            self.assertIn("DONE", render_status(tmp))

    def test_api_answers_count_as_classified(self):
        with tempfile.TemporaryDirectory() as tmp:
            responses = Path(tmp) / "responses"
            responses.mkdir()
            (responses / "api_answers.json").write_text("[]", encoding="utf-8")
            self.assertEqual(stage_status(tmp)["classify"], "done")


# ----------------------------------------------------------------------------- discovery helpers


class TestDiscoveryHelpers(unittest.TestCase):
    def test_discover_studies_finds_the_toy_fixture(self):
        manifests = discover_studies()
        self.assertIn(TOY, manifests)
        self.assertTrue(all(path.name == "manifest.json" for path in manifests))

    def test_find_csvs_skips_pipeline_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "raw_comments.csv").write_text("a\n", encoding="utf-8")
            (root / "raw_posts.csv").write_text("a\n", encoding="utf-8")
            (root / F.ALL_CODED).write_text("a\n", encoding="utf-8")
            (root / "ai_output.csv").write_text("a\n", encoding="utf-8")
            names = [path.name for path in find_csvs(root)]
            self.assertEqual(names, ["raw_comments.csv", "raw_posts.csv"])


# ----------------------------------------------------------------------------- run_wizard


class TestRunWizard(unittest.TestCase):
    def test_quit_at_the_menu_returns_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Scripted(["q"])
            lines, out = collect()
            code = run_wizard(TOY, tmp, input_fn=script, out=out, runner=FakeRunner())
            self.assertEqual(code, 0)
            text = "\n".join(lines)
            self.assertIn("guided workflow", text)
            self.assertIn("Work folder", text)
            self.assertIn("Todo".upper(), text.upper())

    def test_eof_cancels_without_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            lines, out = collect()
            code = run_wizard(
                TOY, tmp, input_fn=Scripted([]), out=out, runner=FakeRunner()
            )
            self.assertEqual(code, 0)
            self.assertIn("Stopped", "\n".join(lines))

    def test_chatbot_branch_runs_export_md(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = FakeRunner()
            # menu -> 2, branch -> Enter (chatbot default), menu -> q
            script = Scripted(["2", "", "q"])
            lines, out = collect()
            code = run_wizard(TOY, tmp, input_fn=script, out=out, runner=runner)
            self.assertEqual(code, 0)
            self.assertEqual(len(runner.calls), 1)
            self.assertEqual(runner.calls[0][0], "export-md")
            self.assertIn("HOW TO GET THE AI ANSWERS", "\n".join(lines))
            self.assertIn("comment_id,is_institutional", "\n".join(lines))

    def test_merge_picks_up_ai_output_and_posts(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / "ai_output.csv").write_text("comment_id\n", encoding="utf-8")
            (work / "ai_posts.csv").write_text("post_key\n", encoding="utf-8")
            runner = FakeRunner()
            # menu -> 3, answer path -> Enter, menu -> q
            script = Scripted(["3", "", "q"])
            lines, out = collect()
            code = run_wizard(TOY, tmp, input_fn=script, out=out, runner=runner)
            self.assertEqual(code, 0)
            self.assertEqual(len(runner.calls), 1)
            argv = runner.calls[0]
            self.assertEqual(argv[0], "merge")
            self.assertIn("--post-ai-responses", argv)
            self.assertIn(str(work / "ai_posts.csv"), argv)

    def test_analyze_and_network_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = FakeRunner()
            # menu -> 4, menu -> 5, menu -> q
            script = Scripted(["4", "5", "q"])
            lines, out = collect()
            code = run_wizard(TOY, tmp, input_fn=script, out=out, runner=runner)
            self.assertEqual(code, 0)
            self.assertEqual(
                [call[0] for call in runner.calls], ["analyze", "export-network"]
            )
            self.assertIn("Final deliverable", "\n".join(lines))

    def test_failed_step_is_reported_and_the_menu_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = FakeRunner(code=2)
            script = Scripted(["4", "q"])
            lines, out = collect()
            code = run_wizard(TOY, tmp, input_fn=script, out=out, runner=runner)
            self.assertEqual(code, 0)
            self.assertIn("did not finish", "\n".join(lines))


class TestExplainAnswerFormat(unittest.TestCase):
    def test_explains_answer_file_and_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            lines = []
            explain_chatbot_branch(tmp, out=lines.append)
            text = "\n".join(lines)
            self.assertIn("ai_output.csv", text)
            self.assertIn("comment_id,is_institutional", text)

    def test_mentions_the_post_answer_only_when_post_batches_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            lines = []
            explain_chatbot_branch(tmp, out=lines.append)
            self.assertNotIn("ai_posts.csv", "\n".join(lines))

            (Path(tmp) / F.POST_BATCH_DIR).mkdir()
            lines = []
            explain_chatbot_branch(tmp, out=lines.append)
            self.assertIn("ai_posts.csv", "\n".join(lines))


class TestCliRouting(unittest.TestCase):
    def test_no_arguments_enters_the_wizard(self):
        with mock.patch("framing.wizard.run_wizard", return_value=0) as wizard:
            self.assertEqual(cli_main([]), 0)
            wizard.assert_called_once_with()

    def test_explicit_subcommand_still_works(self):
        self.assertEqual(cli_main(["validate-config", "--study", str(TOY)]), 0)


if __name__ == "__main__":
    unittest.main()
