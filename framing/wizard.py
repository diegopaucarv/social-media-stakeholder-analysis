"""Interactive guided front-end for `python -m framing` with no arguments.

The wizard asks for the study, the input CSVs and the work folder, shows which
steps are already done, and then runs the same commands as the CLI. It never
performs network access on its own: the chatbot branch is the default, and the
API branch stays off unless the user explicitly turns it on for one run and
provides a key.

Everything here is pure Python on top of :mod:`framing.cli`: the argv builders
are pure functions, and ``run_wizard`` accepts injectable ``input_fn`` / ``out``
/ ``runner`` callables so the whole flow is testable without a terminal.
"""

from __future__ import annotations

import os
from pathlib import Path

from . import files as F
from .config import PKG_ROOT

__all__ = [
    "WORKFLOW",
    "analyze_argv",
    "classify_api_argv",
    "discover_studies",
    "explain_chatbot_branch",
    "export_md_argv",
    "find_csvs",
    "init_argv",
    "merge_argv",
    "network_argv",
    "next_stage",
    "prepare_argv",
    "render_status",
    "run_wizard",
    "stage_status",
    "validate_argv",
]

#: The five user-visible stages, in order: (key, label, one-line description).
WORKFLOW = (
    ("prepare", "prepare", "Link posts and comments, cut the batches"),
    (
        "classify",
        "classify",
        "Bundle the task for a chatbot (default), or call the API (opt-in)",
    ),
    ("merge", "merge", "Ingest the AI answers into coded tables"),
    ("analyze", "analyze", "User profiles, statistics and gap report"),
    ("network", "network", "Write 11_network.csv for network.html"),
)

BUNDLE = "classification.md"
TEMPLATE_CSV = "classification.template.csv"
AI_ANSWER = "ai_output.csv"
POST_ANSWER = "ai_posts.csv"
API_ANSWERS = Path("responses") / "api_answers.json"

# Filenames the pipeline writes. They are excluded when guessing input CSVs so
# the user is never offered its own output as an input.
_OUTPUT_NAMES = {
    F.COMMENTS_ENRICHED,
    F.POSTS_TABLE,
    F.SELECTED,
    F.ALL_CODED,
    F.POSTS_CLASSIFIED,
    F.USERS,
    F.CONTACTS,
    F.PHASES,
    F.REVIEW,
    F.COMMUNITY,
    F.STATS,
    F.COVERAGE,
    F.NETWORK,
    AI_ANSWER,
    POST_ANSWER,
    TEMPLATE_CSV,
    "classification.posts.template.csv",
}

_STATUS_SYMBOL = {"done": "DONE   ", "waiting": "WAITING", "missing": "TODO   "}


class _Cancel(Exception):
    """Raised internally when the user stops the wizard."""


# ----------------------------------------------------------------------------- prompts


def _ask(prompt, default=None, *, input_fn=input, out=print, allow_empty=False):
    """Ask for one value. Enter accepts ``default``; blank is rejected otherwise."""
    suffix = f" [{default}]" if default not in (None, "") else ""
    while True:
        try:
            raw = input_fn(f"{prompt}{suffix}: ")
        except (EOFError, KeyboardInterrupt):
            raise _Cancel() from None
        value = (raw or "").strip()
        if value:
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            return value
        if default is not None:
            return str(default)
        if allow_empty:
            return ""
        out("  (please type a value)")


def _ask_yes_no(prompt, default=False, *, input_fn=input, out=print):
    """Ask a yes/no question. Enter returns ``default``."""
    hint = "Y/n" if default else "y/N"
    while True:
        try:
            raw = input_fn(f"{prompt} ({hint}): ")
        except (EOFError, KeyboardInterrupt):
            raise _Cancel() from None
        value = (raw or "").strip().lower()
        if not value:
            return bool(default)
        if value in ("y", "yes"):
            return True
        if value in ("n", "no"):
            return False
        out("  please answer y or n")


def _ask_choice(prompt, choices, *, default=None, input_fn=input, out=print):
    """Show numbered ``choices`` (key, label) and return the chosen key."""
    default_number = None
    if default is not None:
        for index, (key, _) in enumerate(choices, start=1):
            if key == default:
                default_number = str(index)
                break
    out("")
    for index, (key, label) in enumerate(choices, start=1):
        mark = "  <-- default" if key == default else ""
        out(f"  {index}. {label}{mark}")
    while True:
        value = _ask(
            f"{prompt} (number)", default=default_number, input_fn=input_fn, out=out
        )
        value = value.strip()
        for key, _ in choices:
            if value == key:
                return key
        if value.isdigit() and 1 <= int(value) <= len(choices):
            return choices[int(value) - 1][0]
        out(f"  choose a number from 1 to {len(choices)}")


def _ask_path(
    prompt,
    default=None,
    *,
    must_exist=False,
    allow_empty=False,
    input_fn=input,
    out=print,
):
    """Ask for a path, optionally requiring that it already exists."""
    while True:
        value = _ask(
            prompt,
            default=default,
            input_fn=input_fn,
            out=out,
            allow_empty=allow_empty,
        )
        if value == "":
            return None
        candidate = Path(value).expanduser()
        if not must_exist or candidate.exists():
            return value
        out(f"  not found: {candidate}")
        default = None


# ----------------------------------------------------------------------------- discovery


def discover_studies(root=None):
    """Return the study manifests under ``<root>/studies``, sorted by name."""
    base = Path(root) if root is not None else PKG_ROOT
    studies_dir = base / "studies"
    if not studies_dir.exists():
        return []
    return sorted(
        path
        for path in studies_dir.glob("*/manifest.json")
        if not path.parent.name.startswith((".", "_"))
    )


def find_csvs(*roots, limit=10):
    """Return up to ``limit`` input-looking CSVs under ``roots`` (default: cwd)."""
    search = [Path(root) for root in roots] or [Path.cwd()]
    found = []
    for root in search:
        if not root.exists():
            continue
        for path in sorted(root.glob("*.csv")):
            if path.name in _OUTPUT_NAMES:
                continue
            found.append(path)
            if len(found) >= limit:
                return found
    return found


def _guess_csv(hint):
    """Return one CSV in the current folder whose name contains ``hint``."""
    for path in find_csvs(Path.cwd()):
        if hint in path.name.lower():
            return str(path)
    return None


# ----------------------------------------------------------------------------- status


def _classify_status(work):
    if (work / AI_ANSWER).exists() or (work / API_ANSWERS).exists():
        return "done"
    if (work / BUNDLE).exists():
        return "waiting"
    return "missing"


def stage_status(work):
    """Map each stage key to ``"done"``, ``"waiting"`` or ``"missing"``.

    ``waiting`` is only used by ``classify``: the bundle exists but the AI
    answer CSV has not been saved yet.
    """
    work = Path(work)
    return {
        "prepare": "done" if (work / F.COMMENTS_ENRICHED).exists() else "missing",
        "classify": _classify_status(work),
        "merge": "done" if (work / F.ALL_CODED).exists() else "missing",
        "analyze": "done" if (work / F.STATS).exists() else "missing",
        "network": "done" if (work / F.NETWORK).exists() else "missing",
    }


def next_stage(work):
    """Return the key of the first stage that is not done, or ``None``."""
    statuses = stage_status(work)
    for key, _label, _desc in WORKFLOW:
        if statuses[key] != "done":
            return key
    return None


def render_status(work):
    """Render the stage table as plain text (no I/O, so it is easy to test)."""
    work = Path(work)
    statuses = stage_status(work)
    lines = [f"Work folder: {work}", "  #  Step      Status   What it does"]
    for index, (key, _label, description) in enumerate(WORKFLOW, start=1):
        symbol = _STATUS_SYMBOL[statuses[key]]
        lines.append(f"  {index}.  {key:<9} {symbol}  {description}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- argv builders


def init_argv(study_id, title, language="en", out_dir=None, force=False):
    argv = [
        "init",
        "--id",
        str(study_id),
        "--title",
        str(title),
        "--language",
        str(language),
    ]
    if out_dir:
        argv += ["--dir", str(out_dir)]
    if force:
        argv.append("--force")
    return argv


def validate_argv(study):
    return ["validate-config", "--study", str(study)]


def prepare_argv(
    study,
    comments,
    work,
    posts=None,
    post_meta=None,
    *,
    corpus_dir=None,
    pseudonymize_authors=False,
    no_batches=False,
    no_messages=False,
):
    argv = ["prepare", "--study", str(study)]
    if comments:
        argv += ["--comments", str(comments)]
    if corpus_dir:
        argv += ["--corpus-dir", str(corpus_dir)]
    argv += ["--out", str(work)]
    if posts:
        argv += ["--posts", str(posts)]
    if post_meta:
        argv += ["--post-meta", str(post_meta)]
    if no_batches:
        argv.append("--no-batches")
    if no_messages:
        argv.append("--no-messages")
    if pseudonymize_authors:
        argv.append("--pseudonymize-authors")
    return argv


def export_md_argv(study, work, out):
    return ["export-md", "--study", str(study), "--work", str(work), "--out", str(out)]


def classify_api_argv(study, work, out, *, base_url=None, model=None):
    argv = [
        "classify-api",
        "--study",
        str(study),
        "--work",
        str(work),
        "--out",
        str(out),
        "--enable-api",
    ]
    if base_url:
        argv += ["--base-url", str(base_url)]
    if model:
        argv += ["--model", str(model)]
    return argv


def merge_argv(
    study,
    work,
    responses,
    post_responses=None,
    *,
    out=None,
    min_confidence=None,
    require_coverage=None,
):
    if isinstance(responses, (str, Path)):
        responses = [responses]
    argv = ["merge", "--study", str(study), "--work", str(work), "--ai-responses"]
    argv += [str(item) for item in responses]
    if post_responses:
        if isinstance(post_responses, (str, Path)):
            post_responses = [post_responses]
        argv += ["--post-ai-responses"]
        argv += [str(item) for item in post_responses]
    if out:
        argv += ["--out", str(out)]
    if min_confidence is not None:
        argv += ["--min-confidence", str(min_confidence)]
    if require_coverage is not None:
        argv += ["--require-coverage", str(require_coverage)]
    return argv


def analyze_argv(study, work, out=None):
    argv = ["analyze", "--study", str(study), "--work", str(work)]
    if out:
        argv += ["--out", str(out)]
    return argv


def network_argv(study, work, out=None):
    argv = ["export-network", "--study", str(study), "--work", str(work)]
    if out:
        argv += ["--out", str(out)]
    return argv


# ----------------------------------------------------------------------------- chatbot help


def _response_header(work):
    """The comment-answer CSV header this work folder asks for.

    ``export-md`` writes a header-only template trimmed to the study's selected
    fields; prefer it so the printed instructions match the file the user has.
    Fall back to the package default when the template is missing or unusable.
    """
    from .export_md import RESPONSE_CSV_HEADER

    template = Path(work) / TEMPLATE_CSV
    try:
        first = template.read_text(encoding="utf-8-sig").splitlines()[0]
    except (OSError, IndexError):
        return list(RESPONSE_CSV_HEADER)
    cols = [c.strip() for c in first.split(",")]
    return cols if any(cols) else list(RESPONSE_CSV_HEADER)


def explain_chatbot_branch(work, out=print):
    """Print what to do with ``classification.md`` and where to save the answer."""
    work = Path(work)
    header = _response_header(work)
    bundle = work / BUNDLE
    answer = work / AI_ANSWER
    post_answer = work / POST_ANSWER
    has_posts = (work / F.POST_BATCH_DIR).exists()

    out("")
    out("HOW TO GET THE AI ANSWERS (chatbot branch)")
    out(f"  1. Open {bundle} and copy the whole file.")
    out("  2. Paste it into any chatbot as a single message.")
    out(f"  3. Save the chatbot reply as {answer}")
    out("     The first line of that CSV must be exactly:")
    out("     " + ",".join(header))
    step = 4
    if has_posts:
        out(
            f"  {step}. The bundle also has a post section: save that reply as {post_answer}"
        )
        step += 1
    out(f"  {step}. Come back here and choose step 3 (merge) to ingest the answers.")
    out("")


# ----------------------------------------------------------------------------- wizard flow


def _run(runner, argv, out):
    """Run one CLI command through ``runner`` and report a failure clearly."""
    out("")
    out("$ python -m framing " + " ".join(argv))
    code = runner(argv)
    if code:
        out("")
        out(
            f"That step did not finish (exit code {code}). Fix the message above and try again."
        )
    return code


def _stage_from_choice(choice):
    if choice.isdigit() and 1 <= int(choice) <= len(WORKFLOW):
        return WORKFLOW[int(choice) - 1][0]
    return None


def _choose_study(manifest, *, input_fn, out, runner):
    if manifest:
        return str(Path(manifest))

    studies = discover_studies()
    out("")
    out("Which study do you want to work on?")
    for index, path in enumerate(studies, start=1):
        out(f"  {index}. {path.parent.name}")
    out("  n. create a new study")
    out("  q. quit")

    while True:
        choice = (
            _ask(
                "Study",
                default="1" if studies else "n",
                input_fn=input_fn,
                out=out,
            )
            .strip()
            .lower()
        )
        if choice in ("q", "quit", "exit"):
            raise _Cancel()
        if choice in ("n", "new"):
            study_id = _ask(
                "New study id (short, no spaces)", input_fn=input_fn, out=out
            )
            title = _ask("Study title", input_fn=input_fn, out=out)
            language = _ask(
                "Corpus language code",
                default="en",
                input_fn=input_fn,
                out=out,
            )
            out_dir = PKG_ROOT / "studies" / study_id
            if _run(runner, init_argv(study_id, title, language, out_dir), out):
                continue
            return str(out_dir / "manifest.json")
        if choice.isdigit() and 1 <= int(choice) <= len(studies):
            return str(studies[int(choice) - 1])
        out(f"  choose a number from 1 to {len(studies)}, n for a new study, or q")


def _choose_work(study, work, *, input_fn, out):
    if work:
        return str(Path(work))
    study_id = Path(study).resolve().parent.name
    default = str(PKG_ROOT / "work" / study_id)
    out("")
    return _ask(
        "Work folder (created if it does not exist)",
        default=default,
        input_fn=input_fn,
        out=out,
    )


def _stage_prepare(study, work, *, input_fn, out, runner):
    out("Step 1 - prepare: point the pipeline at your two input CSVs.")
    study_dir = Path(study).resolve().parent
    has_csvs = study_dir.exists() and any(study_dir.glob("*.csv"))
    corpus_dir = _ask_path(
        "Corpus folder (auto-detects the CSVs; blank = type the file names)",
        default=str(study_dir) if has_csvs else None,
        allow_empty=True,
        input_fn=input_fn,
        out=out,
    )
    comments = posts = None
    if not corpus_dir:
        comments = _ask_path(
            "Comments CSV",
            default=_guess_csv("comment"),
            must_exist=True,
            input_fn=input_fn,
            out=out,
        )
        posts = _ask_path(
            "Posts CSV (leave blank if you have none)",
            default=_guess_csv("post"),
            allow_empty=True,
            input_fn=input_fn,
            out=out,
        )
    pseudonymize = _ask_yes_no(
        "Send pseudonyms instead of author names to the AI?",
        default=False,
        input_fn=input_fn,
        out=out,
    )
    argv = prepare_argv(
        study,
        comments,
        work,
        posts=posts,
        corpus_dir=corpus_dir,
        pseudonymize_authors=pseudonymize,
    )
    _run(runner, argv, out)
    template = Path(work) / F.POST_META_TEMPLATE
    if template.exists():
        out("")
        out(f"Optional: label posts in {template}")
        out("  Fill post_topic and post_frame_id, then re-run this step with ")
        out(f"  --post-meta {template} to apply your labels.")


def _stage_classify(study, work, *, input_fn, out, runner):
    from .api import API_KEY_ENV, DEFAULT_BASE_URL, DEFAULT_MODEL

    work = Path(work)
    bundle = work / BUNDLE

    out("Step 2 - classify: the AI reads the prepared batches and returns one CSV.")
    branch = _ask_choice(
        "Which branch?",
        [
            (
                "chatbot",
                "Chatbot - write a .md, paste it, save the answer CSV (default)",
            ),
            ("api", "API - call the model directly (needs a key)"),
        ],
        default="chatbot",
        input_fn=input_fn,
        out=out,
    )

    if branch == "chatbot":
        rebuild = True
        if bundle.exists():
            rebuild = _ask_yes_no(
                f"{bundle} already exists. Rebuild it?",
                default=False,
                input_fn=input_fn,
                out=out,
            )
        if rebuild:
            _run(runner, export_md_argv(study, work, bundle), out)
        explain_chatbot_branch(work, out=out)
        return

    out("")
    out("API calls are OFF by default. This switch applies to this run only.")
    if not _ask_yes_no(
        "Turn API calls ON for this run?",
        default=False,
        input_fn=input_fn,
        out=out,
    ):
        out("API stays off. Use the chatbot branch (step 2) instead.")
        return

    key = os.environ.get(API_KEY_ENV, "")
    if not key:
        key = _ask(
            f"{API_KEY_ENV} (visible while you type)",
            input_fn=input_fn,
            out=out,
        )
        if not key:
            out(f"  no key: set {API_KEY_ENV} and try again. API stays off.")
            return
        os.environ[API_KEY_ENV] = key

    model = _ask("Model", default=DEFAULT_MODEL, input_fn=input_fn, out=out)
    base_url = _ask("Base URL", default=DEFAULT_BASE_URL, input_fn=input_fn, out=out)
    argv = classify_api_argv(
        study,
        work,
        work / "responses",
        base_url=base_url,
        model=model,
    )
    _run(runner, argv, out)


def _stage_merge(study, work, *, input_fn, out, runner):
    work = Path(work)
    default = work / AI_ANSWER
    if not default.exists() and (work / API_ANSWERS).exists():
        default = work / API_ANSWERS
    answer = _ask_path(
        "AI answer file (the CSV the chatbot gave you)",
        default=str(default) if default.exists() else None,
        must_exist=True,
        input_fn=input_fn,
        out=out,
    )
    post = work / POST_ANSWER
    post_answers = [post] if post.exists() else None
    if post_answers:
        out(f"Also ingesting the post answers: {post}")
    _run(runner, merge_argv(study, work, [answer], post_answers, out=work), out)


def _stage_analyze(study, work, *, out, runner):
    out("Step 4 - analyze: user profiles, statistics and the gap report.")
    _run(runner, analyze_argv(study, work, work), out)


def _stage_network(study, work, *, out, runner):
    work = Path(work)
    csv_path = work / F.NETWORK
    out("Step 5 - network: flatten the coding into the viewer CSV.")
    _run(runner, network_argv(study, work, csv_path), out)
    out("")
    out("Final deliverable:")
    out(f"  CSV : {csv_path}")
    out(f"  View: {PKG_ROOT / 'network.html'}")
    out("Open network.html in a browser and upload the CSV.")
    out("The viewer is self-contained: no server and no internet needed.")


def run_wizard(manifest=None, work=None, *, input_fn=input, out=print, runner=None):
    """Run the guided workflow.

    Args:
        manifest: Path to a ``manifest.json``; when omitted the user picks or
            creates a study.
        work: Work folder; when omitted it defaults to ``work/<study_id>``.
        input_fn: Callable used for every prompt (injectable for tests).
        out: Callable used for every message (injectable for tests).
        runner: Callable that accepts an argv list and returns an exit code;
            defaults to :func:`framing.cli.main`.

    Returns:
        ``0`` on a normal exit, including a user cancel.
    """
    if runner is None:
        from .cli import main as cli_main

        runner = cli_main

    try:
        return _run_wizard(
            manifest,
            work,
            input_fn=input_fn,
            out=out,
            runner=runner,
        )
    except _Cancel:
        out("")
        out("Stopped. Nothing was deleted; run `python -m framing` again to continue.")
        return 0


def _run_wizard(manifest, work, *, input_fn, out, runner):
    out("")
    out("FRAMING - guided workflow")
    out("=========================")
    out("input CSV -> prepared CSV -> AI answers -> analysis -> network.html")
    out("API calls are OFF by default: the default branch writes a .md file")
    out("that you paste into a chatbot yourself.")
    out("")

    study = _choose_study(manifest, input_fn=input_fn, out=out, runner=runner)
    work = _choose_work(study, work, input_fn=input_fn, out=out)

    while True:
        out("")
        out(render_status(work))
        suggested = next_stage(work)
        if suggested is None:
            out("")
            out("Everything is done.")
            out(
                f"Open {PKG_ROOT / 'network.html'} and upload {Path(work) / F.NETWORK}."
            )
        out("")
        out("What do you want to do?")
        for index, (key, label, description) in enumerate(WORKFLOW, start=1):
            mark = "  <-- suggested" if key == suggested else ""
            out(f"  {index}. {label:<9} {description}{mark}")
        out("  q. quit")

        choice = _ask(
            "Step",
            default=str(_stage_index(suggested)) if suggested else "q",
            input_fn=input_fn,
            out=out,
        )
        choice = choice.strip().lower()
        if choice in ("q", "quit", "exit"):
            out("")
            out("Bye. Run `python -m framing` again whenever you want to continue.")
            return 0

        key = _stage_from_choice(choice)
        if key is None:
            out(f"  choose a number from 1 to {len(WORKFLOW)}, or q")
            continue

        out("")
        if key == "prepare":
            _stage_prepare(study, work, input_fn=input_fn, out=out, runner=runner)
        elif key == "classify":
            _stage_classify(study, work, input_fn=input_fn, out=out, runner=runner)
        elif key == "merge":
            _stage_merge(study, work, input_fn=input_fn, out=out, runner=runner)
        elif key == "analyze":
            _stage_analyze(study, work, out=out, runner=runner)
        elif key == "network":
            _stage_network(study, work, out=out, runner=runner)


def _stage_index(key):
    for index, (candidate, _label, _desc) in enumerate(WORKFLOW, start=1):
        if candidate == key:
            return index
    return None
