"""Command line interface: `python -m framing <command> ...`"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ConfigError, load_study
from .utils import read_json, write_json


def _cfg(args):
    return load_study(args.study)


def cmd_validate(args) -> int:
    from .progress import done, next_steps

    cfg = _cfg(args)
    print(
        f"OK  study={cfg.study['id']}  hash={cfg.hash}  frames={len(cfg.frames)}  threads={len(cfg.thread_ids)}  "
        f"actor_categories={len(cfg.actor_ids)}  sources={len(cfg.registry)}  examples={len(cfg.codebook['examples'])}"
    )
    if args.save_unified:
        cfg.save_unified(args.save_unified)
        print(f"unified config written to {args.save_unified}")
    done("study configuration validated", {"study": cfg.study["id"], "hash": cfg.hash})
    next_steps(
        [
            (
                "Prepare the corpus: python -m framing prepare "
                f"--study {args.study} --comments <comments.csv> --posts <posts.csv> --out <work>"
            ),
        ]
    )
    return 0


def cmd_init(args) -> int:
    from .init_project import scaffold
    from .progress import done, next_steps, step

    step(0, 5, "init: scaffold a neutral study folder and validate it")
    cfg = scaffold(
        args.study_id,
        args.title,
        args.language,
        args.out_dir,
        force=args.force,
    )
    out = cfg.base_dir
    manifest = out / "manifest.json"
    print(
        f"OK  study={cfg.study['id']}  hash={cfg.hash}  frames={len(cfg.frames)}  "
        f"threads={len(cfg.thread_ids)}  actor_categories={len(cfg.actor_ids)}  "
        f"sources={len(cfg.registry)}  examples={len(cfg.codebook['examples'])}"
    )
    done(
        "study scaffolded and validated",
        {"study": cfg.study["id"], "dir": str(out), "config_hash": cfg.hash},
    )
    next_steps(
        [
            (
                f"Edit the module JSONs in {out} (start with codebook.json: frames, "
                "threads and examples)."
            ),
            (
                "Re-validate after editing: python -m framing validate-config "
                f"--study {manifest}"
            ),
            (
                "Prepare the corpus: python -m framing prepare "
                f"--study {manifest} --comments <comments.csv> --posts <posts.csv> "
                f"--out work/{cfg.study['id']}"
            ),
            (
                "Bundle the task for a chatbot: python -m framing export-md "
                f"--study {manifest} --work work/{cfg.study['id']} "
                f"--out work/{cfg.study['id']}/classification.md"
            ),
        ]
    )
    return 0


def cmd_render(args) -> int:
    from .progress import done, next_steps
    from .prompt_builder import render_all

    print(json.dumps(render_all(_cfg(args), args.out), indent=2))
    done("prompts and response schemas rendered", {"out": args.out})
    next_steps(
        [
            (
                "prepare renders these automatically; to build the corpus now: "
                "python -m framing prepare "
                f"--study {args.study} --comments <comments.csv> --out <work>"
            )
        ]
    )
    return 0


def cmd_prepare(args) -> int:
    from .prepare import prepare
    from .progress import done, next_steps, step
    from .prompt_builder import render_all

    step(1, 5, "prepare: link posts and comments, compute relevance, cut batches")
    cfg = _cfg(args)
    comments = args.comments
    posts = args.posts
    if args.corpus_dir:
        from .prepare import detect_corpus

        found_comments, found_posts = detect_corpus(args.corpus_dir, cfg)
        if not comments:
            comments = str(found_comments)
            print(f"COMMENTS  {comments}")
        if not posts and found_posts is not None:
            posts = str(found_posts)
            print(f"POSTS     {posts}")
    if not comments:
        raise ValueError("pass --comments <comments.csv> or --corpus-dir <folder>")
    summary = prepare(
        cfg,
        comments,
        args.out,
        posts,
        args.post_meta,
        batches=not args.no_batches,
        pseudonymize=args.pseudonymize_authors,
        write_messages=not args.no_messages,
    )
    step(1, 5, "prepare: render prompts and response schemas")
    render_all(cfg, Path(args.out) / "prompts")
    print(json.dumps(summary, indent=2, default=str))
    done("prepare finished: corpus linked, batches ready", summary)
    steps = [
        (
            "Chatbot branch (no API key needed):  python -m framing export-md "
            f"--study {args.study} --work {args.out} --out {args.out}/classification.md"
        ),
        (
            "Then paste classification.md into a chatbot and save its reply as "
            f"{args.out}/ai_output.csv (columns: {args.out}/classification.template.csv)"
        ),
        (
            "API branch (OFF by default):          python -m framing classify-api "
            f"--study {args.study} --work {args.out} --out {args.out}/responses --enable-api"
        ),
        (
            "Then ingest the answers:             python -m framing merge "
            f"--study {args.study} --work {args.out} --ai-responses {args.out}/ai_output.csv"
        ),
    ]
    if posts:
        steps.append(
            "If the bundle also asked for post labels, save that reply as "
            f"{args.out}/ai_posts.csv; merge picks it up automatically."
        )
        steps.append(
            "Optional: label posts in "
            f"{args.out}/post_meta_template.csv, then re-run: python -m framing prepare "
            f"--study {args.study} --comments {comments} --posts {posts} "
            f"--out {args.out} --post-meta {args.out}/post_meta_template.csv"
        )
    next_steps(steps)
    return 0


def cmd_merge(args) -> int:
    from .merge import merge
    from .progress import done, next_steps, step

    step(3, 5, "merge: join the model answers with the corpus")
    post_responses = args.post_responses
    if not post_responses:
        candidate = Path(args.work) / "ai_posts.csv"
        if candidate.exists():
            post_responses = [str(candidate)]
            print(f"NOTE  using post answers found in {candidate}")
    rep = merge(
        _cfg(args),
        args.work,
        args.responses,
        args.out,
        post_responses,
        args.min_confidence,
        require_coverage=args.require_coverage,
    )
    print(
        json.dumps(
            {k: v for k, v in rep.items() if k != "issue_examples"},
            indent=2,
            default=str,
        )
    )
    done(
        "merge finished: coded tables written",
        {
            "files": rep.get("files"),
            "records": rep.get("records"),
            "coverage": round(float(rep.get("coverage", 0.0)), 4),
            "selected": rep.get("selected"),
        },
    )
    out = args.out or args.work
    next_steps(
        [
            (
                "Analyse:   python -m framing analyze "
                f"--study {args.study} --work {out} --out {out}"
            ),
            (
                "Network:  python -m framing export-network "
                f"--study {args.study} --work {out} --out {out}/11_network.csv"
            ),
        ]
    )
    return 0


def cmd_export_md(args) -> int:
    from .export_md import write_classification_bundle
    from .progress import done, next_steps, step

    step(2, 5, "export-md: bundling the whole task for a chatbot")
    info = write_classification_bundle(
        _cfg(args), args.work, args.out, manifest_path=args.study
    )
    done("classification bundle written", info)

    work = Path(args.work)
    answer = work / "ai_output.csv"
    post_answer = work / "ai_posts.csv"
    steps = [
        f"Open {info['markdown']} and paste the whole file into your chatbot.",
        f"Save the chatbot answer as {answer} (expected header is in {info['template_csv']}).",
    ]
    merge_cmd = (
        "python -m framing merge "
        f"--study {args.study} --work {args.work} --ai-responses {answer}"
    )
    if info.get("post_batches"):
        steps.append(f"Also save the post classification answer as {post_answer}.")
        merge_cmd += f" --post-ai-responses {post_answer}"
    steps.append(f"Ingest the answers: {merge_cmd}")
    next_steps(steps)
    return 0


def cmd_classify_api(args) -> int:
    import os

    from .api import (
        API_KEY_ENV,
        DEFAULT_BASE_URL,
        DEFAULT_MODEL,
        classify_batches,
        classify_post_batches,
        require_enabled,
    )
    from .progress import done, next_steps, step

    # Refuse before printing any step marker, so a rejected call shows only the error.
    require_enabled(args.enable_api)
    if args.api_key:
        os.environ[API_KEY_ENV] = args.api_key
    cfg = _cfg(args)
    step(2, 5, "classify-api: sending the batches to the model")
    base_url = args.base_url or DEFAULT_BASE_URL
    model = args.model or DEFAULT_MODEL
    records = classify_batches(
        cfg,
        args.work,
        enabled=args.enable_api,
        api_key=args.api_key,
        base_url=base_url,
        model=model,
    )
    post_records = classify_post_batches(
        cfg,
        args.work,
        enabled=args.enable_api,
        api_key=args.api_key,
        base_url=base_url,
        model=model,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    answers = out / "api_answers.json"
    write_json(answers, records)
    post_answers = out / "api_post_answers.json"
    write_json(post_answers, post_records)
    done(
        "API classification finished",
        {
            "records": len(records),
            "answers": str(answers),
            "post_records": len(post_records),
            "post_answers": str(post_answers),
        },
    )
    merge_cmd = (
        "Ingest the answers: python -m framing merge "
        f"--study {args.study} --work {args.work} --ai-responses {answers}"
    )
    if post_records:
        merge_cmd += f" --post-ai-responses {post_answers}"
    next_steps([merge_cmd])
    return 0


def cmd_export_network(args) -> int:
    from . import files as F
    from .network_export import export_network
    from .progress import done, next_steps, step

    out = args.out or str(Path(args.work) / F.NETWORK)
    step(5, 5, "export-network: flatten the coding into the viewer CSV")
    info = export_network(_cfg(args), args.work, out)
    done("network export written", info)

    viewer = (Path(__file__).resolve().parent.parent / "network.html").resolve()
    csv_path = Path(info["path"]).resolve()
    next_steps(
        [
            "This is the final deliverable.",
            f"Open {viewer} in a browser and upload {csv_path}.",
            (
                "The viewer draws topic -> threads grouped inside argument frames (an edge "
                "per topic/frame pair may exist too), with a stance pie per thread."
            ),
        ]
    )
    return 0


def cmd_users(args) -> int:
    from .progress import done, next_steps
    from .users import run_users

    res = run_users(_cfg(args), args.work, args.out)
    print(json.dumps(res, indent=2))
    done("user profiles finished", res)
    next_steps(
        [
            (
                "Compare two coders: python -m framing compare "
                f"--study {args.study} --a {args.work} --b <second_coder_work> "
                f"--out {args.out or args.work}"
            ),
        ]
    )
    return 0


def cmd_stats(args) -> int:
    from .progress import done, next_steps
    from .stats import run_stats

    res = run_stats(_cfg(args), args.work, args.out)
    print(json.dumps(res, indent=2))
    done("statistics finished", res)
    next_steps(
        [
            (
                "Full analysis: python -m framing analyze "
                f"--study {args.study} --work {args.work} --out {args.out or args.work}"
            ),
        ]
    )
    return 0


def cmd_gaps(args) -> int:
    from .gaps import build_gap_report
    from .progress import done, next_steps

    res = build_gap_report(_cfg(args), args.work, args.out)
    print(json.dumps(res, indent=2))
    done("gap report finished", res)
    next_steps(
        [
            (
                "Full analysis: python -m framing analyze "
                f"--study {args.study} --work {args.work} --out {args.out or args.work}"
            ),
        ]
    )
    return 0


def cmd_analyze(args) -> int:
    from .gaps import build_gap_report
    from .progress import done, next_steps, step
    from .stats import run_stats
    from .users import run_users

    cfg = _cfg(args)
    step(4, 5, "analyze: users, statistics and gap report")
    res = {
        "users": run_users(cfg, args.work, args.out),
        "stats": run_stats(cfg, args.work, args.out),
        "gaps": build_gap_report(cfg, args.work, args.out),
    }
    done("analysis finished", res)
    next_steps(
        [
            (
                "Network view: python -m framing export-network "
                f"--study {args.study} --work {args.work} --out {args.out or args.work}/11_network.csv"
            ),
            (
                "Reliability:  python -m framing compare "
                f"--study {args.study} --a {args.work} --b <second_coder_work> --out {args.out or args.work}"
            ),
        ]
    )
    return 0


def cmd_compare(args) -> int:
    from .progress import done, next_steps
    from .reliability import run_compare

    rep = run_compare(_cfg(args), args.a, args.b, args.out)
    print(
        json.dumps(
            {k: v for k, v in rep.items() if k not in ("confusion_frame", "per_frame")},
            indent=2,
        )
    )
    done("inter-coder reliability written", {"out": args.out})
    next_steps(
        [
            (
                "Re-run analyze so the gap report sees reliability_report.json: "
                "python -m framing analyze "
                f"--study {args.study} --work {args.a} --out {args.out}"
            ),
        ]
    )
    return 0


def cmd_seed(args) -> int:
    from .discovery import seed_codebook
    from .progress import done, next_steps

    write_json(
        args.out,
        seed_codebook(
            read_json(args.brief),
            no_frame_id=args.no_frame_id,
            id_prefix=args.id_prefix,
        ),
    )
    # discovery-input --out is a directory; never hand it the codebook file itself.
    rounds_dir = Path(args.out).parent / "discovery"
    done(
        "seed codebook written", {"codebook": args.out, "round_files": str(rounds_dir)}
    )
    next_steps(
        [
            (
                "Validate the study once the codebook is wired into its manifest: "
                "python -m framing validate-config --study <manifest.json>"
            ),
            (
                "Prepare the corpus, then start the first discovery round: "
                "python -m framing discovery-input "
                f"--study <manifest.json> --brief {args.brief} --work <work> "
                f"--out {rounds_dir} --round 1"
            ),
        ]
    )
    return 0


def cmd_discovery_input(args) -> int:
    from .discovery import build_discovery_input
    from .progress import done, next_steps

    res = build_discovery_input(
        args.study,
        args.brief,
        args.work,
        args.out,
        args.round,
        args.sample_size,
        args.seed,
    )
    print(json.dumps(res, indent=2))
    message = Path(args.out) / f"discovery_round_{args.round:02d}_user_message.md"
    answer = Path(args.out) / f"discovery_round_{args.round:02d}_answer.json"
    done("discovery round input built", res)
    next_steps(
        [
            f"Paste {message} into your chatbot and save its JSON reply as {answer}.",
            (
                "Apply the answer: python -m framing apply-discovery "
                f"--study {args.study} --response {answer} "
                f"--out-codebook <codebook.json> --history-dir {args.out}"
            ),
            (
                "Then run the next round: python -m framing discovery-input "
                f"--study {args.study} --brief {args.brief} --work {args.work} "
                f"--out {args.out} --round {args.round + 1}"
            ),
        ]
    )
    return 0


def cmd_apply_discovery(args) -> int:
    from .discovery import apply_discovery
    from .progress import done, next_steps

    res = apply_discovery(
        args.study, args.response, args.out_codebook, args.history_dir
    )
    print(json.dumps(res, indent=2))
    done("discovery response applied", res)
    out_dir = args.history_dir or str(Path(args.out_codebook).parent)
    next_steps(
        [
            (
                f"Validate the updated study: python -m framing validate-config "
                f"--study {args.study}"
            ),
            (
                "Run the next round: python -m framing discovery-input "
                f"--study {args.study} --brief <research_brief.json> --work <work> "
                f"--out {out_dir} --round <next_round>"
            ),
        ]
    )
    return 0


def cmd_wizard(args) -> int:
    from .wizard import run_wizard

    return run_wizard(manifest=args.study, work=args.work)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="framing",
        description=(
            "Topic-agnostic framing / argument-extraction pipeline. "
            "Run `python -m framing` with no arguments for guided mode."
        ),
        epilog=(
            "Start here: `python -m framing` walks you through study, input CSVs, "
            "AI answers, analysis and the network viewer. API calls are off by default."
        ),
    )
    sub = p.add_subparsers(dest="command")

    def add(name, fn, help_):
        sp = sub.add_parser(name, help=help_)
        sp.set_defaults(func=fn)
        return sp

    s = add(
        "init",
        cmd_init,
        "Scaffold a neutral study folder from scratch and validate it.",
    )
    s.add_argument("--id", dest="study_id", required=True, help="study id")
    s.add_argument("--title", required=True, help="study title")
    s.add_argument(
        "--language",
        default="en",
        help="corpus and analysis language code (default: en)",
    )
    s.add_argument(
        "--dir",
        dest="out_dir",
        help="output folder (default: studies/<id>)",
    )
    s.add_argument(
        "--force",
        action="store_true",
        help="replace an existing non-empty study folder",
    )

    s = add(
        "validate-config",
        cmd_validate,
        "Validate and unify the study modules listed in a manifest.",
    )
    s.add_argument("--study", required=True, help="path to manifest.json")
    s.add_argument("--save-unified", help="write the unified config to this JSON file")

    s = add(
        "render-prompts",
        cmd_render,
        "Render prompts and JSON response schemas from the study config.",
    )
    s.add_argument("--study", required=True)
    s.add_argument("--out", required=True)

    s = add(
        "prepare",
        cmd_prepare,
        "Step 1: link posts/comments, relevance, phases, batches, prompts.",
    )
    s.add_argument("--study", required=True)
    s.add_argument("--comments", help="raw comments CSV (or use --corpus-dir)")
    s.add_argument(
        "--corpus-dir",
        help="folder scanned for the comments/posts CSVs by their header",
    )
    s.add_argument("--posts")
    s.add_argument(
        "--post-meta",
        help="optional CSV keyed by post_key to override source/topic/date",
    )
    s.add_argument("--out", required=True)
    s.add_argument("--no-batches", action="store_true")
    s.add_argument("--no-messages", action="store_true")
    s.add_argument(
        "--pseudonymize-authors",
        action="store_true",
        help="send pseudonyms instead of author names to the LLM",
    )

    s = add(
        "merge",
        cmd_merge,
        "Step 3: merge LLM classifications (JSON, JSONL or the CSV returned by the "
        "chatbot branch) with the enriched comments.",
    )
    s.add_argument("--study", required=True)
    s.add_argument("--work", required=True, help="folder written by `prepare`")
    s.add_argument(
        "--responses",
        "--ai-responses",
        dest="responses",
        required=True,
        nargs="+",
        help="glob(s) of model answers: JSON or the CSV returned by the chatbot branch",
    )
    s.add_argument(
        "--post-responses",
        "--post-ai-responses",
        dest="post_responses",
        nargs="+",
        help="glob(s) of post-level model answers (JSON or CSV)",
    )
    s.add_argument("--out")
    s.add_argument("--min-confidence", type=int)
    s.add_argument(
        "--require-coverage",
        type=float,
        help="fail when less than this share of eligible comments was classified (e.g. 0.95)",
    )

    for name, fn, h in (
        (
            "users",
            cmd_users,
            "Author profiles, contact candidates, community relevance.",
        ),
        ("stats", cmd_stats, "Chi-square (Yates for 2x2) and Cramer's V."),
        ("gaps", cmd_gaps, "Gap report against the study configuration."),
        ("analyze", cmd_analyze, "users + stats + gaps."),
    ):
        s = add(name, fn, h)
        s.add_argument("--study", required=True)
        s.add_argument("--work", required=True)
        s.add_argument("--out")

    s = add(
        "compare", cmd_compare, "Inter-coder reliability between two merged folders."
    )
    s.add_argument("--study", required=True)
    s.add_argument("--a", required=True)
    s.add_argument("--b", required=True)
    s.add_argument("--out", required=True)

    s = add(
        "seed-codebook",
        cmd_seed,
        "Create a seed codebook (frames without threads) from research_brief.json.",
    )
    s.add_argument("--brief", required=True)
    s.add_argument("--out", required=True)
    s.add_argument(
        "--no-frame-id",
        default="NONE",
        help="id of the no-frame pseudo frame (default: NONE)",
    )
    s.add_argument(
        "--id-prefix",
        default="F",
        help="prefix of the generated frame ids; F<i> by default",
    )

    s = add(
        "discovery-input",
        cmd_discovery_input,
        "Build the input of a codebook-discovery round.",
    )
    s.add_argument("--study", required=True)
    s.add_argument("--brief", required=True)
    s.add_argument("--work", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--round", type=int, required=True)
    s.add_argument("--sample-size", type=int, default=80)
    s.add_argument("--seed", type=int, default=1)

    s = add(
        "apply-discovery",
        cmd_apply_discovery,
        "Validate an LLM discovery response and write the new codebook.",
    )
    s.add_argument("--study", required=True)
    s.add_argument("--response", required=True)
    s.add_argument("--out-codebook", required=True)
    s.add_argument("--history-dir")

    s = add(
        "export-md",
        cmd_export_md,
        "Step 2 (default): write one markdown bundle to paste into a chatbot (no API needed).",
    )
    s.add_argument("--study", required=True)
    s.add_argument("--work", required=True)
    s.add_argument("--out", required=True, help="path of the .md bundle to write")

    s = add(
        "classify-api",
        cmd_classify_api,
        "Step 2 (opt-in): classify through an API. Disabled by default.",
    )
    s.add_argument("--study", required=True)
    s.add_argument("--work", required=True)
    s.add_argument("--out", required=True, help="folder for api_answers.json")
    s.add_argument(
        "--enable-api", action="store_true", help="required: allow network calls"
    )
    s.add_argument("--api-key", help="API key (defaults to FRAMING_API_KEY)")
    s.add_argument("--base-url")
    s.add_argument("--model")

    s = add(
        "export-network",
        cmd_export_network,
        "Step 5: write the CSV consumed by network.html.",
    )
    s.add_argument("--study", required=True)
    s.add_argument("--work", required=True)
    s.add_argument("--out")

    s = add(
        "wizard",
        cmd_wizard,
        "Guided mode: ask for the study, the CSVs and the work folder, then run each step.",
    )
    s.add_argument("--study", help="path to manifest.json (optional)")
    s.add_argument("--work", help="work folder (optional)")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if getattr(args, "func", None) is None:
            from .wizard import run_wizard

            return run_wizard()
        return args.func(args)
    except (ConfigError, FileNotFoundError, ValueError, RuntimeError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
