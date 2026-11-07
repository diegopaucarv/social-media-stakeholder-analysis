"""Step 1 - prepare: load raw scraper CSVs, link posts and comments, compute relevance and timeline phases,
resolve sources, and cut classification batches for the LLM. Topic-agnostic: everything comes from the config."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from . import files as F
from .config import REQUIRED_COMMENT_COLUMNS, Config
from .institutional import combine
from .progress import progress
from .prompt_builder import classifier_user_message, post_classifier_user_message
from .provenance import write_run
from .utils import pseudonym_map, write_csv, write_json

ID_COLUMNS = ("comment_id", "post_numeric_id", "user_id", "reply_to_id", "post_id")

# Editable post metadata: every column here can be fed back through --post-meta.
POST_META_COLUMNS = (
    "post_key",
    "source_name",
    "source_type",
    "post_author",
    "post_date",
    "post_text",
    "post_topic",
    "post_frame_id",
)


# ----------------------------------------------------------------------------- loading
def _load(path, mapping: dict, id_cols, required=()) -> pd.DataFrame:
    header = list(pd.read_csv(path, nrows=0).columns)
    use = {
        canon: raw
        for canon, raw in mapping.items()
        if isinstance(raw, str) and raw in header
    }
    missing = [c for c in required if c not in use]
    if missing:
        raise ValueError(
            f"{Path(path).name}: required columns not found for {missing} "
            f"(expected raw names {[mapping[m] for m in missing]}). Adjust study.input_schema."
        )
    raw_cols = sorted(set(use.values()))
    dtype = {raw: str for canon, raw in use.items() if canon in id_cols}
    df = pd.read_csv(path, usecols=raw_cols, dtype=dtype, low_memory=False)
    out = pd.DataFrame({canon: df[raw] for canon, raw in use.items()})
    for canon in mapping:
        if canon not in out and canon != "text_fallbacks":
            out[canon] = np.nan
    return out


def load_comments(path, cfg: Config) -> pd.DataFrame:
    mapping = cfg.study["input_schema"]["comments"]
    df = _load(path, mapping, ID_COLUMNS, REQUIRED_COMMENT_COLUMNS)
    for col in ("thread_depth", "likes", "reply_count"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    df["comment_id"] = df["comment_id"].astype(str).str.strip()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError(
            f"{int(df['timestamp'].isna().sum())} comments have an unparseable timestamp"
        )
    return df


_CORPUS_SKIP = {
    F.COMMENTS_ENRICHED,
    F.POSTS_TABLE,
    F.POST_META_TEMPLATE,
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
    "ai_output.csv",
    "ai_posts.csv",
    "classification.template.csv",
    "classification.posts.template.csv",
}


def detect_corpus(directory, cfg: Config) -> tuple[Path, Path | None]:
    """Find the comments and posts CSVs in ``directory`` from their header.

    Filenames are ignored (scraper exports are long and machine-generated). A CSV is a
    comments file when it has every required comment column, and a posts file when it
    has the configured post id, URL and timestamp columns. Pipeline outputs are never
    considered.

    Returns:
        ``(comments_path, posts_path_or_None)``.

    Raises:
        ValueError: when no comments file exists, or when a role is ambiguous.
    """
    base = Path(directory)
    if not base.is_dir():
        raise FileNotFoundError(f"corpus folder not found: {base}")
    comment_map = cfg.study["input_schema"]["comments"]
    post_map = cfg.study["input_schema"]["posts"]
    need_comments = [comment_map[c] for c in REQUIRED_COMMENT_COLUMNS]
    need_posts = [post_map[c] for c in ("post_id", "url", "timestamp")]

    found_comments, found_posts = [], []
    for path in sorted(base.glob("*.csv")):
        if path.name in _CORPUS_SKIP:
            continue
        try:
            header = set(pd.read_csv(path, nrows=0).columns)
        except (OSError, ValueError, pd.errors.ParserError):
            continue
        if all(col in header for col in need_comments):
            found_comments.append(path)
        elif all(col in header for col in need_posts):
            found_posts.append(path)

    if not found_comments:
        raise ValueError(
            f"no comments CSV found in {base} "
            f"(expected columns {need_comments}). Pass --comments instead."
        )
    if len(found_comments) > 1:
        names = ", ".join(p.name for p in found_comments)
        raise ValueError(
            f"several comments CSVs in {base}: {names}. Pass one with --comments."
        )
    if len(found_posts) > 1:
        names = ", ".join(p.name for p in found_posts)
        raise ValueError(
            f"several posts CSVs in {base}: {names}. Pass one with --posts."
        )
    return found_comments[0], (found_posts[0] if found_posts else None)


def load_posts(path, cfg: Config) -> pd.DataFrame:
    mapping = dict(cfg.study["input_schema"]["posts"])
    fallbacks = mapping.pop("text_fallbacks", [])
    header = list(pd.read_csv(path, nrows=0).columns)
    df = _load(path, mapping, ID_COLUMNS)
    extra = [c for c in fallbacks if c in header]
    if extra:
        fb = pd.read_csv(path, usecols=extra, low_memory=False)
        for col in extra:
            df["text"] = df["text"].fillna(fb[col])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df


# ----------------------------------------------------------------------------- linking posts and comments
def _norm_url(u) -> str:
    if not isinstance(u, str):
        return ""
    p = urlparse(u.strip().lower())
    host = re.sub(r"^(www|m|web|mbasic)\.", "", p.netloc)
    return f"{host}{p.path.rstrip('/')}"


def link_posts(
    comments: pd.DataFrame, posts: pd.DataFrame | None
) -> tuple[pd.DataFrame, dict]:
    """Attach `post_row` (index into `posts`) to every comment using several strategies, in order."""
    comments = comments.copy()
    comments["post_row"] = np.nan
    comments["join_method"] = ""
    report = {
        "n_comments": int(len(comments)),
        "n_post_keys": int(comments.post_url.nunique()),
        "strategies": {},
    }
    if posts is None or posts.empty:
        report["note"] = (
            "no posts file: post metadata comes from comment columns and the optional post metadata CSV"
        )
        report["unmatched_comments"] = int(len(comments))
        return comments, report
    p = posts.reset_index(drop=True)
    maps = {
        "numeric_id": (
            comments["post_numeric_id"].astype(str),
            dict(zip(p["post_id"].astype(str), p.index)),
        ),
        "url_exact": (
            comments["post_url"].astype(str),
            dict(zip(p["url"].astype(str), p.index)),
        ),
        "url_normalised": (
            comments["post_url"].map(_norm_url),
            {_norm_url(u): i for u, i in zip(p["url"], p.index) if _norm_url(u)},
        ),
    }
    title_idx = {}
    for i, t in zip(p.index, p["text"]):
        if isinstance(t, str) and len(t.strip()) >= 20:
            title_idx[t.strip()[:60].lower()] = i
    maps["title_prefix"] = (
        comments["post_title"].map(
            lambda t: t.strip()[:60].lower() if isinstance(t, str) else ""
        ),
        title_idx,
    )
    for name, (keys, lookup) in maps.items():
        todo = comments["post_row"].isna()
        hit = keys.map(lookup)
        newly = todo & hit.notna()
        comments.loc[newly, "post_row"] = hit[newly]
        comments.loc[newly, "join_method"] = name
        report["strategies"][name] = {
            "matched_comments": int(newly.sum()),
            "share_of_all": round(float(newly.mean()), 4),
        }
    report["unmatched_comments"] = int(comments["post_row"].isna().sum())
    report["unmatched_post_keys"] = sorted(
        comments.loc[comments["post_row"].isna(), "post_url"].astype(str).unique()
    )[:50]
    linked = set(comments["post_row"].dropna().astype(int))
    report["posts_without_comments"] = [
        str(p.loc[i, "url"]) for i in p.index if i not in linked
    ][:100]
    report["posts_with_scraper_error"] = (
        int(p["error"].notna().sum()) if "error" in p else 0
    )
    return comments, report


# ----------------------------------------------------------------------------- sources and posts table
def resolve_source(candidates: list, cfg: Config) -> dict | None:
    """Match any candidate string (post URL, page name...) against the source registry keys."""
    lowered = [str(c).lower() for c in candidates if isinstance(c, str) and c]
    for entry in cfg.registry:  # longest keys first
        key = entry["key"].lower()
        if any(key in c for c in lowered):
            return entry
    return None


def build_posts_table(
    comments: pd.DataFrame,
    posts: pd.DataFrame | None,
    cfg: Config,
    post_meta: pd.DataFrame | None = None,
) -> pd.DataFrame:
    first_ts = comments.groupby("post_url")["timestamp"].min()
    rows = []
    for post_key in sorted(comments["post_url"].astype(str).unique()):
        sub = comments[comments["post_url"].astype(str) == post_key]
        prow = sub["post_row"].dropna()
        p = (
            posts.reset_index(drop=True).loc[int(prow.iloc[0])]
            if (posts is not None and len(prow))
            else None
        )
        cands = [post_key]
        if p is not None:
            cands += [p.get("url"), p.get("page_name")]
        entry = resolve_source(cands, cfg)
        text = p.get("text") if p is not None else None
        title = (
            sub["post_title"].dropna().iloc[0]
            if sub["post_title"].notna().any()
            else None
        )
        ts = p.get("timestamp") if p is not None else pd.NaT
        rows.append(
            {
                "post_key": post_key,
                "source_key": entry["key"] if entry else "",
                "source_name": (entry.get("name") or entry["key"])
                if entry
                else (
                    p.get("page_name")
                    if p is not None and isinstance(p.get("page_name"), str)
                    else ""
                ),
                "registered_source": bool(entry),
                "source_type": entry["type"] if entry else "unclassified",
                "post_author": p.get("author")
                if p is not None and isinstance(p.get("author"), str)
                else "",
                "post_date": ts.strftime("%Y-%m-%d") if pd.notna(ts) else "",
                "post_text": text
                if isinstance(text, str)
                else (title if isinstance(title, str) else ""),
                "has_post_text": isinstance(text, str) and bool(text.strip()),
                "post_topic": "",
                "post_frame_id": "",
                "n_comments": int(len(sub)),
                "first_comment_date": first_ts[post_key].strftime("%Y-%m-%d"),
                "join_method": sub["join_method"].iloc[0] if len(sub) else "",
            }
        )
    table = pd.DataFrame(rows)
    if post_meta is not None and len(post_meta):
        meta = post_meta.drop_duplicates("post_key").set_index("post_key")
        for col in (
            "source_name",
            "source_type",
            "post_author",
            "post_date",
            "post_text",
            "post_topic",
            "post_frame_id",
        ):
            if col in meta.columns:
                vals = table["post_key"].map(meta[col])
                has = vals.notna() & (vals.astype(str).str.strip() != "")
                table.loc[has, col] = vals[has].astype(str)
    for i in table.index[
        ~table["registered_source"]
    ]:  # metadata may name a registered source
        entry = resolve_source([table.at[i, "source_name"]], cfg)
        if entry:
            table.at[i, "registered_source"], table.at[i, "source_key"] = (
                True,
                entry["key"],
            )
    table["source_group"] = table["source_type"].map(cfg.source_type_group)
    cutoff = cfg.study["period"]["cutoff_date"]
    if cutoff:
        ref = pd.to_datetime(
            table["post_date"].where(
                table["post_date"] != "", table["first_comment_date"]
            ),
            utc=True,
        )
        table["period"] = np.where(
            ref < pd.Timestamp(cutoff, tz="UTC"),
            cfg.study["period"]["before_label"],
            cfg.study["period"]["after_label"],
        )
    else:
        table["period"] = cfg.study["period"]["after_label"]
    table.insert(0, "post_id", range(1, len(table) + 1))
    return table


# ----------------------------------------------------------------------------- relevance and phases
def compute_relevance(c: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rel = cfg.study["relevance"]
    c = c.copy()
    observed = c["reply_to_id"].value_counts()
    c["reply_count"] = np.maximum(
        c["comment_id"].map(observed).fillna(0), c["reply_count"].fillna(0)
    ).astype(int)
    c["relevance_score"] = (
        c["likes"] * rel["likes_weight"] + rel["reply_weight"] * c["reply_count"]
    )
    c["relevance_tier"] = np.where(
        c["relevance_score"] >= rel["high_threshold"],
        "high",
        np.where(c["relevance_score"] >= rel["medium_threshold"], "medium", "low"),
    )
    return c


def compute_phases(c: pd.DataFrame) -> pd.DataFrame:
    """Per post, in chronological order: phase 1 starts at the first comment; every non-institutional
    high-relevance comment opens a new phase that lasts until the next one (or the end of the thread)."""
    c = c.sort_values(["post_id", "timestamp", "comment_seq"]).reset_index(drop=True)
    c["chronological_order"] = c.groupby("post_id").cumcount() + 1
    phases = {}
    for _, g in c.groupby("post_id", sort=True):
        n, started = 0, False
        for seq, tier, inst in zip(
            g["comment_seq"], g["relevance_tier"], g["is_institutional"]
        ):
            if not started:
                n, started = 1, True
            elif tier == "high" and not inst:
                n += 1
            phases[seq] = n
    c["phase"] = c["comment_seq"].map(phases)
    return c


def compile_gazetteer(cfg: Config):
    out = []
    for g in cfg.gazetteer:
        names = sorted({g["name"], *g.get("aliases", [])}, key=len, reverse=True)
        pat = re.compile(
            r"(?<!\w)(?:" + "|".join(re.escape(n) for n in names) + r")(?!\w)",
            re.I | re.U,
        )
        out.append((g["name"], pat))
    return out


def locations_in(text: str, gaz) -> str:
    return "; ".join(name for name, pat in gaz if pat.search(text or ""))


# ----------------------------------------------------------------------------- batches
def _post_context(row: pd.Series) -> dict:
    return {
        "post_id": int(row["post_id"]),
        "post_key": row["post_key"],
        "source_type": row["source_type"],
        "source_name": row["source_name"],
        "post_date": row["post_date"],
        "post_text": row["post_text"],
    }


def _clear_batches(directory: Path, patterns: tuple[str, ...]) -> None:
    """Remove only the files this stage owns, so LLM responses saved next to their
    prompts survive a re-run."""
    for pattern in patterns:
        for old in directory.glob(pattern):
            old.unlink()


def write_batches(
    comments: pd.DataFrame,
    posts_table: pd.DataFrame,
    cfg: Config,
    out: Path,
    pseudonymize: bool = False,
    write_messages: bool = True,
) -> dict:
    ccfg = cfg.study["classification"]
    bdir = out / F.BATCH_DIR
    bdir.mkdir(parents=True, exist_ok=True)
    _clear_batches(bdir, ("batch_*.json", "batch_*.user_message.txt"))
    text_by_id = comments.set_index("comment_id")["text"].to_dict()
    n_batches = 0
    eligible = comments[(~comments["is_institutional"]) & comments["has_text"]]
    groups = list(eligible.groupby("post_id"))
    for post_id, g in progress(groups, total=len(groups), desc="cutting batches"):
        prow = posts_table[posts_table["post_id"] == post_id].iloc[0]
        for k in range(0, len(g), ccfg["batch_size"]):
            sub = g.iloc[k : k + ccfg["batch_size"]]
            batch = []
            for _, r in sub.iterrows():
                par = r["reply_to_id"]
                batch.append(
                    {
                        "comment_id": r["comment_id"],
                        "timestamp": r["timestamp"].isoformat(),
                        "thread_depth": int(r["thread_depth"]),
                        "likes": int(r["likes"]),
                        "reply_count": int(r["reply_count"]),
                        "reply_to_id": None if pd.isna(par) else par,
                        "parent_text": ""
                        if pd.isna(par)
                        else str(text_by_id.get(par) or "")[
                            : ccfg["parent_text_max_chars"]
                        ],
                        "author": r["user_pseudonym"]
                        if pseudonymize
                        else r["user_name"],
                        "text": r["text"],
                    }
                )
            payload = {"post_context": _post_context(prow), "comments_batch": batch}
            stem = f"batch_P{int(post_id):02d}_{k // ccfg['batch_size'] + 1}"
            write_json(bdir / f"{stem}.json", payload)
            if write_messages:
                (bdir / f"{stem}.user_message.txt").write_text(
                    classifier_user_message(payload), encoding="utf-8"
                )
            n_batches += 1
    pdir = out / F.POST_BATCH_DIR
    pdir.mkdir(parents=True, exist_ok=True)
    _clear_batches(pdir, ("post_batch_*",))
    plist = [
        {
            "post_key": r["post_key"],
            "post_id": int(r["post_id"]),
            "source_type_hint": r["source_type"] if r["registered_source"] else "",
            "source_name": r["source_name"],
            "post_date": r["post_date"],
            "post_text": r["post_text"],
            "n_comments": int(r["n_comments"]),
        }
        for _, r in posts_table.iterrows()
    ]
    n_post_batches = 0
    for k in range(0, len(plist), ccfg["post_batch_size"]):
        chunk = plist[k : k + ccfg["post_batch_size"]]
        write_json(
            pdir / f"post_batch_{k // ccfg['post_batch_size'] + 1}.json",
            {"posts": chunk},
        )
        if write_messages:
            (
                pdir / f"post_batch_{k // ccfg['post_batch_size'] + 1}.user_message.txt"
            ).write_text(post_classifier_user_message(chunk), encoding="utf-8")
        n_post_batches += 1
    return {
        "comment_batches": n_batches,
        "post_batches": n_post_batches,
        "comments_to_classify": int(len(eligible)),
    }


# ----------------------------------------------------------------------------- orchestration
def write_post_meta_template(posts_table: pd.DataFrame, path) -> bool:
    """Write the editable post-metadata CSV derived from the prepared posts table.

    All columns except ``post_topic`` and ``post_frame_id`` are already inferred from
    the posts CSV and the study's source registry; the two empty columns are the
    interpretive ones a researcher fills in by hand. An existing file is never
    overwritten, so edits survive a re-run of `prepare`.

    Returns:
        True when the file was written, False when it was left untouched.
    """
    path = Path(path)
    if path.exists():
        return False
    template = posts_table.copy()
    for col in POST_META_COLUMNS:
        if col not in template:
            template[col] = ""
    write_csv(template[list(POST_META_COLUMNS)], path)
    return True


def prepare(
    cfg: Config,
    comments_csv,
    out_dir,
    posts_csv=None,
    post_meta_csv=None,
    batches: bool = True,
    pseudonymize: bool = False,
    write_messages: bool = True,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    c = load_comments(comments_csv, cfg)
    posts = load_posts(posts_csv, cfg) if posts_csv else None
    c, join_report = link_posts(c, posts)
    c = c.sort_values(["post_url", "timestamp"], kind="stable").reset_index(drop=True)
    c["comment_seq"] = range(1, len(c) + 1)
    meta = pd.read_csv(post_meta_csv, dtype=str) if post_meta_csv else None
    posts_table = build_posts_table(c, posts, cfg, meta)
    key_to_id = dict(zip(posts_table["post_key"], posts_table["post_id"]))
    c["post_key"] = c["post_url"].astype(str)
    c["post_id"] = c["post_key"].map(key_to_id).astype(int)
    c = compute_relevance(c, cfg)
    c["is_institutional"] = combine(c["user_name"].isin(cfg.institutional.keys()))
    c["is_reply"] = (c["thread_depth"] > 0).astype(int)
    c["has_text"] = c["text"].notna() & (c["text"].astype(str).str.strip() != "")
    c["text"] = c["text"].fillna("")
    c["has_attachment"] = c["attachment_type"].notna()
    c["parent_author"] = c["parent_author"].fillna(c["parent_author_alt"])
    gender_map = {"MALE": "M", "FEMALE": "F", "NEUTER": "undeclared"}
    c["gender"] = c["user_gender"].map(
        lambda g: (
            gender_map.get(str(g).upper(), "unknown") if pd.notna(g) else "unknown"
        )
    )
    c["verified"] = c["user_verified"]
    c["user_pseudonym"] = c["user_id"].astype(str).map(pseudonym_map(c["user_id"]))
    gaz = compile_gazetteer(cfg)
    c["locations_mentioned"] = c["text"].map(lambda t: locations_in(t, gaz))
    c = compute_phases(c)
    keep = [
        "comment_seq",
        "comment_id",
        "post_id",
        "post_key",
        "timestamp",
        "chronological_order",
        "phase",
        "likes",
        "reply_count",
        "relevance_score",
        "relevance_tier",
        "thread_depth",
        "is_reply",
        "reply_to_id",
        "user_id",
        "user_pseudonym",
        "user_name",
        "gender",
        "verified",
        "user_url",
        "text",
        "has_text",
        "has_attachment",
        "attachment_type",
        "comment_url",
        "parent_author",
        "is_institutional",
        "locations_mentioned",
    ]
    enriched = c[keep].rename(columns={"user_url": "profile_url"})
    write_csv(enriched, out / F.COMMENTS_ENRICHED)
    write_csv(posts_table, out / F.POSTS_TABLE)
    template_written = write_post_meta_template(posts_table, out / F.POST_META_TEMPLATE)
    join_report["config_hash"] = cfg.hash
    write_json(out / F.JOIN_REPORT, join_report)
    summary = {
        "comments": int(len(enriched)),
        "posts": int(len(posts_table)),
        "phases": int(enriched.groupby("post_id")["phase"].max().sum()),
        "relevance": enriched["relevance_tier"].value_counts().to_dict(),
        "unmatched_comments": join_report["unmatched_comments"],
        "post_meta_template": str(out / F.POST_META_TEMPLATE),
        "post_meta_template_written": template_written,
    }
    if batches:
        summary.update(
            write_batches(enriched, posts_table, cfg, out, pseudonymize, write_messages)
        )
    write_run(
        cfg, out, "prepare", comments=int(len(enriched)), posts=int(len(posts_table))
    )
    return summary
