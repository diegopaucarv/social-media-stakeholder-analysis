"""English network contract for the HTML viewer (``network.html``).

The viewer renders the debate as a graph (post -> argument frame -> sub-argument
thread -> comments) with a supports/opposes/qualifies stance breakdown and the
most active authors ("super users"). It reads one flat CSV, so this module
flattens the coded corpus into that file using the stable column set in
``NETWORK_COLUMNS`` instead of the legacy Spanish header the HTML used to expect.

Contract summary (one row per coded comment):

- ``post_topic`` / ``post_url`` identify the original post; ``post_topic`` comes from
  ``03_posts_classified.csv`` when merge wrote it (else from ``posts_table.csv``);
- ``frame_id`` / ``frame_label`` the argument frame;
- ``thread_id`` / ``thread_label`` the sub-argument thread. Emergent threads get a
  synthetic, stable id ``<emergent_thread_id>:<slug>`` (the configured sentinel, the
  model proposal lower-cased and slugified) while ``thread_label`` keeps the proposal
  text, so two different proposals no longer collapse into one node;
- ``author``, ``comment_text`` and ``published_at`` describe the comment;
- ``stance`` keeps the raw coded value while ``stance_group`` normalises it to
  ``support``, ``against`` or ``qualify`` for the pie chart;
- ``relevance_score``, ``likes``, ``reply_count``, ``thread_depth`` and
  ``is_reply`` feed node sizing and ranking.

Rows codable as no-frame, empty, or emergent at the frame level are excluded:
the viewer only draws actual arguments.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from . import files as F
from .config import Config
from .progress import progress
from .users import load_coded
from .utils import as_bool, write_csv

NETWORK_COLUMNS = [
    "post_topic",
    "post_url",
    "frame_id",
    "frame_label",
    "thread_id",
    "thread_label",
    "author",
    "comment_text",
    "published_at",
    "stance",
    "stance_group",
    "relevance_score",
    "likes",
    "reply_count",
    "thread_depth",
    "is_reply",
]

# Order is precedence: the first group whose aliases contain the normalised key wins.
_STANCE_ALIASES = (
    (
        "support",
        (
            "support",
            "supports",
            "supporting",
            "pro",
            "in_favour",
            "defends",
            "defiende",
            "a_favor",
        ),
    ),
    (
        "against",
        (
            "against",
            "oppose",
            "opposes",
            "opposing",
            "contra",
            "counter",
            "rebuts",
            "en_contra",
        ),
    ),
    (
        "qualify",
        (
            "qualify",
            "qualifies",
            "qualifying",
            "nuance",
            "matiza",
            "conditional",
        ),
    ),
)


def _stance_key(value) -> str:
    """Lower-case a stance key and unify spaces/hyphens with the alias spelling."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):  # arrays and other exotic values
        pass
    return re.sub(r"[\s\-]+", "_", str(value).strip().lower())


def stance_group(value: str) -> str:
    """Map a configured stance key to ``support``, ``against`` or ``qualify``.

    Matching is case-insensitive and tolerates spaces/hyphens (``in favour`` ->
    ``in_favour``). Unknown or empty values default to ``qualify`` so the viewer's
    pie chart always has a bucket.
    """
    key = _stance_key(value)
    for group, aliases in _STANCE_ALIASES:
        if key in aliases:
            return group
    return "qualify"


def _text_column(df: pd.DataFrame, name: str) -> pd.Series:
    """String view of a column; missing columns and NaN become empty strings."""
    if name not in df.columns:
        return pd.Series("", index=df.index, dtype=object)
    return df[name].fillna("").astype(str)


def _int_column(df: pd.DataFrame, name: str, default: int) -> pd.Series:
    """Integer view of a column; missing or invalid values use ``default``."""
    if name not in df.columns:
        return pd.Series(default, index=df.index, dtype="int64")
    return pd.to_numeric(df[name], errors="coerce").fillna(default).astype(int)


def _post_meta(work_dir) -> pd.DataFrame:
    """Post id -> (post_topic, post_url), preferring the classified post table.

    ``03_posts_classified.csv`` carries the model-assigned ``post_topic`` /
    ``post_frame_id`` written by ``merge``; ``posts_table.csv`` is the fallback when
    post classification was skipped. ``post_key`` is the post URL as built by
    ``prepare``. When ``post_topic`` is empty, fall back to the (truncated) post text,
    then the URL's last segment, then ``Post <id>``, so the viewer always has a label
    for the topic node.
    """
    base = Path(work_dir)
    source = base / F.POSTS_CLASSIFIED
    if not source.exists():
        source = base / F.POSTS_TABLE
    posts = pd.read_csv(
        source,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    if "post_id" not in posts.columns:
        raise ValueError(f"{F.POSTS_TABLE} has no post_id column")

    def column(name: str) -> pd.Series:
        if name not in posts.columns:
            return pd.Series("", index=posts.index, dtype=object)
        return posts[name].fillna("").astype(str)

    post_id = column("post_id").str.strip()
    topic = (
        posts["post_topic"].fillna("").astype(str).str.strip()
        if "post_topic" in posts.columns
        else pd.Series("", index=posts.index, dtype=object)
    )
    url = (
        posts["post_key"].astype(str).str.strip()
        if "post_key" in posts.columns
        else pd.Series("", index=posts.index, dtype=object)
    )

    text = (
        posts["post_text"].fillna("").astype(str)
        if "post_text" in posts.columns
        else pd.Series("", index=posts.index, dtype=object)
    )
    collapsed = text.str.replace(r"\s+", " ", regex=True).str.strip()
    slug = (
        posts["post_key"]
        .fillna("")
        .astype(str)
        .str.rstrip("/")
        .str.rsplit("/", n=1)
        .str[-1]
        if "post_key" in posts.columns
        else pd.Series("", index=posts.index)
    )
    slug = slug.fillna("")

    label = slug.where(text.str.strip() == "", text.str.slice(0, 60))
    label = label.where(label.str.strip() != "", "Post " + post_id)
    topic = topic.where(topic != "", label)

    return pd.DataFrame(
        {
            "post_id": post_id,
            "post_topic": topic,
            "post_url": url,
        }
    ).drop_duplicates("post_id")


def _emergent_slug(proposal: str) -> str:
    """Stable id fragment for an emergent thread proposal (<= 60 chars)."""
    slug = re.sub(r"[^a-z0-9]+", "-", str(proposal).strip().lower()).strip("-")
    return slug[:60]


def build_network_rows(cfg: Config, work_dir) -> pd.DataFrame:
    """Flatten the coded corpus into the stable ``NETWORK_COLUMNS`` contract."""
    c = load_coded(work_dir, cfg)
    c = c.drop(columns=[col for col in ("post_topic", "post_url") if col in c.columns])

    meta = _post_meta(work_dir)
    c["post_id"] = c["post_id"].astype(str).str.strip()
    c = c.merge(meta, on="post_id", how="left")
    c["post_topic"] = c["post_topic"].fillna("")
    c["post_url"] = c["post_url"].fillna("")

    frame_all = _text_column(c, "frame_id").str.strip()
    text_all = _text_column(c, "text")
    keep = (
        (frame_all != "")
        & (frame_all != str(cfg.no_frame_id))
        & (frame_all != cfg.emergent_thread_id)
        & (text_all.str.strip() != "")
        & (_text_column(c, "post_topic").str.strip() != "")
    )
    c = c.loc[keep].reset_index(drop=True)

    frame_id = _text_column(c, "frame_id").str.strip()
    thread_id = _text_column(c, "thread_id").str.strip()
    frame_label = frame_id.map(lambda fid: cfg.frame_label.get(fid, fid))
    thread_label = thread_id.map(lambda tid: cfg.thread_label.get(tid, tid))
    if "emergent_thread_proposal" in c.columns:
        proposal = _text_column(c, "emergent_thread_proposal").str.strip()
        use_proposal = (thread_id == cfg.emergent_thread_id) & (proposal != "")
        thread_label = thread_label.mask(use_proposal, proposal)
        slug = proposal.map(_emergent_slug)
        use_slug = use_proposal & (slug != "")
        thread_id = thread_id.mask(use_slug, f"{cfg.emergent_thread_id}:" + slug)

    pseudonym = _text_column(c, "user_pseudonym").str.strip()
    user_name = _text_column(c, "user_name").str.strip()
    author = pseudonym.where(pseudonym != "", user_name)
    author = author.where(author != "", "Anonymous")

    stamps = pd.to_datetime(c["timestamp"], utc=True, errors="coerce")
    published_at = stamps.dt.strftime("%Y-%m-%dT%H:%M:%SZ").fillna("")

    stance = _text_column(c, "stance")
    depth = _int_column(c, "thread_depth", 0)
    flagged = (
        as_bool(c["is_reply"])
        if "is_reply" in c.columns
        else pd.Series(False, index=c.index)
    )

    out = pd.DataFrame(
        {
            "post_topic": _text_column(c, "post_topic"),
            "post_url": _text_column(c, "post_url"),
            "frame_id": frame_id,
            "frame_label": frame_label,
            "thread_id": thread_id,
            "thread_label": thread_label,
            "author": author,
            "comment_text": _text_column(c, "text"),
            "published_at": published_at,
            "stance": stance,
            "stance_group": stance.map(stance_group),
            "relevance_score": _int_column(c, "relevance_score", 1),
            "likes": _int_column(c, "likes", 0),
            "reply_count": _int_column(c, "reply_count", 0),
            "thread_depth": depth,
            "is_reply": ((depth > 0) | flagged).astype(int),
        }
    )
    out = out[NETWORK_COLUMNS]
    return out.sort_values(
        ["post_topic", "frame_label", "thread_label", "published_at"],
        kind="stable",
    ).reset_index(drop=True)


def export_network(cfg: Config, work_dir, out_path) -> dict:
    """Write the network contract to ``out_path`` and return a small summary.

    Counts are graph identities, not raw columns: ``posts`` is distinct
    ``post_url``, ``frames`` distinct ``frame_label`` and ``threads`` distinct
    ``thread_label`` (so emergent proposals count as their own threads).
    """
    path = Path(out_path)
    coded = Path(work_dir) / F.ALL_CODED
    if not coded.exists():
        raise ValueError(
            f"cannot export the network: {coded} was not found. "
            f"Run `merge` in {work_dir} first."
        )
    try:
        rows = build_network_rows(cfg, work_dir)
    except FileNotFoundError as exc:
        raise ValueError(
            f"cannot export the network: {exc}. Run `merge` in {work_dir} first."
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, path)

    # The summary counts graph identities. Walk the flattened rows (the only per-item
    # pass left in this stage) so long corpora report progress while counting.
    posts_n: set[str] = set()
    frames_n: set[str] = set()
    threads_n: set[str] = set()
    columns = rows[["post_url", "frame_label", "thread_label"]]
    for post_url, frame_lbl, thread_lbl in progress(
        columns.itertuples(index=False, name=None),
        total=len(rows),
        desc="summarising network rows",
    ):
        posts_n.add(post_url)
        frames_n.add(frame_lbl)
        threads_n.add(thread_lbl)
    return {
        "rows": len(rows),
        "posts": len(posts_n),
        "frames": len(frames_n),
        "threads": len(threads_n),
        "path": str(path),
    }
