"""Console feedback helpers: progress bars, step markers, DONE blocks and next steps.

Progress and warnings go to stderr, so stdout stays readable on its own; the
``DONE`` block and the ``NEXT STEPS`` list go to stdout. ``tqdm`` is optional:
when it is not installed, :func:`progress` still yields every item, simply
without drawing a bar.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable, Iterator
from typing import TypeVar

try:  # optional dependency
    from tqdm import tqdm
except ImportError:  # pragma: no cover - depends on the environment
    tqdm = None  # type: ignore[assignment]

__all__ = ["done", "next_steps", "progress", "step", "warn"]

T = TypeVar("T")


def progress(
    iterable: Iterable[T],
    *,
    total: int | None = None,
    desc: str = "",
) -> Iterator[T]:
    """Yield ``iterable`` while reporting progress.

    When ``tqdm`` is available the items are wrapped in a bar written to stderr;
    otherwise they are passed through unchanged. Works with one-shot iterators
    (generators, file handles), in which case ``total`` may be supplied by the
    caller if it is known.

    Args:
        iterable: Any iterable or iterator, sized or not.
        total: Expected number of items; ``None`` lets tqdm count on the fly.
        desc: Label shown in front of the bar.

    Yields:
        The items of ``iterable``, in order.
    """
    if tqdm is None:
        # No tqdm installed: consume the iterable silently, item by item.
        yield from iterable
        return

    yield from tqdm(iterable, total=total, desc=desc, file=sys.stderr)


def step(index: int, total: int, title: str) -> None:
    """Print a single step marker such as ``[2/5] merge`` to stdout.

    Args:
        index: 1-based position of the current step.
        total: Number of steps in the stage.
        title: Short human-readable name of the step.
    """
    print(f"[{index}/{total}] {title}", flush=True)


def _render(value) -> str:
    """Render a value for display without raising on exotic types."""
    if isinstance(value, (dict, list, tuple, set)):
        try:
            return json.dumps(value, default=str, ensure_ascii=False)
        except (TypeError, ValueError):  # pragma: no cover - last-resort safety
            return str(value)
    return str(value)


def done(title: str, details: dict | list[str] | str | None = None) -> None:
    """Print a success block to stdout, starting with ``DONE  `` + ``title``.

    Args:
        title: What finished, e.g. ``"merge"``.
        details: Optional payload. A dict is printed as ``key: value`` lines, a
            list as bullet lines, and a string verbatim. ``None`` prints only
            the title line. Values are rendered with ``str`` (JSON with
            ``default=str`` for containers), so nothing here can raise.
    """
    print(f"DONE  {title}", flush=True)
    if details is None:
        return
    if isinstance(details, dict):
        for key, value in details.items():
            print(f"  {key}: {_render(value)}", flush=True)
    elif isinstance(details, str):
        print(details, flush=True)
    else:
        try:
            items = list(details)
        except TypeError:  # not iterable: show it as a scalar
            print(_render(details), flush=True)
            return
        for item in items:
            print(f"  - {_render(item)}", flush=True)


def next_steps(commands: list[str]) -> None:
    """Print a numbered ``NEXT STEPS`` block of follow-up commands to stdout.

    Args:
        commands: Commands (or instructions) to list in order. An empty list
            still prints the header.
    """
    print("NEXT STEPS", flush=True)
    for i, command in enumerate(commands, start=1):
        print(f"  {i}. {command}", flush=True)


def warn(message: str) -> None:
    """Print ``WARNING: <message>`` to stderr.

    Args:
        message: The warning text.
    """
    print(f"WARNING: {message}", file=sys.stderr, flush=True)
