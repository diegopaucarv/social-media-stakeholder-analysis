"""Contract test for the final deliverable: `network.html`.

The viewer keeps its graph construction in the pure function
``buildGraphElements(rows)`` so it can be tested without a browser. This module
extracts that function from the HTML and runs it under Node against a small
fixture in the exact shape of ``11_network.csv`` (every value a string).

The test is skipped when Node is not installed, so the Python-only CI stays
green. No network access is involved: `network.html` is read from disk and the
function under test never touches the DOM.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VIEWER = ROOT / "network.html"


def _row(
    url,
    topic,
    frame_id,
    frame_label,
    thread_id,
    thread_label,
    author,
    text,
    stance,
    stance_group,
    relevance=1,
):
    """One row in the exact shape `network.html` receives from PapaParse."""
    return {
        "post_topic": topic,
        "post_url": url,
        "frame_id": frame_id,
        "frame_label": frame_label,
        "thread_id": thread_id,
        "thread_label": thread_label,
        "author": author,
        "comment_text": text,
        "published_at": "2024-01-01T10:00:00Z",
        "stance": stance,
        "stance_group": stance_group,
        "relevance_score": str(relevance),
        "likes": "1",
        "reply_count": "0",
        "thread_depth": "0",
        "is_reply": "0",
    }


# Two posts, two frames, five threads (including one emergent) and four authors.
# Alice and Bob each comment three times, so they are the two super users.
ROWS = [
    _row(
        "p1",
        "Topic one",
        "F1",
        "Cost",
        "F1_T1",
        "Parking",
        "Alice",
        "text a",
        "supports",
        "support",
    ),
    _row(
        "p1",
        "Topic one",
        "F1",
        "Cost",
        "F1_T1",
        "Parking",
        "Alice",
        "text b",
        "supports",
        "support",
    ),
    _row(
        "p1",
        "Topic one",
        "F1",
        "Cost",
        "F1_T1",
        "Parking",
        "Alice",
        "text c",
        "opposes",
        "against",
    ),
    _row(
        "p1",
        "Topic one",
        "F2",
        "Safety",
        "F2_T1",
        "Kids",
        "Bob",
        "text d",
        "qualifies",
        "qualify",
    ),
    _row(
        "p2",
        "Topic two",
        "F1",
        "Cost",
        "F1_T2",
        "Shops",
        "Carol",
        "text e",
        "supports",
        "",
    ),
    _row(
        "p2",
        "Topic two",
        "F2",
        "Safety",
        "F2_T2",
        "Crossing",
        "Bob",
        "text f",
        "opposes",
        "against",
    ),
    _row(
        "p2",
        "Topic two",
        "F2",
        "Safety",
        "EMERGENT:new-idea",
        "New idea",
        "Dave",
        "text g",
        "supports",
        "support",
        relevance=3,
    ),
    _row(
        "p2",
        "Topic two",
        "F2",
        "Safety",
        "EMERGENT:new-idea",
        "New idea",
        "Bob",
        "text h",
        "qualifies",
        "qualify",
    ),
]

HARNESS = r"""
const fs = require('fs');
const [htmlPath, rowsPath] = process.argv.slice(2);
const html = fs.readFileSync(htmlPath, 'utf8');
const rows = JSON.parse(fs.readFileSync(rowsPath, 'utf8'));

function done(payload, code) {
  console.log(JSON.stringify(payload));
  process.exit(code);
}

const marker = 'function buildGraphElements';
const start = html.indexOf(marker);
if (start < 0) done({ ok: false, error: 'buildGraphElements not found' }, 1);

let depth = 0;
let end = -1;
for (let i = html.indexOf('{', start); i < html.length; i += 1) {
  const ch = html[i];
  if (ch === '{') depth += 1;
  else if (ch === '}') {
    depth -= 1;
    if (depth === 0) { end = i + 1; break; }
  }
}
if (end < 0) done({ ok: false, error: 'unbalanced braces' }, 1);

const source = html.slice(start, end);
let build;
try {
  build = new Function(source + '\nreturn buildGraphElements;')();
} catch (error) {
  done({ ok: false, error: 'eval failed: ' + error.message }, 1);
}

const before = JSON.stringify(rows);
const elements = build(rows);
const again = build(rows);

if (!Array.isArray(elements)) done({ ok: false, error: 'not an array' }, 1);

const nodes = elements.filter((e) => !e.data.source);
const edges = elements.filter((e) => e.data.source);
const ofType = (kind) => nodes.filter((n) => n.data.type === kind);
const ids = new Set(nodes.map((n) => n.data.id));
const distinct = (values) =>
  new Set(values.map((v) => String(v == null ? '' : v).trim()).filter(Boolean));

const authorCounts = rows.reduce((acc, row) => {
  const author = String(row.author || '').trim();
  if (author) acc[author] = (acc[author] || 0) + 1;
  return acc;
}, {});
const superUsers = Object.keys(authorCounts).filter((a) => authorCounts[a] >= 3);

const threads = ofType('thread');

done({
  ok: true,
  topics: ofType('topic').length,
  frames: ofType('frame').length,
  threads: threads.length,
  users: ofType('user').length,
  wantTopics: distinct(rows.map((r) => r.post_url)).size,
  wantFrames: distinct(rows.map((r) => r.frame_label)).size,
  wantThreads: distinct(rows.map((r) => r.thread_id)).size,
  wantUsers: superUsers.length,
  parentsOk: threads.length > 0 && threads.every((n) => ids.has(n.data.parent)),
  stanceOk: threads.every((n) =>
    ['support', 'against', 'qualify'].every((k) => typeof n.data[k] === 'number')),
  endpointsOk: edges.every((e) => ids.has(e.data.source) && ids.has(e.data.target)),
  uniqueIds: ids.size === nodes.length,
  edgeIdsUnique: new Set(edges.map((e) => e.data.id)).size === edges.length,
  topicFrameEdges: edges.filter(
    (e) => String(e.data.source).startsWith('TOPIC_') && String(e.data.target).startsWith('FRAME_')).length,
  topicThreadEdges: edges.filter(
    (e) => String(e.data.source).startsWith('TOPIC_') && String(e.data.target).startsWith('THREAD_')).length,
  emptyStanceBecomesQualify: threads
    .flatMap((n) => n.data.comments)
    .filter((c) => c.text === 'text e')
    .every((c) => c.stanceGroup === 'qualify'),
  deterministic: JSON.stringify(elements) === JSON.stringify(again),
  inputUnchanged: JSON.stringify(rows) === before,
}, 0);
"""


@unittest.skipUnless(shutil.which("node"), "node is not installed")
class NetworkViewerTests(unittest.TestCase):
    """`buildGraphElements` must match the 16-column CSV contract exactly."""

    @classmethod
    def setUpClass(cls):
        if not VIEWER.exists():
            raise unittest.SkipTest(f"{VIEWER} is missing")

    def test_graph_matches_the_csv_contract(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        rows_path = tmp / "rows.json"
        rows_path.write_text(json.dumps(ROWS), encoding="utf-8")
        harness = tmp / "harness.js"
        harness.write_text(HARNESS, encoding="utf-8")

        result = subprocess.run(
            [shutil.which("node"), str(harness), str(VIEWER), str(rows_path)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(
            result.returncode, 0, result.stderr[-2000:] or result.stdout[-2000:]
        )
        lines = result.stdout.strip().splitlines()
        self.assertTrue(lines, "harness produced no output")
        verdict = json.loads(lines[-1])
        self.assertTrue(verdict.get("ok"), verdict)

        # Node identity is the contract id, not the label: two posts that share a
        # topic label must still be two nodes, and each emergent thread is its own.
        self.assertEqual(verdict["topics"], verdict["wantTopics"])
        self.assertEqual(verdict["frames"], verdict["wantFrames"])
        self.assertEqual(verdict["threads"], verdict["wantThreads"])
        self.assertEqual(verdict["users"], verdict["wantUsers"])
        self.assertEqual(verdict["topics"], 2)
        self.assertEqual(verdict["frames"], 2)
        self.assertEqual(verdict["threads"], 5)
        self.assertEqual(verdict["users"], 2)

        self.assertTrue(verdict["parentsOk"])
        self.assertTrue(verdict["stanceOk"])
        self.assertTrue(verdict["endpointsOk"])
        self.assertTrue(verdict["uniqueIds"])
        self.assertTrue(verdict["edgeIdsUnique"])
        self.assertTrue(verdict["deterministic"])
        self.assertTrue(verdict["inputUnchanged"])
        self.assertTrue(verdict["emptyStanceBecomesQualify"])

        # One edge per (topic, frame) pair and one per (topic, thread) pair.
        self.assertEqual(verdict["topicFrameEdges"], 4)
        self.assertEqual(verdict["topicThreadEdges"], 5)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
