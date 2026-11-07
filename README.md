# Framing pipeline

Extracts the arguments people make in social-media comments: which **frame** each
comment sustains, what **stance** it takes, how confident the coding is, and who is
talking. Point it at two CSV files from your scraper and it produces coded tables,
statistics, and an interactive network you open in a browser.

The only project-specific material is a **study folder** (a few JSON files). The code
is topic-agnostic. **API calls are off by default**: the normal path writes a Markdown
file that you paste into any chatbot.

```text
input CSV  ->  prepared CSV  ->  AI answers  ->  coded tables  ->  11_network.csv  ->  network.html
```

Every stage prints a `DONE` block with a short summary. `prepare`, `merge`,
`classify-api` and `export-network` also draw a progress bar on stderr, but only when
`tqdm` (in `requirements.txt`) is installed. Stages that chain into another step end
with copy-pasteable `NEXT STEPS`; a few still contain `<placeholders>` for paths only
you know.

---

## What you actually need to start

The pipeline needs only three things: two CSVs from your scrape and one study folder.
Everything else on this page is something the pipeline produces from them.

| Input               | Required    | Where it comes from                                                                     | What it does                                                                                                  |
| ------------------- | ----------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| Comments file (CSV) | yes         | your scraper, one row per comment                                                       | The corpus: the text that gets coded                                                                          |
| Posts file (CSV)    | recommended | the same scrape, one row per post                                                       | Lets comments be linked to their post, and prefills most of `post_meta_template.csv`                          |
| Study folder        | yes         | `python -m framing init --id <name> --title "<title>"`, or an existing `studies/<name>` | Holds the codebook (topics, frames, threads) and the source/actor registries. The only project-specific part. |

Two clarifications that matter in practice:

- **No API key is an input.** The default path is offline: you get a Markdown file,
  paste it into a chatbot, and save the reply as a CSV.
- **The pipeline never reads its own output as input.** `prepare` refuses to treat
  `comments_enriched.csv`, `ai_output.csv` or `11_network.csv` as a corpus, even when
  they sit in the same folder.

If your CSVs come from a different scraper, the column names are mapped in
`study.input_schema` (see [Configuration knobs](#configuration-knobs)); the default
map is for Apify's Facebook scraper.

## Quick start (recommended)

Open a terminal in this folder (the one containing `framing/` and
`requirements.txt`), then:

```sh
pip install -r requirements.txt
python -m framing
```

That is the whole interface for a non-expert. The guided mode asks for the study, the
input CSVs and the work folder, shows which steps are already done, and then runs the
same commands as the CLI. There are no flags to memorise and nothing to look up;
type `q` at the study menu or the step menu to stop.

If you prefer to point it at a specific folder:

```sh
python -m framing wizard --study studies/<id>/manifest.json --work work/<id>
```

Both flags are optional: `python -m framing wizard` starts the same menu without
pre-filling anything.

### What the menu looks like

```text
Work folder: work/toy_bikelanes
  #  Step      Status   What it does
  1.  prepare   DONE     Link posts and comments, cut the batches
  2.  classify  WAITING  Bundle the task for a chatbot (default), or call the API (opt-in)
  3.  merge     TODO     Ingest the AI answers into coded tables
  4.  analyze   TODO     User profiles, statistics and gap report
  5.  network   TODO     Write 11_network.csv for network.html
```

| Status    | Meaning                                                       |
| --------- | ------------------------------------------------------------- |
| `TODO`    | Not started.                                                  |
| `WAITING` | The chatbot bundle exists, but you have not saved its answer. |
| `DONE`    | The step finished and wrote its files.                        |

The next sensible step is marked `<-- suggested`.
Below the table the wizard prints the same steps as a numbered menu ending in
`q. quit`, then prompts `Step [n]:`. Type a number to run that step, or `q` to stop.
You can still run any step, in any order, as many times as you like: every stage only
reads the files of the previous one.

---

## The five steps

| #   | Step             | What it does                                                                                       | You get                                                                 | Then                                     |
| --- | ---------------- | -------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- | ---------------------------------------- |
| 1   | `prepare`        | Links comments to posts, adds thread/phase info, cuts batches                                      | `comments_enriched.csv`, `batches/`, `prompts/`                         | Step 2.                                  |
| 2   | `classify`       | Gives the AI the batches: as a `.md` bundle (default) or an API call                               | `classification.md` or `responses/api_answers.json`                     | Paste / save the AI answer, then step 3. |
| 3   | `merge`          | Joins the AI answer with the corpus, marks rows above the confidence floor, fills the review queue | `02_all_comments_coded.csv`, `06_review_queue.csv`, `merge_report.json` | Step 4.                                  |
| 4   | `analyze`        | Author profiles, hypothesis tests, gap report                                                      | `04_users_profile.csv`, `08_stats.csv`, `10_gap_report.md`              | Step 5.                                  |
| 5   | `export-network` | Flattens the coding into the viewer CSV                                                            | `11_network.csv`                                                        | Open `network.html` and upload that CSV. |

The first column of that table is the command name, except for steps 2 and 5: the menu
labels `classify` and `network` are stage names, not subcommands. `classify` runs
`export-md` (default) or `classify-api`; `network` runs `export-network`.

---

## The chatbot branch (the default)

This is what step 2 does unless you explicitly choose the API.

1. The pipeline writes `work/<id>/classification.md`: one file containing the
   instructions, the codebook and every batch of comments.
2. Open it, select all, copy, and paste it into any chatbot as a single message.
3. Save the chatbot's reply as `work/<id>/ai_output.csv`.
   If the bundle also contains a post section, save that reply as
   `work/<id>/ai_posts.csv`.
4. Choose step 3 (`merge`) in the wizard, or run the `merge` command below.

The answer must be a CSV whose first line is exactly the header the bundle prints
under "Required output format" (it is generated from `study.output_fields`; with
the default field set it is):

```csv
comment_id,is_institutional,frame_id,secondary_frame_id,thread_id,stance,evidence_level,confidence,rationale,literal_excerpt,emergent_thread_proposal,actor_category,actor_self_declared,actor_evidence,needs_human_review
```

`classification.template.csv` in the work folder contains that header, ready to use.
If the chatbot returns prose or a Markdown table, ask it to "return only a CSV with
exactly these columns and no extra text".

The post answer (when there is one) uses:

```csv
post_key,topic,primary_frame_id,source_type_guess
```

`merge` notices `work/<id>/ai_posts.csv` automatically, in guided mode and on the
command line alike. Use `--post-ai-responses` only when the post answer lives
somewhere else.

---

## The API branch (opt-in)

Network access is **disabled by default**. Nothing contacts a server unless you pass
`--enable-api` **and** a key is available.

| Flag           | Required              | Default                     | What it does                           |
| -------------- | --------------------- | --------------------------- | -------------------------------------- |
| `--enable-api` | yes, to allow network | off                         | The explicit switch that permits calls |
| `--api-key`    | no                    | `FRAMING_API_KEY` env       | Key; prefer the environment variable   |
| `--base-url`   | no                    | `https://api.openai.com/v1` | OpenAI-compatible endpoint             |
| `--model`      | no                    | `gpt-4o-mini`               | Model name                             |

```sh
# bash / sh
export FRAMING_API_KEY=...
# PowerShell: $env:FRAMING_API_KEY = "..."
# cmd.exe:    set FRAMING_API_KEY=...

python -m framing classify-api \
  --study studies/<id>/manifest.json \
  --work work/<id> \
  --out work/<id>/responses \
  --enable-api
```

Writes `responses/api_answers.json` and `responses/api_post_answers.json`, which
`merge` accepts exactly like the chatbot CSVs. In guided mode the key is typed at a
prompt that says it is visible while you type; it is kept only for that run (in the
process environment), never placed on a command line. Setting `FRAMING_API_KEY`
before starting is safer on a shared machine.

---

## Command reference

The wizard runs these for you. They are listed so you can script the pipeline or
re-run one stage. Every command accepts `--help`.

### Create and check a study

```sh
python -m framing init --id <id> --title "<title>" --language en --dir studies/<id>
python -m framing validate-config --study studies/<id>/manifest.json
python -m framing render-prompts --study studies/<id>/manifest.json --out work/<id>/prompts
```

| Command           | Flag             | Required | Default        | What it does                                     |
| ----------------- | ---------------- | -------- | -------------- | ------------------------------------------------ |
| `init`            | `--id`           | yes      | -              | Study id (also the folder name).                 |
|                   | `--title`        | yes      | -              | Human-readable title.                            |
|                   | `--language`     | no       | `en`           | Corpus language code.                            |
|                   | `--dir`          | no       | `studies/<id>` | Where the manifest and five modules are written. |
|                   | `--force`        | no       | off            | Replace an existing non-empty folder.            |
| `validate-config` | `--study`        | yes      | -              | Path to `manifest.json`.                         |
|                   | `--save-unified` | no       | -              | Write the merged config to this JSON file.       |
| `render-prompts`  | `--study`        | yes      | -              | Path to `manifest.json`.                         |
|                   | `--out`          | yes      | -              | Folder for the rendered prompts and schemas.     |

`init` writes `manifest.json` plus `study.json`, `sources.json`, `actors.json`,
`locations.json` and `codebook.json`, then validates the result. The content is a
neutral placeholder meant to be edited, not a finished study.

### 1. Prepare

```sh
python -m framing prepare \
  --study studies/<id>/manifest.json \
  --comments <comments.csv> \
  --posts <posts.csv> \
  --out work/<id>
```

If your two CSVs already sit next to `manifest.json`, let the pipeline find them by
their header instead of typing their long scraper names:

```sh
python -m framing prepare \
  --study studies/<id>/manifest.json \
  --corpus-dir studies/<id> \
  --out work/<id>
```

The command prints which file it picked for each role:

```text
COMMENTS  studies/<id>/dataset_...-comments-....csv
POSTS     studies/<id>/dataset_...-posts-....csv
```

It never mistakes its own outputs (`comments_enriched.csv`, `ai_output.csv`, …) for
inputs. If two CSVs could be the comments file it stops and asks you to pass
`--comments` explicitly.

| Flag                     | Required | Default | What it does                                                   |
| ------------------------ | -------- | ------- | -------------------------------------------------------------- |
| `--study`                | yes      | -       | Path to `manifest.json`.                                       |
| `--comments`             | no*      | -       | Raw comments CSV. *Required unless `--corpus-dir` is used.     |
| `--corpus-dir`           | no*      | -       | Folder whose CSVs are identified by header wording.            |
| `--posts`                | no       | -       | Raw posts CSV; needed to link comments to posts.               |
| `--post-meta`            | no       | -       | CSV keyed by `post_key` that overrides source, topic and date. |
| `--out`                  | yes      | -       | Work folder (created if missing).                              |
| `--pseudonymize-authors` | no       | off     | Send pseudonyms instead of author names to the AI.             |
| `--no-batches`           | no       | off     | Skip batch files. Do **not** use with the chatbot branch.      |
| `--no-messages`          | no       | off     | Skip the per-batch `*_user_message.txt` files.                 |

`prepare` also writes `post_meta_template.csv`: one row per post with `post_key` (the
comment's post URL), `source_name`, `source_type`, `post_author`, `post_date` and
`post_text` already filled in from the posts CSV and the study's source registry. Only
`post_topic` and `post_frame_id` are empty, because they are interpretive rather than
scraped. Fill them in, then re-run the same `prepare` command adding
`--post-meta work/<id>/post_meta_template.csv` (keep `--comments` or `--corpus-dir`, and
`--out`). An existing template is never overwritten, so your edits survive re-runs.
The wizard has no prompt for this flag; apply post labels on the command line.

### 2. Classify

```sh
# default: chatbot bundle
python -m framing export-md \
  --study studies/<id>/manifest.json \
  --work work/<id> \
  --out work/<id>/classification.md

# opt-in: API calls
python -m framing classify-api \
  --study studies/<id>/manifest.json \
  --work work/<id> \
  --out work/<id>/responses \
  --enable-api
```

| Command        | Flag           | Required | Default                     | What it does                        |
| -------------- | -------------- | -------- | --------------------------- | ----------------------------------- |
| `export-md`    | `--study`      | yes      | -                           | Path to `manifest.json`.            |
|                | `--work`       | yes      | -                           | Folder written by `prepare`.        |
|                | `--out`        | yes      | -                           | Path of the `.md` bundle to write.  |
| `classify-api` | `--study`      | yes      | -                           | Path to `manifest.json`.            |
|                | `--work`       | yes      | -                           | Folder written by `prepare`.        |
|                | `--out`        | yes      | -                           | Folder for `api_answers.json`.      |
|                | `--enable-api` | yes      | off                         | Allow network calls.                |
|                | `--api-key`    | no       | env                         | Key; defaults to `FRAMING_API_KEY`. |
|                | `--base-url`   | no       | `https://api.openai.com/v1` | OpenAI-compatible endpoint.         |
|                | `--model`      | no       | `gpt-4o-mini`               | Model name.                         |

### 3. Merge

```sh
python -m framing merge \
  --study studies/<id>/manifest.json \
  --work work/<id> \
  --ai-responses work/<id>/ai_output.csv
```

Add `--post-ai-responses work/<id>/ai_posts.csv` only if you also produced a post
answer; without the flag `merge` looks for that file itself.

| Flag                  | Required | Default                                     | What it does                                                                   |
| --------------------- | -------- | ------------------------------------------- | ------------------------------------------------------------------------------ |
| `--study`             | yes      | -                                           | Path to `manifest.json`.                                                       |
| `--work`              | yes      | -                                           | Folder written by `prepare`.                                                   |
| `--ai-responses`      | yes      | -                                           | One or more answer files: CSV, JSON or JSONL. Alias `--responses`.             |
| `--post-ai-responses` | no       | `<work>/ai_posts.csv` when present          | Post-level answers. Alias `--post-responses`.                                  |
| `--out`               | no       | `--work`                                    | Where the coded tables are written.                                            |
| `--min-confidence`    | no       | `study.classification.min_confidence` (`1`) | Hide records below this confidence (1 = low … 3 = high).                       |
| `--require-coverage`  | no       | -                                           | Fail if less than this share of eligible comments was classified, e.g. `0.95`. |

`merge` also reports **coverage**: the share of eligible comments (with text, not
institutional) that received an answer. A low number usually means a batch was skipped
or an answer file was incomplete. Records that need a human decision go to
`06_review_queue.csv` (flagged by the model or coded at confidence 1).

**Confidence** is `1` (low) to `3` (high). The default floor of `1` keeps every code.
**Institutional** comments are those by the official accounts listed in `actors.json`;
they are excluded from the coverage denominator because they are the page, not the
public.

If `--post-ai-responses` is omitted, `merge` uses `ai_posts.csv` from the work folder
when that file exists, so the usual chatbot workflow needs no extra flag.

### 4. Analyse

```sh
python -m framing analyze \
  --study studies/<id>/manifest.json \
  --work work/<id> \
  --out work/<id>
```

`analyze` runs the three smaller commands below. All four take the same flags:

| Flag      | Required | Default  | What it does               |
| --------- | -------- | -------- | -------------------------- |
| `--study` | yes      | -        | Path to `manifest.json`.   |
| `--work`  | yes      | -        | Folder written by `merge`. |
| `--out`   | no       | `--work` | Where the tables go.       |

| Command   | Writes                                                                             |
| --------- | ---------------------------------------------------------------------------------- |
| `users`   | `04_users_profile.csv`, `04b_contact_candidates.csv`, `07_community_relevance.csv` |
| `stats`   | `08_stats.csv`, `09_contingency_tables.txt`                                        |
| `gaps`    | `10_gap_report.md`, `10_gap_report.json`, `10_source_coverage.csv`                 |
| `analyze` | everything the three commands above write                                          |

### 5. Visualise

```sh
python -m framing export-network \
  --study studies/<id>/manifest.json \
  --work work/<id> \
  --out work/<id>/11_network.csv
```

| Flag      | Required | Default                    | What it does               |
| --------- | -------- | -------------------------- | -------------------------- |
| `--study` | yes      | -                          | Path to `manifest.json`.   |
| `--work`  | yes      | -                          | Folder written by `merge`. |
| `--out`   | no       | `work/<id>/11_network.csv` | Path of the viewer CSV.    |

Then open `network.html` (in the repo root, beside `framing/`) in a browser and upload
`11_network.csv`. That is the final deliverable.

`11_network.csv` is a filtered view, not the whole corpus: it leaves out comments with
no codebook frame, emergent threads at frame level, and comments with empty text. It
therefore has fewer rows than `02_all_comments_coded.csv`.

### Optional: inter-coder reliability

```sh
python -m framing compare \
  --study studies/<id>/manifest.json \
  --a work/<id> \
  --b work/<id>_second_coder \
  --out work/<id>
```

| Flag      | Required | What it does                                     |
| --------- | -------- | ------------------------------------------------ |
| `--study` | yes      | Path to `manifest.json`.                         |
| `--a`     | yes      | First merged folder.                             |
| `--b`     | yes      | Second merged folder (a second coder's answers). |
| `--out`   | yes      | Where `reliability_report.json` is written.      |

Folder `--b` is produced the same way as `--a`: run steps 1-3 a second time into
`work/<id>_second_coder`, giving `merge` the second coder's answer file.
Re-run `analyze` afterwards so the gap report sees the reliability file.

### Optional: codebook discovery

```sh
# 1. Write the seed where validate-config and discovery-input read it. Back up first.
cp studies/<id>/codebook.json studies/<id>/codebook.json.bak
python -m framing seed-codebook \
  --brief studies/<id>/research_brief.json \
  --out studies/<id>/codebook.json

# 2. Prepare the corpus first: discovery-input reads the prepared work folder.
python -m framing prepare \
  --study studies/<id>/manifest.json \
  --corpus-dir studies/<id> \
  --out work/<id>

# 3. Build one discovery round.
python -m framing discovery-input \
  --study studies/<id>/manifest.json \
  --brief studies/<id>/research_brief.json \
  --work work/<id> \
  --out work/<id>/discovery \
  --round 1

# 4. Save the chatbot's JSON reply as
#    work/<id>/discovery/discovery_round_01_answer.json, then apply it.
python -m framing apply-discovery \
  --study studies/<id>/manifest.json \
  --response work/<id>/discovery/discovery_round_01_answer.json \
  --out-codebook studies/<id>/codebook.json

python -m framing validate-config --study studies/<id>/manifest.json
```

| Command           | Flag                                               | Default | What it does                                     |
| ----------------- | -------------------------------------------------- | ------- | ------------------------------------------------ |
| `seed-codebook`   | `--brief` / `--out`                                | -       | Research brief in, seed codebook out.            |
|                   | `--no-frame-id`                                    | `NONE`  | Id of the "no frame" pseudo-frame.               |
|                   | `--id-prefix`                                      | `F`     | Prefix of generated frame ids (`F1`, `F2`, ...). |
| `discovery-input` | `--study`, `--brief`, `--work`, `--out`, `--round` | -       | Build one discovery round.                       |
|                   | `--sample-size`                                    | `80`    | Comments sampled for the round.                  |
|                   | `--seed`                                           | `1`     | Sampling seed (repeatable rounds).               |
| `apply-discovery` | `--study`, `--response`, `--out-codebook`          | -       | Validate the answer and rewrite the codebook.    |
|                   | `--history-dir`                                    | -       | Where `codebook_history.jsonl` is appended.      |

`discovery-input` writes `discovery_round_01_input.json`,
`discovery_round_01_user_message.md`, `discovery_system.md`,
`discovery_response_schema.json` and `discovery_state.json`. Paste the user message into
any chatbot, save its JSON reply as `discovery_round_01_answer.json` in that same
folder, and pass that path to `apply-discovery`, which validates it.

---

## Output files

The numbered deliverables (`01_…` through `11_network.csv`) are defined once in
`framing/files.py`; the exchange files (`classification.md`, `ai_output.csv`, …) are
written by the stage that owns them.

| File                                | Written by         | What it contains                                                  | Read by                     |
| ----------------------------------- | ------------------ | ----------------------------------------------------------------- | --------------------------- |
| `comments_enriched.csv`             | `prepare`          | Comments linked to posts, with relevance and phase info           | `merge`                     |
| `posts_table.csv`                   | `prepare`          | One row per post                                                  | `merge`, `analyze`          |
| `post_meta_template.csv`            | `prepare`          | Editable per-post metadata; topic and frame left blank            | You, then `--post-meta`     |
| `join_report.json`                  | `prepare`          | How many comments matched a post, and why some did not            | You                         |
| `batches/`                          | `prepare`          | Comment batches ready for the AI                                  | `export-md`, `classify-api` |
| `post_batches/`                     | `prepare`          | Post batches, when posts were supplied                            | `export-md`, `classify-api` |
| `prompts/`                          | `prepare`          | System prompts, user-message templates, schemas, manifest         | You                         |
| `run.json`                          | `prepare`, `merge` | Config hash and run metadata                                      | Later stages (staleness)    |
| `classification.md`                 | `export-md`        | The whole task in one file, to paste into a chatbot               | You / your chatbot          |
| `classification.template.csv`       | `export-md`        | Empty CSV with the exact expected header                          | You                         |
| `classification.posts.template.csv` | `export-md`        | Same, when posts exist; header follows the selected post fields   | You                         |
| `ai_output.csv`                     | You                | The chatbot's comment answer                                      | `merge`                     |
| `ai_posts.csv`                      | You                | The chatbot's post answer (optional)                              | `merge`                     |
| `responses/api_answers.json`        | `classify-api`     | The API's comment answers                                         | `merge`                     |
| `responses/api_post_answers.json`   | `classify-api`     | The API's post answers                                            | `merge`                     |
| `01_selected_comments.csv`          | `merge`            | Comments that took a codebook frame at/above the confidence floor | You                         |
| `02_all_comments_coded.csv`         | `merge`            | The coded corpus; the main table of the pipeline                  | `analyze`, `export-network` |
| `03_posts_classified.csv`           | `merge`            | Post-level classification                                         | `analyze`, `export-network` |
| `05_phases_timeline.csv`            | `merge`            | Comments per phase over time                                      | You                         |
| `06_review_queue.csv`               | `merge`            | Records that need a human decision                                | You                         |
| `merge_report.json`                 | `merge`            | Coverage, rejected records, issue examples                        | You                         |
| `04_users_profile.csv`              | `analyze`          | One row per author with activity and framing profile              | You                         |
| `04b_contact_candidates.csv`        | `analyze`          | Authors worth contacting, with evidence                           | You                         |
| `07_community_relevance.csv`        | `analyze`          | Which communities the frames appear in                            | You                         |
| `08_stats.csv`                      | `analyze`          | Hypothesis tests (chi-square, Cramer's V)                         | You                         |
| `09_contingency_tables.txt`         | `analyze`          | The tables behind the tests                                       | You                         |
| `10_source_coverage.csv`            | `analyze`          | Which sources and locations are actually represented              | You                         |
| `10_gap_report.md`                  | `analyze`          | What the data cannot support yet, in plain language               | You                         |
| `10_gap_report.json`                | `analyze`          | The same gaps as structured data                                  | You                         |
| `reliability_report.json`           | `compare`          | Agreement between two coders                                      | `analyze`                   |
| `11_network.csv`                    | `export-network`   | The exact input of `network.html`                                 | `network.html`              |

---

## Troubleshooting

| Symptom                                       | Likely cause                             | Fix                                                                                                                         |
| --------------------------------------------- | ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `ERROR: <file>: required columns not found`   | Your CSV uses different column names     | Map them with `study.input_schema` (see knobs below), or rename.                                                            |
| `WARNING: only 94.7% ... classified`          | A batch or an answer file was skipped    | Check the answer file covers every `comment_id`; then re-run `merge`.                                                       |
| `merge` fails with `--require-coverage`       | Same as above, promoted to an error      | Only pass `--require-coverage` when full coverage is expected.                                                              |
| `WARNING: ... was built with config hash ...` | The work folder predates a study edit    | Re-run `prepare` and `merge` into a fresh `--work` folder.                                                                  |
| `network.html` says columns are missing       | The CSV header was edited or reordered   | Re-run `export-network`; never rename its columns.                                                                          |
| `ERROR: API classification is disabled`       | API calls are off by default             | Add `--enable-api` and set `FRAMING_API_KEY`.                                                                               |
| `ERROR: ... FRAMING_API_KEY`                  | No key in the environment                | Export the key, or let the wizard ask for it.                                                                               |
| Real-corpus test is skipped                   | Its environment variables are unset      | Set both `FRAMING_REAL_COMMENTS` and `FRAMING_REAL_POSTS`.                                                                  |
| `FileNotFoundError` on `manifest.json`        | Wrong study path                         | Run from the repo root, or pass the full path to `--study`.                                                                 |
| The wizard says `WAITING` forever             | The answer is not inside the work folder | Save it as `<work>/ai_output.csv`; pointing `merge --ai-responses` at another path ingests it but does not clear `WAITING`. |

---

## The network viewer

`network.html` is the final deliverable and is fully self-contained: Cytoscape and
Papaparse are inlined, so there is no build step, no server and no network access.
Open it directly in a browser (double-click it), then upload `11_network.csv`.

It draws the topic, the argument frames and the threads inside them, with a stance pie
per thread. It checks that all 16 required columns are present before drawing (extra
columns are tolerated, but the names must match):

```csv
post_topic,post_url,frame_id,frame_label,thread_id,thread_label,author,comment_text,published_at,stance,stance_group,relevance_score,likes,reply_count,thread_depth,is_reply
```

A missing or renamed column produces an explicit English error listing the missing
columns, so a bad header fails loudly instead of drawing a partial graph.

---

## Configuration knobs

Behaviour that used to be fixed is driven by keys in the `study` / `actors` /
`locations` / `codebook` modules. Anything omitted falls back to the default shown.

| Key                                          | Default            | What it controls                                                     |
| -------------------------------------------- | ------------------ | -------------------------------------------------------------------- |
| `study.analysis_language`                    | `en`               | Language of labels, rationales and review notes.                     |
| `study.corpus_language`                      | required           | Language of the comments.                                            |
| `actors.levels[].coverage_check`             | `true`             | Set `false` to exempt an actor level from the self-declaration gap.  |
| `locations.coverage_audit.level`             | off                | Gazetted level checked for "never mentioned"; omit to disable.       |
| `study.stats.tests`                          | built-in           | Replaces the hypothesis set (`default_tests` in `framing/stats.py`). |
| `study.stats.min_expected_warning`           | `5`                | Expected-cell threshold behind the low-cell warnings.                |
| `study.ethics.enabled` / `study.ethics.text` | `true` / generic   | Whether and how the ethics gap is reported.                          |
| `study.gap_checks.join_warning_share`        | `0.05`             | Share of unlinked comments above which the join gap is raised.       |
| `study.gap_checks.min_frame_comments`        | `30`               | Frames below this are skipped by the frame-level tests.              |
| `study.gap_checks.sparse_frame_comments`     | `5`                | Frames below this are flagged as sparse.                             |
| `study.gap_checks.severe_frame_comments`     | `10`               | Frames below this raise a `high` (not `medium`) gap.                 |
| `study.input_schema`                         | Apify Facebook map | Per-section column override for other scrapers.                      |
| `study.classification.emergent_thread_id`    | `EMERGENT`         | Thread id the model uses to propose a brand-new thread.              |

Allowed `study.stats.tests[].kind`: `omnibus`, `one_vs_rest`, `any_frame`,
`thread_within_frame`. Allowed `pop`: `citizen`, `coded`, `citizen_gender`,
`coded_dedup`. The full schema is `schemas/modules.schema.json`.

### Answer fields

The fields the classifier is asked to fill are declared once, in
`framing/fields.py`, and a study can trim, reorder or re-word them without touching
code:

```json
"output_fields": ["comment_id", "frame_id", "thread_id", "stance", "confidence"],
"output_field_help": {"stance": "one of {stances}, or \"\" for no-frame"}
```

Omit `output_fields` to keep the full default vocabulary. `comment_id` and
`frame_id` are structural (the join key and the analysis pivot) and cannot be
dropped; every other field is optional. A dropped field is not asked for, but the
record keeps a safe default, the analysis tables keep the (empty) column, and the
gap report stops running the checks that depend on it instead of inventing values.
Dropping `confidence` counts every record as `2` ("probable"), so the confidence
filter keeps working.

| field                      | tier       | in CSV         | default when absent              |
| -------------------------- | ---------- | -------------- | -------------------------------- |
| `comment_id`               | structural | yes            | `""`                             |
| `frame_id`                 | structural | yes            | `""`                             |
| `is_institutional`         | core       | yes            | `false`                          |
| `thread_id`                | core       | yes            | `""`                             |
| `stance`                   | core       | yes            | `""`                             |
| `evidence_level`           | core       | yes            | `""`                             |
| `confidence`               | core       | yes            | `2` (when not asked for)         |
| `needs_human_review`       | core       | yes            | `false`                          |
| `secondary_frame_id`       | optional   | yes            | `""`                             |
| `emergent_thread_proposal` | optional   | yes            | `""` (needs `thread_id`)         |
| `literal_excerpt`          | optional   | yes            | `""`                             |
| `rationale`                | optional   | yes            | `""`                             |
| `review_reason`            | optional   | no (JSON only) | `""`                             |
| `actor_category`           | optional   | yes            | `""`                             |
| `actor_self_declared`      | optional   | yes            | `false` (needs `actor_category`) |
| `actor_evidence`           | optional   | yes            | `""` (needs `actor_category`)    |

Three generic coding rules are dropped with the field they exist for
(`literal_excerpt`, `is_institutional`, the actor trio). Rules that only _mention_
a field as part of the wider method stay, so trim with care: the JSON schema is
the authority on which fields are required.

#### Stance and evidence vocabulary

The values of `stance` and `evidence_level` are not fixed either: they are the keys
of two codebook sections, in the order they are declared there.

```json
"stance_definitions": {
  "a_favor": "Sustains the argument of the thread.",
  "en_contra": "Rejects the argument of the thread.",
  "matiza": "Accepts it only in part."
},
"evidence_definitions": {
  "core": "States the thread's own claim.",
  "adjacent": "Related evidence, weaker than core."
}
```

That single definition drives the prompt (`{{STANCES}}` / `{{EVIDENCE}}` blocks),
the `enum` of the JSON response schema, the "permitted values" block of the chatbot
CSV instructions, the record checks and the gap report. Two conventions matter:

- **Order is meaning.** The prompt and the schema list the ids in declaration order,
  and the **last** evidence level is the _weakest_ one (the gap report warns when a
  frame rests mostly on it). Declare the levels strongest first.
- **A value outside the vocabulary is never silently kept.** It fails the record
  check, lands in the review queue and is blanked, so a mis-typed id is visible
  instead of being counted as a real category.

`EMERGENT` is a different kind of value: it is a _protocol sentinel_ — the thread id
that means "no existing thread fits, the proposal is in `emergent_thread_proposal`".
Rename it with `study.classification.emergent_thread_id` when the corpus language or
the convention needs another token; every consumer (schema enum, prompt, merge
labels, statistics exclusion, network export) follows the configured value.

Both layers are plain files you can edit: `schemas/modules.schema.json` says what a
study is allowed to declare (widen it when you need a new key), and
`prompts/*.template.md` holds the prose of the prompts — the `{{NAME}}` blocks are
filled from the study config, so keep their names and change the words around them.
The generic reasoning rules adapt to the declared order on their own: the calibration
rule says "choose the cautious stance" / "choose the weakest level" using your ids.

#### Post answer fields

The post classifier uses the same mechanism (`framing/post_fields.py`), with
`study.output_post_fields` (an ordered list) and `study.output_post_field_help`
(a field-to-wording map; the help text may use `{frames}`, `{frames_list}`
and `{source_types}`). Omit both to keep the full default vocabulary, in this order:
`post_key`, `topic`, `primary_frame_id`, `related_frame_ids`, `source_type_guess`,
`rationale`, `confidence`. Only `post_key` is structural (the join key) and cannot be
dropped; every other field may be. The JSON prompt asks for every selected field; the
chatbot CSV prompt carries only the fields asked for in CSV, four by default. A
dropped field is not asked for, and every consumer tolerates its absence:

| field               | in JSON | in CSV | default when absent | fallback                                                               |
| ------------------- | ------- | ------ | ------------------- | ---------------------------------------------------------------------- |
| `post_key`          | yes     | yes    | `""`                | - (join key; cannot be dropped)                                        |
| `topic`             | yes     | yes    | `""`                | viewer labels the topic node from the post text                        |
| `primary_frame_id`  | yes     | yes    | `""`                | `post_frame_id` stays empty (dominant frame still comes from comments) |
| `related_frame_ids` | yes     | no     | `()`                | never ingested                                                         |
| `source_type_guess` | yes     | yes    | `""`                | unregistered posts stay `unclassified` / group `other`                 |
| `rationale`         | yes     | no     | `""`                | never ingested                                                         |
| `confidence`        | yes     | no     | `0`                 | never ingested                                                         |

```json
"output_post_fields": ["post_key", "topic", "primary_frame_id", "source_type_guess"],
"output_post_field_help": {"topic": "short label, at most 8 words"}
```

With the defaults untouched the rendered post prompt, the post JSON schema and the
post CSV header are byte-identical, so existing studies need no re-classification.

---

## Glossary

| Term           | Meaning                                                               |
| -------------- | --------------------------------------------------------------------- |
| Study          | One folder under `studies/` with a `manifest.json` and five modules.  |
| Frame          | An argumentative frame from your codebook; a comment may sustain one. |
| Thread         | A sub-topic inside a frame.                                           |
| Stance         | The position a comment takes, using the study's stance vocabulary.    |
| Batch          | A chunk of comments small enough to send to the AI at once.           |
| Work folder    | The one folder every stage reads and writes (`work/<id>` by default). |
| Chatbot branch | Human-in-the-loop classification through a pasted `.md` file.         |
| API branch     | Automatic classification, off unless explicitly enabled.              |

---

## Project layout

```
framing/       The package. `python -m framing` runs the CLI and the wizard.
studies/       One folder per study: manifest + study/sources/actors/locations/codebook.
prompts/       Prompt templates (do not rename).
schemas/       JSON-Schema for every study module.
tests/         Offline unittest suite; one opt-in real-corpus regression.
outputs/       Study deliverables and QA artefacts (not code).
network.html   Self-contained viewer for 11_network.csv.
```

`framing/config.py` sets `PKG_ROOT` to the repository root, which is why `prompts/`,
`schemas/` and `network.html` sit beside `framing/` rather than inside it.

The shipped studies are `studies/toy_bikelanes` (2 frames, 3 threads; the test
fixture) and `studies/waterscope_viana` (4 frames, 17 threads; one example project,
including its `research_brief.json` and legacy ETL helpers).

---

## Tests

```sh
python -m unittest discover -s tests -v
```

The suite is offline. `tests/test_waterscope_regression.py` is skipped unless both
`FRAMING_REAL_COMMENTS` and `FRAMING_REAL_POSTS` point at the original scraper CSVs
**and** `studies/waterscope_viana` exists.
