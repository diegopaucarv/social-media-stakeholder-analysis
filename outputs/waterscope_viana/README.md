# waterscope_viana — archived legacy run

This directory holds an archived run made before the file contract was standardised: its deliverables use
Spanish filenames and Spanish column headers. Current runs write the canonical English names defined in
`framing/files.py`, so everything under `legacy/` is history only. The files were moved verbatim, not renamed
or rewritten, so the archived data is byte-for-byte the original.

## Legacy → canonical mapping

| legacy file | current canonical file | note |
| --- | --- | --- |
| `legacy/00_informe_metodologia_hallazgos_y_faltantes.md` | `10_gap_report.md` (+ `10_gap_report.json`) | One narrative combining methodology, findings and gap inventory; current runs emit a machine-readable `10_gap_report.json` plus a generated `10_gap_report.md`. |
| `legacy/01_comentarios_seleccionados.csv` | `01_selected_comments.csv` | Same content (185 selected comments); Spanish column headers. |
| `legacy/02_comentarios_todos_codificados.csv` | `02_all_comments_coded.csv` | All 694 coded comments. |
| `legacy/03_posts_clasificados.csv` | `03_posts_classified.csv` | 29 posts with comments plus post-less rows. |
| `legacy/04_usuarios_perfil.csv` | `04_users_profile.csv` | Also carries a `candidato_contacto` column; current runs split that out into `04b_contact_candidates.csv`. |
| `legacy/05_fases_linea_de_tiempo.csv` | `05_phases_timeline.csv` | Per-post timeline phases. |
| `legacy/06_cola_revision_humana.csv` | `06_review_queue.csv` | Low-confidence / flagged comments queued for human review. |
| `legacy/07_estadistica_chi2_cramer.csv` | `08_stats.csv` | Chi-square / Cramér's V results, renumbered. The `07` slot is now `07_community_relevance.csv` (per-location relevance), which this run did not produce. |
| `legacy/08_tablas_contingencia.txt` | `09_contingency_tables.txt` | Contingency tables; the canonical file also carries the adjusted standardised residuals. |
| `legacy/08b_residuos_ajustados_T1.csv` | n/a | T1 (frame × space type) residuals extract, superseded by the adjusted-residuals section of `09_contingency_tables.txt`. |
| `legacy/docs/` | n/a | Process diagram (`proceso_tres_fases.puml`); study documentation, not a pipeline deliverable. |
| `legacy/intermediate/` | n/a (canonical intermediates are `comments_enriched.csv`, `posts_table.csv`, `join_report.json`, `batches/`, `post_batches/`) | Holds an ad-hoc Gemini simulation dump (`gemini_sim.json`); this run was coded manually, so none of the canonical intermediates were written. |
| `legacy/qa/` | n/a (canonical QA files are `reliability_report.json` and `merge_report.json`) | Blind-test extracts (`diferencias_extra_ciega.csv` and `prueba_ciega/`) from the old manual QA pass. |

This run predates source-coverage reporting and the network export, so `10_source_coverage.csv` and
`11_network.csv` have no legacy counterpart either.

`python -m framing analyze` and `python -m framing export-network` regenerate the canonical files, so the legacy
copy is reference-only and must not be read by the pipeline.
