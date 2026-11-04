# Research notes

This folder documents the research behind unbooru-tagger:

- what it set out to do;
- what was tried, in order, with the results, including the failures and the mistakes;
- what data was gathered and how;
- what the evidence supports.

For using the tagger, see the [README](../README.md). For training it, see [docs/TRAINING.md](../docs/TRAINING.md).

## Contents

| Page | What it covers |
|---|---|
| [goals.md](goals.md) | The original goals, why a dual encoder, and the maintainer's rules that shaped the research. |
| [timeline.md](timeline.md) | The story from the first commit through run 1, run 2, run 3, run 4, run 5 (glimpses) and the current work. |
| [datasets-and-evaluation.md](datasets-and-evaluation.md) | The crawled dataset, label corrections (ratings, implications, site coverage, tag parentage), the gold sets (gold, gold-small, gold-auto), the labelled context check and the missing-labels pass. |
| [experiments-localization.md](experiments-localization.md) | Heatmaps and boxes, the localization loss, tower fusion, the separate map, crop consistency, background suppression, mosaics. |
| [experiments-context-shortcut.md](experiments-context-shortcut.md) | Deliberate targets, context negatives, negative gamma, context prior, classification pooling, out-of-context and excluder sampling, the context-margin and refit checks. |
| [experiments-erased-views.md](experiments-erased-views.md) | Erased-view (occlusion) training, its patch-cue artifact and the anchor term. |
| [experiments-glimpses.md](experiments-glimpses.md) | Run 5's model-chosen glimpses, locality negatives and the glimpse pooling investigation. |
| [experiments-window-labels.md](experiments-window-labels.md) | The ceiling test, mosaic and twin window labels, pasted twins, and the open-vocabulary plan (self-map windows, presence x window, 3x3 mosaics). |
| [experiments-measurement.md](experiments-measurement.md) | Every metric, when and why it changed, and the evaluation bugs. |
| [engineering-and-hardware.md](engineering-and-hardware.md) | TorchSharp on an Intel Arc Pro B70 and an RTX A4000: speed, memory, driver faults, pitfalls. |
| [conclusions.md](conclusions.md) | The core findings, with links to the evidence. |
| [comparison.md](comparison.md) | A stub comparing this project with other taggers and approaches. This project's own numbers are filled in (run 4, epoch 4); the other approaches' are still to do. |
| [mistakes-and-lessons.md](mistakes-and-lessons.md) | Data incidents, bugs, evaluation bugs, artifacts mistaken for gains, and operational mistakes. |
| [citations.md](citations.md) | The papers the notes reference, with arXiv links where the sources give them. |

## Conventions

- **Steps** are global optimizer steps of a run.
- **An "arm"** is one branch of an A/B: a copy of a checkpoint trained for a fixed number of steps with one
  change, against a control trained the same way without it.
- **"Gold pointing"** is the share of gold (image, tag) pairs whose heatmap peak lands in a gold box.
- **"Object effect"** is mean logit(control blanked) − logit(object blanked) over every gold pair.
- **Every number** comes from a source listed below. Where a value was not found, the text says "not recorded".
  Where two sources disagree, both are given.
- **Numbers from different measurement protocols are not comparable.**
  [experiments-measurement.md](experiments-measurement.md) lists the protocol changes.
- **People are not named.** "The maintainer" is the project's owner. "Claude subagents" and "a Sonnet labeller"
  are AI agents used for labelling and analysis.

## Sources

### In git (this repository)

- **[`CLAUDE.md`](../CLAUDE.md):** architecture rationale and the detailed experiment log. It is the richest
  single source.
- **The commit history** (`git log`): especially `c618a2b` (deliberate training), `5efeacf` (XPU), `c40d156`
  (mosaics and context negatives), `3e350d6` (object-or-context checks), `a8764f0` (bf16, determinism) and
  `85713a4` (glimpses and the window-label experiments).
- **The code:** option descriptions in `unbooru-tagger-training/Program.cs` and
  `unbooru-tagger-inference/Program.cs`, and class doc comments.

### Outside git

These are external artifacts in the maintainer's data directory, outside the repository.
They are named here by their relative paths. None of this data is in git, and none was copied into this folder.

| Path | Content |
|---|---|
| `dataset/` | The crawled corpus (`images.bin`, `crawl.sqlite`, `tag_rows.sqlite`, the vocabulary and the exported label files). |
| `gold/`, `gold-small/`, `gold-auto/` | The evaluation box sets, with their README, labelling rules and tools. |
| `context-negatives/`, `context-negatives-2/`, `context-negatives-3/`, `context-check/`, `ctxfp/`, `context-margin/` | The context false-positive investigation. |
| `missing-labels/` | The SAM 3 missing-labels pass (results in datasets-and-evaluation.md). |
| `review/` | Experiment notes and A/B arms: `open-vocab-research/PLAN.md`, `pooling/PLAN.md`, `pointing/NOTES.md`, `window-probe/`, `locality/`, `selfmap/`, `selfmap-neg/`, `mosaic3/`, `run4-twins/`, `twin-check/`, `content-pool/`, `glimpse-dump/`, and the run comparison tables (`v4-e2s365k.txt`, `v5-e1s200k.txt`). |
| `ab-*/` | Earlier A/B arms. `ab-erasure/README.txt` and its `compare*.txt` files hold the erased-view results. |
| `run1-small-convnext-epoch12-COMPARISON-ONLY/README.txt` | Run 1's baseline. |
| `train-v2.sh` to `train-v5.sh`; `xpu-test/evaluate-on-cuda.sh`, `evaluate-v5-on-cuda.sh`, `eval-queue.sh`, `watchdog-v5.sh` | The real training and evaluation commands, with a dated change log in their header comments. |
| `overnight-2026-10-28.md` | One night's log: memory, determinism, step profiling, bf16. |
| `tag-parentage/` | The reviewed tag-parentage rules. |

Two further sources are outside that directory:

- **The project's memory notes** for its AI assistant, which record decisions, the maintainer's rules and
  incidents. They live in the assistant's own configuration, not in the repository.
- **The occlusion-noise literature notes** (`unbooru-scratch/occlusion-noise/literature.md` on the scratch
  drive).
