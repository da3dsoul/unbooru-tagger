# Comparison with other taggers and approaches (stub)

> **Status: stub.** This page sets out *what* to compare and *what is already known about this project*. Accuracy
> and performance data are still being updated and improved. Cells marked **TBD** must be filled from measured
> runs or from the other projects' own published figures, with a citation. Nothing here has been benchmarked
> head-to-head yet.

## What this project does differently

These points are design facts, from `CLAUDE.md` and the code. They are not benchmark results.

- **Tags can be added later without retraining.** Each tag is a row in an embedding table. `add-tag` appends one
  row and fine-tunes only that row, with the image encoder frozen: 200 steps by default, optionally warm-started
  from a frozen text embedder. A fixed classification head (the WD14 tagger and DeepDanbooru are this project's
  named prior art) has to be expanded and retrained to learn a new tag.
- **Localization without box labels.** Heatmaps come from the same dot product as the tag scores, at every map
  cell. Approximate boxes are extracted from the heatmaps. No box annotations were used in training. The gold sets
  are evaluation-only.
- **Training cost.** The maintainer's position is that this model is much faster to train than the alternatives.
  The project's own figures are below; the comparison figures are **TBD**.
  - Run 4 trains at about 142 ms per step (batch 10 x 2 gradient accumulation, 512px, ConvNeXt-T with 3 fused
    stages, bf16) on one Intel Arc Pro B70.
  - Run 2's configuration ran at 158 ms/step on the B70 and 278 ms/step on an RTX A4000.
  - Run 4's first 4 epochs (1,939,684 steps, held-out mAP 0.801) took about 123 B70-hours. This is an estimate
    from wall clock and recorded step times, not a measured total; see
    [timeline.md](timeline.md#run-4-mosaics-and-context-negatives-10-26-to-now) for how it was made. Evaluations
    ran on a second GPU (an RTX A4000) and are not counted.
- **Single .NET stack.** It trains with TorchSharp and serves with ONNX Runtime, from one C# CLI, on CUDA or the
  CPU. Training also runs on Intel Arc (XPU).

## Comparison table (to fill)

| Approach | Vocabulary | Adding a tag | Localization | Box labels needed | Training cost | Tag accuracy | Localization accuracy |
|---|---|---|---|---|---|---|---|
| **unbooru-tagger** (this project, run 4 epoch 4) | Embedding table (run 4: 18,452 rows, from the dataset's 24,899) | One row, encoder frozen (`add-tag`) | Heatmaps and approximate boxes | No | About 123 B70-hours for 4 epochs (estimate); about 145 ms/step | Held-out mAP 0.801, hard-negative mAP 0.862, under this project's own protocol (not comparable to other projects' metrics without re-scoring) | Gold pointing 67.7% (78.0% within 16px); gold-small 55.6%; gold-auto (SAM 3 boxes, held out) 65.9% |
| Fixed-head booru taggers (WD14 tagger, DeepDanbooru) | Fixed head | Retrain or expand the head | TBD | TBD | TBD | TBD | TBD |
| JoyTag | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| CLIP/SigLIP-style zero-shot scoring | Open (text) | Write a prompt | TBD | TBD | TBD | TBD | TBD |
| Open-vocabulary detectors (OWLv2, Grounding DINO) | Open (text) | Write a prompt | Boxes | Yes: the open-vocabulary plan notes that both are trained from boxes or pseudo-boxes | TBD | TBD | Calibrated here only as gold-auto proposers (see below) |
| SAM 3 | Open (noun phrases) | Write a prompt | Boxes and masks | Yes (boxes and masks) | TBD | TBD | Calibrated here only as a gold-auto box source (see below) |

## What has been measured against other models, and how

These are the only head-to-head numbers in the sources. They measure the other models **as box sources for
evaluation data**, on gold-small's verified boxes (instance recall / precision at IoU ≥ 0.5), not as taggers:

| Box source | Found | Right |
|---|---|---|
| OWLv2 base + Grounding DINO base, score ≥ 0.2 | 66% | 27% |
| OWLv2 large, ≥ 0.05 | 80% | 25% |
| Qwen2.5-VL 3B as a detector | 37% | 51% |
| SAM 3, ≥ 0.7 | 55% | 88% |
| SAM 3, ≥ 0.8 | 43% | 95% |

See [datasets-and-evaluation.md](datasets-and-evaluation.md#gold-auto-sam-3-boxes-gold-auto) for the full
calibration.

## To do before this page can make claims

- [ ] **Tag accuracy against other taggers.** Pick a shared protocol, for example per-tag AP on run 4's held-out
  images, with the same label completion (implications, site-blind tags). Run each other tagger's released model
  through it. Tags outside a model's vocabulary need a rule.
- [ ] **Localization against detectors.** Score OWLv2, Grounding DINO and SAM 3 as *taggers with boxes* on
  gold-small and the main gold set, with the pointing and box metrics `evaluate --gold` uses. Note that gold-auto
  was built from SAM 3, so SAM 3 can't be scored on it.
- [ ] **Training cost.** Run 4's own total is estimated (about 123 B70-hours to epoch 4); it still needs comparing
  against what other projects report, with their hardware named.
- [ ] **Adding tags.** Measure `add-tag` on held-out new tags (time, images needed, resulting AP), against
  retraining a fixed head.
- [ ] **Inference speed.** Images/s for `tag-batch` and `detect-batch` on CPU and on CUDA, against the other
  taggers on the same hardware.
