# Timeline: run 1 to the current work

This page tells the story in order. Experiment details are in the themed pages, linked below. All dates are 2026.
"Step" means the global optimizer step of the run. Numbers come from the commit messages, `CLAUDE.md`, the
memory notes and the review notes. A number marked "not recorded" was not found in any of them.

**A caveat for every gold-pointing number below:** the main gold set was picked from run 1's validation split.
Only 30 of its 289 images are in run 4's held-out split, so run 4 (and run 5, which uses the same split) trained
on the tags of the other 259 gold images. Gold-small and gold-auto were built later from run 4's held-out
images, partly to fix this. See [datasets-and-evaluation.md](datasets-and-evaluation.md).

## July: scaffolding, data import, crawler

| Date | Event |
|---|---|
| 07-17 | Four-project solution scaffolded: Core, Inference CLI, Training CLI, Tests. TorchSharp has no ONNX exporter, so the ONNX graph is built by hand (OnnxSharp protobuf types) and tested against TorchSharp's forward pass. |
| 07-17 | First real bugs (see [mistakes-and-lessons.md](mistakes-and-lessons.md)). The pooled embedding and the spatial map were in different spaces; fixed by making `Projection` a 1x1 conv applied before pooling. A NaN loss at step 1 led to LayerScale. Training ran only on the CPU until a `Device` parameter was threaded through. |
| 07-17 | `unbooru-tagger-data` (later `-unbooru-import`) pulls images and tags from the maintainer's `unbooru` SQL Server database. Early stopping on validation loss. |
| 07-19 | Resumable large-cache builds; checkpoint resume with optimizer state. |
| 07-20 | `detect` (boxes from heatmaps, percentile thresholds, bilateral refinement). Two label-free auxiliary losses: an MIL/log-sum-exp localization loss and a SimSiam consistency loss. |
| 07-20 | `unbooru-tagger-data-booru-downloader`: a crawler for Danbooru and Gelbooru that writes the same cache format. |
| 07-21 to 07-29 | Crawler hardening: perceptual-hash dedup, crash-safe checkpointing, Gelbooru pagination fixes, SQLite tag rows, co-occurrence hard-negative mining. See [datasets-and-evaluation.md](datasets-and-evaluation.md#the-dataset-and-crawler). |

## August: data incidents

| Date | Event |
|---|---|
| 08-04 | **Vocabulary corruption.** A system crash mid-crawl left `tag_vocabulary.json` at 0 bytes, because the save was not atomic. The vocabulary was rebuilt from `crawl.sqlite` with a new `rebuild-vocabulary` command; 263 of 2,611,746 images (0.01%) had no recoverable tags. The rebuild grew the vocabulary from 6,341 tags (an earlier checkpoint, `checkpoint-medium`) to 15,299, which made that checkpoint impossible to resume as is. |
| 08-06 | **Gelbooru silent search redirect.** A search for `water_fountain` returned exactly the posts of `fountain`, with no trace in any alias listing. Fixed by trusting the site's own search result for the searched tag. |
| 08-24 | **Damaged pixel cache.** A `prune-unsupported-media` compaction reported success at the right row count but wrote empty 0x0 records for part of the corpus. `recover-corrupted-images` rebuilt the cache in a staging directory and re-downloaded the rest, MD5-verified. The number of affected rows is given only as "millions of rows" in the commit message. |
| 08-31 / 09-01 | Crawl and alias fixes. Mid-epoch checkpoints (every 15 minutes by default), and a persisted validation-split seed so a resume doesn't leak validation images into training. |
| 09-11 | Inference reads the input size from the ONNX graph instead of assuming 224. |

## Run 1: small ConvNeXt (August to October)

- **Model:** a 1.7M-parameter tower (64/128/256 channels, 2/2/2 blocks), 512px input, 32x32 heatmaps.
- **Vocabulary:** 24,899 rows, of which 6,468 were stale rows that no image carried. No rating labels.
- **Training:** random-image batches. Every untagged tag counted as an equally certain negative. Early stopping on
  mean validation loss, which stopped it after epoch 12 at about 0.005.
- **Epoch 12** (`run1-small-convnext-epoch12-COMPARISON-ONLY/README.txt`), held-out, 20K images:
  - mean per-tag AP 0.72; hard-negative AP 0.80; image precision 0.38 / recall 0.43;
  - gold pointing 41.8%; box precision 1.8% / recall 19.5%.

  These numbers were measured under run 1's own protocol. Later runs changed the negatives and the label
  completion, so they are not directly comparable.
- **Why it ended:** noisy boxes, false tags and misses, while validation loss looked fine. Mean loss over about
  25K tags per image is about 600:1 easy negatives; an all-negative predictor scores about 0.014.

This led to the "deliberate training" rework (commit `c618a2b`, 10-24): tag-centric batches, certainty-weighted
targets, label corrections from the crawl data, and ranking-based evaluation. See
[experiments-context-shortcut.md](experiments-context-shortcut.md#deliberate-targets) and
[experiments-measurement.md](experiments-measurement.md).

## Run 2: ConvNeXt-T with fusion, on the A4000 (10-22 to 10-24)

- **Configuration:**
  - ConvNeXt-Tiny (96/192/384/768 channels, 3/3/9/3 blocks) with 2 fused stages, giving 64x64 heatmaps;
  - batch 10 x 2 gradient accumulation (batch 20 ran out of memory on the 16 GB A4000);
  - lr 1e-4 cosine with 2,000 warmup steps; deliberate targets with tag-centric sampling;
  - sparse localization loss (weight 0.3, temperature 0.25); self-supervised loss off; ratings on;
  - vocabulary compacted from 24,902 to 18,434.
- **Speed:** about 278 ms per micro-step, about 37 hours per epoch.
- **Created before the GroupNorm init fix.** TorchSharp left norm weights uninitialized, which is likely zeros on
  CUDA. It was judged recoverable by training and not restarted.
- **Changes made while it ran:**
  - `--localization-map fused` won a 3,000-step A/B: gold pointing 27.0% → 29.0%, deletion win 75% → 82%.
  - The context-negative band was introduced.
  - Danbooru implication completion added 523,867 labels on 507,666 images.
  - Site-blind tags were excluded from the negatives.
- **Stopped** 10-24 01:19 at epoch 2, step about 119K. Evaluated then: held-out mAP 0.551, gold pointing 29.9%.

## Run 3: fused-3 on the Intel Arc Pro B70 (10-24 to 10-26)

- **Configuration:** run 2's recipe, fresh, with 3 fused stages. That gives a stride-4 map: 128x128 heatmaps at
  512px, aimed at small objects.
- **Hardware:** the first run on the 32 GB Arc Pro B70 (XPU), with epoch evaluations offloaded to the A4000. Speed
  172 ms/step; fused-3 holds 17.9 GiB at batch 10.
- **Results:**

  | | Epoch 1 | Epoch 2 |
  |---|---|---|
  | mAP | 0.673 | 0.748 |
  | Gold pointing | 27.0% | 32.9% |
  | Context recall gap | +20.8 pts | +29.4 pts |
  | Context false-positive gap | +20.7 pts | +25.6 pts |
  | Still called with the object blanked | 87% | 83% |

- **Why it was paused** (epoch 3, step 26,019): small objects were detected but localized at about 0-5%. The new
  occlusion and context-reliance checks showed the classifier answering from character and artist context, and
  the shortcut grew from epoch 1 to epoch 2.
- **A/Bs from run 3:** mosaics, full-strength context negatives, and the local map. See
  [experiments-localization.md](experiments-localization.md) and
  [experiments-context-shortcut.md](experiments-context-shortcut.md).

## Run 4: mosaics and context negatives (10-26 to now)

Run 4 is run 3's recipe restarted fresh with `--mosaics-per-step 2` and `--context-negative-weight 3`. Its real
command is `train-v4.sh` (outside git; reproduced in [docs/TRAINING.md](../docs/TRAINING.md#run-4s-recipe)).

| Date / step | Change |
|---|---|
| 10-26 ~16:08 | Launched on the B70. The dashboard showed about 244 ms/step right after launch; `train-v4.sh`'s header gives about 218 ms/step "with mosaics", measured in the mosaic A/B arms. Both are right for what they measured: step time is noisy with mosaics and the variable-width localization pairs (identical arms differ by several percent), and a fresh process includes warm-up. Later figures track builds: 223 (fp32, 10-28) → 199 (`--bf16`) → 221 by 11-02 (why it rose is not recorded) → 142 (run 5's speed work, 11-02) → about 145 (padded pair rows, 11-03). |
| 10-27, ~155K | Snapshot evaluations every 10,000 steps, and trend charts on the dashboard. |
| 10-27, ~275K | `--negative-gamma 2` (was 4), after an A/B. |
| 10-28, 315,834 | Restarted on dispose-scoped forwards, deterministic evaluation seeds and `ForwardPooled`. |
| 10-28, 04:17-08:23 | Host RAM pressure: about 3 s/step for about 4 hours, until a graceful restart. |
| 10-28, 359,388 | `--bf16`: 223 → 199 ms/step. |
| 10-28, ~502K | `--tag-parentage`. Evaluations from 510K on score families, so they are not comparable with earlier points. |
| 10-29 22:30 | The gold checks pool over content only. The charted object effect steps up about 0.15 here. |
| 10-30, ~1.325M | Labelled context-check charts. |
| 10-30 | Epoch 3 review: mAP 0.785, gold pointing 65.6%, still called with the object blanked 51.8%. |
| 10-31, 1,495,198 | Paused so the B70 could run glimpse work. |
| 11-02 22:19 (~1,499,226) | Resumed at the maintainer's choice, on the build with run 5's speed work: 221 → 142 ms/step, same recipe. A 12K-step arm on that build reached mAP 0.788, against run 4's own 0.787. |
| 11-03, ~1.78M | Localization pair rows padded to multiples of 64. Before that, VRAM ratcheted from 12 to 24 GB in 10 minutes and the trainer died with xe "VM worker error: -12". |
| 11-03 | Pointing arms from 1,507,198 (12K steps each, not adopted): context-negative weights 4/6/10 and a stride-2 localization level. A 2x zoom test on the gold sets showed small objects gain 6-12 points from more pixels. See [experiments-localization.md](experiments-localization.md#higher-context-negative-weights-2026-11-03-not-adopted). |
| 11-03 ~23:00, 1,939,684 | Epoch 4 complete. Review: mAP 0.801 (epoch 3: 0.785), gold pointing 67.7% (65.6%). |

**Latest recorded readouts** (epoch 4, step 1,939,684; epoch 3 in brackets):

- held-out mAP 0.801 (0.785); hard-negative mAP 0.862 (0.850); image precision / recall 45.8% / 52.8% (44.0% / 51.5%);
  exclusive violations 4.5% (5.4%);
- gold pointing 67.7% (65.6%), within 16px 78.0%; by size: small 37.4% (40.7%), medium 60.5% (57.3%), large 71.1%
  (67.5%), characters 91.0% (87.9%);
- gold-small pointing 55.6% (56.3%), within 16px 71.5%; gold-auto (SAM 3 boxes, 500 held-out images) 65.9%, within
  16px 74.8%, small 43.7%;
- object effect: gold 1.29 (1.13), gold-small 0.89 (0.77), gold-auto 1.28;
- absent objects called on the labelled context set 8.1% (10.2%); visible-but-untagged 76% (75%);
- context recall / false-positive gaps 28.9 / 23.5 points (27.5 / 22.6).

Per tag on gold (>= 8 pairs), epoch 3 to 4: wings 30.4% to 52.2%, tail 32.3% to 45.2%, flower 56.8% to 67.6%, but
choker 76.2% to 52.4%, umbrella 75.0% to 50.0%, shoes 75.6% to 67.4%. Box precision and count error jump between the
two epochs only because of the box-speck fix (see [experiments-measurement.md](experiments-measurement.md)).

**Training cost so far, an estimate:** about 123 B70-hours for 4 epochs (1,939,684 steps). That is the wall clock
from launch to the 10-31 pause (105.7 h for 1,495,198 steps, about 254 ms/step with restarts, warm-ups and the 4-hour
host-RAM slowdown included) plus the steps since the resume at the recorded ~145 ms/step (440,458 steps, about 17.7 h;
the A/B arms that shared the B70 then are left out). Offloaded evaluations on the A4000 are not counted.

The [experiments-measurement.md](experiments-measurement.md) page explains each metric.

## The first glimpse run, deleted (10-30 to 10-31)

The first glimpse run was an attempt to score object tags from close-up windows. Two things were wrong with it:

- **Training:** the labels picked its training windows. Half framed the positives' peaks; the rest were random.
- **Evaluation:** it scored a fixed 378-window grid, about 6x a whole-image pass and about 30 minutes per
  evaluation on the A4000.

So the model never learned where to look, and training and evaluation scored different windows. The maintainer
had it deleted ("V5 was a mistake"). See [experiments-glimpses.md](experiments-glimpses.md#the-first-glimpse-run-deleted).

## Run 5: model-chosen glimpses (10-31 to 11-01/02)

- **Recipe:** run 4's recipe, fresh, with `--classification-glimpses`:
  - a proposal head ranks suspect tags;
  - the localization map gives each suspect's peak;
  - a sensor looks at 32/64/128/256px windows, 48 per image;
  - a log-sum-exp pools the windows.

  It uses the same validation split as run 4.
- **Speed:** built for the B70 (padding, matmul 1x1 convs, channels-last, pinned uploads): from 426 down to
  205 ms/step.
- **Copy-engine faults:** these recurred until `OverrideCopyOffloadMode=0`.
- **At 200K:** mAP 0.490 against run 4's 0.512 at the same step; gold pointing 27.7% against 30.1%.
- **The maintainer saw run 5's charts tracking run 4's.** The per-window dumps showed every glimpse learning the
  image's label. The pooling was nearly a mean: the temperature is in dot units, before the learned scale of
  about 8.6.
- **Locality negatives** were in run 5 for a while. They were removed because of a device-memory leak, and their
  gain later turned out to be an artifact.
- **Stopped at step 277,022.** See [experiments-glimpses.md](experiments-glimpses.md).

## Window labels and the open-vocabulary plan (11-01 to 11-03)

- **Pooling investigation (11-01/02):** 6K and 12K-step arms from run 5 at 189,803. No label-only change made the
  windows read the object.
- **Ceiling test:** fine-tuning on gold-box windows raised medium 64px on/off AUC from 0.66 to 0.80. So the
  sensor could read objects; the training signal was missing.
- **Tag-derived window labels:** mosaic tiles and matched twins cost mAP. Pasted twins on run 4 were noise.
- **Open-vocabulary research plan (11-03):**
  - Self-map windows (labels from the model's own detached map) reached 0.82 AUC with no hand labels. Nothing
    run 4 scores uses windows, though.
  - Presence x window scoring didn't beat whole-image scoring.
  - Negative windows and 3x3 mosaics were null.

See [experiments-window-labels.md](experiments-window-labels.md).

## Resolution, not aliasing: hi-res stores and mixed scales (11-03 to 11-04)

- **11-03:** a 2x zoom of the gold images at evaluation raised small-object pointing 6-12 points: the object's pixels,
  not the map's grid, limit small objects.
- **11-04 night:** the gold originals (1,435, downloaded with the maintainer's approval) separated the cache's
  nearest-neighbour aliasing from resolution. A clean 512px shrink scored like the cache; real 768/1,024px pixels
  gained 10-16 points. Lanczos was chosen for the stores.
- **11-04:** fine-tuning run 4 at 768/1,024 (from the cache, enlarged) kept mAP and gained 10-20 points on small
  objects, losing up to 7 on large ones. A ≤1-day pull of six weak small-object tags' originals at 1,024
  (`store1024`, ~192K rows) raised those tags' AP in proportion to coverage (wristband 0.47 → 0.61). A model
  trained only on enlarged images gained nothing from real pixels at test time.
- **11-04 13:13:** run 4 moved to mixed scales (`--train-scales`: 512/768/1,024/2,048 per micro-batch) with the store,
  and a second pull was queued: 38K collage/crowd/comic images at 2,048, 310K small-object images at 1,024 (the six
  tags plus ~19 more), 36K medium-object images at 768. Snapshot evaluations add the gold checks at 1,024.

See [experiments-localization.md](experiments-localization.md#aliasing-or-resolution-the-gold-originals-2026-11-04).

## Current work (11-04)

- Run 4 is training on mixed scales on the B70, under the watchdog. The 768 and 2,048 stores join it once the second
  pull has rows in them.
- The plan is to gather resolution where evaluations show weakness: by tag (small objects at 1,024, medium at 768)
  and by kind of image (collages, comics, crowds at 2,048), and to adjust the scale mix as the 1,024 trend charts
  and per-tag AP move.
- Open: whether 2,048 gains over 1,024 on the gold images that have originals that large (scoring in progress); the
  occlusion object effect falls a little at higher input sizes (1.32 → 1.17), not yet explained.
