# Experiments: localization

This page covers how the heatmaps and boxes were trained and read. "Gold pointing" means the share of gold
(image, tag) pairs whose heatmap peak falls inside a gold box. The metrics themselves are defined in
[experiments-measurement.md](experiments-measurement.md).

## Heatmaps and boxes at inference (July)

- **Question:** can the dual encoder localize without box labels?
- **Setup:** the heatmap is `sigmoid(scale * dot(tag, map cell) + bias)` at every cell (MaskCLIP-style). Boxes are
  connected components over a thresholded heatmap, built in three steps:
  1. a per-tag percentile cut (`--box-percentile`, default 0.6);
  2. an absolute floor (`--box-threshold` 0.5);
  3. a joint bilateral filter that snaps the heatmap to the image's edges (the label-free analogue of
     CAM+DenseCRF).

  Later additions:
  - a minimum region of 0.2% of the map;
  - at most 3 boxes per tag;
  - a minimum heatmap range of 0.05;
  - whole-image tags reported without boxes.
- **Results:** run 1's box precision was 1.8% at recall 19.5% on gold. The gold checks' box metrics were
  meaningless until 2026-11-01 (the speck bug; see
  [experiments-measurement.md](experiments-measurement.md#gold-box-metrics-and-specks)).
- **Verdict:** kept. Boxes stay "rough, approximate localization, not a trained detector" (`CLAUDE.md`).

## MIL / log-sum-exp localization loss (July, revised October)

- **Question:** can a loss reward a sharp spatial response directly, instead of leaving localization to the
  pooling geometry?
- **Setup:**
  - Each tag is scored from every location and pooled by a log-sum-exp at a temperature (default 0.35; runs 2-5
    use 0.25 with weight 0.3).
  - At a high temperature this equals the main loss, because the dot product distributes over the average.
  - It is trained in vocabulary chunks (`--localization-tag-chunk-size` 4096), each backpropagated at once. The
    result is exact; this was checked by a gradient-equivalence test.
- **Sparse version** (with deliberate targets, `LocalizationPairSelector`): only localizable positives, certain
  negatives, and the 32 highest-scoring wrong tags per image. That is about 100 pairs instead of about 25K. Step
  time fell from 1,413 to 468 ms on the old tower; equivalence with the dense loss is tested.
- **Verdict:** adopted, in sparse form, from run 2 on.

## SimSiam self-supervised loss

- **Setup:** two random-crop views of a batch predict each other's pooled embedding, through a predictor head with
  stop-gradient on the target. No labels are used.
- **Verdict:** on by default in code, **off in runs 2-5** (`--self-supervised-weight 0`). The run-2 notes say it
  "conflicts with crop-sensitive localization". No A/B numbers are recorded.

## Larger tower and FPN-style fusion (run 2, October)

- **Question:** small objects (earrings, hairclips, chokers) pointed at near chance on run 1's 32x32 map.
- **Setup:** a ConvNeXt-T tower. `--fused-stages` adds a top-down path: the last stage's map is upsampled 2x and
  added to a 1x1 projection of the previous stage, once per fused stage. 2 fused stages give 64x64 at 512px
  (run 2); 3 give 128x128 (runs 3-5).
- **Results:** no isolated A/B of fusion alone is recorded. Run 3 with fused-3 at epoch 2 reached gold pointing
  32.9%, against run 2's 29.9%, but other things also differed.
- **Verdict:** adopted, fused-3 from run 3 on.

## A separate localization map: `--localization-map fused` (run 2, epoch 2)

- **Question:** every cell of the last stage sees most of the 512px image. Should localization and heatmaps read
  the fused laterals alone?
- **Setup:** a 3,000-step A/B. Classification still pools the full map. Exported as an extra ONNX output
  (`localization_features`).
- **Results:** gold pointing 27.0% → 29.0% (clean 29.7% → 33.9%); deletion win 75% → 82%.
- **Verdict:** adopted from run 2's epoch 2.

## `--localization-map local` (run 3)

- **Question:** does leaving out the coarsest (stride-16) lateral, so a cell sees about 100px, fix small objects?
- **Setup:** A/B in `ab-local/` (run 3).
- **Results:** it fixed shoes and gloves pointing, but lost glasses, blush, hats and characters. No aggregate
  numbers are recorded in the sources.
- **Verdict:** not adopted. The open-vocabulary plan later suggested the loss may have been "the price of
  dropping the only level that sees them". That was not measured.

## Background suppression (run 2)

- **Setup:** `--localization-background-weight`. For each tag an image has, its least tag-like 40% of locations
  are trained as negatives.
- **Results:** it shrank the heatmaps but didn't move the peaks. No numbers are recorded.
- **Verdict:** not adopted.

## Crop consistency (run 2)

- **Question:** can the classifier's verdict on a random crop teach the map where a tag is?
- **Setup:** `--crop-consistency-weight` (`SigmoidContrastiveLoss.ComputeWindowConsistency`).
- **Results:**
  - **Ungated**, it made localization worse. The crop-teacher check showed the epoch-1 classifier at chance on
    crops (51% balanced accuracy): it answered from context.
  - **Gated** (`CropTeacherGate`: held-out AP ≥ 0.8, ≥ 20 positives, context AP ≥ 0.8, confident answers only),
    it cropped 72% of images via common tags (long_hair, breasts), for +26% step time and about 1% confident
    answers.
- **Verdict:** off. The maintainer later added a rule: don't add zoom-in crops trained on whole-image labels,
  because they teach context.

## Mosaic localization training (run 3 A/B, adopted in run 4)

- **Question:** can the localization map learn "the gloves are in that tile, not this one", even on a character
  who usually wears them?
- **Setup:**
  - Each step, `--mosaics-per-step` canvases of 4 batch images, shrunk 2x into a 2x2 grid.
  - Each tile keeps all its tags (unlike a crop, nothing has to be learned from context).
  - Each tile is its own localization bag, with the other tiles' positive tags as negatives.
  - A/B at 6,000 steps from run 3 (`ab-mosaic/`).
- **Results** (mosaic against none):
  - gold pointing 33.1% → 35.7%: small 6.6 → 7.8, medium 16.1 → 20.5, large 24.1 → 28.8;
  - mAP flat (0.750 vs 0.751); +28% step time for 2 mosaics;
  - the context gaps and occlusion didn't move (recall gap 29.7 → 30.0; still called 83.9% → 85.9%).

  Mosaics train only the localization map, never the classifier.
- **Verdict:** adopted in run 4 (2 per step). It fixes pointing, not the context shortcut. That took full-strength
  context negatives: see [experiments-context-shortcut.md](experiments-context-shortcut.md).

### Variants tried later on run 4

- **Tile classification and full-size mosaics** (`--mosaic-classification-weight`, `--mosaic-full-size`;
  6,000-step A/Bs from 271,754): small gains. Full size cost 1.4 mAP points, and 2 full-size canvases ran out of
  memory on the B70. Not adopted.
- **3x3 mosaics** (`--mosaic-large-grid 3`: every 2nd step, one 3x3 canvas of 4x-shrunk images in place of the two
  2x2 mosaics; after OWLv2's mixed grids). 12K steps from ctl4; `review/mosaic3/`. ctl5 / m3:

  | Readout | ctl5 | m3 |
  |---|---|---|
  | mAP | 0.7886 | 0.7882 |
  | gold pointing | 67.1% | 67.6% |
  | gold pointing, small | 37.7% | 38.8% |
  | gold-small pointing | 56.1% | 53.5% |
  | gold-auto pointing, small | 42.7% | 43.0% |
  | gold-auto pointing, large | 76.0% | 71.8% |
  | step time | 139 ms | 134 ms |

  Null: small objects fall below the map's resolution at 4x shrink, and large objects lose a little. Not adopted.

## Where the map points wrong (2026-11-03)

See [experiments-measurement.md](experiments-measurement.md#pointing-by-size-tolerant-pointing-and-where-misses-land).

- **Near misses:** about half of small-object misses land within 16px, usually on the adjacent body part (the hand
  rather than the glove).
- **Far misses:** usually on the character's face or body, or on another character. The map answers "where is the
  character who would have this tag", which is the context problem at map level.

## Pointing A/B: stronger localization terms (2026-11-03, not adopted)

The maintainer would trade a little overall quality for better small and medium pointing. Three arms of 12,000 steps
each ran on the B70 from run 4's ctl4 (the self-map A/B's start, ~1.78M), against ctl5 (run 4's recipe from the same
start). Each was scored on gold, gold-small and gold-auto, strict and within 16px (`review/pointing-ab/`,
`compare.py`):

- `--localization-weight 0.6` (run 4: 0.3);
- `--mosaic-weight 3` (run 4: 1);
- `--localization-temperature 0.1` (run 4: 0.25).

| pointing, strict / 16px | ctl5 | lw06 | mw3 | t01 |
|---|---|---|---|---|
| gold small (228) | 37.7 / 57.3 | 42.3 / 61.3 | 41.3 / 61.4 | 31.5 / 55.0 |
| gold medium (452) | 60.1 / 73.9 | 63.5 / 75.4 | 59.7 / 72.7 | 57.9 / 72.1 |
| gold-small small (217) | 37.3 / 54.4 | 35.5 / 55.3 | 35.9 / 54.4 | 23.5 / 47.5 |
| gold-small medium (303) | 67.3 / 76.2 | 65.7 / 74.9 | 64.0 / 75.6 | 62.4 / 74.6 |
| gold-auto small (611) | 42.7 / 61.2 | 40.6 / 61.2 | 43.5 / 63.5 | 35.2 / 58.3 |
| gold-auto medium (1,506) | 69.0 / 75.3 | 70.1 / 76.0 | 68.8 / 75.8 | 65.5 / 74.7 |
| mAP | 0.789 | 0.789 | 0.789 | 0.789 |
| object effect, gold / gold-small | 1.17 / 0.81 | 1.17 / 0.80 | 1.12 / 0.74 | 1.15 / 0.80 |
| exclusive violations | 5.1% | 6.1% | 6.5% | 5.6% |
| absent objects called | 10.7% | 11.7% | 10.7% | 11.2% |

- **lw06 and mw3:** the gains on gold small (+4.6 and +3.6) didn't repeat on gold-small or gold-auto. With a
  standard error of about 3 points at n = 228, that's noise. Pooled over the three sets, small strict is
  40.5 / 39.9 / 41.5% for ctl5 / lw06 / mw3, and medium 67.0 / 68.2 / 66.3%.
- **t01:** a sharper pool made small-object pointing worse on every set (gold-small small −14 points).
- No arm cost mAP, so the trade on offer wasn't needed; none of them bought anything.

A further lever was dropped before it was tried: mosaic canvases whose tiles share a character (one tile with the
tag, one without). Context negatives (a character's images without the tag) are already certain negatives in the
localization loss at weight 3, so such a canvas repeats a "no" the map already gets on the whole image.

Still untried: a label that says "not the hand" for near misses.

### Higher context-negative weights (2026-11-03, not adopted)

The maintainer asked whether a much higher `--context-negative-weight` would do anything (run 4: 3). Three more arms ran
from ctl4, the same way: 4, 6 and 10 (`review/pointing-ab/ab-cnw.sh`, `compare-cnw.py`). Gold and gold-small pointing
are from the reports (ctl5's gold-small pair dump is truncated, and its report predates the 16px figure, so its 16px
column is from the dumps, as above).

| | ctl5 (3) | cnw4 | cnw6 | cnw10 |
|---|---|---|---|---|
| mAP | 0.789 | 0.787 | 0.787 | 0.785 |
| micro precision / recall | 0.444 / 0.518 | 0.440 / 0.516 | 0.435 / 0.512 | 0.431 / 0.507 |
| context recall gap (pts) | 27.0 | 26.2 | 24.7 | 21.4 |
| context false-positive gap (pts) | 22.9 | 21.9 | 20.9 | 19.5 |
| object effect, gold / gold-small | 1.17 / 0.81 | 1.14 / 0.72 | 1.19 / 0.82 | 1.21 / 0.79 |
| absent objects called | 10.7% | 11.7% | 10.7% | 11.2% |
| gold small, strict / 16px | 37.7 / 57.3 | 39.8 / 63.4 | 41.6 / 61.3 | 39.7 / 58.9 |
| gold medium | 60.1 / 73.9 | 61.5 / 74.0 | 59.7 / 72.0 | 61.1 / 73.6 |
| gold-small small | 37.3 / 54.4 | 36.4 / 60.8 | 35.5 / 55.8 | 34.1 / 56.7 |
| gold-small medium | 67.3 / 76.2 | 64.4 / 75.6 | 64.4 / 77.9 | 67.0 / 76.6 |
| gold-auto small | 42.7 / 61.2 | 41.4 / 61.9 | 42.1 / 62.2 | 41.7 / 62.4 |
| gold-auto medium | 69.0 / 75.3 | 66.9 / 73.4 | 68.6 / 75.8 | 69.6 / 76.3 |

Pointing didn't move on any set beyond the noise (gold small's +2-4 points again didn't repeat on gold-small or
gold-auto). The weight does what it did in run 3 and at 6 in run 4: the context gaps shrink with the dose (recall gap
27 → 21 points at 10), but by calling less, since precision and recall both fall and absent objects aren't called less.
A context lever, not a pointing one.

### A stride-2 localization level (2026-11-03, not adopted)

The 128x128 map's misses within 4px are only ~10-13% of small pairs, so a finer map could buy a few strict points on small
objects at most. `--fine-localization` (`Model/FineLocalization.cs`) tests it: a stride-2 level under the localization map
(a 2x2 patchify of the pixels plus the stage-0 features, one 64-channel ConvNeXt block, a zero-initialised 1x1 projection
added to the nearest-upsampled coarse map), so the map is 256x256; classification is unchanged. Its own Adam, warmed up
over 1,000 steps.

Built the obvious way it didn't fit: 427 ms/step in a 60-step smoke and out of memory on the B70 within a minute of the
arm. Training now keeps the map in its parts (`FineLocalizationMap`): a tag's fine logits are `(eW) · features`, the
projection moved onto the tag side (64 channels instead of 512 at 256x256), and the log-sum-exp is taken over each
coarse cell's four fine cells and then over the cells; mosaic tiles are scored on their own quadrant of the canvas
instead of the whole canvas with three quarters masked. Same value and gradients (tested). 222 ms/step in a 300-step
smoke (run 4: 144), ~200 over the arm; VRAM 17.9 GB, then 23.3, then 32.0 GB (the card's limit, held for the last 24
minutes; the allocator's reserve, but no headroom). Options that read the whole map (glimpses, erased views, crop
consistency, self-map windows and the like) are refused with it.

12,000 steps from ctl4, against ctl5:

| | ctl5 | fine |
|---|---|---|
| mAP | 0.789 | 0.789 |
| object effect, gold / gold-small | 1.17 / 0.81 | 1.17 / 0.80 |
| absent objects called | 10.7% | 9.1% |
| exclusive violations | 5.1% | 6.4% |
| gold small, strict / 16px | 37.7 / 57.3 | 43.4 / 63.1 |
| gold medium | 60.1 / 73.9 | 62.8 / 74.7 |
| gold-small small | 37.3 / 54.4 | 35.5 / 55.3 |
| gold-small medium | 67.3 / 76.2 | 64.4 / 76.6 |
| gold-auto small | 42.7 / 61.2 | 40.4 / 60.9 |
| gold-auto medium | 69.0 / 75.3 | 67.9 / 75.8 |

The same pattern as the other arms: gold small up (+5.7), but not on gold-small (-1.8) or gold-auto (-2.3, n = 611).
Nothing else moved. Kept in the code (off by default) for a longer test if the map's resolution becomes the limit.

### Is it resolution? The 2x zoom test (2026-11-03)

The maintainer asked whether more small-object images would help. Per tag, small-object pointing barely follows how many
images a tag has (Spearman 0.41 over tags; gloves 907K images, 35%; ring 62K, 7%), and the worst tags are tiny jewelry at the hands, wrists and ears,
a few pixels at 512px. So the test was whether the model points better when the objects are bigger: ctl5 (run 4's recipe
at ~1.79M) scored on all three gold sets from their 512px images, and from 2x copies at 1024px input (`evaluate
--gold-images DIR --gold-input-size 1024`, the boxes doubled), upscaled by bicubic and by Real-ESRGAN x4plus anime 6B
(resized to 2x). Sizes and the 16px tolerance are in the 512px image's pixels; the same 4,511 pairs in every run
(`review/zoom/`, `analyze.py`).

| strict / 16px | 512 | bicubic 2x | AI 2x |
|---|---|---|---|
| gold small (214) | 38.8 / 58.9 | 50.0 / 61.2 | 50.9 / 63.1 |
| gold-small small (217) | 37.3 / 54.4 | 42.4 / 57.1 | 46.1 / 58.1 |
| gold-auto small (611) | 42.7 / 61.2 | 48.6 / 56.5 | 48.6 / 58.9 |
| gold medium (431) | 60.3 / 73.8 | 58.5 / 66.1 | 57.1 / 66.8 |
| gold-auto medium (1,505) | 69.0 / 75.3 | 56.9 / 61.9 | 58.7 / 64.5 |
| gold-auto large (600) | 76.0 / 78.7 | 55.7 / 61.5 | 58.3 / 63.0 |
| gold characters (383) | 90.9 / 92.2 | 83.3 / 84.3 | 86.2 / 86.9 |

- Small objects point better at 2x on all three sets, +6 to +12 strict points, from a model that never saw that scale. This
  is the first change that moved small-object pointing on every set. The stride-2 level's 256x256 map at 512px didn't, so
  it is the object's size in the input, not the map's grid.
- Medium and large objects and characters lose 2-20 points: at 1024px they are bigger than anything the model trained on.
- The AI upscale beat bicubic by 1-4 points on most rows. The cache's 512px images look aliased (thin lines broken into
  dots); `ImagePreprocessing.BuildLetterboxCanvas` resizes with `SKSamplingOptions.Default`, which as far as known is
  nearest-neighbour without mipmaps. Not yet confirmed or changed.
- Within 16px gains less (gold-auto small falls): zoom fixes near misses more than far ones. Hand-and-wrist jewelry (ring,
  bracelet, watch, wristband) stays near 0 at any scale; neckwear, necktie, bow and hair_flower gain 15-30 points.

Next: training on zoomed views (multi-scale), so small objects get the gain without large ones losing it.
