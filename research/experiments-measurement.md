# Measurement: metrics, how they changed, and evaluation bugs

Most of the research turned on finding a metric that measures the right thing with little noise. This page
defines the metrics, records when and why each changed, and lists the evaluation bugs. When comparing numbers
across a change listed here, check which side of the change each number is on.

## Held-out ranking (from 2026-10-24)

Mean validation loss was replaced. Early stopping now tracks **held-out mean per-tag AP**, scored against each
tag's hard negatives plus random negatives, with likely-missing images left out (`TagRankingEvaluator`). The
same pass also produces:

- per-tag best-F1 thresholds (`tag_thresholds.json`, shipped in the model bundle);
- image-level precision and recall at those thresholds;
- exclusive-pair violations;
- rating calibration, tuned for recall.

**Why.** Mean loss over about 25K tags per image is about 600:1 easy negatives. An all-negative predictor scores
about 0.014, and run 1 stopped at about 0.005.

**Comparability breaks in the mAP series:**

- **Site-blind negatives excluded** (run 2, epoch 2): epoch-2 mAP is not directly comparable with epoch 1.
- **Tag parentage** (run 4, from 510K): evaluations score families, so mAP and gold numbers are not comparable with
  earlier points.

## Gold-set localization

Scored by `evaluate --gold` and `--gold-small`. Boxes have certain, group and ignore kinds.

- **Pointing:** the heatmap peak falls in a certain box. Reported by stratum (clean/messy) and by size: small
  < 32px, medium 32-96px, large ≥ 96px, and characters.
- **Energy in box:** the share of heatmap mass inside the boxes.
- **Box precision, recall and count error:** from the extracted boxes.

**Label-free checks** on held-out images (`--localization-samples`):

- deletion: blanking the top cells beats blanking random cells;
- two-character distinct peaks;
- crop consistency;
- solo counts.

### Gold box metrics and specks

- **The bug:** until 2026-11-01, the gold checks extracted boxes with none of `detect`'s speck filtering. On a
  128x128 map, every above-cutoff speck became a box: about 150-190 per pair. Box precision was 0.1-0.3% and the
  count error about 100-190 in every run, run 4 at 1.49M included.
- **The fix:** use `detect`'s defaults (regions ≥ 0.2% of the map, at most 3 boxes per tag).
- **Effect on run 5 at about 105K:**

  | | before | after |
  |---|---|---|
  | count error | 176.7 | 1.7 |
  | box precision | 0.2% | 7.9% (characters 20.6%) |
  | box recall | 21.5% | 15.5% |

  Recall fell because the specks had hit some boxes by chance. Pointing and energy were unchanged.
- **Consequence: box numbers from before 2026-11-01 aren't comparable.** That includes run 1's 1.8% box precision.

### Whole-canvas vs. content pooling

- **The bug:** until 2026-10-29 22:30, the gold checks pooled whole-image scores over the whole canvas, letterbox
  bars included. Training and the held-out evaluation pool content only.
- **The fix:** the gold checks now pool content only. The kept checkpoints were re-scored both ways
  (`review/content-pool/`).
- **Gold object effect, whole canvas → content only:**

  | Checkpoint | Gold | Gold-small |
  |---|---|---|
  | 485K | 0.33 → 0.49 | 0.15 → 0.24 |
  | 550K | 0.41 → 0.52 | 0.25 → 0.34 |
  | 850K | 0.59 → 0.76 | 0.38 → 0.50 |

  The order is the same; the values are about 0.15 higher. Run 4's charted series steps up at the switch.
- **Not affected:** gold pointing and the crop teacher's crops.

### Tolerant pointing (2026-11-03)

- **Setup:** `GoldResult.TolerantPointing` counts a peak within 16 canvas pixels (about 4 map cells) of a box.
- **ctl5** (run 4 at 1.519M), strict → within 16px:

  | Set | All | Small | Medium | Large |
  |---|---|---|---|---|
  | gold | 67.1 → 76.3 | 37.7 → 57.3 | 60.1 → 73.9 | 68.4 → 72.0 |
  | gold-small | 56.1 → 68.1 | 37.3 → 54.4 | 67.3 → 76.2 | |
  | gold-auto | 64.6 → 72.9 | 42.7 → 61.2 | 69.0 → 75.3 | 76.0 → 78.7 |

  Characters on gold: 90.8 → 92.1.
- **Chance**, a random point in the content with the same boxes: gold small 0.5 → 2.4, medium 2.7 → 6.2.
- **Reading:** the tolerance adds about 2 points by chance on small objects, but about 19-20 for the model. About
  half of the small-object misses are real near misses.

### Pointing by size, tolerant pointing and where misses land

From `review/pointing/NOTES.md`, ctl5:

**Distance of misses from the nearest box** (gold / gold-auto):

| Size | ≤ 16px | 16-48px | 48-128px | > 128px |
|---|---|---|---|---|
| small | 31 / 32% | 15 / 17% | 26 / 28% | 27 / 22% |
| medium | 35 / 21% | 13 / 19% | 26 / 30% | 27 / 30% |

**Far misses** (beyond 16px): the object usually ranks 4th or lower among the map's separate peaks (gold / gold-auto: small 61 /
67%, medium 60 / 51%). The map responds more strongly at three or more other places.

**The object among the top 2 / top 3 peaks:**

| Set and size | Top 2 | Top 3 | Strict pointing |
|---|---|---|---|
| gold, small | 60.0% | 70.5% | 37.7% |
| gold, medium | 77.8% | 83.4% | 60.1% |
| gold-auto, small | 61.7% | 71.5% | 42.7% |
| gold-auto, medium | 81.7% | 86.5% | 69.0% |

The first gold-auto dump behind these figures was cut short (two runs of the same evaluation wrote one output
directory at once, so it held 2,382 of 2,718 pairs); the figures above are from a clean rerun
(`review/pointing/ctl5-auto-recheck/`) and agree with the report's strict pointing.

**What far misses look like:** the pointing peak sits on a character's face or body, often in multi-character,
crowded or comic images. In about 3-4 of 16 gold-auto examples, the peak was on a real instance that SAM 3 didn't
box, so gold-auto pointing is somewhat understated.

**Gold, non-character pairs** (hit / near miss / far miss):

| Stratum and size | Hit | Near miss | Far miss |
|---|---|---|---|
| clean, medium | 62.5% | 13.4% | 19.2% |
| clean, small | 47.3% | 20.9% | 20.9% |
| messy, medium | 56.7% | 14.4% | 23.3% |
| messy, small | 31.0% | 18.6% | 41.9% |

## Object or context?

### Occlusion: "still called with the object blanked" (2026-10-27)

- **Setup:** for each gold pair the classifier finds (p ≥ 0.5), blank the tag's boxes and score again. A control
  blanks the same sizes elsewhere.
- **Problem: noise.** The rate counts only pairs found at p ≥ 0.5. The median found score is 0.65 and the blanked
  one 0.54. Between checkpoints 4,500 steps apart, 15% of shared pairs flip and a quarter of the found set changes.
  Two identical 6,000-step controls differed by 7 points.

### Object effect (2026-10-29)

- **Definition:** mean logit(control blanked) − logit(object blanked), over **every** gold pair. It is
  threshold-free and computed over a fixed set.
- **Results:** it rose at every run-4 checkpoint: epoch 1 0.33 → 550K 0.41 → 622K 0.48 → 700K 0.51 (gold-small
  0.15 → 0.34), before the content-pooling switch. A "win rate" variant stayed flat at about 0.82 and was useless.
- **How to read it:** with the control drop, and on an unseen fill (`--occlusion-fill zero|mean|noise`). See
  [experiments-erased-views.md](experiments-erased-views.md).
- **Map-scored occlusion:** the version that scores from the localization map is off by default. 97% still called
  showed the fused map isn't local enough.

### Context reliance (2026-10-27)

- **Setup:** each held-out image's prior for a tag is the highest P(tag | its character/artist/series) in
  training. Recall and false-positive rate are compared across prior bands: recall gap 60%+ vs < 5%; FP gap 60-90%
  vs < 5%.
- **Why it replaced per-tag context AP:** on a 20K sample with a median of 2 context negatives per tag, context
  AP's chance level is about 0.98.
- **Caveat:** many high-prior "negatives" show the object untagged, so the FP gap also rises as the model gets
  better at finding untagged objects. See
  [experiments-context-shortcut.md](experiments-context-shortcut.md#context-false-positives-are-partly-missing-labels-2026-10-30).

### Labelled context check (2026-10-30)

- **Setup:** `--context-check`. Calls on hand- or Sonnet-labelled context negatives where the object is absent
  (true context false positives) versus visible but untagged.
- **What the charts show:** the **mean log-odds past the threshold**, not the called rates. About 200 pairs at
  thresholds recalibrated every evaluation flip by points. On run 4's snapshots, the absent rate jumped between
  neighbours by 0.44 of its trend's range; the log-odds by 0.08.

### Exclusive violations

- **Setup:** which mutually exclusive pairs fire together, and which side the image carries.
- **What it found:** the wrong `solo, no_humans` rule, and the people-count cost of context negatives on
  whole-image tags.

### Crop teacher and glimpse funnel

- **Crop teacher:** the classifier's balanced accuracy on crops; it gates crop consistency.
- **Glimpse funnel** (run 5): suspected → covered → best glimpse on the object → called. Its chance level is over
  all windows, not over the winning zoom. That flaw made the locality-negative artifact read as a gain; see
  [experiments-glimpses.md](experiments-glimpses.md).

### Window AUC (2026-11-02)

The within-pair on/off-object window AUC at a fixed zoom (64/128px). Mixed-size gaps and the effective number of
windows depend on the scale, so they mislead.

## Determinism and evaluation speed

- **Non-reproducible seeds:**
  - Seeds came from `HashCode.Combine`, which .NET randomizes per process. Two evaluations of one checkpoint
    differed by about 0.001 mAP and by dozens of images per context band.
  - Fixed 2026-10-28 with `DeterministicSeed` (SplitMix64), checked against an independent implementation.
  - Also: TorchSharp's `manual_seed` doesn't seed the XPU generator.
- **Evaluation memory:**
  - TorchSharp frees a tensor only when its dispose scope ends, so a forward held every intermediate. Gold scoring
    used 13.2 GB and ran the 16 GB A4000 out of memory.
  - Scoped forwards cut it to 4.9 GB, with byte-identical localization reports.
  - Measured live, with the CUDA cache off, a whole evaluation is about 1.5 GB median and 3.3-4.3 GB peak on the
    GPU; nvidia-smi's 7.5 GB is the caching allocator.
- **Evaluation speed (2026-10-29):**
  - A snapshot evaluation spent about 10 of its 14.5 minutes on 415 gold images. Every pass copied full fp32
    feature maps (about 33 MB per image) to the CPU.
  - After moving the pooling and heatmaps onto the device: 362 s instead of 839 s, with every number on the 850K
    checkpoint unchanged.
  - Run 5's evaluations add `--precision tf32` (held-out embedding -27%, gold -24%, mAP equal to 4 decimals) and
    `--no-crop-teacher`.

## Noise floor of an A/B

These figures limit what a 6,000-step arm can show:

- Two identical controls: per-tag AP rms 0.015, per-tag FP gap rms 0.07, still-called 54.3% vs 61.3%.
- The labelled context check differs by about 4 points between identical 6K controls.
- The B70 is power-limited: its clock swings between 2.2 and 2.8 GHz, so speed comparisons were made interleaved.
  Run-to-run ms/step at different times carries about 10% noise.
