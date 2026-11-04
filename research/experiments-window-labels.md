# Experiments: window-level labels and the open-vocabulary plan

**Sources:** `review/pooling/PLAN.md` (15:45 onward), `review/open-vocab-research/PLAN.md` (its "Results"
section), `review/window-probe/`, `review/twin-check/`, `review/run4-twins/`, `review/selfmap/`,
`review/selfmap-neg/` and `review/mosaic3/`, all outside git. The window AUC readout is the within-pair on/off
window AUC defined in [experiments-glimpses.md](experiments-glimpses.md#how-arms-were-judged).

## The ceiling test (2026-11-02)

- **Question:** is the 64px glimpse sensor itself the limit, or is the training signal missing?
- **Setup** (`window-probe`, `WindowProbeCommandHandler`):
  - ctl12's glimpse path fine-tuned on windows labelled from the gold boxes. "Yes" means the window holds half a box
    or is half filled by one; "no" means it touches none.
  - Tag rows and the logit scale frozen; balanced BCE.
  - 5 folds by image over 402 gold images. Held-out images re-embedded on ctl12's own chosen windows.
- **Results:**

  | | base | 400 steps | 1,500 steps |
  |---|---|---|---|
  | medium on/off AUC, 64px / 128px | 0.66 / 0.62 | 0.80 / 0.78 | 0.78 / 0.77 |
  | small on/off AUC, 64px / 128px | 0.67 / 0.60 | 0.78 / 0.73 | 0.74 / 0.69 |
  | medium, against windows on *another* labelled object, 64/128px | 0.65 / 0.63 | 0.80 / 0.79 | |

  The tag-agnostic "objectness" probe didn't move (0.57 → 0.55). The gain is in the tag-specific readout.
- **Verdict:** the sensor isn't the floor. Asked directly, with about 330 images and 400 steps, a 64px glimpse
  tells the tag's object from other objects at 0.80. **Run 5's losses never give a window a within-image "no".**
- **Constraint added by the maintainer:** training must not depend on the hand-made gold set. So the next arms
  derived window labels from the tags.

## Mosaic window labels (mg1)

- **Setup:** `--mosaic-glimpse-weight 1`, `Mosaic.GlimpseTerm`. Glimpses on the mosaic canvases; a tag's windows
  on other tiles are a "no", and its own tile is a "yes" bag. 12K steps from 189,803 against ctl12.
- **Results:**

  | | ctl12 | mg1 |
  |---|---|---|
  | mAP | 0.490 | 0.432 |
  | precision | 0.180 | 0.137 |
  | absent called | 13% | 21% |
  | medium 64px window AUC | 0.66 | 0.68 (within noise) |
  | window gap | +0.59 | +0.07 |

  Calls rose everywhere. The "no" windows show other images, which the ordinary negatives already cover. There is
  still no *within-image* "no".
- **Verdict:** not adopted.

## Matched twins (the maintainer's idea)

- **Setup** (`--twin-every`, `TagCentricBatchSampler.SampleTwins`, `TwinWindows`):
  - Every Nth batch holds one image with a localizable general tag and 9 "twins" without it, whose tags overlap it
    most (Jaccard; at most one character each).
  - The half of the positive's windows most like 3 twins' windows (cosine, leaving out the one holding the tag's
    map peak) are "no".
- **Results** (12K steps from 189,803):

  | Arm | mAP | Absent called | Notes |
  |---|---|---|---|
  | ctl12 | 0.490 | 13% | |
  | tw0 (twin batches, weight 0) | 0.482 | 17% | The batches alone are about neutral. |
  | tw2 (weight 1) | 0.442 | 23% | The window term costs about 4 mAP points; the on/off window gap widened a little (+0.75 vs +0.59). |
  | tw02 (weight 0.2) | 0.477 | 18% | No cost, no gain. |
  | tw1d (weight 1, logit scale/bias detached) | 0.438 | 20% | So the cost isn't the shared bias. |

- **Twin check** (`evaluate --twin-check`; ctl12; 1,041 gold pairs):
  - The twins' "no" windows hold the object 12% of the time, against a 16% base rate (medium 9% vs 13%).
  - 31% of the object's own windows get a "no".

  That is too weak a label.
- **Verdict:** not adopted.

## Pasted twins, on run 4

- **Why run 4:** run 5's map can't place a paste. Its map peak lands on held-out gold-small objects 8.5% of the
  time (ctl12), against run 4's 55% (medium 67%, at 1.49M).
- **Setup** (`--twin-paste 2`):
  - The positive's 64/96/128px region at its tag's map peak is pasted, feathered, into 2 twins, which then carry
    the tag.
  - A region from elsewhere goes into 2 other twins as certain negatives, so a paste cue can't pay.
  - Only when the peak's own logit is ≥ 0.
  - 12K steps from run 4 at 1,495,198: ctl4 / tw4 / paste4.
- **Results:**

  | | ctl4 | tw4 | paste4 |
  |---|---|---|---|
  | mAP | 0.788 | 0.787 | 0.787 |
  | absent called | 10% | 9% | 10% |
  | object effect | 1.14 | 1.16 | 1.10 |
  | gold pointing | 67.8% | 66.9% | 65.8% |

  paste4 made 11,993 pastes. All arms are within the noise of two identical controls. Run 4's classifier is
  whole-image, so it had nothing to learn the pastes with.
- **Verdict:** not adopted.

## The open-vocabulary research plan (2026-11-03)

### The question

The maintainer asked what SAM- and OWL-style open-vocabulary training could teach the tagger. The plan is
`review/open-vocab-research/PLAN.md`.

### The plan's reading of the literature

Every open-vocabulary detector surveyed trains its per-region classifier from boxes or pseudo-boxes, never from
image-level labels assigned to its own best region. Detic shows that prediction-based assignment of image labels
is worse than not using them at all on novel classes. So image labels should train only the whole-image score,
and windows should be trained from a localizer.

### The constraint

Run 4's map is the only localizer that points. It was used here only as the *same model's* own detached map in
the same step. The maintainer later ruled out a separate frozen teacher model; see [goals.md](goals.md).

### Experiment 0: are window labels from the model's own map good enough?

- **Setup:** `evaluate --map-window-check`, no training. Run 4 at about 1.507M; gold plus gold-small, 1,041 pairs.
- **Results:**

  | Size | Peak window holds the object | MIST top-15% | Bottom-40% "no" touching the object (base) | Map's window on/off AUC |
  |---|---|---|---|---|
  | all | 70% | 25% | 8% (18%) | 0.85 |
  | large ≥ 96px | 69% | 44% | 24% (37%) | 0.81 |
  | medium | 72% | 22% | 5% (15%) | 0.88 |
  | small < 32px | 67% | 20% | 3% (9%) | 0.84 |

- **Verdict:** go for medium and small objects; "no" contamination is three times below the base rate (the twins
  had 12% at a 16% base). MIST's top-p rule is far worse than the single peak at this density.

### Plan #1: self-map windows

- **Setup** (`--self-map-weight 1`, `SelfMapWindows`):
  - Per labelled localizable positive, the 64/128px windows at the tag's detached map peak are "yes", and two of
    its bottom-40% 64px windows are "no".
  - Each window is embedded as a 64px glimpse and trained by per-window BCE. The image-level losses are unchanged.
  - 12K steps from ctl4 on the B70. Controls: ctl4 (start) and ctl5 (a 12K control).
- **Results** (ctl4 / ctl5 / selfmap):
  - **Student window AUC**, 64 + 128px: all 0.57 / 0.57 / **0.82**; medium 0.59 / 0.58 / **0.84**; small 0.56 /
    0.56 / 0.83; large 0.57 / 0.57 / 0.77. The ceiling test's gold-supervised figure was 0.80 medium.
  - **Whole-image readouts** (ctl5 / selfmap):

    | Readout | ctl5 | selfmap |
    |---|---|---|
    | mAP | 0.7886 | 0.7871 |
    | image precision / recall | 44.4 / 51.8% | 44.0 / 51.7% |
    | absent called (n = 197) | 10.7% | 8.6% |
    | gold pointing | 67.1% | 66.6% |
    | gold-small pointing | 56.1% | 52.4% |
    | gold-auto pointing | 64.6% | 62.2% |
    | step time | 139 ms | 162 ms (+16%) |

  - Exclusive violations were 5.07 / 7.10%, but the 3x3 mosaic arm also showed 7.06%, so the control was the low
    one.
- **Verdict:** the windows learned to tell the object's window from the rest of the image with no hand labels,
  past the gold-supervised ceiling. But **run 4 scores whole images, so nothing uses the windows.** Not adopted by
  itself.

### Plan #2: presence x window scoring (SAM 3's factorization), no training

- **Setup:** `evaluate --peak-window-scores`. 20K held-out images, 10 object tags. Product = log sigmoid(whole) +
  log sigmoid(window).
- **Results** (mean per-tag AP, ctl5 / selfmap):

  | Score | ctl5 | selfmap |
  |---|---|---|
  | whole image | 0.570 | 0.570 |
  | 64px window alone | 0.159 | 0.288 |
  | 128px window alone | 0.156 | 0.271 |
  | product, 64px | 0.376 | 0.548 |
  | product, 128px | 0.368 | 0.558 |

  - The product calls no fewer absent objects: selfmap whole 8.6%, product128 10.2%.
  - The windows were never shown an image without the tag, so their scores there are uncalibrated.
  - Scoring the whole vocabulary by product would need a glimpse per tag per image.
- **Verdict:** not pursued.

### Step 1 after #2: "no" windows on images without the tag

- **Setup:** `--self-map-negatives certain|all`, after SAM 3's supervised-negatives row. Per image, up to 4
  localizable tags it is a certain negative for; the windows at that tag's map peak are "no". negc uses certain
  negatives only; nega adds context negatives.
- **Results** (selfmap / negc / nega):

  | Readout | selfmap | negc | nega |
  |---|---|---|---|
  | product128 AP | 0.558 | 0.544 | 0.505 |
  | absent called by product128 | 10.2% | 9.6% | 12.7% |
  | mAP (ctl5 0.7886) | 0.7871 | 0.7848 | 0.7861 |
  | gold-auto pointing (ctl5 64.6%) | 62.2% | 59.8% | 61.9% |

- **Verdict:** fails. This closes the presence x window route on run 4.

### Plan #3: 3x3 mosaics

Null; see [experiments-localization.md](experiments-localization.md#variants-tried-later-on-run-4).

### Plans #4 and #5

- **#4:** self-training round 2 with MIST top-p.
- **#5:** large-loss rejection on the "no" windows.

Neither was run. They refine #1's windows, which have no consumer yet.

### Where it stands

- **The one result that worked:** self-map windows. The windows learn object vs. background inside an image from
  the model's own map.
- **The open question:** a consumer for those windows, for example a run-5-style glimpse classifier that starts
  from self-map windows instead of image labels. That would be a new run, not an A/B, and the self-map labels are
  themselves pending the maintainer's ruling on self-generated labels.

The remaining entries of the plan are below. Each has a verdict in the plan, and none was run:

- tag-conditioned glimpses (GLIP/Grounding DINO-style FiLM);
- SPARC-style pooling for the map loss;
- CLIP dense-feature surgery readouts;
- CONTA/CDA/TDE context debiasing;
- attention-MIL.
