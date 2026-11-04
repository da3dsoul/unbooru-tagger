# Experiments: glimpses (run 5) and glimpse pooling

**Sources:** `CLAUDE.md` ("Glimpses", "Glimpse funnel", "Locality negatives", "Glimpse pooling investigation"),
`train-v5.sh`, `review/pooling/PLAN.md`, `review/locality/`, `review/glimpse-dump/` and the memory notes. All of
these except `CLAUDE.md` are outside git.

## Why glimpses

- Every cell of a whole-image pass sees most of a 512px image, including the localization map's cells.
- So the averaged score, the map-cell score (classification pooling) and the last-layer refit all kept the
  context.
- A 64px glimpse of a window physically can't see the face. The design was the maintainer's: 64px glimpses at
  zooms covering 32-256px windows, box filter down and bilinear up, a learned code per zoom.
- **The bar:** "We want training to know what gloves are, not just that the entire image contains it."

## Measuring the window choice before the run (`glimpse-frames`, on run 4's map)

The question: how do you choose where to look?

**Which tags to suspect** (gold tags in an image's top 32):

| Ranked by | Gold tags in the top 32 |
|---|---|
| the localization map | 6% |
| the whole-image classifier | 68% |

The map is trained on about 100 pairs per image, so its scores for the rest aren't comparable. This is why the
proposal head ranks from the whole image.

**Which zoom:** choosing one zoom per tag from the map's response blob matched the right zoom at best 41% of the
time, and no blob rule worked. Hence the sensor: one window per zoom at the peak.

**How many windows** (share of gold tags framed):

| Windows per image | 16 | 32 | 48 | 64 |
|---|---|---|---|---|
| Gold tags framed | 55% | 75% | 87% | 91% |

Run 5 uses 48.

## The first glimpse run (deleted)

- **When and where:** started 2026-10-30 16:58 on the A4000, 444 ms/step.
- **What was wrong:**
  - The labels picked its training windows: half framed the positives' peaks, the rest were random.
  - Evaluation scored a fixed 378-window grid, about 6x a whole-image pass and about 30 minutes per evaluation.
    bf16 didn't speed it up.
  - With 16 windows in training and 378 at evaluation, the score scale differed between the two.
- **Result:** the model never learned where to look, and a wrong tag's peak was never looked at closely.
- **Verdict:** the maintainer had it deleted. The lesson became a rule: training and evaluation must run the same
  model forward.

## Run 5 (from 2026-10-31)

### The forward pass (the same in training, evaluation and inference)

1. The **proposal head** (`ImageTower.Proposals`, a D x D matrix on the pooled embedding, trained on every
   localizable tag) ranks what the image might hold, context included.
2. The **localization map** gives each suspect's peak.
3. The **glimpse sensor** looks there with one window per zoom (32/64/128/256px), 48 windows per image after
   merging repeats. The zoomed-out grid fills any left over.
4. The localizable tags' scores are a **log-sum-exp over the glimpses**: temperature 0.5, annealed from 4 over
   30K steps.

Whole-image tags keep the whole image. Characters are localizable, so they are glimpse-scored too.

### Training rules

- A labelled positive that no close window frames gets **no glimpse loss** (`--glimpse-unframed-positive-weight
  0`).
- While the map can't point yet (the first 20,000 steps), a share of the windows looks at labelled positives,
  half at step 0.
- The localization loss's mined negatives are ranked by the classifier's actual scores.

### Engineering

**Speed on the B70** (detail in [engineering-and-hardware.md](engineering-and-hardware.md)):

| Change | ms/step |
|---|---|
| fixed glimpse batch size | 426 → 237 |
| 1x1 convs as matmul; pooled-only glimpse forward | 293.8 → 243.2 (profiled) |
| no per-step device read-backs | 247 → 237 |
| pinned non-blocking uploads | 235 → 228 |
| channels-last tower | 228 → 205 |

**The copy-engine faults** were cured by `OverrideCopyOffloadMode=0`.

**A mistake:** run 5's snapshot evaluations at 130K-180K all failed silently. The CUDA evaluation inherited the XPU
`LD_LIBRARY_PATH`.

### Results

**The glimpse funnel at about 118K, medium objects:**

| Stage | Result |
|---|---|
| Suspected (in the proposal head's top 48) | 59% |
| Covered by a close window | 82% |
| Best glimpse on the object | 28% (chance 13%) |
| Called when covered / not covered | 36% / 17% |

Small objects: best glimpse on the object 31% (chance 13%); large 32% (chance 28%). The windows find the
objects, but the classifier's strongest evidence is mostly elsewhere.

**Character tags:** AP matched run 4's at the same step (0.507 at 110K vs 0.500 at 103K). Few clear 0.5, though:
0.8% of gold character pairs. That is calibration, which per-tag thresholds absorb.

**Run 5 at 200K against run 4 at 200K:**

| Readout | Run 5 | Run 4 |
|---|---|---|
| mAP | 0.490 | 0.512 |
| image precision | 17.7% | 16.7% |
| gold pointing | 27.7% | 30.1% |
| gold pointing, small | 2.7% | 3.9% |
| still called, object blanked | 79.5% | 85.4% |
| context recall gap | 9.4% | 8.8% |
| context FP gap | 15.0% | 13.1% |

Gold box precision 7.9% vs 0.2% here reflects the speck-filter fix, not the model.

**Verdict:** stopped at step 277,022, after the maintainer saw run 5's charts tracking run 4's. Run 4 resumed
2026-11-02.

## Locality negatives (2026-11-01; the maintainer's idea)

- **Question:** some tags can't share a close look (glasses and shoes; gloves and a face can). Can a close window
  where one labelled tag is confidently found train each other labelled tag "no" there, weighted by how rarely the
  two are found together?
- **Setup** (`LocalityNegatives`, `--locality-*`):
  - The together-table is learned during training, or seeded from run 4's map (`evaluate --locality-prior-out`:
    385,691 pairs over 20K training images). Examples: shoes/glasses together 1.5% of the time, shoes/hat 2.8%,
    gloves/hat 7%, hat/glasses 23%, shoes/boots 61%.
  - A/B on the B70: 6,000 steps from run 5 at 189,803. ctl / online / online + run 4 prior, weight 1.
- **Results:**

  | | ctl | online | + prior |
  |---|---|---|---|
  | Best glimpse on object, medium | 31.9% | 60.6% | 59.5% |
  | Best glimpse on object, small | 28.5% | 59.2% | 68.9% |
  | mAP | 0.483 | 0.412 | 0.370 |
  | Micro precision | 0.173 | 0.086 | 0.069 |
  | Absent objects called | 18% | 57% | 75% |
  | Object effect | 0.130 | 0.097 | 0.045 |

  A negative that only says where a tag *isn't* is cheapest to satisfy by firing everywhere else.
- **Dose arms** (per pair, gate p ≥ 0.9):

  | Weight | Best on object (medium / small / gold-small) | mAP | Absent called |
  |---|---|---|---|
  | 0.1 | 40.7% / 47.8% / 45.3% | 0.481 | 19% |
  | 0.3 | 54.6% / 64.5% / 56.5% | 0.475 | 26% |
  | 1 | 57.5% / 64.0% / 61.4% | 0.453 | 37% |

  - Weight 0.3 with the logit scale and bias detached matched plain 0.3.
  - In no arm did the object effect rise.
- **Run 5 trial:** weight 0.1 went into run 5 from about 221K (20:10 on 2026-11-01, recorded in `train-v5.sh` when
  it was switched on; `CLAUDE.md`'s "~238K" was wrong and is corrected). VRAM grew until the trainer died: xe "VM worker error: -12" and
  `UR_RESULT_ERROR_OUT_OF_RESOURCES` after 14-18 minutes unpadded, and after 47 minutes with rows padded to 32. It
  was off again from about 256K.
- **Correction (2026-11-01 23:30): the gain was an artifact.**
  - The term only penalizes close windows (32-128px), so the model moved its evidence into the 256px window, a
    quarter of the image with the context in it.
  - The winning zoom for small and medium objects was 256px in 5-12% of pairs in the control, about 50% at weight
    0.1, 90%+ at 0.3, and all at weight 1. In run 5 itself it went 10% (230K) → 75% (250K).
  - The funnel's chance level is computed over all windows, not over the winning zoom, so bigger windows read as
    "on the object".
- **Verdict:** not adopted.

## The glimpse pooling investigation (2026-11-01/02)

### Diagnosis

From per-window gold dumps (`evaluate --dump-occlusion` → `glimpse_pairs*.jsonl`), on medium objects:

- window logits on and off the object differ by only +0.6;
- 31% of windows that don't touch the object fire for it;
- about 26 of 48 windows carry each score;
- the best window alone would call 98% of labelled gold pairs, against 41% pooled.

The log-sum-exp temperature (0.5) is in dot units, before the learned scale (about 8.6). In logit units it is
about 4.3: nearly a mean. Every glimpse learns the image's label.

### How arms were judged

The maintainer's rules for judging windows:

- a window is on the object only if the object fills it;
- a 256px window is for large objects.

The readout became the **within-pair on/off-object window AUC at a fixed zoom** (64/128px), not mixed-size gaps
or the funnel.

### Arms

6K or 12K steps from run 5 at 189,803. The 12K control, ctl12: mAP 0.490, absent called 13%, medium 64px AUC
0.66.

| Arm | Setup | Result | Verdict |
|---|---|---|---|
| lse012 / lse006 / lse003 | Sharper pooling: T = 0.12 / 0.06 / 0.03 (dot units). | T 0.5 / 0.12 / 0.06 / 0.03: mAP 0.483 / 0.484 / 0.469 / 0.446; off-object firing 31 / 24 / 19 / 9%; window gap never wider. A window the object fills scores at chance (AUC 0.51-0.58) in every arm. | Monotone trade; 0.12 is the only free setting. |
| top1 / top3 | Hard max; mean of the top 3. | mAP 0.409 / 0.445; off-object firing 2-3%, but the window gap halved. | Pooling alone ruled out. |
| wild | WILDCAT top-2 + 0.7 x bottom-2. | mAP 0.423; window gap +0.18. | Worst pooling arm after top1. |
| no256 | No 256px window in the pool. | mAP 0.462; nothing over lse006. | Not adopted. |
| floor, fpp | Label framing kept on (8 windows); framed positives pooled over their 64/128px windows. | fpp mAP 0.471, a null. The map's peak frames a medium object about 1 time in 12 (medium gold pointing 7.9-8.1%), so framed positives mostly trained context windows "yes". | Not adopted. |
| pdetach | The proposal loss kept off the tag rows. | Null (mAP 0.485). | Not adopted. |
| sepg / sepg2 / sepgs | A separate glimpse tower (`--separate-glimpse-tower`), at 1x LR, 2x LR, and 2x LR with T = 0.06. | sepg: no different from ctl. sepg2: the only probe gain (+0.04) for -1.3 mAP. sepgs: learned the image's *identity* instead (same-image nearest neighbour 32% → 45%; cosine to its own pooled embedding up to 0.899). | The cheapest window feature for an image label is the character, palette or style that every crop shows. |
| pairs / pairs2 | Same-source pairs: one image of a character with the tag, one without, and a hinge on their pooled glimpse logits. | Weight 1: every window collapsed (mAP 0.315). Weight 0.1: -5 mAP points. | Not adopted. |
| split3 / psplit3 / pxpass | Singh et al. 2020's feature split on the glimpse embedding. "Exclusive" means a known character/artist/series with prior < 0.2, limited to the 1,980 tags with context sources. | psplit3 / pxpass mAP 0.477 / 0.471, absent called 18 / 20%. The object half alone can't call the tags (mAP 0.21-0.25). The first version's "exclusive" rule was wrong, because `.Take(8)` kept only 8 context sources (split3: mAP 0.327). | Null to negative. |

### Probes on the dumped glimpse embeddings

- **The information exists but the rows don't read it.** Filled windows probe AUC 0.70 vs the tag row's own 0.52;
  "holds" windows 0.81 vs 0.66.
- **Readability tracks the tag, not the size.**
  - Readable: fixed-shape objects at fixed body places (glasses 0.92, choker 0.90, blush, animal_ears, shoes,
    hat).
  - Unreadable at every zoom: gloves (0.57-0.66), flower, earrings, wings, hair_ornament.
  - 256px is never the most readable size.

### Crashes in this batch

- Three arms died with exit 1 or `UR_RESULT_ERROR_OUT_OF_RESOURCES` (one with xe "VM worker error: -12"). The
  causes are not established; reruns of all three survived.
- The pairs arm's VRAM rose in steps from 14 to 21 to 28 GB, then stayed flat.

### Conclusion

No label-only change makes object tags read the object. That covers pooling, zooms, framing, proposal detach, a
separate tower, same-source pairs and the feature split. This led to the ceiling test and to window-level labels:
see [experiments-window-labels.md](experiments-window-labels.md).
