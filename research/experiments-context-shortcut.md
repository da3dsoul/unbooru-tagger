# Experiments: the context shortcut

**The problem.** The classifier calls object tags (gloves, hat, shoes, earrings) from who is in the image (the
character, the artist, the series) rather than from the object. Run 3's new checks showed this, and showed it
growing between epochs. The maintainer's diagnosis (2026-10-27): the remaining problem is *classifying* tags like
gloves, not localizing them.

**How the arms are judged.** By the context gaps, the occlusion check (later the object effect) and the labelled
context check before mAP. All are defined in [experiments-measurement.md](experiments-measurement.md).

**A word on noise.** Two identical 6,000-step controls differed by about 7 points on "still called with the
object blanked", and by about 4 points on the labelled context check. Small differences in the tables below are
within that noise.

## Deliberate targets

These came out of the run 1 post-mortem (commit `c618a2b`, 2026-10-24).

- **Question:** mean loss over about 25K tags per image is about 600:1 easy negatives. How do you train on what
  the corpus actually knows?
- **Setup:**
  - **Tag-centric batches.** Anchor tags are drawn uniformly over the vocabulary, each with 2 positives and 2 hard
    negatives (images with a commonly co-occurring tag but not this one).
  - **Pair weights** (`PairWeightBuilder`):
    - positives and verified negatives (anchor hard negatives, exclusive groups) get full BCE;
    - a tag on at least 90% of the images of a tag the image carries (`--missing-threshold`) is a likely-missing
      label, with weight 0;
    - other negatives go through the asymmetric loss (ASL: clip 0.05, gamma 4);
    - tags no training image carries get weight 0.
  - **Context band.** Between `--implied-threshold` (0.6) and the missing threshold, the source tag is a *context
    source*. Its images without the tag are unverified negatives, sampled per anchor.
- **Verdict:** adopted from run 2 on. `--sampling rare-tag --targets legacy` reproduces run 1.

## Full-strength context negatives (run 3 A/B, adopted in run 4)

- **Question:** focal damping left "this character, therefore gloves" nearly free, because a confident wrong
  "yes" on a context negative cost almost nothing. Does training those pairs at full strength help?
- **Setup:** `--context-negative-weight W`. Every batch pair where the image has one of the tag's context sources
  but not the tag becomes full BCE at weight W. A/Bs from run 3 (`ab-context/`), all with 2 mosaics; the control is
  the mosaic arm.
- **Results:**

  | Arm | Still called, object blanked | Recall gap | FP gap | Gold pointing | mAP | Exclusive violations |
  |---|---|---|---|---|---|---|
  | mosaic (control) | 85.9% | 30.0 | 26.3 | 35.6-35.7% | 0.750 | 6.7% |
  | ctx (1 sampled per anchor) | 77.8% | 27.8 | 24.5 | 35.6% | 0.749 | 8.2% |
  | ctxw3 (weight 3) | 70.3% | 24.1 | 21.1 | 36.1% | 0.745 | 9.7% |
  | ctxw3loc (weight 3, localizable tags only) | 74.0% | 24.0 | 21.6 | 37.3% | 0.746 | 6.2% |

  - ctx3 (3 sampled per anchor) was no better than ctx.
  - Recall where context makes a tag unlikely didn't rise at any weight (27.1-27.8%). So the gaps shrink by
    suppressing context-driven calls, not by finding more objects out of context.
  - ctxw3's extra exclusive violations were people counts: 1girl + 2girls rose from 170 to 316, and 1boy + 2boys
    from 38 to 182. People counts are often just untagged, so the term was limited to localizable tags.
- **Verdict:** adopted in run 4 at weight 3, localizable tags only.

## `solo` and `no_humans` are not exclusive

The exclusive-violations breakdown showed at least 54 held-out images carrying both: a lone creature or robot is
tagged both. The default rule had been training true tags as certain negatives. It was removed (commit `c5edd12`).

## Run 4 context-shortcut A/Bs (2026-10-27/28)

All arms ran 6,000 steps from run 4 at step 271,754 (`ab-prior/start`). The control was mAP 0.572, image
precision 19.7%, still called with the object blanked 87%.

| Arm | Question | Result | Verdict |
|---|---|---|---|
| `--negative-gamma 2` (was 4) | At gamma 4, a wrong "yes" on an unverified negative was pushed back 12-27x more weakly than a missed "yes". Cross-site checks put forgotten labels at about 1% of these tags' negatives. | mAP 0.577; image precision 22.8%; still called 74%; exclusive violations 8.3% → 12.0%. Gamma 1 went further on occlusion but cost 2.5 points of gold pointing. | **Adopted** in run 4 from about 275K. |
| Context prior (`--context-prior-weight`, product of experts with P(tag \| character/artist/series), training only) | Let context explain what it can, so the object has to explain the rest. | Closed the gaps only by suppressing calls; mAP -2 to -4 points; gloves and shoes unmoved. | Rejected. |
| Map-scored occlusion | Is the fused map local enough to score from? | 97% still called: it isn't. | Set aside. |
| Focus anchors (`--focus-share`), tile classification, full-size mosaics | Group context negatives with positives; classify mosaic tiles. | Small gains; full size cost 1.4 mAP points. | Not adopted. |
| Per-tag classification pooling (`--classification-pooling-*`): a log-sum-exp over a 16x16 cell grid instead of the average | Make the classifier depend on where the object is. | Abrupt switch at T = 0.25 or 1: mAP 0.577 → 0.537. Annealed 8 → 0.25 over 12K steps with 256 mined negatives, at 20K steps: mAP 0.567 vs run 4's 0.586 at 290K. Still called 52% vs 72% (better), but object tags no better (gloves AP 0.349 vs 0.367; glasses 0.146 vs 0.134; earrings 0.165 vs 0.169). | Rejected. |

**Checks done along the way:**

- The evaluator's pooled path was verified: at T = 1000 it equals the average, and scoring at T = 3.13 alone costs
  nothing. So the mAP loss came from training.
- Run 4 on its own kept improving: still called 79% (280K) → 56% (310K), mAP 0.576 → 0.598.
- Gloves AP stayed about 0.35 in every arm, at a 17% base rate.

## Positive and negative selection (2026-10-29)

All arms ran 6,000 steps from run 4 at 626,771 on the A4000 (`ab-ooc/`). The controls were ctl and ctl2. The
maintainer's preferred direction was "clever selection of positives and negatives with strong weights".

| Arm | Setup | Result | Verdict |
|---|---|---|---|
| ooc | `--out-of-context-share 0.5`, weight 3: anchor positives whose character/artist/series prior is low. | No effect (low-prior recall 24.2 vs 24.0): each tag got about one such positive per run. | Not adopted. |
| oocall | `--out-of-context-all --out-of-context-weight 3`: every low-prior positive at weight 3. | Recall gap 17.2 (controls 20.2/21.0), FP gap 17.8 (19.7/20.1), mAP 0.707 (equal), but object effect 0.728 vs 0.771/0.763 (worse) and image precision -0.9. It calls more; it doesn't look harder. | Not adopted. |
| nox | `--excluder-share 0.2 --excluded-negative-weight 3`: anchors on tags an excludes rule rules out, with the `no_X` images as certain negatives. | Object effect 0.796/0.587 (gold/gold-small) vs about 0.767/0.510; footwear AP -0.03 to -0.05, since half its anchors were kinds of shoe. | Fixed into nox2. |
| nox2 | Same, with the ruled-out parents drawn uniformly. | Lost the gain: object effect 0.749 vs 0.774/0.768; shoes AP 0.618 vs 0.64. | Rejected. |

## Context false positives are partly missing labels (2026-10-30)

**The observation.** Averaged per tag over a fixed 243-tag set, run 4's false-positive rate where the character
usually has the tag (prior 60-90%) rose from 20.4% (170K) to 24.6% (1,030K). Where the prior is under 5%, it
fell from 8.1% to 3.0%. High-prior recall went 64% → 77%, low-prior recall 19% → 27%. The pooled rate, dominated
by common tags, was flat, which hid this.

**What the calls actually are.** The maintainer checked 180 high-prior negatives by eye: 27 visible, 50 unclear,
103 absent. On that set:

- Run 4 calls most of the visible ones: 48% at epoch 1, 59-70% from 700K on.
- It calls few of the absent ones: 5-9% up to 600K, and 11-12% from 700K on (n = 103, so about ±3 points).

So most of the rise is the model finding objects nobody tagged. True context false positives rose a little, to
about 11%, against well under 4% where the prior is low.

**Context weight 6 (cn6)**, 6,000 steps, to test the consequence: full-strength context negatives train every one
of those visible objects as a "no".

- FP rate where the prior is high: 22.1% (controls 23.4/23.7). But high-prior recall fell as much: 49.9% vs
  52.2/52.7%.
- Low-prior recall, object effect and mAP: unchanged.
- Checked set: absent objects called 12% (controls 16/10%), visible 56% (59/56%).
- With an erasure term added (hinge6), it was worse than either alone: the noise-fill control drop came back
  (0.33 vs 0.12), and mAP was 0.698.
- Solo, non-comic images are no cleaner: 6 visible, 5 unclear and 32 absent of 43.

**Verdict:** a higher weight pushes against real objects too. Run 4 stays at weight 3.

## The labelled context check and the cheap-first investigation (`ctxfp/`, 2026-10-30)

The hand-checked set was grown with a calibrated Sonnet labeller into `context-check.json` (see
[datasets-and-evaluation.md](datasets-and-evaluation.md#hand-checked-context-negatives-and-the-labelled-context-check)).

**On it, from 700K to 1.09-1.24M, run 4's calls went:**

- on absent objects: 10% → 7-9%;
- mean log-odds past the threshold: -0.94 → -1.28;
- on visible-but-untagged objects: about 70-77%, with the gap between visible and absent logits widening
  1.36 → 1.98.

Over the same steps, the band false-positive gap rose. **The band gap mostly counts untagged objects that the
model finds.** True context false positives are flat to slightly falling.

**Ruled out cheaply:**

- **A last-layer refit** on frozen pooled embeddings (fit on held-out images 20K-100K, test on 0-20K; DFR-style).
  Upweighting context negatives traded high-prior recall for false positives one for one, and group balancing
  cost 3-8 AP. The entanglement is in the features.
- **Scoring tags from their best local cells** at inference: no fewer absent calls, and hat AP 0.53 → 0.37.
- **Cleaning labels by tag count:** the median is 27 vs 30 general tags, so count doesn't separate untagged-visible
  from absent.
- **The `neg` erasure arm:** its lead on the maintainer's set didn't hold on the added images.

**Other findings:**

- Absent-called on the 197 labelled absents: controls 9-13%, every A/B arm 9-13%. **6,000-step arms can't resolve
  differences of this size.**
- `--erasure-loss negative-only` was started and stopped after 113 steps, once the larger set showed no erasure arm
  moves the absent-called rate.

## A stricter threshold where context makes a tag likely (`context-margin/`, 2026-10-30)

- **Setup:** run 4 at 1.316M; 100K held-out images; 35 object tags. The prior comes from the
  characters/artists/series the model itself calls (precision 0.63, recall 0.54). An object tag's logit is
  lowered by 0.25 where its prior is 60-90%.
- **Results:**
  - absent objects called 10.5% → 7.4%;
  - visible-but-untagged called 71% → 66%;
  - band recall 59% → 53%; macro AP 0.467 → 0.460; F1 down on every tag.

  Larger margins trade the same way.
- **Verdict:** not viable. It changes the operating point, not the model.

## Built but not tested

- `--negative-loss hill`: the Hill loss for unverified negatives.
- `--hard-negative-source-weight`: every pair whose image carries a hard-negative source of the tag becomes a
  full-strength negative.

No A/B results are recorded for either.

## Where this left the problem

- Every loss or threshold change that narrowed the context gaps did it by **suppressing calls**, not by making the
  call depend on the object. This covers the context prior, weight 6, the margin and the refit.
- The two things that moved the object-dependence measures were:
  - full-strength context negatives at weight 3;
  - plain continued training. Run 4's object effect rose at every checkpoint: 0.33 at epoch 1, 0.41 at 550K, 0.48
    at 622K, 0.51 at 700K, before the content-pooling change raised all values.
- The next attempts changed what the classifier *looks at*: erased views ([experiments-erased-views.md](experiments-erased-views.md))
  and glimpses ([experiments-glimpses.md](experiments-glimpses.md)).
