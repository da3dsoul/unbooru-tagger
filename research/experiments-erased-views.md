# Experiments: erased-view training

**Dates:** 2026-10-29/30, on the A4000. Notes in `ab-erasure/README.txt` and `compare*.txt` (outside git), plus
`CLAUDE.md`.

## Question

Can the occlusion check become a training loss? The check blanks a found tag's region and sees whether the score
drops. If the model is trained so that blanking its own evidence region lowers the score more than blanking a
control region elsewhere, does its call come to depend on the object?

Literature behind it:

- Shetty et al. 2019, "Not Using the Car to See the Sidewalk": a ranking hinge on object-removed views;
- GAIN (Li et al. 2018): an erased view trained as a negative;
- CSS (Chen et al. 2020): counterfactual samples with a different label.

See [citations.md](citations.md).

## Setup

- **Code:** `ErasureContrast`, with the `--erasure-*` options.
- **Which tags:** called localizable positives.
- **The two blanks:**
  - the object blank: the region the tag's own map points at, detached and capped at 15% of the image;
  - the control blank: a same-size region, either random or on another of the image's tags.
- **The loss:** a hinge makes the object-blanked score drop at least `--erasure-margin` (1 logit) more than the
  control-blanked score.
- **Fills:** zero, mean or noise. No blur, because censoring is itself a tag. No patches of other images, which
  can carry the object.
- **Start:** run 4 at 626,771, 6,000 steps unless named -12k or -18k.
- **Controls:** ctl and ctl2 (mAP 0.706/0.707, gold object effect about 0.73).

## How to read the results

The object effect is a difference: object-blanked versus control-blanked. A model that learns "any filled patch
lowers every score" isn't caught by it. So each arm is also read on two things:

- the **control drop:** how far the score falls with only the control blanked;
- a **fill the arm never trained with:** zero fill for the mean/noise arms, and noise for the zero-fill arm.

## Results

| Arm | Dose | Object effect (trained fill / unseen fill) | Control drop | mAP | Notes |
|---|---|---|---|---|---|
| hinge | zero fill, random control, 2 pairs/step | 0.87 (zero) / **0.37 (noise)**, below the controls' 0.42 | not recorded | 0.686 | The whole gain was "a grey patch lowers a score". |
| neg | GAIN-style: object-blanked view a negative, control-blanked a positive | 1.08 | (no patch cue) | 0.667 | Every called score rose: median logit 1.8 vs 0.6. |
| hinge2 | mean/noise fills, control on another tag, 2 pairs/step, weight 1 | 0.92 (unseen zero fill) | **0.58** (controls 0.09) | 0.672 | Blanking anywhere lowered scores; object hurts more 79% vs 88%; pointing -3 points. |
| hinge3 | 1 pair/step, weight 0.25 | 1.11 (zero) / 0.78 (noise); controls 0.73 / 0.42 | 0.28 | 0.692 (-1.4) | Pointing -0.8; object hurts more 86% (controls 89/87%); +19% step time. |
| hinge4 | hinge3 every 2nd step | 0.92 (zero) / 0.79 (noise); gold-small 0.61 vs controls 0.15/0.16 | 0.10 with zero fill (controls' level) | 0.698 (-0.8) | Exclusive violations 9.0% vs 7.8/6.6%. |
| hinge5 | hinge3 every 4th step | 0.92 (zero) / 0.84 (noise); gold-small 0.61/0.47 | 0.12 (noise) | 0.703 (-0.3) | Pointing 48.6% (+0.8); exclusive violations at the controls' level; 336 ms/step (+2%). |

### hinge5 extended (2026-10-30)

hinge5 and a control were continued to 12,000 and 18,000 steps:

- **mAP gap didn't close:** 12K 0.702 vs 0.707; 18K 0.703 vs 0.708. Pointing -1.2 at both.
- **The patch cue came back.** Noise-fill control drop: 0.12 (6K) → 0.67 (12K) → 0.31 (18K), against the
  control's 0.06. The hinge is relative, so a cue that lowers both views costs it nothing.
- **At 18K,** over all 1,030 gold pairs with zero fill: object effect 0.74 vs 0.46, control drop 0.16 vs 0.06.

### The anchor term

`--erasure-anchor-weight` holds the control-blanked score to the unblanked one. Both versions used hinge5's dose
and ran 12K steps.

- **anchor1** (one-sided: only a drop is penalized):
  - the patch cue went and overshot: noise control drop -0.13, so a blank elsewhere *raised* scores;
  - the best object effect of any arm: zero fill 1.09 vs 0.69, gold-small 0.71 vs 0.45; noise 0.85 vs 0.41;
  - but mAP 0.687 vs 0.707, image precision -2.2, and low-prior recall -1.3.
- **anchor2** (symmetric: |unblanked - control|):
  - control drop at the control's level: zero fill 0.06 vs 0.07, noise 0.05 vs 0.06;
  - over all 1,030 pairs, object effect 0.65 with zero fill (control 0.46), 0.56 with noise (0.41), gold-small
    0.43/0.26 (0.30/0.15);
  - but mAP 0.687 again, image precision -2.6, and "general called explicit/questionable" 21.2% vs 17.8%.

### Context false positives

None of the doses moved them:

- Averaged per tag, the FP rate where the character usually has the tag was 22.5-22.7% in every erasure arm, vs
  23.4/23.7% in the controls.
- Recall where context makes a tag unlikely was unchanged (32-33%), at 12K and 18K too.
- On the maintainer's checked set, absent / present called (n = 103 / 27): ctl 16/59, ctl2 10/56, hinge3 11/52,
  hinge4 9/44, hinge5 9/56; at 12K and 18K, hinge5 10-11/63, ctl 10-11/56-63, anchor1 10/59, anchor2 12/67.

  No arm moved the absent rate outside the controls' spread.

### Speed (A4000, 400 steps each)

| Variant | ms/step |
|---|---|
| no erasure | 330 |
| 1 pair, own pass | 390 |
| 1 pair, batched with the mosaics | 377 |
| 2 pairs | 418 |
| 1 pair every 2nd step | 351 |

The A4000 is compute-bound, so batching saved little; the extra images themselves are the cost.

### A measurement caveat

The 6,000-step zero-fill numbers are over the 455 gold pairs (216 gold-small) that the controls' older reports
dumped; the noise numbers are over all pairs. Numbers over different pair sets aren't comparable.

## Verdict

**Not adopted.** Erased views make a called tag *depend* on the object, but they don't stop the call.

- Unanchored, they learn the patch cue with more training.
- Anchored, they cost about 2 mAP points.

## What came out of it

- `evaluate --occlusion-fill` scores the occlusion check with any fill. Use it to test whether a gain carries to a
  fill the model never trained on.
- The rule for reading the object effect: always with its control drop, and on an unseen fill.
