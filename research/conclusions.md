# Core findings

These are the conclusions the evidence supports so far. Each links to the page with the numbers. Several are
negative results, which are as useful here as the positive ones.

## About the data

1. **Booru labels are incomplete in structured ways, and fixing that is worth more than loss tricks.**
   - Gelbooru doesn't apply Danbooru's implications: completing them added 523,867 labels on 507,666 images.
   - 950 tags are (almost) only used by one site (925 Gelbooru-only, 25 Danbooru-only), so their absence on the other site's images means nothing.
   - Danbooru's ontology isn't the maintainer's: `shoes` excludes boots.

   Each was handled as a label correction (implications, site-blind weights, tag parentage), not as a loss
   change. [datasets-and-evaluation.md](datasets-and-evaluation.md)
2. **Missing labels are rare overall but common exactly where context says "yes".**
   - Cross-site checks: forgotten labels are about 1% of an object tag's negatives.
   - But among context negatives (the character usually has the tag), the maintainer's check found at least 15%,
     and up to about 40%, showing the object. Among the decidable ones, 21% (the maintainer) to 43% (the
     calibrated Sonnet labeller) carry it.

   [experiments-context-shortcut.md](experiments-context-shortcut.md#context-false-positives-are-partly-missing-labels-2026-10-30)
   - Measured with SAM 3 on 2,417 clean held-out images (2026-11-03): 0.63 sure untagged objects per labelled one
     over 34 object tags (a lower bound: SAM is sure on only 40% of labelled objects). The model calls 23% of them,
     so it mostly learns the labels' conventions; scored against labels completed with them, precision is 59.9%
     instead of 48.1%. [datasets-and-evaluation.md](datasets-and-evaluation.md#missing-labels-with-sam-3-missing-labels)

## About measurement

3. **Mean loss is the wrong stopping metric** for a 25K-tag sigmoid model: about 600:1 easy negatives. Per-tag AP
   against hard negatives replaced it.
4. **Thresholded rates are too noisy to judge a 6,000-step arm.** "Still called with the object blanked" varied
   7 points between identical controls. The threshold-free **object effect**, read with its control drop and on an
   unseen fill, replaced it.
5. **The context false-positive gap mostly measures untagged objects the model finds.** A labelled check of absent
   vs. visible-but-untagged pairs separates the two. On it, true context false positives in run 4 are about 7-11%
   and flat to falling.
6. **Several evaluation bugs changed the numbers, not the models.** Check which side of each fix a number is on:
   - box specks (box precision 0.2% → 7.9%);
   - whole-canvas pooling (object effect about +0.15);
   - non-reproducible seeds;
   - the seed-0 validation fallback.

   [experiments-measurement.md](experiments-measurement.md)

## About the model

7. **The classifier learns the context shortcut, and the shortcut grew with training in run 3.** Calls of
   gloves, hats and shoes come from the character, artist and series. Run 3's recall gap rose from 20.8 to 29.4
   points between epochs 1 and 2.
8. **Things that improved object-dependence without suppressing calls:**
   - full-strength context negatives at weight 3 (still called 85.9% → 70.3%);
   - focal gamma 2 instead of 4;
   - plain continued training. Run 4's object effect rose at every checkpoint, and its still-called rate fell from
     79% to 56% between 280K and 310K.
9. **Every loss or threshold change that narrowed the context gaps further did it by calling less.** That covers
   the context prior, weight 6, the margin, the last-layer refit, out-of-context weighting and classification
   pooling. [experiments-context-shortcut.md](experiments-context-shortcut.md)
10. **Erased-view training makes a call depend on the object but doesn't stop the call.**
    - Unanchored, it learns "any filled patch lowers scores".
    - Anchored, it costs about 2 mAP points.

    [experiments-erased-views.md](experiments-erased-views.md)
11. **Mosaics train the localization map.** Gold pointing went 33.1% → 35.7%. Mosaics never train the
    classifier, so they don't touch the shortcut. 3x3 mosaics at 4x shrink were null.
    [experiments-localization.md](experiments-localization.md)
12. **Small-object pointing is limited by how many pixels an object has, more than by the map's grid.**
    - A finer (stride-2, 256x256) localization level didn't move small objects on every set.
    - Higher context-negative weights (4, 6, 10) narrowed the context gaps but left pointing within noise.
    - Showing the gold images at 2x (AI-upscaled), with no retraining, raised small-object pointing 6-12 points on
      all three gold sets; medium, large and characters fell 2-20 points, since they became bigger than anything
      trained on. Wrist and hand jewelry stayed near 0 at either scale.
    - Small objects' pointing barely tracks how many training images a tag has (Spearman 0.41 per tag), so more
      booru images of them are unlikely to help as much as training at more than one scale.

    [experiments-localization.md](experiments-localization.md#is-it-resolution-the-2x-zoom-test-2026-11-03)

## About glimpses and window labels

13. **Multiple-instance pooling over model-chosen windows, trained by image labels, teaches every window the image
    label.**
    - Sharper pooling trades mAP for fewer off-object firings and never separates the windows.
    - A free glimpse tower under sharp pooling learns the image's identity instead.

    [experiments-glimpses.md](experiments-glimpses.md)
14. **The 64px glimpse can read objects when told per window.** Gold-supervised windows reach 0.80 medium AUC.
    What was missing was a within-image "no". Tag-derived window labels (mosaic tiles, matched twins) were too
    weak or cost mAP; the twins' "no" windows held the object 12% of the time against a 16% base rate.
    [experiments-window-labels.md](experiments-window-labels.md)
15. **The model's own map is a good window labeller, once the map points.** Run 4's map: peak window on the
    object 70% of the time; "no" contamination 3-5% for small and medium objects. Self-map windows reach 0.82 AUC
    with no hand labels. But nothing run 4 scores reads windows, and presence x window scoring didn't beat the
    whole-image score.
16. **Apparent gains have to be checked for a cheaper explanation.**
    - Locality negatives "doubled" best-on-object by moving evidence into 256px windows.
    - Zero-fill erasure gains were a patch cue.

    In both cases, a metric whose chance level or reference didn't match the change hid the artifact.

## About engineering

17. **TorchSharp on an Intel Arc Pro B70 is about 1.8x an RTX A4000** for this model. It needs the anti-spill
    variables, the legacy oneDNN TF32 switch, fixed tensor shapes and `OverrideCopyOffloadMode=0`. Run 4 now
    trains at about 145 ms/step there; its first 4 epochs took about 123 B70-hours (an estimate).
    [engineering-and-hardware.md](engineering-and-hardware.md)
