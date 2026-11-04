# Goals

This page records what the project set out to do, and the rules the maintainer added along the way. The
original plan is in [`CLAUDE.md`](../CLAUDE.md), which also notes where the implementation departed from it.

## The original goal

A multi-label image tagger for anime/booru images. It trains on booru-style string tags only; there are no
bounding-box labels. It must provide:

- **Per-tag confidence** for every tag in the vocabulary.
- **Rough per-tag localization:** where in the image a tag's evidence is.
- **An open, growing vocabulary.** The vocabulary has hundreds of thousands of tags in the long tail. New
  tags must be addable later without full retraining, for example a new costume, a new character or a new
  series.

## Why not a fixed classification head

A head with N sigmoid outputs (the WD14 tagger and DeepDanbooru are named as prior art) has to be expanded and
retrained for every new tag. The project chose a **dual encoder with a sigmoid contrastive loss** instead. JoyTag
and SigLIP are the references studied. The loss is a sigmoid, not a softmax, because tags are multi-label and
should not compete for probability mass.

- **Image tower:** a ConvNeXt-style CNN. It produces a pooled embedding and the pre-pool spatial map, and both
  live in the tag embedding space.
- **Tag tower:** one learned embedding row per tag. New rows can optionally be warm-started from a frozen text
  embedder.
- **Score:** `sigmoid(scale * dot(image, tag) + bias)`.
- **Heatmap:** the same dot product taken at every location of the spatial map (MaskCLIP-style).

The original build order:

1. Validate the dual encoder on a bounded vocabulary.
2. Heatmap localization.
3. An "add a tag" pipeline.
4. Long-tail handling: minimum-image thresholds and oversampling.
5. Not done: a partial, top-layers-only unfreeze for periodic fine-tunes.

The original plan named Python, PyTorch, `timm` and `open_clip`. The project was built entirely in C#/.NET
instead: TorchSharp for training, and ONNX Runtime for inference.

## How the goal sharpened during the research

After run 3, the maintainer stated the problem as **classification, not localization**. The model detected
gloves, shoes and earrings, but it called them from *who* was in the image (the character, artist or series),
not from the object. Two statements by the maintainer set the bar from then on:

- "We want training to know what gloves are, not just that the entire image contains it. It's necessary for
  localization." (2026-10-31, when run 5 was designed.)
- A tag should mean that the object is present, not that the character usually has it (2026-10-30).

## Rules the maintainer set

Each of these rules shaped which experiments were allowed:

| Rule | Where it came from | Effect |
|---|---|---|
| **Training must not depend on the hand-made gold sets.** | Stated after the window-probe ceiling test (2026-11-02), which fine-tuned on gold boxes. | Gold sets are evaluation only. Window labels for training had to come from tags or from the model itself (mosaic tiles, matched twins, self-map windows). |
| **No frozen teacher model or distillation from another trained model.** | 2026-11-03: "Knowing the goal of the project, training a model to train another isn't valid." | Rules out "run 4's map labels run 5's windows" and Detic/OWL-ST-style distillation from a separate model. Labels may come only from the tags, or from the same model's outputs in the same step, and even those are pending the maintainer's ruling. |
| **Claude-labelled data at scale is too expensive.** | The `gold-auto/` calibration: about 15-18K tokens per image, about 18M tokens for 1,020 images. | SAM 3 boxes replaced Claude-checked boxes for `gold-auto/`. SAM 3 was also used for the missing-labels measurement. Both are evaluation data, not training data. |
| **The maintainer's tag ontology wins over Danbooru's.** | 2026-10-28, while reviewing gold-small ("Boots are shoes, but shoes aren't boots"). | The tag-parentage rules: see [datasets-and-evaluation.md](datasets-and-evaluation.md#tag-parentage). |
| **Judge context-shortcut fixes on the context gaps and the object, not on mAP alone.** | Run 3 review (2026-10-26) and the overnight brief of 2026-10-30. | An mAP drop was acceptable if it recovered with more training. Arms were extended to 12K and 18K steps to check that. |
| **Training and evaluation must run the same model forward.** | The deleted first glimpse run (2026-10-31). | Window choice in evaluation must be the model's own, as in training. |
| **Measure design choices before a multi-day run depends on them.** | Same incident. | Run 5's window budget was measured against gold boxes with `glimpse-frames` before launch. |

Not every rule has a recorded date. Where a date is given above, it comes from the memory notes and the
review notes listed in [README.md](README.md#sources).
