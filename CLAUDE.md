# Anime Tagging Model — Project Context

This file originally described the *planned* architecture before any code existed.
Most of it has since been built; where the implementation differs from the original
plan, that's called out explicitly below rather than silently updated, so the
reasoning behind a deviation stays visible. See [`README.md`](README.md) for how to
actually build/run/train the thing — this file is architecture rationale and status.

## Goal
Multi-label image tagger trained on booru-style string tags (no bounding box
labels). Needs: per-tag confidence, rough per-tag localization, and an
open-ended, ever-growing tag vocabulary (currently hundreds of thousands of
tags, long-tail distributed) that must support adding new tags later
(e.g. a new Hatsune Miku costume, a new character, a new series) without
full retraining.

## Why not a plain sigmoid classification head
A fixed N-output classification head doesn't scale to a vocabulary this
large or this dynamic — every new tag means expanding and retraining the
head. Reference prior art for that approach: WD14 tagger, DeepDanbooru.
This project deliberately moves past that design.

## Architecture: dual-encoder, sigmoid contrastive tagging
Reference implementations studied: **JoyTag**, **SigLIP** (sigmoid loss,
not softmax — matters because tags are multi-label and shouldn't compete
for probability mass the way softmax forces them to).

**Image tower — implemented, ConvNeXt only**
- `unbooru-tagger-training/Model/ImageTower.cs`: a compact ConvNeXt-inspired
  stack — a stride-4 stem, `GroupNorm`-normalized stride-2 downsample convs
  between stages of `ConvNeXtBlock`s (depthwise 7x7 conv, 4x-expanding
  pointwise MLP, LayerScale). Default config (`ModelConfig.Default`):
  `stemChannels=64, stageChannels=[64,128,256], blocksPerStage=[2,2,2]`.
- The original plan named "ViT or ConvNeXt" as the backbone choice; only
  ConvNeXt is actually built. Swapping in a ViT later would mean a new
  `Module<Tensor, (Tensor Pooled, Tensor Spatial)>` implementation exposing
  the same pooled+spatial contract — nothing else in the codebase is
  ConvNeXt-specific.
- Exposes both a pooled global embedding (`adaptive_avg_pool2d` over the
  spatial map) AND the pre-pool spatial feature map, as planned — this is
  what localization needs.
- `Projection` (a 1x1 conv) is applied to the full spatial map BEFORE
  pooling, not to the pooled vector afterward, so the spatial output and
  the pooled output land in the same embeddingDim-dimensional space as tag
  embeddings — required for the heatmap dot product to be dimensionally
  valid at all.

**Tag tower — implemented, warm-start prior NOT wired in by default**
- `TagTower`: a learned embedding table, one row per tag, trained jointly —
  as planned.
- The warm-start mechanism itself exists (`ITagTextEmbedder` /
  `OnnxTagTextEmbedder` in `unbooru-tagger-core`: runs a frozen ONNX
  text-embedding model to produce a tag's prior vector), **but no CLI flag
  currently wires a concrete model into it.** Both call sites that could use
  it — `train`'s fresh-vocabulary path and `add-tag` — pass
  `warmStartEmbedder: null` today, which falls back to
  `EmbeddingInit.RandomRow` (small uniform random noise). In the current
  build, every new tag starts from random noise, not a semantic prior from
  related tags. Wiring in a real frozen embedding model (and a CLI flag to
  point at it + a tokenizer) is unfinished work, not a design change.

**Training objective — implemented, plus two label-free auxiliary losses added later**
- Primary loss: SigLIP-style sigmoid loss between every image embedding and
  every tag embedding in the batch (`SigmoidContrastiveLoss.Compute`) — each
  image-tag pair is an independent binary match/no-match decision, as
  planned. Confidence for a pair = `sigmoid(dot(pooled image embedding, tag
  embedding))` (`TagScorer.Score`).
- `--localization-weight` / `--localization-temperature` (on by default —
  see README for current defaults): an MIL/log-sum-exp pooled auxiliary
  loss (`SigmoidContrastiveLoss.ComputeLocalized`), scored from every
  spatial location instead of the one pooled embedding, so sharp
  localization is directly rewarded by the loss instead of only being a
  side effect of pooling geometry. At high temperature this is
  mathematically identical to the primary loss (dot product distributes
  over an average, so mean-of-per-location-logits == dot(mean-pooled
  embedding, tag)); lower temperature pulls it toward a max over locations,
  concentrating gradient on a tag's best-matching spot.
- `--self-supervised-weight` (on by default): a SimSiam-style consistency
  loss between two random-crop-augmented views of the same already-loaded
  batch (`RandomCropAugmentation`), predicting each other's pooled
  embedding through a small predictor head (`PredictionHead`) with
  stop-gradient on the target side — no momentum/teacher network needed.
  Uses zero tag labels, just images already in the corpus.
- Neither auxiliary loss needed new training data of any kind — both are
  pure restructurings of the existing sigmoid-contrastive setup, added
  after the original plan was written to sharpen localization/
  representation quality without bounding-box labels.

## Localization — implemented: heatmap, and (beyond the original scope) approximate boxes
- **Heatmap** (`TagScorer.Heatmap`; `unbooru-tagger-inference heatmap`
  command): dot product between a tag's embedding and the spatial feature
  map at every location, sigmoid-scored. No separate CAM/Grad-CAM step —
  falls directly out of the dual-encoder design (MaskCLIP-style), exactly
  as originally planned.
- **Bounding boxes** (`unbooru-tagger-inference detect` command) — the
  original plan said "rough region only, no bounding boxes needed"; boxes
  were added later anyway, still entirely label-free:
  - Connected components over a thresholded heatmap grid produce the boxes
    in the first place.
  - Per-tag percentile thresholding (`--box-percentile`) cuts within each
    tag's own heatmap range instead of one global absolute cutoff, so a
    box tightens around a tag's own peak.
  - A joint bilateral filter (`HeatmapRefiner`) uses the source image's own
    edges as a guide to snap the heatmap boundary to real object
    contours — the label-free version of the CAM+DenseCRF refinement trick
    from weakly-supervised segmentation.
- All of this is still rough, approximate localization, not a trained
  detector — that part of the original framing still holds even though
  boxes now exist.

## Handling tag growth
- **New tag pipeline — implemented** (`unbooru-tagger-training add-tag`):
  appends a new row (random-init today — see the tag-tower warm-start gap
  above), fine-tunes ONLY that row against newly tagged images with the
  image encoder fully frozen (every `imageTower` parameter gets
  `requires_grad = false`), and promotes the tag from warm-start-only to
  trained once it crosses `--min-image-threshold` (default 15) images.
  Cheap and fast, as planned.
- **Periodic full fine-tune passes — implemented differently than planned.**
  The original plan: unfreeze the tag table plus *a few top image-encoder
  layers* so new tags settle into a globally consistent embedding space.
  What actually exists is all-or-nothing: `train` unfreezes the entire
  image tower and tag table together (a full/periodic retrain), and
  `add-tag` freezes the entire image tower. There's no partial /
  top-layers-only unfreeze mode — that middle ground from the original
  plan was never built.

## Long-tail handling — implemented
- Minimum image thresholds gate vocabulary growth at two different points:
  `--min-tag-images` (data pipeline, default 100) decides whether a tag
  gets a vocabulary row at all; `--min-images-per-tag` (data pipeline,
  default 15) reserves rarest-first images per tag when `--max-images` caps
  the corpus; `--min-image-threshold` (`add-tag`, default 15) gates
  promotion from warm-start-only to trained. All in the ~10-20 image range
  the original plan called for.
- Batch sampler oversamples rare tags (`RareTagOversamplingBatchSampler`):
  each image's sampling weight is `1 / frequency(its rarest tag)`, so
  common tags (1girl, solo, ...) don't dominate gradients and starve the
  long tail — exactly the concern the original plan called out as
  mattering "more than almost anything else" at this vocabulary scale.

## Build order — status
1. ✅ Sigmoid contrastive dual-encoder validated on a bounded vocabulary
   (via `--min-tag-images`, not literally "top N" but the same effect).
2. ✅ Spatial-similarity localization — done (heatmap), and extended past
   the original "heatmap only" scope with approximate bounding boxes
   (percentile thresholding + bilateral refinement), still entirely
   label-free.
3. ✅ "Add a tag" pipeline (warm-start + row fine-tune) — done as its own
   `add-tag` command; warm-start currently defaults to random init (see
   the tag-tower gap above), not the planned text-embedding prior.
4. ✅ Long-tail handling (minimum-image thresholds + oversampling) — done.
5. ✅ *(beyond original scope)* Two label-free training-time objectives
   added to sharpen localization and representation quality further,
   without any new training data: the MIL/log-sum-exp pooled localization
   loss, and the SimSiam self-supervised consistency loss.
6. ⬜ Not yet done: wire a real frozen text-embedding model into the
   tag-tower warm-start path (the interface exists; nothing plugs into it
   by default, and there's no CLI flag for it yet); a partial /
   top-layers-only unfreeze mode for periodic fine-tunes (currently
   all-or-nothing, see the tag-growth section above).

## Stack — actual, differs from the original plan
- **Originally planned**: Python/PyTorch training, `timm` for backbones,
  `open_clip` as a contrastive-training-loop reference.
- **Actually built**: entirely C#/.NET. Training uses TorchSharp directly —
  no `timm`/`open_clip` dependency; the ConvNeXt block and training loop
  are hand-written against TorchSharp's `torch.nn` API. Serving exports to
  ONNX and loads it via ONNX Runtime from a .NET CLI
  (`unbooru-tagger-inference`), not a Python inference server. There is no
  Python code anywhere in this repository.
