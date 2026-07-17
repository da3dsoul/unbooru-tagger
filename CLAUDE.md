# Anime Tagging Model — Project Context

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
We are deliberately moving past that design.

## Chosen architecture: dual-encoder, sigmoid contrastive tagging
Reference implementations to study: **JoyTag**, **SigLIP** (sigmoid loss,
not softmax — matters because tags are multi-label and shouldn't compete
for probability mass the way softmax forces them to).

**Image tower**
- ViT or ConvNeXt backbone.
- Needs to expose BOTH a pooled global embedding AND the pre-pool spatial
  feature map (the latter is required for localization — see below).

**Tag tower**
- NOT a full text transformer per tag — tags are short atomic strings
  ("hatsune_miku", "twintails", "nurse_costume"), not sentences.
- Learned embedding table: one vector per tag, trained jointly.
- Warm-start each row from a small frozen language model's embedding of
  the tag string, so rare/new tags inherit a semantic prior from similar
  tags before they've seen much training data.

**Training objective**
- Sigmoid loss between every image embedding and every tag embedding in
  the batch (SigLIP-style). Each image-tag pair is an independent binary
  match/no-match decision.
- Confidence for a given image-tag pair = sigmoid(dot product of the two
  embeddings).

## Localization (rough region only — no bounding boxes needed)
Dot product between a tag's embedding and the SPATIAL (pre-pool) image
feature map produces a heatmap of where that tag's evidence concentrates.
No separate CAM/Grad-CAM step needed — falls directly out of the
dual-encoder design (same trick MaskCLIP-style zero-shot segmentation
uses on top of CLIP). This is intentionally approximate, not tight boxes.

## Handling tag growth
- **New tag, related to existing concepts** (e.g. new costume for an
  existing character): add a new row to the tag embedding table,
  warm-start from the text-embedding prior, fine-tune ONLY that row
  against newly tagged images, image encoder frozen. Cheap, fast.
- **Genuinely new tag family** (new series/characters never seen before):
  same process, just expect it needs more images before the embedding
  stabilizes since there's no correlated tag to lean on.
- **Periodic full fine-tune passes** (e.g. weekly/monthly): unfreeze the
  tag table plus a few top image-encoder layers so newly added tags
  settle into a globally consistent embedding space instead of drifting.

## Long-tail handling (important — vocabulary is large and growing)
- Minimum image threshold (~10-20 images) before a tag gets its own
  trained embedding; below that, it rides on the text warm-start prior
  and correlated tags.
- Batch sampler must OVERSAMPLE rare tags — natural frequency sampling
  lets common tags (1girl, solo, etc.) dominate gradients and starve
  everything else. This matters more than almost anything else at this
  vocabulary scale.

## Build order
1. Working sigmoid contrastive dual-encoder trained on the top few
   thousand most common tags first — validates the pipeline without
   long-tail pain.
2. Add spatial-similarity localization on top.
3. Build the "add a tag" pipeline (warm-start + row fine-tune) as its own
   tool; test by adding one held-out tag and confirming no full retrain
   is needed.
4. Scale vocabulary out with the minimum-image threshold and batch
   sampling fix already in place.

## Stack notes
- Training: Python/PyTorch. `timm` for backbones, `open_clip` as a
  reference for contrastive training loops.
- Serving: export to ONNX; can be served from a .NET service if that
  fits the rest of the stack better than a Python inference server.