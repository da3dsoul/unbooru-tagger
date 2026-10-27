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
- **Configurable size and multi-scale fusion (run 2):** `--stem-channels`,
  `--stage-channels`, `--blocks-per-stage` set the tower for a fresh run (the
  checkpoint's `model_config.json` wins on resume; `ModelConfig.From(tower)` saves
  the real shape — it used to always save `ModelConfig.Default`). `--fused-stages`
  adds an FPN-style top-down path (`ImageTower.Laterals`): the last stage's projected
  map is nearest-upsampled 2x and added to a 1x1 projection of the previous stage,
  once per fused stage, so a 4-stage stride-32 tower with 2 fused stages emits a
  stride-8 (64x64 at 512px) heatmap. The run-1 gold set showed small objects
  (earrings, hairclips, chokers) at near-chance pointing on the old 32x32 map. The
  hand-built ONNX exporter mirrors it with `Resize` (nearest, asymmetric/floor) +
  `Add`; `ImageTowerOnnxExportTests` checks the fused graph against TorchSharp.
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

**Tag tower — implemented, warm-start prior wired in but optional**
- `TagTower`: a learned embedding table, one row per tag, trained jointly —
  as planned.
- The warm-start mechanism (`ITagTextEmbedder` / `OnnxTagTextEmbedder` in
  `unbooru-tagger-core`: runs a frozen ONNX text-embedding model to produce
  a tag's prior vector) is now wired into both call sites that can use it —
  `train`'s fresh-vocabulary path and `add-tag` — via
  `--warm-start-model`/`--warm-start-vocab` (`TagWarmStartEmbedderFactory`
  in `unbooru-tagger-training`). `--warm-start-vocab` is a standard
  BERT-family "vocab.txt"; tokenization is a from-scratch minimal WordPiece
  implementation (`WordPieceTokenizer`), since the interface deliberately
  leaves tokenization to the caller (the right tokenizer depends on which
  frozen model gets chosen) rather than baking one algorithm in. Booru tags
  use underscores where the text model expects spaces
  (`hatsune_miku_nurse_costume` → `hatsune miku nurse costume`) — that
  normalization lives in the factory, not the generic tokenizer. Both
  flags are optional and must be given together; omitting them keeps the
  old `EmbeddingInit.RandomRow` (small uniform random noise) behavior
  exactly as before. No projection step: the text model's output dim must
  equal the tag tower's `--embedding-dim`, or `OnnxTagTextEmbedder.Embed`
  throws immediately with the actual vs. expected size, rather than either
  silently truncating/padding or failing later with a cryptic tensor-shape
  error inside `TagTower`.

**Training objective — implemented, plus two label-free auxiliary losses added later**
- Primary loss: SigLIP-style sigmoid loss between every image embedding and
  every tag embedding in the batch (`SigmoidContrastiveLoss.Compute`) — each
  image-tag pair is an independent binary match/no-match decision, as
  planned. Confidence for a pair = `sigmoid(scale * dot(pooled image
  embedding, tag embedding) + bias)` (`TagScorer.Score`).
- **Learnable logit scale/bias** (`LogitScaleBias`, `LogitScaling`) —
  matches the real SigLIP paper's own parametrization, added after the
  initial implementation shipped without it. With a vocabulary in the tens
  of thousands and small batches, almost every (image, tag) pair is a
  negative; a plain unscaled/unbiased sigmoid has no parameter dedicated to
  representing that imbalance, so it was found pushing rare/mid-frequency
  tags' own embeddings toward "always predict negative" instead — those
  tags scored near-zero on images that obviously matched, while common tags
  stayed well-calibrated. Scale is stored in log-space (can't cross zero);
  bias starts at -10 so an untrained tag defaults to "no match." Persisted
  in its own checkpoint/optimizer files, deliberately kept separate from
  the main image/tag-tower optimizer so resuming an older checkpoint (one
  written before this existed) can't corrupt that optimizer's already-
  accumulated momentum — see `TrainingState.SaveLogitScaleBiasOptimizerState`'s
  doc comment. Exported alongside the rest of the model bundle
  (`logit_scaling.json`) so inference reproduces the same operating point;
  a model bundle exported before this existed loads as `LogitScaling.Identity`
  (`Scale=1, Bias=0`), matching its old unscaled/unbiased behavior exactly.
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
- Trained via `SigmoidContrastiveLoss.ComputeLocalizedChunked`
  (`--localization-tag-chunk-size`, default 4096), not `ComputeLocalized`
  directly — the `[batch, vocabSize, locations]` tensor `ComputeLocalized`
  builds is the single largest memory user at this vocabulary's scale, so
  training processes the vocabulary in chunks instead, backpropagating
  each one immediately (its own `backward(retain_graph: true)` call, not
  one combined `loss.backward()`) so a chunk's memory is freed before the
  next chunk's forward pass. Exact, not an approximation — each tag's
  pooling only ever depends on its own locations, never other tags, so
  this is just a different order of the same sum; verified directly by a
  gradient-equivalence test against `ComputeLocalized`, not just by
  argument. `ComputeLocalized` itself is unchanged and still used directly
  by the test suite as the reference implementation that verifies this
  against.
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

## Deliberate batches, targets and evaluation — added after the first full run
The first full run (12 epochs, `rare-tag` sampling, every untagged tag a
uniform -1) early-stopped on mean validation loss while still producing
noisy boxes, false tags and misses. Mean loss over 25K tags per image is
~600:1 easy negatives; an all-negative predictor scores ~0.014 and the run
stopped at ~0.005. Training now uses the per-tag structure the crawler
built the corpus around:
- `TagStatistics`: per-tag positive lists, plus two things estimated from a
  bounded sample of each tag's images rather than an exact corpus-wide pair
  table (which is too big at this scale): implied tags (P(t|s) ≥ 0.6) and
  hard-negative sources (the crawler's own `FindHardNegativeSources` definition).
- `TagCentricBatchSampler` (default): batches are built around anchor tags
  drawn uniformly over the vocabulary, each with positives AND hard negatives.
- `PairWeightBuilder` + `SigmoidContrastiveLoss.ComputeWeighted` (default):
  pairs are weighted by certainty.
  - Positives and verified negatives (anchor hard negatives,
    `ExclusiveTagGroups`) use full BCE.
  - Likely-missing tags get weight 0.
  - Other negatives use the asymmetric loss (ASL).
  - Tags no training image carries get weight 0.
  The same targets drive the localization loss, with `rating:*` excluded.
- `TagRankingEvaluator`: early stopping tracks held-out mean per-tag AP,
  scored against each tag's hard negatives. It also calibrates per-tag
  best-F1 thresholds (`TagThresholds`, shipped in the model bundle). The
  `evaluate` command runs it on any checkpoint.
- `--sampling rare-tag --targets legacy` reproduces the original behavior.
- **Sparse localization loss (default with deliberate targets):**
  `LocalizationPairSelector` keeps, per image:
  - localizable positives;
  - certain negatives;
  - the `--localization-mined-negatives` (32) highest-scoring wrong tags.

  `SigmoidContrastiveLoss.ComputeLocalizedPairs` scores only those, about 100
  pairs per image instead of 25K. That cut a step from 1,413ms to 468ms on the
  old tower; most of the old step time was this loss. Equivalent to the dense loss
  when every pair is selected (tested).
- **Localizability rules (`TagLocalizability`, `localizable_tags.txt`):**
  whole-image tags never reach the localization loss, and `detect` reports them
  without boxes. Covered: artist/series/meta/rating, backgrounds, mediums,
  styles, lighting, weather, framing, people counts and format. Rules are anchored
  (exact / prefix* / *suffix / !exception), because substring rules misfire
  (`eyeshadow`, `covered navel`).
- **Content ratings:** the crawler's `export-ratings` writes `ratings.bin`, the
  strictest rating across each image's sources. Training learns three cumulative
  labels, with cutoffs calibrated for recall rather than F1 (`RatingEvaluator`).
  Before this, ratings never reached training at all.
- **Vocabulary:** a fresh run drops rows no image carries (`VocabularyRemap.Compact`;
  ~6.5K stale rows such as "long hair" beside "long_hair"). Dataset rows are matched
  to a checkpoint's vocabulary by tag name, not by row index.
- **LR schedule:** `--lr-schedule cosine` (default) gives linear warmup
  (`--warmup-steps`) then cosine decay to 5% across `--epochs`, computed from the
  global step, so a resume lands on the same rate. Raising `--epochs` mid-run
  reshapes the remaining decay.
- **Data loading:** `BatchPrefetcher` + `PreprocessedDatasetCacheReader.ReadEncoded`
  sample ahead on a background thread, then read each window offset-sorted with
  32 reads in flight. That gave ~450–590 img/s vs ~48 for the old synchronous path
  on the 6-disk RAID5; a step now waits ~1ms on data.
- **Gold set** (`/media/da3dsoul/Golias/unbooru-tagger/gold`, not in git): 289
  hand-checked validation images (164 clean, 125 messy) with certain, group and
  ignore boxes (COCO-style crowd/ignore semantics). Scored by `evaluate --gold`,
  rendered side by side with `render-gold`. Run 1's epoch-12 baseline is in
  `run1-small-convnext-epoch12-COMPARISON-ONLY/README.txt`. With a model, `render-gold`
  writes the detections to `boxes.json` instead of PNGs (unless `--png`), which the gold
  set's `compare.html` draws on a canvas, one list entry per box: in a PNG, overlapping
  boxes from several tags can't be told apart.
- **Separate localization map (`--localization-map fused`, run 2 from epoch 2):** the
  localization loss, `evaluate`'s gold scoring and inference heatmaps use the fused
  laterals alone (`ImageTower.ForwardMaps`), without the upsampled last stage, whose
  locations each see most of the image. Classification still pools the full map. Exported
  as an extra ONNX output (`localization_features`); older bundles still work. A 3,000-step
  A/B: gold pointing 27.0% → 29.0% (clean 29.7% → 33.9%), deletion win 75% → 82%.
  `--localization-map local` (`ImageTower.LocalizationSkippedLaterals`, saved in `model_config.json`,
  mirrored by the exporter) also leaves out the coarsest (stride-16) lateral, so a location sees
  about 100px. Tried in run 3, not adopted: it fixed shoes/gloves pointing but lost glasses,
  blush, hats and characters.
- **Context negatives:** `TagStatistics` splits the old "implied" band. P(t|s) ≥
  `--missing-threshold` (0.9) is still a likely-missing label (weight 0). Between
  `--implied-threshold` (0.6) and that, s is a *context source* for t: a character who
  usually wears a hat. Images with s but not t are trained as unverified (focal-damped)
  negatives and sampled per anchor (`--context-negatives-per-anchor`). Before, a
  character's hatless images were ignored, so "this character, therefore hat" was free to
  learn. `evaluate` reports a per-tag context AP (positives vs. those images); hard/random
  negatives still leave the whole band out, so the early-stopping mAP stays comparable.
- **Context negatives at full strength (`--context-negative-weight`, run 4 uses 3):** focal
  damping left "this character, therefore gloves" nearly free, since a confident wrong "yes" on
  a context negative cost almost nothing. With a weight, `PairWeightBuilder` trains every such
  pair in a batch (not just the sampled ones) as full BCE at that weight. Localizable tags only:
  on people counts it doubled 1girl + 2girls contradictions, since those are often just
  untagged. Run 3 A/B, weight 3 vs. off (both with mosaics): "still called with the object blanked" 85.9% → 70.3%,
  context recall/false-positive gaps 30.0/26.3 → 24.1/21.1 pts, mAP 0.750 → 0.745. Recall where
  context makes a tag unlikely didn't rise: the gaps shrink by suppressing context-driven calls.
- **Label completion from Danbooru implications:** the crawler's `export-implications`
  writes `tag_implications.json` (46,895 active rules; 6,625 with both tags in run 2's
  vocabulary, e.g. witch_hat → hat, red_hat → hat). Danbooru applies them to every post,
  Gelbooru doesn't, so `TagImplications.Complete` adds each image's implied tags when the
  tags are loaded (train and `evaluate`). Ground truth, not a statistic.
- **Site coverage:** `export-image-sites` writes `image_sites.bin` (per cache row:
  Danbooru-only 1.18M, Gelbooru-only 1.33M, both 2.87M) and `export-tag-agreement` writes
  `tag_agreement.json` (per tag, over the 2.9M images on both sites: listed by both / one
  site only). 1,136 vocabulary tags exist on only one site's tag system: 993 Gelbooru-only
  (silver_hair, black_legwear, *_footwear, female_focus, 10s, gen_N_pokemon) and 130
  Danbooru-only (covered_nipples, photoshop_(medium)). An image from only the other site
  can never carry them, so `TagStatistics.IsSiteBlind` pairs get weight 0 in training,
  are never sampled as negatives, and are left out of evaluation negatives, threshold
  calibration and image-level precision. The agreement counts are NOT usable as tagging
  recall: Gelbooru largely copies Danbooru's tags, so agreement is 96-99% for nearly every
  tag. (`TagStatistics.Build`'s `tagRecall` posterior rule exists but is unused for that
  reason; the fixed `--missing-threshold` applies.)
- **Gated crop consistency (`--crop-consistency-weight`):** the classifier's verdict on a
  random crop teaches the localization map where a tag is
  (`SigmoidContrastiveLoss.ComputeWindowConsistency`). Ungated, it made localization worse:
  `evaluate --gold`'s crop-teacher check showed the epoch-1 classifier at chance (51%
  balanced) on crops, answering from context. `CropTeacherGate` trusts only tags with held-out
  AP ≥ 0.8, ≥ 20 positives and (if measurable) context AP ≥ 0.8 in the latest epoch
  evaluation, recomputed each epoch; only confident answers (p > 0.9 / < 0.1) train; only
  images with a trusted tag are cropped, so its cost tracks how much the teacher is trusted.
- **Background suppression (`--localization-background-weight`, off):** tested and not used;
  it shrank heatmaps but didn't move peaks.
- **Mosaic localization training (`--mosaics-per-step`, `--mosaic-weight`; `Mosaic`):** each step,
  canvases of four of the batch's images shrunk 2x into a 2x2 grid. Unlike a crop, a tile keeps
  all its tags, so no label has to be learned from context. Each tile is its own localization
  bag, with the other tiles' positive tags as negatives: "gloves are in that tile, not this
  one", even on the character who usually wears them. Run 3 A/B at 6,000 steps: gold pointing
  33.1% → 35.7% (small 6.6 → 7.8, medium 16.1 → 20.5), mAP flat, +28% step time for 2 mosaics.
  It trains only the localization map, so the context gaps didn't move; that took full-strength
  context negatives. Run 4 uses both.
- **Intel GPU / XPU training (run 3):** `-p:TorchBackend=xpu` builds against the `torch 2.10.0+xpu`
  wheel's libtorch with stock TorchSharp. The device is `(DeviceType)12`, since TorchSharp's enum has no XPU.
  `OneDnnPrecision` turns on oneDNN TF32 by calling libtorch's exported `at::Context` members, because
  TorchSharp only exposes the CUDA switches. Only the legacy `allowTF32OneDNN` flag reaches XPU convs.
  Measured on an Arc Pro B70: 158 ms/step for run 2's config vs. 278 on the A4000. It also needs the
  NEO anti-migration env vars (README) so VRAM overflow fails instead of spilling to host RAM.
- **Offloaded evaluation (`--evaluation-command`, `OffloadedEvaluations`):** each epoch's evaluation
  runs as a separate process on a model snapshot (`history/epoch-N/checkpoint/`) while training
  continues. Results are applied in epoch order at the next checkpoint save, and marker files make
  that survive restarts. `evaluate` now reads the validation seed from a progress-only snapshot
  (`TrainingState.HasProgress`); it used to need `optimizer.dat` too and fall back to seed 0.
- **Snapshot evaluations and trend charts (`--snapshot-evaluation-steps`):** with `--evaluation-command`, the
  trainer also snapshots the model every N global steps and runs the same command on it in the background
  (`SnapshotEvaluations`, reports in `snapshots/step-N/`, weights deleted once evaluated). They never feed early
  stopping. The dashboard charts mAP, gold pointing, occlusion, the context gaps and exclusive violations over
  every epoch and snapshot evaluation (`BrailleChart`), and the loss over the whole run (`LossHistory`, saved
  with each checkpoint). All of it is read back from disk on resume and cut back to the resumed step.
- **Graphical charts (`DashboardScreen`, `charts` command):** in a Sixel-capable terminal (`TerminalGraphics`;
  inside tmux, the attached client's reported features) the dashboard draws the charts as an image
  (`TrainingCharts`, SkiaSharp, encoded by `Sixel` with an adaptive 256-color palette) instead of braille text.
  Spectre's live display can't host it: it rewrites every line each refresh, and overwriting an image's lines
  erases it (tmux drops it). So the trainer draws the screen itself: the Spectre-rendered panel in the top rows
  every second, the image in the rows below, redrawn only when its data, the panel's height or the terminal size
  changes. `charts` draws the same image from a checkpoint directory alone (or to a PNG).
- **Per-epoch gold scoring (`train --gold`):** each epoch evaluation also scores the gold set
  and the crop teacher (`localization_report.json`); every epoch's reports are archived
  under `history/epoch-N/`.
- **Object or context? (every epoch evaluation):** two checks of whether a tag is found from the
  object or from who is in the image and who drew it.
  - *Occlusion* (gold, `LocalizationReport.Occlusion`): blank a found tag's boxes, keep everything
    else, and see whether the classifier still calls it; a control blanks the same sizes elsewhere.
  - *Context reliance* (held-out, `EvaluationReport.ContextReliance`): each image's prior for a
    tag is the highest P(tag | its character/artist/series) in training. Recall and false-positive
    rate are compared across prior bands. Gaps near 0 mean the object is what's seen.

  These replace per-tag context AP for this purpose: on a 20K sample that has a median of 2
  context negatives per tag, so its chance level is about 0.98. `CropTeacherGate` still uses it.
  Run 3 epoch 1 → 2: recall gap +20.8 → +29.4 pts, still called with the object blanked 87% →
  83%, while mAP and gold pointing rose. `evaluate` prints both, plus
  `EvaluationReport.ExclusiveViolations` (which exclusive pairs fire together, and which side is
  true). That breakdown showed `solo, no_humans` was a wrong default rule (a lone creature or
  robot is tagged both), so `ExclusiveTagGroups.DefaultText` no longer has it.

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
- **Batched inference (`detect-batch`, `tag-batch`):** the exporter declares a symbolic
  batch dimension, so `OnnxImageEncoder.EncodeBatch` runs many images per call; every op is
  per image (GroupNorm included), tested against running each image alone. Bundles exported
  before this have the batch pinned to 1 and run one image at a time (`SupportsBatches`).
  `BatchPipeline` overlaps decoding, batched encoding and per-image scoring/boxes through
  bounded queues. `--device Auto` uses ONNX Runtime's CUDA provider when it loads (the core
  references `Microsoft.ML.OnnxRuntime.Gpu` 1.24.2, the newest built for CUDA 12), else CPU.
  `detect-batch` gets tags and boxes from one model run per image, one JSON line per image.

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
   `add-tag` command; warm-start defaults to random init unless
   `--warm-start-model`/`--warm-start-vocab` point at a real frozen
   text-embedding model (see the tag-tower section above).
4. ✅ Long-tail handling (minimum-image thresholds + oversampling) — done.
5. ✅ *(beyond original scope)* Two label-free training-time objectives
   added to sharpen localization and representation quality further,
   without any new training data: the MIL/log-sum-exp pooled localization
   loss, and the SimSiam self-supervised consistency loss.
6. ⬜ Not yet done: a partial / top-layers-only unfreeze mode for periodic
   fine-tunes (currently all-or-nothing, see the tag-growth section
   above).

## Stack — actual, differs from the original plan
- **Originally planned**: Python/PyTorch training, `timm` for backbones,
  `open_clip` as a contrastive-training-loop reference.
- **Actually built**: entirely C#/.NET. Training uses TorchSharp directly —
  no `timm`/`open_clip` dependency; the ConvNeXt block and training loop
  are hand-written against TorchSharp's `torch.nn` API. Serving exports to
  ONNX and loads it via ONNX Runtime from a .NET CLI
  (`unbooru-tagger-inference`), not a Python inference server. There is no
  Python code anywhere in this repository.
