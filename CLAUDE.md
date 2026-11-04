# Anime Tagging Model — Project Context

This file originally described the *planned* architecture before any code existed.
Most of it has since been built; where the implementation differs from the original
plan, that's called out explicitly below rather than silently updated, so the
reasoning behind a deviation stays visible. See [`README.md`](README.md) for how to
build and run the thing, [`docs/TRAINING.md`](docs/TRAINING.md) for how to train it, and
[`research/`](research/README.md) for the research history — this file is architecture
rationale and status.

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
  see docs/TRAINING.md for current defaults): an MIL/log-sum-exp pooled auxiliary
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
  rendered side by side with `render-gold`. A third set, gold-auto (2026-11-02/03, `gold-auto/`, not in git; its README.txt):
  1,020 held-out images, 34 object tags x 30 at full-body/cowboy/upper-body framing, boxes from SAM 3 (facebook/sam3, text
  prompts, score >= 0.7, no ignore boxes; `gold.json`, 1,830 pairs with a box). Checking OWLv2/Grounding DINO proposals with
  Claude subagents (`tools/CHECK.md`) was the first plan, dropped at ~18M tokens for the set; SAM 3 scored 55% found / 88% right
  at >= 0.7 on gold-small's verified boxes (`tools/score.py`), and its boxes score checkpoints like the user's on the same pairs
  (pointing 60.4 vs 59.4%, object effect 0.93 vs 0.93), its bias being which pairs it keeps (easier ones). A second set, gold-small (`gold-small/`, 126 small-object images held out from run 4), is
  scored on its own by `evaluate --gold-small` into `localization_report_small.json` and charted beside the main set
  (most main-set images are in run 4's training split). Run 1's epoch-12 baseline is in
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
- **Tag parentage (`--tag-parentage`, `TagParentage`/`TagFamilies`, run 4 from step ~502K):** the user's
  reviewed is-a / implies / excludes rules (the tag-parentage review page's `rules.json`, outside git), where the
  user's ontology wins over Danbooru's: on Danbooru `shoes` means non-boot shoes (9% of boot images carry it) and
  `hair_ornament` leaves out ribbons and bows. Is-a and implies rules complete labels like Danbooru's implications
  (boots → shoes, chaining through them). An image tagged with a family parent but no kind of it (shoes, not which)
  leaves the kinds unknown (weight 0), not negative, and a parent is never a hard-negative source for its kinds; this
  applies only through reviewed is-a edges, so Danbooru-only families (costume → character) keep their negatives.
  Excludes rules (no_shoes, barefoot → shoes) make the parent and every kind of it certain negatives unless the image
  is tagged with the parent (a second character), and their images are sampled first as hard negatives: the user's
  point that no_X means the character usually has X. Evaluation reads labels the same way, and the gold set too (a
  boots box is a shoes box; a parent-only box is ignored for its kinds). Saved into the checkpoint; `none` turns it off.
- **Site coverage:** `export-image-sites` writes `image_sites.bin` (per cache row:
  Danbooru-only 1.18M, Gelbooru-only 1.33M, both 2.87M) and `export-tag-agreement` writes
  `tag_agreement.json` (per tag, over the 2.9M images on both sites: listed by both / one
  site only). By `TagAgreement.LoadSiteCoverage`'s rule (>= 200 uses on those images, < 2% of them on one
  site), 950 vocabulary tags are single-site: 925 Gelbooru-only (silver_hair, black_legwear, *_footwear,
  female_focus, 10s, gen_N_pokemon) and 25 Danbooru-only (covered_nipples, photoshop_(medium)) (recounted
  2026-11-03; this line used to say 1,136 = 993 + 130, which neither adds up nor reproduces). An image from only the other site
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
  NEO anti-migration env vars (docs/TRAINING.md) so VRAM overflow fails instead of spilling to host RAM.
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
- **Context-shortcut experiments (run 4, all off by default, none adopted).** Following the
  user's diagnosis that the remaining problem is *classifying* tags like gloves, not localizing
  them, each was A/B tested from run 4 at step 271,754 for 6,000 steps (control: mAP 0.572,
  still called with the object blanked 87%):
  - `--negative-gamma 2` (was 4) **was adopted** from ~275K: mAP 0.577, image precision
    19.7% → 22.8%, still called 87% → 74%, exclusive violations 8.3% → 12%. Cross-site checks
    put forgotten labels for gloves/hat/earrings at ~3% (~1% of their negatives), so a strong
    focal damping protected few true positives.
  - Context prior (`ContextPrior`, `--context-prior-weight`): a product of experts with
    P(tag | character/artist/series). It closed the gaps only by suppressing calls; mAP -2 to -4.
  - Focus anchors, tile classification, full-size mosaics (`FocusTags`, `--focus-share`,
    `--mosaic-classification-weight`, `--mosaic-full-size`): small gains; full size cost 1.4 pts.
  - Per-tag classification pooling (`ClassificationPooling`, `--classification-pooling-*`): a
    log-sum-exp over a 16x16 cell grid instead of the average, annealed from temperature 8 to
    0.25. Still-called 52% vs run 4's 72%, but mAP 0.567 vs 0.586 at 290K and gloves AP no better
    (0.349 vs 0.367). The evaluator's pooled path was checked (temperature 1000 equals the
    average), so the loss came from training. The ONNX exporter refuses pooled checkpoints.

  Run 4 on its own kept improving occlusion (still called 79% at 280K → 56% at 310K) while mAP
  rose. Gloves AP stayed ~0.35 in every arm.
- **Occlusion object effect (2026-10-29).** The still-called rate is noisy: it counts only pairs found at p ≥ 0.5, the median
  found score is 0.65 and the blanked one 0.54, so between checkpoints 4,500 steps apart 15% of shared pairs flip and a quarter
  of the found set changes; two identical 6,000-step A/B controls differed by 7 points. The occlusion check now scores every
  gold pair and reports `ObjectEffect`: mean logit(control blanked) − logit(object blanked), threshold-free over a fixed set.
  It rose at every checkpoint (epoch 1 0.33 → 550K 0.41 → 622K 0.48 → 700K 0.51; gold-small 0.15 → 0.34) and the trend charts
  show it instead of still-called. `evaluate --dump-occlusion` writes each pair.
- **Positive/negative selection A/B (2026-10-29, A4000, 6,000 steps from 626,771, two controls; all off by default).**
  `--out-of-context-share` (anchor positives with a low character/artist/series prior, `ContextPriorTable`): no effect (each
  tag got about one per run). `--out-of-context-all --out-of-context-weight 3` (every low-prior positive, weight 3): context
  recall gap 20.6 → 17.2 pts, false-positive gap 19.9 → 17.8, mAP equal, but gold object effect 0.767 → 0.728 (calls more,
  doesn't look harder). `--excluder-share 0.2 --excluded-negative-weight 3` (anchors on tags an excludes rule rules out, with
  no_X images as certain negatives): object effect +0.03 (gold) / +0.08 (gold-small), but footwear AP −0.03 to −0.05, since
  half its anchors were kinds of shoe; the sampler now draws the ruled-out parents uniformly. That version lost the gain
  (object effect 0.749 vs controls 0.774/0.768, shoes AP 0.618 vs 0.64), so neither is used.
- **Erased-view training (`--erasure-*`, `ErasureContrast`, 2026-10-29; not adopted).** The occlusion check
  as a loss (Shetty et al. 2019; GAIN/CSS for the negative variant). For called localizable positives, the region the tag's
  own map points at (detached; capped at 15% of the image) is blanked, and so is a same-size control on another of the
  image's tags (random if it has none). A hinge then makes the score drop at least `--erasure-margin` more for the object.
  A/B on the A4000, 6,000 steps from 626,771, controls ctl/ctl2 (mAP 0.706/0.707, gold object effect 0.73):
  - Zero-filled blanks, random control: object effect 0.87 but mAP 0.686. Re-scored with noise-filled blanks it hadn't
    seen, 0.37, below the controls' 0.42: it had learned "a grey patch lowers a score".
  - Negative variant (object-blanked view a negative, control-blanked a positive): object effect 1.08, no patch cue, but
    every called score rose (median logit 1.8 vs 0.6) and mAP fell to 0.667.
  - Mean/noise fills (`OcclusionFill`; no blur, since censoring is a tag, and no patches of other images, which can carry
    the object), control on another tag, 2 pairs/step, weight 1: object effect 0.92 with unseen zero fills, but blanking
    anywhere lowered scores (0.58 vs 0.09) and mAP 0.672.
  - Same, 1 pair/step, weight 0.25: the best so far. Object effect 1.11 with unseen zero fills, 0.78 with noise (controls
    0.73/0.42); control drop 0.28; object hurts more in 86% of pairs (controls 89/87%); mAP 0.692 (-1.4), pointing -0.8.
  - Same, every 2nd step only (`--erasure-every 2`): object effect 0.92 with unseen zero fills (hinge3 1.11), 0.79 with
    noise (hinge3 0.78; gold-small 0.61 vs controls 0.15/0.16); control drop 0.10 with zero fills, the controls' level
    (hinge3 0.28); object hurts more 88%; mAP 0.698 (-0.8), pointing equal, exclusive violations 9.0% vs 7.8/6.6%.
    Halving the dose kept the whole gain on the fill it trains with and half of it on the unseen one, with no "blanking
    anywhere lowers scores" left.
  - Every 4th step: object effect 0.92 with zero fills (same as every 2nd), 0.84 with noise and a control drop of 0.12
    (every 2nd: 0.79/0.26); gold-small 0.61/0.47, below every 2nd's 0.75/0.61; mAP 0.703 (-0.3), pointing 48.6% (+0.8),
    exclusive violations back to the controls'. 336 ms/step (+2%). Nearly free at 6,000 steps, but not with more training.
  - Every 4th step continued with a control to 12,000 and 18,000 steps (2026-10-30): the mAP gap didn't close (12K 0.702
    vs 0.707, 18K 0.703 vs 0.708; pointing -1.2 at both), and "blanking anywhere lowers scores" came back: the noise-filled
    control drop went 0.12 (6K) → 0.67 (12K) → 0.31 (18K), against the control's 0.06. The hinge is relative, so a cue that
    lowers both views costs it nothing. The object effect is a difference too, so it can't see such a cue; read it with the
    control drop, and on a fill the arm never trained with. At 18K, over all 1,030 gold pairs, zero fill: object effect 0.74
    vs 0.46, control drop 0.16 vs 0.06.
  - `--erasure-anchor-weight` adds a term holding the control-blanked score to the unblanked one. One-sided (only a drop
    penalized), weight 1, every 4th step, 12,000 steps: the patch cue went and overshot (noise control drop -0.13, so a blank
    elsewhere raised scores), the object effect was the best of any arm (zero fill 1.09 vs 0.69, gold-small 0.71 vs 0.45;
    noise 0.85 vs 0.41), but mAP 0.687 vs 0.707, image precision -2.2, recall where context makes a tag unlikely -1.3. The
    term is now symmetric (|unblanked - control|). That fixed the overshoot: the control drop sat at the control's level
    (zero fill 0.06 vs 0.07, noise 0.05 vs 0.06). Over all 1,030 gold pairs the object effect was 0.65 with zero fill (control
    0.46, one-sided anchor 0.72, unanchored 0.59), 0.56 with noise (0.41), gold-small 0.43/0.26 (0.30/0.15). But mAP was
    0.687 again, image precision -2.6, "general called explicit/questionable" 21.2% vs 17.8%. Anchored either way, the views
    cost 2 mAP points; unanchored, they learn the patch cue. Context false positives and the hand-checked set: unchanged.
  None of the doses moved the context false positives: averaged per tag, the false-positive rate where the character
  usually has the tag is 22.5-22.7% in every erasure arm vs 23.4/23.7% in the controls, and recall where context makes a
  tag unlikely is unchanged (32-33%), at 12K and 18K too. Erased views make a called tag depend on the object; they don't stop the call.
  The 6,000-step zero-fill numbers are over the 455 gold pairs (216 gold-small) the controls' older reports dumped; noise
  over all. Numbers over different pair sets aren't comparable.
  `evaluate --occlusion-fill` sets the check's fill, to test whether a gain carries to a fill the model never trained on.
  The views are picked right after the forward pass, so reading their maps back doesn't wait on the queued losses, and they
  share the mosaic canvases' forward pass. `--erasure-every N` runs them every Nth step only. A4000, 400 steps each:
  none 330 ms/step; 1 pair 390 in its own pass, 377 batched with the mosaics; 2 pairs 418; 1 pair every 2nd step 351.
  The A4000 is compute-bound, so batching saved little; the extra images themselves are the cost.
- **Context false positives are partly missing labels (2026-10-30).** Averaged per tag, run 4's false-positive rate where the
  character usually has the tag (prior 60-90%) rose from 20.4% at 170K to 24.6% at 1,030K, while where the prior is under 5%
  it fell from 8.1% to 3.0%. (The pooled rate, dominated by common tags, was flat, which hid it.) To see what those calls
  are, 180 such negatives (36 each of gloves, hat, glasses, wings, tail) were checked by eye (`context-negatives/`, outside
  git): the object is visible in 27, can't be told in 50 (crowds, comics, tiny figures) and absent in 103. A model calls
  most of the visible ones (run 4: 48% at epoch 1, 59-70% since 700K) and few of the absent ones (5-9% up to 600K, 11-12%
  since 700K; n=103, so ±3 pts). So most of the rise is the model finding objects nobody tagged; the real context false
  positives rose a little, to about 11%, against well under 4% where the prior is low. Context negatives at full strength
  train every one of those visible objects as a "no", so a higher `--context-negative-weight` pushes against real objects
  too. A 6,000-step A/B at weight 6 (vs. 3) bore that out: false-positive rate where the prior is high 22.1% vs 23.4/23.7%,
  with high-prior recall down as much (49.9% vs 52.2/52.7%), low-prior recall, object effect and mAP unchanged, and on the
  checked set 12% of absent objects called (controls 16/10%) and 56% of visible ones (controls 59/56%). With hinge5's erased views added it was worse than either alone: blanking anywhere lowered scores again (noise-fill control drop 0.33 vs 0.12) and mAP 0.698. Solo, non-comic images are no cleaner (6 visible, 5 unclear, 32 absent of 43). `score.py` there scores any
  `evaluate --dump-scores` output on this set.
- **Labelled context check (2026-10-30, `evaluate --context-check`, `LabelledContextCheck`).** A cheap-first investigation
  (`ctxfp/README.txt`, outside git) grew the hand-checked set with a Sonnet labeller, calibrated on the user's 180 (of the
  user's 103 absent it marked 37 absent, 66 unclear, 0 present; of 27 present, 23 present, 4 unclear): 257 absent / 142
  visible-but-untagged pairs over gloves/hat/glasses/wings/tail (`context-check/context-check.json`; 197/88 fall in the first
  20K held-out images that evaluations score). On it run 4's calls on absent objects went 10% (700K) → 7-9% (1.09-1.24M),
  mean log-odds past the threshold -0.94 → -1.28, while visible-but-untagged calls stayed ~70-77%; over the same steps the
  band false-positive gap rose. So the band gap mostly counts untagged objects the model finds, and true context false
  positives are flat to slightly falling. Evaluations report both rates and chart the mean log-odds past the threshold (on run 4's snapshots, the absent rate
  jumped between neighbours by 0.44 of its trend's range, the log-odds by 0.08); reports from before were backfilled
  from their scores.jsonl (`context-check/backfill.py`). The same investigation ruled out, cheaply: a last-layer refit on
  held-out data (upweighting context negatives cut false positives and recall 1:1, as at training time), scoring tags from
  their best local cells at inference (no fewer absent calls; hat AP 0.53 → 0.37), cleaning labels by tag count, and the
  `neg` erasure arm (its lead on the user's set didn't hold on the added images). Two identical 6K controls differ by 4
  points on it, so 6K-step arms can't resolve this. The agent added `--erasure-loss negative-only` (off by default, untested
  beyond 113 steps).
- **A stricter threshold where context makes a tag likely (2026-10-30, `context-margin/`, outside git): not viable.** Run 4
  at 1.316M, 100K held-out images, 35 object tags, the prior from the characters/artists/series the model itself calls
  (precision 0.63, recall 0.54 against the labels). Lowering an object tag's logit by 0.25 where its prior is 60-90%: absent
  objects called 10.5% → 7.4%, visible-but-untagged 71% → 66%, band recall 59% → 53%, macro AP 0.467 → 0.460, F1 down on
  every tag; larger margins trade the same way. An operating point, not a better model.
- **Glimpses (`--classification-glimpses`, `Model/Glimpses.cs`; run 5, `train-v5.sh`, from 2026-10-31).** Localizable tags
  are decided from close looks the model chooses, so "gloves" is decided from the gloves, not the character. A whole-image
  pass can't separate them: every map location, the localization map's included, sees most of a 512px image, so the averaged
  score, the map-cell score (classification pooling) and the last-layer refit all kept the context. A 64px glimpse of a
  window physically can't see the face.
  - The model's forward, the same in training, evaluation and inference: (1) the proposal head (`ImageTower.Proposals`, a
    D x D matrix on the pooled embedding, trained on every localizable tag from the labels) ranks what the image might hold,
    context included; (2) the localization map gives each suspect's peak; (3) the glimpse sensor looks there with one window
    per zoom (32/64/128/256px, centred on the peak; box filter to shrink, bilinear to enlarge; a learned zoom code each),
    48 windows per image after merging repeats (`Glimpses.Propose`, `Sensor`, `SameLookAs`), the zoomed-out grid filling
    any left over; (4) the localizable tags' scores are a log-sum-exp over those glimpses, so which zoom shows a tag best is
    learned. Whole-image tags (`localizable_tags.txt`'s exclusions) keep the whole image; characters are localizable, so they
    are glimpse-scored too (the docs said otherwise until 2026-11-01). Their AP matched run 4's at the same step (0.507 at 110K
    vs 0.500 at 103K), but few clear 0.5 (on gold, 0.8% of character pairs): calibration, which per-tag thresholds absorb. A wrong suspicion ("this character, therefore gloves")
    gets a close look and is trained as a "no".
  - Training: a labelled positive the choice didn't look at closely (`Glimpses.Frames`: a window other than the 256px one
    at its own peak) gets no glimpse loss (`--glimpse-unframed-positive-weight 0`): the classifier never saw it, and training
    it "yes" would teach it from whatever the windows showed. The proposal and localization losses teach the model to find
    it. While a fresh run's map can't point, a share of the windows (half at step 0, none from `--glimpse-label-framing-steps`)
    looks at labelled positives by the same sensor rule. The localization loss's mined negatives are ranked by the
    classifier's actual scores (they used to be ranked by the whole-image embedding, which a glimpse tower never trains for
    localizable tags).
  - The choice was measured against the gold boxes on run 4's map (`glimpse-frames`), not guessed. Ranking by the map put
    6% of gold tags in an image's top 32 (it is trained on ~100 pairs per image, so its scores for the rest aren't
    comparable); the whole-image classifier 68%. Choosing one zoom per tag from the map's response blob matched the right
    zoom at best 41% of the time (no blob rule worked), hence the sensor. 16/32/48/64 windows frame 55/75/87/91% of gold
    tags.
  - **Run 5 replaces a first glimpse run (deleted 2026-10-31).** It let the labels pick its training windows (half framing
    the positives' peaks, the rest random) and scored a fixed 378-window grid at evaluation (~6x a whole-image pass, ~30 min
    on the A4000; bf16 didn't speed it up). The model never learned where to look, a wrong tag's peak was never looked at
    closely, and with 16 windows in training and 378 at evaluation its scores' scale differed between the two.
  - B70 speed: the glimpse batch is padded to a fixed size (`Glimpses.Embed`'s `padTo`; every new batch size compiled new
    oneDNN kernels, ~2.5s each, more sizes than the cache holds: 426 -> 237 ms/step), and `ConvNeXtBlock` runs its MLP's 1x1
    convolutions as one matrix multiply over every pixel's channels (`MixedPrecision.Rows`; oneDNN's 1x1 weight gradient ran
    ~7x the forward, and a per-image batched multiply wasted the device on 480 glimpses with 2x2-16x16 maps). Glimpses take
    only the pooled embedding (`ImageTower.ForwardPooledOnly`: each level's masked sum is taken over its input features and
    projected after, exact since the projections are 1x1, so no embedding-wide maps or localization chain are built), and all
    480 of a step run in one pass. B70, forward + backward: glimpse pass 91.7 -> 63.8 ms, the 10-image 512px pass 127 -> 109 ms
    (2026-10-31, run 5 at step 4,331). Nothing in a step reads the device back: the window choice, crops, framing check and the
    localization and mosaic pairs' mined negatives run on the device, and the dashboard's losses are read every 25 steps (each
    read made the device finish the step and idle while the CPU set up the next). Uploads go from pinned memory without
    blocking (`DeviceUpload`): on the B70 a copy from ordinary memory waits for the whole queue (147 ms behind 146 ms of
    queued work). The pinned staging copies are held until the device has finished (released after the every-25-step read
    back): freed as soon as the copy was queued, the copy engine faulted after ~25 minutes of training (xe "Engine memory CAT
    error, class=bcs", engine reset, trainer hung in backward()), though a stress test of exactly that had passed. 247 -> 237 -> 228 ms/step. The tower runs channels-last on the device (`ImageTower.ChannelsLast`): bf16
    convolutions on the B70 return channels-last, and group norm put each block back in NCHW for its MLP to copy out again;
    with one group it equals a layer norm over (C, H, W), which `MixedPrecision.Norm` now runs on the NHWC view (same values
    and gradients, tested). 228 -> 205 ms/step. A tanh GELU was no faster (elementwise ops are bound by memory traffic). The copy engine itself turned out to be the fault: the same bcs CAT error, at nearly the same GPU address,
    had hit runs 3 and 4 (2026-10-24, 10-27) and two A/B arms on 2026-11-01 (one 16 minutes in, one 26), with or without the
    pinned staging copies, so holding them wasn't the fix (the same signature is reported on other B70s: intel/torch-xpu-ops#5867,
    vllm#61300; compute runtime 26.05.37020 here, a vLLM user saw none after 26.35). The trainer kept the bcs about as busy as the
    compute engine (`drm-cycles-bcs` in /proc/PID/fdinfo). The NEO key `OverrideCopyOffloadMode=0` (train-v5.sh) keeps every copy
    on the compute engine: no bcs cycles at all, the same upload timings and 204 ms/step. `UR_L0_USE_COPY_ENGINE=0` and
    `ForceCopyOperationOffloadForComputeCmdList=0` don't stop it. Checkpoints are written to a staging directory and moved in
    once complete (`StagedSave`), so a fault mid-save leaves the previous one whole; `xpu-test/watchdog-v5.sh` restarts the run
    in place after a fault, a hang or a crash. Batch sampling isn't seeded, so per-run statistics such as the unframed share
    differ by a few points between identical runs. `benchmark-tower --glimpses N --vary` times the glimpse path. Not
    exported to ONNX; `export-onnx --ignore-pooling` exports the boxes (from the localization map) and `render-gold --scores`
    takes the evaluation's glimpse-scored gold pairs. `evaluate --glimpse-grid` scores the old fixed grid instead.
- **Faster, content-pooled gold checks (2026-10-29).** The gold checks had grown slow: a snapshot evaluation spent ~10 of
  its ~14.5 minutes on 415 gold images. Every pass copied the full fp32 feature maps (about 33 MB per image) to the CPU and
  averaged them there, and each heatmap re-copied the map per tag. The map-scored occlusion check also re-ran the whole
  check one pair at a time, though nothing charts it. Scores now come from `ImageTower.ForwardPooled`, 32 images per pass,
  heatmaps are computed on the device, the map-scored check is off unless `--map-score-temperature` is given, and
  `evaluate` prints each phase's time. The held-out embedding uses `ForwardPooled` too. On the 850K checkpoint every
  reported number was unchanged, and the whole evaluation took 362s instead of 839s. Now: held-out embedding 216s (about
  93 images/s, compute-bound in fp32), rankings 37s, gold 55s, gold-small 28s.
  The gold checks also pooled scores over the whole canvas, letterbox bars included, while training and the held-out
  evaluation pool content only; they now pool content only. The old snapshots' weights are gone, so the checkpoints that
  kept theirs were re-scored to bridge the trend (`review/content-pool/`). Gold object effect, whole canvas → content only:
  485K 0.33 → 0.49, 550K 0.41 → 0.52, 850K 0.59 → 0.76 (272K 0.21 and 627K 0.60 content only); gold-small 0.15 → 0.24,
  0.25 → 0.34, 0.38 → 0.50. Same order, about 0.15 higher: run 4's charted series steps up where the switch happened
  (snapshots after 22:30 on 2026-10-29). Gold pointing and the crop teacher's crops don't depend on it.
- **Glimpse funnel (2026-11-01, `LocalizationReport.GlimpseFunnel`, in place of the crop teacher in run 5's snapshot
  evaluations).** Per labelled localizable gold pair, by size: is the tag among the proposal head's top 48 (suspected); does a
  close window (<=128px) sit on its own map peak (what training's "framed" share counts); does a close window chosen for any
  tag cover its gold box (`Covers`: overlap >= half the smaller); is the tag's highest-scoring glimpse on the object, against
  the share of glimpses that are (chance); which zoom that glimpse is; and the called rate with and without a covering window.
  Run 5 at ~118K, medium objects: suspected 59%, covered 82%, best glimpse on the object 28% (chance 13%), called 36% when
  covered vs 17% when not; small 31% (chance 13%); large 32% (chance 28%). The windows find the objects; the classifier's
  strongest evidence is mostly elsewhere. Its "called" matches the occlusion dump's scores exactly.
- **Locality negatives (`--locality-negative-weight`, `LocalityNegatives`, 2026-11-01; not adopted yet).** The user's
  idea: some tags can't share a close look (glasses and shoes; gloves and a face can), so a close window where one labelled tag
  is confidently found trains each other labelled tag "no" there, weighted by how rarely the two are together. That table is
  learned as training runs (each pair's best close windows within 64px, weighted by confidence, decaying), optionally seeded
  from a trained map (`evaluate --locality-prior-out`: run 4's peaks over 20K training images, 385,691 pairs; shoes/glasses 1.5%
  together, shoes/hat 2.8%, gloves/hat 7%, hat/glasses 23%, shoes/boots 61%). A/B on the B70, 6,000 steps from run 5 at
  189,803, ctl / online / online + run 4 prior, weight 1:
  - Best glimpse on the object (funnel, chance ~13% medium/small, 28% large): medium 31.9% / 60.6% / 59.5%, small 28.5% /
    59.2% / 68.9%, large 38.6% / 58.1% / 56.7%, gold-small 34.9% / 63.4% / 60.2%. It does what it was meant to.
  - But mAP 0.483 / 0.412 / 0.370, micro precision 0.173 / 0.086 / 0.069 (recall 0.373 / 0.420 / 0.452), absent objects
    called on the labelled context set 18% / 57% / 75%, object effect 0.130 / 0.097 / 0.045 (gold-small 0.082 / 0.00 / -0.02).
    The prior (more pairs known from the start, so more negatives) was worse: a dose effect. A negative that only says where
    a tag isn't is cheapest to satisfy by firing everywhere else, and the term sums over every pair of an image's labelled
    tags, so busy images got the most.
  - Dose arms (`--locality-per-pair`: each image's term the mean over its pairs; `--locality-confidence-gate 0.9`: only
    windows where A is called at p >= 0.9 give negatives; same start and control). Weight 0.3: best glimpse on the object
    medium 54.6%, small 64.5%, large 54.8%, gold-small 56.5%, most of the weight-1 gain; mAP 0.475 (-0.008), micro precision
    0.155 / recall 0.380, absent objects called 26% (controls of one recipe differ by ~4 pts there), object effect 0.091 /
    gold-small 0.043, control drop 0.009 (the control's 0.013). Finding the object better didn't yet make the call depend on
    it more. Weight 1 (the rerun; the first try died to an xe "VM worker error: -12" with the host at 5 GB free): best on
    object about the same (medium 57.5%, small 64.0%, gold-small 61.4%), every cost larger: mAP 0.453, precision 0.128,
    absent called 37%, exclusive violations 17.0%, object effect 0.072. The gain is full by 0.3; the costs grow with the dose.
    Weight 0.1: nearly free and still a real gain: best on object medium 40.7%, small 47.8%, large 51.4%, gold-small 45.3%;
    mAP 0.481, precision 0.168, absent called 19%, exclusive violations 13.5%, object effect 0.116 / 0.072 (all within the
    noise of two identical controls). Weight 0.3 with `--locality-detach-scaling` (no gradient into the shared logit scale
    and bias) matched plain 0.3 (mAP 0.474, precision 0.151, absent called 26%), so the cost isn't the shared bias. In no arm
    did the object effect rise: the model looks at the object more, but its call doesn't depend on it more yet. Candidate:
    weight 0.1, per pair, gate 0.9; in run 5 from step ~221K (20:10, `train-v5.sh`). The term's per-image tag rows must be padded to
    fixed widths (`PaddedRows.Upload`'s `padTo`, 32): unpadded, their count changed nearly every step, VRAM grew ~2 GB a minute
    and the trainer died after 14-18 minutes with UR_RESULT_ERROR_OUT_OF_RESOURCES / xe "VM worker error: -12" (pp1's first
    try and run 5 at 21:20). Padded, VRAM held at 23.4 GB for the first 11 minutes, but the trainer died the same way after 47
    (22:11), so the term is off in run 5 again from ~22:15 (step 256K) until the remaining growth is found (its masked_select /
    index_put_ table updates also change size every step). At 250K, with ~12K steps of it: best glimpse on the object, medium,
    59.5% (25-31% over the 100K steps before); glimpse-scored non-character AP still 0.039 behind run 4's. `review/locality/compare-arms.py` prints the arms side by side.
  - Correction (2026-11-01 23:30): the best-on-object gain is an artifact. The term only penalizes close windows (32-128px),
    so the model moved its evidence into the 256px window, a quarter of the image with the context in it: the winning zoom for
    small/medium objects was 256px in 5-12% of pairs in the control, ~50% at 0.1, 90%+ at 0.3, all at weight 1, and in run 5
    itself 10% (230K) -> 75% (250K). The funnel's chance level is over all windows, not the winning zoom, so bigger windows
    read as "on the object". Not adopted.
- **Glimpse pooling investigation (2026-11-01/02, `review/pooling/PLAN.md`; run 5 stopped at 277,022).** The user saw run 5's
  charts track run 4's. Per-window gold dumps (`evaluate --dump-occlusion` also writes `glimpse_pairs*.jsonl` and
  `glimpse_embeddings*.jsonl`; `review/glimpse-dump/`, `review/pooling/compare.py`) showed every glimpse learning the image's
  label: on medium objects, window logits on and off the object differ by +0.6, 31% of windows that don't touch the object fire
  for it, ~26 of 48 windows carry each score, and the best window alone would call 98% of labelled gold pairs vs 41% pooled. The
  log-sum-exp temperature (0.5) is in dot units, before the learned scale (~8.6), so in logit units it is ~4.3, nearly a mean.
  Judged on within-pair on/off-object window AUC at a fixed zoom (64/128px), not mixed-size gaps or the funnel. Arms, 6K or 12K
  steps from run 5 at 189,803, control ctl12 (12K: mAP 0.490, absent objects called 13%, medium 64px AUC 0.66); none adopted:
  - Sharper pooling (`--classification-pooling-temperature` 0.12/0.06/0.03, `--classification-pooling-top-k` 1/3, WILDCAT
    `--classification-pooling-bottom-weight`, `--glimpse-pool-zooms 3`, `--glimpse-pool-min-zoom`): monotone trade. T 0.5 / 0.12 /
    0.06 / 0.03: mAP 0.483 / 0.484 / 0.469 / 0.446, off-object windows firing 31 / 24 / 19 / 9%, window gap never wider; hard max
    mAP 0.409. A window the object fills scores at chance (AUC 0.51-0.58) in every arm. 0.12 is the only free setting.
  - Label framing kept on (`--glimpse-label-framing-floor 8`, `--glimpse-framed-positive-pooling`): null; the map's peak frames
    a medium object ~1 time in 12, so framed positives mostly trained context windows "yes".
  - Proposal loss kept off the tag rows (`--glimpse-proposal-detach-tags`): null.
  - A separate glimpse tower (`--separate-glimpse-tower`, `ImageTower.GlimpseTower`, `--glimpse-tower-lr-scale`): at T 0.5 a
    small probe gain (+0.04) for -1.3 mAP; under sharp pooling it learned the image's identity instead (same-image nearest
    neighbour 32% -> 45%): the cheapest window feature for an image label is the character/palette/style every crop shows.
  - Same-source pairs (`--context-pairs-per-batch`, `--context-pair-weight/-margin`: one image of a character with the tag, one
    without, a hinge on their pooled glimpse logits): weight 1 collapsed every window (mAP 0.315); 0.1 cost 5 mAP points.
  - Singh et al. 2020's feature split (`--glimpse-feature-split*`, `--glimpse-exclusive-per-batch`, `evaluate
    --glimpse-object-only`), exclusive = a known character/artist/series with prior < 0.2, limited to the 1,980 tags with context
    sources: mAP 0.477 / 0.471, absent called 18 / 20%; the object half alone can't call the tags (mAP 0.21-0.25).
  - The ceiling test (`window-probe`, `WindowProbeCommandHandler`, `review/window-probe/`): ctl12's glimpse path fine-tuned on
    gold-box windows (yes = holds half a box or is half filled by one, no = touches none; rows and scale frozen; 5 folds by image
    over 402 gold images). Medium 64px on/off AUC 0.66 -> 0.80, also against windows on another labelled object (0.65 -> 0.80).
    The sensor isn't the floor: run 5's losses never give a window a within-image "no". Training mustn't depend on the hand-made
    gold set (the user), so the next arms derived window labels from tags:
  - Mosaic window labels (`--mosaic-glimpse-weight`, `Mosaic.GlimpseTerm`: glimpses on the mosaic canvases, a tag's windows on
    other tiles "no", its own tile a "yes" bag): mAP 0.432, calls up everywhere; the "no" windows show other images, which
    ordinary negatives already cover.
  - Matched twins (the user's idea; `--twin-every`, `TagCentricBatchSampler.SampleTwins`, `TwinWindows`): every Nth batch is one
    image with a localizable general tag and 9 without it whose tags overlap it most (Jaccard, at most one character each); the
    half of the positive's windows most like 3 twins' windows (cosine, excluding the one holding the tag's map peak) are "no".
    Weight 1 (tw2): mAP 0.442; the batches alone (tw0, weight 0) 0.482; weight 0.2 0.477 (no gain); weight 1 with
    `--twin-detach-scaling` 0.438 (so not the shared bias). `evaluate --twin-check` (`TwinCheck`) scores the rule against gold:
    its "no" windows hold the object 12% of the time vs a 16% base rate, too weak a label.
  - Pasted twins (`--twin-paste`, `TwinPaste`): the positive's 64/96/128px region at its tag's map peak is pasted (feathered) into
    2 twins, which then carry the tag, and a region from elsewhere into 2 others as certain negatives, so a paste cue can't pay;
    only when the peak's own logit is >= 0. Run 5's map can't place it (gold-small peak on the object 8.5% vs run 4's 55%), so it
    ran on run 4 (`review/run4-twins/`, 12K from 1,495,198): control / twin batches / twins + pastes (11,993 pastes) mAP
    0.788 / 0.787 / 0.787, absent called 10 / 9 / 10%, object effect 1.14 / 1.16 / 1.10, pointing 67.8 / 66.9 / 65.8%: noise.
  Built, no A/B result recorded: `--negative-loss hill` (Hill loss for unverified negatives) and `--hard-negative-source-weight`.
  Run 5's "ideal" for an object tag (gap >= 2 logits, < 10% off-object firing, few windows carrying the score) wasn't reached by
  any label-only change; window-level supervision is what moved it.
- **Gold box metrics fixed (2026-11-01).** The gold checks extracted boxes with none of `detect`'s speck filtering, so on a
  128x128 map every above-cutoff speck was a box: ~150-190 per pair, box precision 0.1-0.3% and a count error of ~100-190 in
  every run, run 4 at 1.49M included. They now use `detect`'s defaults (`TagScorer.Default*`: regions of at least 0.2% of the
  map, at most 3 boxes per tag), still without its edge-guided refinement, which needs the source image. Run 5 at ~105K: count
  error 176.7 -> 1.7, box precision 0.2% -> 7.9% (characters 20.6%), recall 21.5% -> 15.5% (the specks had hit some boxes by
  chance); pointing and energy unchanged. Box numbers from before aren't comparable. Run 5's snapshot evaluations also pass
  `--no-crop-teacher` (the crop teacher only matters to crop consistency, which run 5 doesn't use; ~40% of its gold checks)
  and `--precision tf32` (held-out embedding -27%, gold -24%, mAP equal to 4 decimals).
- **TorchSharp memory and determinism.** TorchSharp frees a tensor when its dispose scope
  ends, not when it goes out of use, so `ConvNeXtBlock.forward` and the tower's forwards open
  their own scopes; before that, a forward held every intermediate until the caller's scope ended
  (gold scoring 13.2 GB on the A4000, now 4.9). Seeds used to come from `HashCode.Combine`, which
  .NET randomizes per process, so evaluating one checkpoint twice gave different numbers;
  `DeterministicSeed` (SplitMix64) replaces it. `manual_seed` doesn't seed the XPU generator.
- **Faster step and bf16 (run 4).** Profiling (`UNBOORU_PROFILE_STEPS`, `benchmark-tower`) put
  88% of a step in the tower's forward and backward, limited by memory bandwidth:
  elementwise ops run at about a fifth of the B70's bandwidth, and the matrix multiplies use
  a small fraction of its compute. That is why int8 wouldn't help. `ImageTower.ForwardPooled`
  pools the full map without building it (each fused level times the mask sum-pooled to its
  grid; exact for 2x nearest upsampling), about 2%. Running mosaic canvases in the batch's own
  forward was slower on the B70, so they keep a second pass. Channels-last made no difference.
  `--bf16` (`ImageTower.ComputeType`, `MixedPrecision`) runs the tower in bfloat16 from fp32
  weights cast per call, with fp32 outputs, gradients and checkpoints: 223 → 199 ms/step, and a
  6,000-step A/B kept mAP (0.576 vs 0.575) and gold pointing (32.3% vs 32.6%). Run 4 uses it from
  step 359,388. It isn't saved in the checkpoint.
  From step ~1,499,226 (2026-11-02) run 4 trains on the build with run 5's speed work (the blocks' 1x1 convolutions as one
  matrix multiply, channels-last with the norm on the NHWC view, pinned non-blocking uploads, no per-step device read-backs;
  see Glimpses' B70 speed): 221 -> 142 ms/step with the same recipe and checkpoint. A 12,000-step arm of it reached run 4's own
  mAP (0.788 vs 0.787 at 1.49M), and its checkpoints still load in the older evaluation build that run 4's trend charts use.

- **Small/medium pointing arms (2026-11-03, `research/experiments-localization.md`; none adopted).** 12K steps from run 4 at
  1,507,198, control ctl5 (mAP 0.789). `--context-negative-weight` 4/6/10: recall gap 27.0 -> 21.4 pts at 10, mAP 0.785,
  pointing within noise. `--fine-localization` (`Model/FineLocalization.cs`, a stride-2 level giving a 256x256 map; its loss
  is computed factored, `FineLocalizationMap` + `Mosaic.TileCrops`, so the dense map is never built, and it still fills the
  B70's 32 GB): gold small 43.4% vs 37.7%, but gold-small and gold-auto small fell. A 2x zoom of the gold images at evaluation
  (`evaluate --gold-images --gold-input-size`, Real-ESRGAN anime 6B or bicubic) raised small-object pointing 6-12 points on
  all three sets and cost medium/large 2-20: the object's pixels, not the grid, limit small objects. Next candidate:
  multi-scale training views. Run 4's epoch 4 (1,939,684): mAP 0.801, gold pointing 67.7%, gold-auto 65.9%.
- **Resolution and mixed-scale training (2026-11-04, `research/experiments-localization.md`).** The cache's 512px shrink is
  nearest-neighbour, but the gold originals showed the limit is resolution, not aliasing (a clean 512px shrink scored like the
  cache; real 768/1,024px pixels +10 to +16 points on small objects). Fine-tuning at 1,024 kept mAP, gained 15-20 points on small
  objects and lost up to 7 on large ones; 768 was balanced. Chosen rows are re-downloaded into side stores (`HiResImageStore`,
  core: JPEGs at a long side, written by `review/hires/pull*.py` outside git), and training draws each micro-batch at one of
  several sizes with its own batch (`--train-scales size:batch:share[:stored]`, `TrainingScales`; `--hires-store a,b` reads rows
  from the smallest store reaching the size, refreshed every 5 minutes while a pull writes). Real pixels raised a tag's AP in
  proportion to the store's coverage of it (wristband 0.47 -> 0.61 at 53%); a model trained on enlarged images alone gained
  nothing from real pixels at test time. Run 4 trains this way from ~2.09M (`512:10:0.3,768:5:0.3:0.2,1024:3:0.3:0.3,2048:1:0.1:1`;
  2,048 only on stored rows). `ScaleStats` logs what each size did; the snapshot evaluations add the gold checks at 1,024
  (`HigherScalePoint`, four chart panels; every 20,000 steps, since at every snapshot the A4000's queue fell behind). The plan: gather resolution where evaluations show weakness, by tag and by kind of image.

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
  - **From 2026-11-04 the cut is in log-odds, 4 below the tag's own peak (`--box-below-peak`, `TagScorer.DefaultBoxBelowPeak`).**
    Drawn as regions over the gold images (`render-gold --heatmaps`, `gold/compare.html` outside git, docs/EXAMPLES.md), the maps
    mostly peak on their objects, but the probability-range cut kept about a fifth of the image for every tag. On run 4 at 2.12M:
    box precision 0.08 -> 0.19, recall 0.12 -> 0.25, small objects 0 -> 0.11 / 0.14 (minimum region 0.2% -> 0.05% of the map).
    The gold evaluation also reports region precision/recall at that cut (`GoldResult.RegionPrecision`), charted by size.
    A tag's further regions are boxed only when their own peak is within 1 log-odds of its peak (`--box-peak-margin`,
    `TagScorer.DefaultBoxPeakMargin`): the band took in faint blobs the drawn map shows as clear. Box precision 0.19 -> 0.27, recall
    0.25 -> 0.22; no cut, smoothing or trim did much better, since a map's extent isn't its object's.
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
