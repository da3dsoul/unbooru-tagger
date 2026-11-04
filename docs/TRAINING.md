# Training unbooru-tagger

This document covers building a dataset, training, evaluating and exporting a model. For running a trained model,
see the [README](../README.md). For the research behind the options below (what was tried and why the defaults
are what they are), see [research/](../research/README.md).

## Contents

- [Hardware](#hardware)
- [Building the training CLI](#building-the-training-cli)
  - [NVIDIA / CUDA](#nvidia--cuda)
  - [Intel Arc / XPU](#intel-arc--xpu)
  - [CPU only](#cpu-only)
- [Building a dataset](#building-a-dataset)
- [Training](#training)
- [Run 4's recipe](#run-4s-recipe)
- [Evaluation](#evaluation)
- [The dashboard and charts](#the-dashboard-and-charts)
- [Offloaded and snapshot evaluations](#offloaded-and-snapshot-evaluations)
- [Keeping a long run alive](#keeping-a-long-run-alive)
- [Exporting to ONNX](#exporting-to-onnx)
- [Adding a tag](#adding-a-tag)
- [Checkpoint directory contents](#checkpoint-directory-contents)
- [Experimental options](#experimental-options-off-by-default)
- [Measuring speed](#measuring-speed)
- [Tests](#tests)

## Hardware

These are the machines the project has actually trained and evaluated on. **No minimum configuration has been
tested.** The "minimum" column says what is known to be too small, or leaves it open.

| | Known too small / minimum | What the project uses (recommended) |
|---|---|---|
| **GPU for training** | 16 GB (RTX A4000) is too small for run 4's recipe (batch 10, mosaics, localization pairs) without lowering the batch size. It ran run 2's ConvNeXt-T fused-2 tower at batch 10 x 2 accumulation; batch 20 ran out of memory. 24 GB (RTX 4090) is untested: run 4 peaked at 25.3 GB before the localization-pair cap (build 39) and holds about 21.5 GB with it, so it might just fit. | A 32 GB card: the Intel Arc Pro B70 the project trains on, or an RTX 5090. Run 3 (fused-3) holds 17.9 GiB at batch 10. fused-2 fits about batch 32 (batch 30 = 27.9 GiB). `--fine-localization` (a 256x256 map) at run 4's recipe reaches the B70's full 32 GB. |
| **GPU for evaluation** | 16 GB (A4000) works with `--batch-size 32`, after the 2026-10-28 memory fix. Measured live, an evaluation uses about 1.5 GB median and 3.3-4.3 GB peak. | A second GPU, so evaluations run beside training (see [offloaded evaluations](#offloaded-and-snapshot-evaluations)). |
| **Host RAM** | Not recorded. The trainer slowed to about 3 s/step when the host ran out of memory (swap full). | Enough to keep free memory with training and evaluations running. An A4000 evaluation process holds about 13.5-22 GB RSS (run 4's tower / run 5's glimpse checkpoints). |
| **Disk** | The current corpus: `images.bin` 3.10 TB (about 5.39M images at 512px, about 576 KB average), `crawl.sqlite` 8.5 GB, `tag_rows.sqlite` 0.9 GB. A checkpoint holds the tower, embeddings and optimizer state. | A striped array read with high queue depth. On the 6-disk RAID5, random reads gave about 85 img/s with 1 in flight and about 590 img/s with 32; the loader reaches about 450-590 img/s, and a step waits about 1 ms on data. |
| **CPU** | Not recorded. | Decoding runs in parallel in the loader. |

**Speed for reference** (one training step at batch 10 x 2 accumulation, 512px, ConvNeXt-T):

| Configuration | ms/step |
|---|---|
| Run 2 (fused-2) on the A4000 | 278 |
| Run 2 (fused-2) on the B70 | 158 |
| Run 3 (fused-3) on the B70 | 172 |
| Run 4 (fused-3, 2 mosaics, bf16) on the B70, current build | about 142 |

Epochs take days; run 2 was about 37 hours per epoch on the A4000.

## Building the training CLI

Requires the **.NET 10 SDK** ([`global.json`](../global.json) allows a newer major and prereleases).
`unbooru-tagger-training` doesn't need the sibling `unbooru` repository; only `unbooru-tagger-data-unbooru-import`
does.

### NVIDIA / CUDA

```sh
dotnet build unbooru-tagger-training -c Release
```

- On Linux the project references `TorchSharp-cuda-linux` 0.107.0, which bundles its own CUDA 12.8 runtime. Only a
  compatible NVIDIA driver is needed.
- On other operating systems it references `TorchSharp-cpu`, so training on Windows is CPU-only.
- `DeviceSelector` uses CUDA when it is available. cuDNN's TF32 convolutions are on by default.

### Intel Arc / XPU

Training runs on Intel Arc GPUs; it was tested on an Arc Pro B70. Stock TorchSharp 0.107.0 works against the
libtorch shipped in PyTorch's `torch==2.10.0+xpu` wheel: 2.10.0 is the libtorch TorchSharp 0.107 is built
against, and the C++ ABI matches.

**1. Get the native libraries.** Download the `torch==2.10.0+xpu` wheel and its Intel runtime dependencies from
`https://download.pytorch.org/whl/xpu`. Unzip them, and copy every `torch/lib/*.so*` and `*.data/data/lib/*.so*`
into one folder. No oneAPI install is needed. The system needs the `xe` kernel driver, `libze-intel-gpu1` and
`libze1`.

**2. Build** without a bundled libtorch, in its own output folder, so an IDE or CUDA build never replaces the
binary that a resume runs:

```sh
dotnet build unbooru-tagger-training -c Release -p:TorchBackend=xpu --artifacts-path <xpu-build-dir>
```

**3. Run** with that folder on `LD_LIBRARY_PATH` and these variables set:

```sh
# The Intel runtime must not silently migrate VRAM allocations into host RAM. Over-allocation must fail
# (UR_RESULT_ERROR_OUT_OF_RESOURCES) instead of thrashing over PCIe.
export NEOReadDebugKeys=1
export EnableSharedSystemUsmSupport=0
export EnableImplicitMigrationOnFaultableHardware=0
# Run host<->device copies on the compute engine, not the copy engine (bcs).
export OverrideCopyOffloadMode=0
export LD_LIBRARY_PATH=<folder with the wheel's .so files>
```

**What the XPU path does, and what to watch for:**

- **Device.** TorchSharp's `DeviceType` enum has no XPU, so the device is `(DeviceType)12`. `DeviceSelector`
  picks CUDA, then XPU (by probing an allocation), then the CPU.
- **TF32.** Only libtorch's legacy `at::Context::setAllowTF32OneDNN(true)` reaches the XPU convolutions.
  `OneDnnPrecision` calls the exported symbol, because TorchSharp exposes only the CUDA switches. The newer
  `setFloat32Precision` reads back as set but has no effect. TF32 took the tower from 203 to 138 ms/step.
- **Out of memory** aborts the process rather than throwing a .NET exception. Resume from the last checkpoint.
- **Copy-engine faults.**
  - The symptom: the kernel log shows xe "Engine memory CAT error, class=bcs", and the trainer hangs, often in
    `backward()` or in the next process's `Checkpoint.Load`. On this machine (compute runtime 26.05.37020) it
    recurred on several days until `OverrideCopyOffloadMode=0`. The same signature is reported on other B70s
    (intel/torch-xpu-ops#5867, vllm#61300).
  - `UR_L0_USE_COPY_ENGINE=0` does **not** stop copy-engine use.
  - To check: `drm-cycles-bcs` in `/proc/<pid>/fdinfo` should stay 0.
- **Keep tensor shapes fixed.** Every new input shape compiles new oneDNN kernels (about 2.5 s for this tower),
  and the cache holds only a few tower shapes. A dimension that varies step to step makes steps slow at random.
  The code pads variable-size inputs to fixed widths (glimpse batches, localization pair rows to multiples of 64)
  for this reason. Keep that in any new loss: pad, don't trim.
- **Varying-size allocations ratchet VRAM.** If VRAM keeps growing (for example 12 → 24 GB in 10 minutes), the
  trainer will die with `UR_RESULT_ERROR_OUT_OF_RESOURCES` and xe "VM worker error: -12" in the kernel log. Look
  for tensors whose size changes every step (`masked_select`, `index_put_` with changing counts) and give them
  fixed shapes. Check VRAM with `drm-resident-vram0` in `/proc/<pid>/fdinfo`.
- **Uploads.** An upload from ordinary host memory blocks until the whole device queue drains, and so does a
  read-back. The trainer uploads from pinned memory without blocking, and reads losses back only every few steps.
- **Stopping.** Stop with **Ctrl+C**: the trainer finishes the current step, saves, and exits. A second Ctrl+C
  quits at once.
  - Killing it mid-step left a copy-engine fault, and the next load hung for hours.
  - After a stop, wait a little before starting a new trainer; one started seconds later can fail with
    `UR_RESULT_ERROR_OUT_OF_RESOURCES`.
- **Host memory.** Keep free RAM. Under host memory pressure the B70 went mostly idle at about 3 s/step until a
  graceful restart. `/tmp` on many Linux systems is RAM-backed (tmpfs), so don't put scratch builds there.
- **Clock speed.** The B70 is power-limited, with clocks between about 2.2 and 2.8 GHz. Compare speeds by
  alternating the variants back to back.
- **Two GPU backends can't share a process.** A CUDA process started from the XPU environment inherits
  `LD_LIBRARY_PATH`, loads the XPU libtorch, and fails with "Cannot have both XPU and cuda". Every CUDA launcher
  started from an XPU script must `unset LD_LIBRARY_PATH`.

### CPU only

Without CUDA or XPU, TorchSharp runs on the CPU (`TorchSharp-cpu` on non-Linux builds; a CUDA build falls back to
the CPU when no GPU is found). No CPU training speed is recorded. At this model size and corpus, CPU training is
not practical; it is useful for tests and for small manifests.

## Building a dataset

Training reads a **preprocessed cache directory**:

- `images.bin`: uint8 pixels, content region only; non-square images are letterboxed at load time;
- `tag_rows.sqlite`;
- `tag_vocabulary.json`;
- optionally `ratings.bin`, `image_sites.bin`, `tag_implications.json` and `tag_agreement.json`.

There are two ways to build one.

### From Danbooru and Gelbooru (`unbooru-tagger-data-booru-downloader`)

Needs network access only. Gelbooru requires `--gelbooru-api-key` and `--gelbooru-user-id`, even for read-only
listings. `--danbooru-login` and `--danbooru-api-key` raise Danbooru's rate-limit tier.

**1. Survey tags.** This records each tag's per-site post count and eligibility (`--min-images`, default 500). It
also fetches Danbooru's tag-alias table, and fails without changes if that fetch fails.

```sh
dotnet run --project unbooru-tagger-data-booru-downloader -- survey-tags \
  --output-dir ./data/crawled --min-images 500 \
  --gelbooru-api-key <key> --gelbooru-user-id <user-id>
```

**2. Crawl.** Up to `--max-images` (default 1000) images per eligible tag, rarest tag first, then a negative
top-up per tag:

```sh
dotnet run --project unbooru-tagger-data-booru-downloader -- crawl \
  --output-dir ./data/crawled --min-images 500 --max-images 1000 --input-size 512 \
  --gelbooru-api-key <key> --gelbooru-user-id <user-id>
# later:
dotnet run --project unbooru-tagger-data-booru-downloader -- crawl --output-dir ./data/crawled --resume
```

- **Saved options.** The options are saved to `last-crawl-command.json`, and `--resume` reuses them. That file
  holds your API keys; keep it private.
- **Per-site workers.** Each site runs its own worker at its own rate (`--rate-danbooru` 4, `--rate-gelbooru` 2
  requests/s), and must contribute its own share of each tag.
- **Dedup.** Images are deduplicated by MD5 and by a 64-bit perceptual hash, so a cross-posted re-encode becomes
  an extra source of the existing row.
- **Negatives.**
  - `--negative-target` defaults to 2 x `--min-images`.
  - The negative phase prefers hard negatives from the corpus's own co-occurrence counts (`--negative-cooccurrence-ratio`
    0.5, `--negative-cooccurrence-min-examples` 15, `--max-hard-negative-sources` 3; set the last to 0 for plain
    `-tag` queries).
- **Resilience.** Transient failures retry with backoff. If a site's retries are exhausted, it waits 20 minutes and
  retries, indefinitely, logging to `crawl-errors.log`. Progress is checkpointed durably every page.
- **Tag identities** are category-prefixed: `character:`, `series:`, `artist:`, `meta:`; general tags are bare.
  - Every `meta:` tag is excluded except `*_(medium)` tags and a short curated list.
  - Edit `excluded_tags.txt` to change that: a bare line excludes a tag; a `!` line includes it.

**3. Refresh tags** (optional): re-fetches crawled posts by id, to pick up tag edits. Unlike the crawl, it can
remove tags. `--only-tags` scopes it; `--reset` restarts a site's sweep.

**4. Export the label files training uses.** Run each once, and again after anything that changes the cache's
rows:

| Command | Writes | Used for |
|---|---|---|
| `export-ratings --output-dir <dataset>` | `ratings.bin` | Content ratings: the strictest rating across an image's sources. |
| `export-implications --output-dir <dataset>` | `tag_implications.json` | Danbooru's implications (witch_hat → hat), filled in for Gelbooru posts. On run 2 this added 523,867 labels. |
| `export-image-sites --output-dir <dataset>` | `image_sites.bin` | Which sites each image came from. |
| `export-tag-agreement --output-dir <dataset>` | `tag_agreement.json` | Per tag, how often both sites list it. Together with the site file, it marks tags only one site's tag system has, so their absence on the other site's images isn't trained or evaluated as a negative. |

**Maintenance and recovery commands.** Most default to a dry run and write only with `--apply`.
`recover-corrupted-images` keeps the dataset read-only until `--finalize`, and `shrink-cache` converts in place.

| Command | Purpose |
|---|---|
| `shrink-cache` | Converts an old float32/full-canvas `images.bin` to uint8/content-only, in place, resumably. |
| `prune-aliased-tags` | Removes surveyed tags that a site has since aliased away. |
| `detect-deprecated-redirects` | Finds tags a site silently redirects, by exact post-id matches, and merges them into the aliases. |
| `prune-unsupported-media` | Removes video and GIF images (SkiaSharp decodes a GIF's first frame silently). |
| `rebuild-vocabulary --seed-vocabulary <old vocab>` | Rebuilds a lost or corrupted `tag_vocabulary.json` and `tag_rows.sqlite` from `crawl.sqlite`, keeping known row indices. |
| `recover-corrupted-images --staging-dir <dir>` | Rebuilds a damaged `images.bin` in a staging directory, re-downloading and MD5-verifying what's missing. |

### From an `unbooru` database (`unbooru-tagger-data-unbooru-import`)

This needs a sibling checkout of the [`unbooru`](https://github.com/da3dsoul/unbooru) repository at `../unbooru`,
and a reachable `unbooru` SQL Server database.

```sh
# A small manifest: images with the given tags plus an equal sample without them (for quick runs, add-tag).
dotnet run --project unbooru-tagger-data-unbooru-import -- build-small-dataset \
  --tags hatsune_miku twintails --connection-string "Server=...;Database=unbooru;..." \
  --out ./data/miku-small --max-images 500

# The full (or a capped) corpus as a preprocessed cache; re-running resumes.
dotnet run --project unbooru-tagger-data-unbooru-import -- build-large-cache \
  --connection-string "Server=...;Database=unbooru;..." --out ./data/large-cache \
  --input-size 224 --min-tag-images 100
```

- `--min-tag-images` (default 100) is the corpus-wide count a tag needs to get a vocabulary row.
- `--min-images-per-tag` (default 15) reserves the rarest-first images per tag when `--max-images` caps the corpus.

## Training

```sh
dotnet run --project unbooru-tagger-training -c Release -- train \
  --cache-dir ./data/crawled --checkpoint-dir ./checkpoint
```

`--manifest <manifest.json>` trains from a small manifest instead of a cache.

**Resuming.** Re-running against the same `--checkpoint-dir` resumes everything:

- model weights and optimizer state;
- the epoch and the step within it;
- the validation split;
- early-stopping history and the loss chart.

A checkpoint is saved at every epoch end, and every `--checkpoint-interval-minutes` (default 15). Saves go to a
staging directory and are moved in once complete, so a crash mid-save leaves the previous checkpoint whole. A
resume matches the dataset's rows to the checkpoint's vocabulary by tag name.

### Basic options

| Option | Default | Notes |
|---|---|---|
| `--epochs` | 5 | Run 4 uses 6. Changing it mid-run reshapes the remaining LR decay. |
| `--batch-size` | 16 | Run 4: 10. |
| `--gradient-accumulation-steps` | 1 | Effective batch = batch x this. Run 4: 2. |
| `--lr`, `--lr-schedule`, `--warmup-steps` | 1e-4, `cosine`, 2000 | Linear warmup, then cosine decay to 5% over `--epochs`, computed from the global step. |
| `--embedding-dim` | 512 | |
| `--validation-fraction` | 0.1 | |
| `--validation-max-images` | 0 (all) | Run 4: 20000 (the same fixed subset every epoch). |
| `--validation-split-seed` | random | Reuse another run's `ValidationSplitSeed` (from its `training_progress.json`) to hold out the same images. Run 5 used run 4's. |
| `--early-stopping-patience` | 3 | Epochs without held-out mean AP improving. |
| `--max-steps` | 0 | Stop after N steps and save (for short A/B arms). |
| `--keep-full-vocabulary` | off | A fresh run otherwise drops vocabulary rows no image carries. |
| `--no-ratings` | off | Needed to resume a checkpoint from before ratings existed. |

### Tower size (fresh runs only)

A resume uses the checkpoint's `model_config.json`.

| Option | Default | Run 4 |
|---|---|---|
| `--stem-channels` | 64 | 96 |
| `--stage-channels` | `64,128,256` | `96,192,384,768` |
| `--blocks-per-stage` | `2,2,2` | `3,3,9,3` |
| `--fused-stages` | 0 | 3 |

- Each change of stage width downsamples 2x.
- Each fused stage halves the heatmap stride: 4 stages with 3 fused give 128x128 heatmaps at 512px.
- `--input-size` (default 224) applies only to manifests; a cache has its own size, 512 for the current corpus.

### Batches and targets

- **`--sampling tag-centric`** (default; `rare-tag` is the old sampler) builds each batch around anchor tags drawn
  uniformly over the vocabulary. Per anchor:
  - `--positives-per-anchor` (2) positives;
  - `--hard-negatives-per-anchor` (2) hard negatives;
  - `--context-negatives-per-anchor` (1) context negatives.
- **`--targets deliberate`** (default; `legacy` = every untagged tag is an equally certain negative) weights each
  (image, tag) pair:

  | Pair | Treatment |
  |---|---|
  | Positives; anchor hard negatives; tags exclusive with one the image carries | Full BCE (`--certain-negative-weight` 1). |
  | A tag on ≥ `--missing-threshold` (0.9) of the images of a tag the image carries | Likely missing: weight 0. |
  | A tag on between `--implied-threshold` (0.6) and 0.9 | Context negative: unverified. With `--context-negative-weight W`, full BCE at weight W for localizable tags. |
  | Other unverified negatives | Asymmetric loss: `--negative-clip` 0.05, `--negative-gamma` 4 (run 4: 2). |
  | Tags only one site's tag system has, on the other site's images | Weight 0. |

- **`--exclusive-tags <file>`:** mutually exclusive groups, one comma-separated group per line. The default is the
  checkpoint's saved copy, else built-in people-count and rating groups. The file is shipped with the model.
- **`--localizable-tags <file>`:** which tags reach the localization loss and get boxes, one rule per line:
  `tag`, `prefix*`, `*suffix`, `!exception`. Whole-image tags (artist, series, meta, rating, backgrounds,
  mediums, styles, lighting, framing, people counts) are excluded by default. The rules are shipped with the model.
- **`--tag-parentage <rules.json>`:** reviewed is-a, implies and excludes rules.
  - Kinds complete their parent: boots → shoes.
  - A parent with no kind named leaves the kinds unknown.
  - An excludes rule makes the parent and its kinds certain negatives.
  - The rules are saved into the checkpoint; `none` turns them off.
- **Content ratings:** with `ratings.bin`, the model learns three cumulative labels
  (`rating:at_least_sensitive`, `rating:at_least_questionable`, `rating:is_explicit`). Their cutoffs are
  calibrated for recall (90% / 97% / 97%).

### Localization

| Option | Default | Notes |
|---|---|---|
| `--localization-weight` | 0.1 | MIL / log-sum-exp loss over map locations. Run 4: 0.3. |
| `--localization-temperature` | 0.35 | Lower = more max-like. Run 4: 0.25. |
| `--localization-mined-negatives` | 32 | With deliberate targets, the loss trains only on localizable positives, certain negatives and this many top-scoring wrong tags per image. |
| `--localization-tag-chunk-size` | 4096 | For the dense (legacy) loss: processes the vocabulary in chunks; exact. |
| `--localization-map` | `full` on a fresh run | `fused`: the fused laterals only, without the last stage (run 4). `local`: also without the coarsest lateral (tried, not adopted). |
| `--mosaics-per-step` | 0 | Canvases of 4 batch images at half size; each tile's tags are negatives in the other tiles. Run 4: 2 (+28% step time on run 3). `--mosaic-weight` 1. |
| `--self-supervised-weight` | 0.1 | SimSiam consistency between two crops. Runs 2-5 use 0. |
| `--crop-consistency-weight` | 0 | Gated by `--crop-teacher-min-ap` 0.8, `--crop-teacher-min-positives` 20, `--crop-teacher-confidence` 0.9. Off in all runs. |
| `--localization-background-weight` | 0 | Tested; it shrank heatmaps without moving peaks. |
| `--gold <gold.json>` | none | Scores the gold set at every epoch evaluation. |

### Data loading

With `--cache-dir`, batches are sampled and read ahead on a background thread:

- `--prefetch-batches` (8) batches per window, read in one burst;
- images deduplicated and sorted by file offset;
- `--io-parallelism` (32) reads in flight, then decoded in parallel.

The dashboard shows how long each step waited on data. If that isn't near zero, raise either option.

### Mixed precision

`--bf16` runs the image tower in bfloat16; weights, gradients, the optimizer, checkpoints and the output maps stay
fp32. On the B70 it took a run 4 step from 223 to 199 ms, with mAP and gold pointing unchanged in a 6,000-step
A/B. **It isn't saved in the checkpoint: pass it on every resume.**

### Higher resolution and mixed scales

The cache stores every image at 512px, shrunk with nearest-neighbour sampling. Small objects lose their lines
there, and tests on the gold originals showed the limit is resolution, not aliasing: real 768/1,024px pixels raised
small-object pointing 10 to 16 points, a clean 512px shrink didn't
([research/experiments-localization.md](../research/experiments-localization.md)). Re-pulling every image is too
big, so chosen rows are re-downloaded into side stores.

**Hi-res stores** (`HiResImageStore`, `unbooru-tagger-core`): a directory holding `store.json`
(`{"LongSide": 1024}`), `images.bin` (JPEGs, concatenated) and `index.bin` (24-byte records: row, offset, length,
width, height). Each image is its letterbox content for a `LongSide` canvas, shrunk with Lanczos and kept as JPEG
(q92, 4:4:4; ~238 KB at 1,024). They're written by `review/hires/pull.py` / `next/pull2.py` (outside git), which
re-download originals by the crawl's file URLs, check the md5, and keep the site rate limits.

| Option | Purpose |
|---|---|
| `--train-input-size N` | Train at N px instead of the cache's size. Cached images are enlarged (Mitchell cubic). |
| `--hires-store a,b,...` | Stores to read rows from instead of the cache: the smallest that reaches the size, else the largest. Rows a pull appends while training runs are picked up every 5 minutes. |
| `--train-scales size:batch:share[:stored],...` | Mixed-scale training (below). |
| `--focus-tags a,b` | With `--focus-share`, draw that share of the anchors from these tags instead of the evaluation's worst. |
| `--lr-schedule constant` | A fixed `--lr`, which now also overrides the rate saved in the optimizer on a resume. |

**Mixed scales.** With `--train-scales 512:10:0.3,768:5:0.3:0.2,1024:3:0.3:0.3,2048:1:0.1:1`, each micro-batch
is drawn at one size, by share, with that size's batch:

- `stored` is the share of a batch drawn from rows a store holds at that size or larger; the rest are sampled as
  usual, and still come from a store when one holds them.
- A size with `stored` 1 trains only on real pixels, and is skipped while no store holds rows at that size.
- Each image counts the same in the gradient whatever its batch size. `--batch-size` still sets the epoch length
  (in micro-batches), and the checkpoint keeps `--train-input-size` (or the cache's size) as its input size.
- Mosaics need four images per batch, so a batch of 1-3 has none.
- Every 2,000 steps the log gets a summary: each size's share drawn against the plan, its fitted ms per
  micro-batch, its loss, the share of its images from a store, rows per store, img/s and VRAM peak.

B70 batches that fit beside run 4's mosaics: 10 at 512, 5 at 768, 3-4 at 1,024, 1-2 at 2,048 (2,048 at batch 1:
11.9 GB, 399 ms for the tower alone). 4,096 doesn't fit at batch 1.

`evaluate --input-size N --hires-store <dir>` scores the held-out images at N; `--gold-images <dir>
--gold-input-size N` scores the gold sets from images prepared at N (`review/aliasing/area1024`, from the originals).

### Glimpses (run 5; off by default)

`--classification-glimpses` (fresh runs only) scores localizable tags from close looks the model chooses:

- a proposal head ranks suspect tags;
- the localization map gives their peaks;
- 64px views of 32/64/128/256px windows are pooled by a log-sum-exp at `--classification-pooling-temperature`.

Run 5's settings:

- `--glimpses-per-image 48`;
- `--classification-pooling-temperature 0.5 --classification-pooling-start-temperature 4
  --classification-pooling-anneal-steps 30000`;
- `--glimpse-label-framing-steps 20000`.

Not exported to ONNX (`export-onnx --ignore-pooling` exports the boxes only). Run 5 was stopped: its windows
learned the image label. See [research/experiments-glimpses.md](../research/experiments-glimpses.md).

## Run 4's recipe

Run 4 is the current model. This is its real command (`train-v4.sh`, outside git), with the machine's paths
replaced by placeholders:

```sh
# XPU environment (see Intel Arc / XPU above)
export NEOReadDebugKeys=1 EnableSharedSystemUsmSupport=0 EnableImplicitMigrationOnFaultableHardware=0
export OverrideCopyOffloadMode=0
export LD_LIBRARY_PATH=<xpu libtorch folder>

dotnet <xpu-build>/unbooru-tagger-training.dll train \
  --cache-dir <dataset> \
  --checkpoint-dir <checkpoint-v4> \
  --embedding-dim 512 \
  --epochs 6 \
  --stem-channels 96 --stage-channels 96,192,384,768 --blocks-per-stage 3,3,9,3 --fused-stages 3 \
  --batch-size 10 --gradient-accumulation-steps 2 \
  --lr 1e-4 --lr-schedule cosine --warmup-steps 2000 \
  --localization-weight 0.3 --localization-temperature 0.25 \
  --self-supervised-weight 0 \
  --validation-max-images 20000 \
  --checkpoint-interval-minutes 15 \
  --localization-map fused \
  --mosaics-per-step 2 \
  --context-negative-weight 3 \
  --negative-gamma 2 \
  --bf16 \
  --train-scales 512:10:0.3,768:5:0.3:0.2,1024:3:0.3:0.3,2048:1:0.1:1 \
  --hires-store <store768>,<store1024>,<store2048> \
  --tag-parentage <tag-parentage/rules.json> \
  --gold <gold/gold.json> \
  --evaluation-command <evaluate-on-cuda.sh> \
  --snapshot-evaluation-steps 10000 \
  --chart-reference v3-e1=<context-check/v3-e1> \
  --chart-reference v3-e2=<context-check/v3-e2>
```

**How the recipe grew** (from the script's header):

| Date / step | Change |
|---|---|
| 2026-10-26 | Run 3's recipe, restarted fresh with `--mosaics-per-step 2` and `--context-negative-weight 3`. |
| ~275K | `--negative-gamma 2`. |
| ~358-359K | `--bf16`. |
| ~502K | `--tag-parentage`. |
| 2026-11-02 | The build with run 5's speed work: 221 → 142 ms/step, same recipe. |
| ~1.78M | Localization pair rows padded to multiples of 64. |
| ~1.79M | An image's labelled pairs capped at 192, a mosaic tile's pairs from other tiles at 128. |
| ~2.09M (2026-11-04) | Mixed scales (`--train-scales`) and the hi-res stores; until the 768 and 2,048 stores held rows, `store1024` alone. 206 ms/step, 48 img/s (512 alone: 142 ms, 70 img/s). |

Evaluations stay on one evaluation build, so the trend charts stay comparable. The A/B results behind each flag
are in [research/](../research/README.md).

## Evaluation

`evaluate` scores any checkpoint on the held-out split and calibrates its thresholds:

```sh
dotnet run --project unbooru-tagger-training -c Release -- evaluate \
  --checkpoint-dir ./checkpoint --cache-dir ./data/crawled --max-images 20000 \
  --gold gold/gold.json --gold-small gold-small/gold.json --localization-samples 0
```

`--max-images` and `--validation-fraction` must match the training run's `--validation-max-images` and
`--validation-fraction`.

### What it writes

To `--out-dir`, which defaults to the checkpoint directory:

- **`tag_thresholds.json`:** per-tag best-F1 cutoffs. `export-onnx` ships it, and inference uses it by default.
- **`evaluation_report.json`:**
  - per-tag AP against hard plus random negatives, hard-negative AP, F1 and threshold (worst tags first);
  - image-level precision and recall;
  - exclusive violations;
  - context reliance (recall and false-positive rate by context-prior band);
  - with `--context-check`, the labelled context check.
- **`rating_report.json`:** the rating confusion matrix.
- **`localization_report.json`** (and `_small.json` for `--gold-small`):
  - gold pointing, by stratum and size;
  - energy in the boxes;
  - region precision and recall: the map within 4 log-odds of its peak (the cut `detect` boxes come from) against the
    boxes, by cell (from 2026-11-04);
  - box precision, recall and count error (boxes cut as `detect` cuts them; before 2026-11-04 at 60% of the probability
    range, so earlier box numbers aren't comparable);
  - the occlusion check's object effect and control drop;
  - with `--localization-samples > 0`, the label-free checks (deletion, two characters, crop consistency, solo
    counts).

### Useful options

| Option | Default | Purpose |
|---|---|---|
| `--batch-size` | 10 | Images per forward pass. The evaluation scripts use 32 on a 16 GB A4000. |
| `--precision fp32\|tf32\|bf16` | fp32 | tf32 cut the held-out embedding about 27% at equal mAP. |
| `--embeddings-cache <file>` | none | Reuse the held-out embeddings when re-evaluating the same checkpoint. |
| `--dump-occlusion` | off | Writes each gold pair's scores (`occlusion_pairs*.jsonl`, glimpse windows, pointing pairs). |
| `--occlusion-fill zero\|mean\|noise` | zero | Fill for blanked boxes. Use a fill the model never trained with. |
| `--dump-scores tag1,tag2,...` | none | `scores.jsonl` with every held-out image's probability for those tags. |
| `--context-check <file>` | none | Labelled context negatives (`{cacheIndex, tag, label: present\|absent}`). |
| `--map-score-temperature T` | 0 (off) | Also run the occlusion check scored from the localization map (slow). |
| `--no-crop-teacher` | off | Skip the crop-teacher check. |
| `--gold-images <dir> --gold-input-size N` | none | Score the gold sets from images prepared at N px (e.g. from the originals); the checks' batches shrink with N. |
| `--input-size N`, `--hires-store <dir>` | the checkpoint's | Score the held-out images at N px, rows a store holds from it. |
| `--show` | 25 | Worst tags to print. |

Diagnostics used in the research:

- `--glimpse-grid`, `--glimpse-object-only`;
- `--twin-check`, `--map-window-check`, `--peak-window-scores`;
- `--locality-prior-out`, `--locality-prior-images`;
- the `glimpse-frames` and `window-probe` commands.

`evaluate` prints each phase's time. On run 4's 850K checkpoint a full evaluation took 362 s on the A4000.

### Reading the numbers

See [research/experiments-measurement.md](../research/experiments-measurement.md).

- The occlusion **object effect** must be read with the control drop, and on an unseen fill.
- Box metrics before 2026-11-01, and gold object effects before 2026-10-29 22:30, came from older protocols.

## The dashboard and charts

In a terminal, `train` shows a full-width panel. It covers:

- the phase, epoch and whole-run progress with finish times;
- ms/step and the data wait;
- the loss, both this step and a 1,000-step mean;
- the learning rate, checkpoints, the last evaluation and recent events.

**Charts below the panel:**

- the loss over the whole run (`loss_history.json`, kept across resumes);
- at every epoch and snapshot evaluation:
  - held-out mAP;
  - gold and gold-small pointing (also split by size);
  - exclusive violations;
  - the occlusion object effect;
  - the context gaps;
  - the labelled context check's log-odds;
  - the gold checks at 1,024px, when the evaluation command wrote them (`at1024/`, `at1024-auto/`): gold and
    gold-auto pointing by size, gold-small pointing and the object effect;
  - region precision by size, recall dashed, at 512 (gold) and at 1,024 (gold-auto).

In a terminal that shows Sixel images (Konsole, xterm, WezTerm, foot, or tmux 3.4+ built with sixel), the charts
are drawn as an image. Other terminals get braille text charts, as does `--text-dashboard`. With output
redirected, the trainer prints a one-line summary every 5 minutes. Run the trainer in tmux or a terminal: the
live panel needs a TTY.

**Chart options:**

- `--chart-reference label=report-dir` (repeatable) adds another run's evaluation as a dashed line.
- `--chart-compare label=checkpoint-dir` draws another run's evaluations and loss as curves.

**The same charts from any checkpoint directory,** alongside a running training:

```sh
dotnet run --project unbooru-tagger-training -- charts --checkpoint-dir <checkpoint> --watch \
  [--png charts.png] [--reference label=report-dir ...] [--compare label=checkpoint-dir ...] [--epochs 6]
```

## Offloaded and snapshot evaluations

**Offloaded epoch evaluations.** `--evaluation-command <exe>` runs each epoch's evaluation as a separate process,
so training continues; for example `evaluate` on a second GPU. A CUDA and an XPU libtorch can't share one
process.

1. At each epoch end, the trainer writes a model-only snapshot to `history/epoch-N/checkpoint/`.
2. It runs `<exe> <snapshot dir> <history/epoch-N>`.
3. The command must write the reports there and exit 0.
4. Results are applied in epoch order at the next checkpoint save: early stopping, and the top-level
   `evaluation_report.json` and `tag_thresholds.json` that `export-onnx` reads.

So early stopping lags by up to an epoch. The best epoch's weights stay in its snapshot. Marker files
(`evaluation.done`, `.failed`, `.applied`) let a restarted trainer apply finished evaluations and rerun
interrupted ones.

A minimal command:

```sh
#!/usr/bin/env bash
exec dotnet unbooru-tagger-training.dll evaluate --checkpoint-dir "$1" --out-dir "$2" \
  --cache-dir ./dataset --max-images 20000 --localization-samples 0 --gold gold.json
```

Run 4's real one (`xpu-test/evaluate-on-cuda.sh`, outside git) adds:

- `--batch-size 32`, `--gold-small`, `--dump-occlusion`;
- `--dump-scores gloves,hat,earrings,shoes,glasses,hairclip,wings,tail,choker,blush,1girl`;
- `--context-check`;
- `exec setsid flock <a4000.lock> ...`. `setsid` keeps a Ctrl+C in the training window from killing a running
  evaluation, and `flock` runs one evaluation on the GPU at a time.
- If the trainer runs on XPU, `unset LD_LIBRARY_PATH` in a CUDA evaluation script.
- From 2026-11-04 (mixed scales): after the 512px evaluation, the gold sets again at 1,024 from the originals
  (`--gold-images review/aliasing/area1024 --gold-input-size 1024`; gold + gold-small into `at1024/`, gold-auto into
  `at1024-auto/`). It re-runs itself under `flock`, so the 1,024 checks hold the lock too, and writes
  `evaluate-on-cuda.done`: a restarted trainer relaunches its unfinished evaluations while the old trainer's copies
  still wait on the lock, and whichever runs second stops there.

Evaluations of experiments should wait while a run's own evaluations are queued, or the trend charts fall hours behind.

**Snapshot evaluations.** `--snapshot-evaluation-steps N` (needs `--evaluation-command`) also evaluates a
model-only snapshot every N global steps, for the trend charts.

- Reports go to `snapshots/step-N/`, and the weights are deleted once evaluated.
- Snapshot evaluations never affect early stopping.
- On a resume, finished ones are reloaded, unfinished ones restart, and ones past the resumed step are deleted.
- Check the snapshot directories for `evaluation.failed`, not only `evaluation.done`.

## Keeping a long run alive

The maintainer's setup uses a watchdog script (`xpu-test/watchdog-v5.sh`, outside git). Its behaviour is
described here so it can be rebuilt. Once a minute it looks for:

- new xe kernel errors (a copy-engine fault, an engine reset or a timed-out job), treated as a hang at once;
- a stall: the dashboard says "training" but the step hasn't moved for 5 minutes;
- an unexpected exit: a non-zero status or a signal. Exit 0 (Ctrl+C with a save, `--max-steps`, the end of the
  run, early stopping) and a NaN loss are left alone.

When it finds one, it:

1. saves the evidence (kernel lines, the managed stack, the dashboard screen);
2. ends the trainer (SIGTERM, then SIGKILL);
3. checks the GPU with a short job;
4. restarts the training script in the same tmux pane (`tmux respawn-pane`, without closing the window).

It gives up after 3 restarts within 120 minutes, or if the device check fails. A pause file stops it during
deliberate stops.

**Lessons from running it:**

- Never leave the watchdog paused across a long job.
- Anything that waits on a job must also fire on failure and on stalls.
- Don't edit a bash script while it runs: bash reads scripts as it goes.

## Exporting to ONNX

```sh
dotnet run --project unbooru-tagger-training -- export-onnx --checkpoint-dir ./checkpoint --model-dir ./model
```

- **What it writes:** the bundle the inference CLI reads. That is `image_encoder.onnx`, `tag_vocabulary.json`,
  `tag_embeddings.bin` and `logit_scaling.json`, plus `exclusive_tags.txt`, `localizable_tags.txt` and
  `tag_thresholds.json` when the checkpoint has them.
- **How the graph is built:** by hand, because TorchSharp has no ONNX exporter. Tests check it against
  TorchSharp's forward pass, including the fused laterals and a symbolic batch dimension.
- **Pooled checkpoints:** checkpoints that score localizable tags per tag (classification pooling, glimpses) are
  refused. `--ignore-pooling` exports them anyway with correct boxes but not correct scores for those tags.

## Adding a tag

Adds one embedding row and fine-tunes only that row, with the image encoder frozen:

```sh
dotnet run --project unbooru-tagger-training -- add-tag hatsune_miku_nurse_costume \
  --checkpoint-dir ./checkpoint --images ./new-tag-manifest.json --cache-dir ./data/crawled --steps 300
```

| Option | Default | Notes |
|---|---|---|
| `--images` | (required) | A manifest of newly tagged images. |
| `--steps` | 200 | |
| `--lr` | 1e-3 | |
| `--early-stopping-patience` | 5 | |
| `--min-image-threshold` | 15 | Images needed before the tag is promoted from warm-start-only to trained. |
| `--cache-dir` | none | The corpus the checkpoint was trained on. It adds hard negatives (images with the tags that usually accompany the new tag, but not the new tag) and random negatives, `--negatives-per-positive` (4) per image. The option's own help calls it "strongly recommended: without negatives the new tag learns to fire on everything similar". |
| `--warm-start-model`, `--warm-start-vocab` | none | Start the row from a frozen ONNX text-embedding model's encoding of the tag name (BERT-family WordPiece `vocab.txt`; underscores become spaces). Its output size must equal the embedding size. The same options apply to `train`'s fresh vocabulary. |

## Checkpoint directory contents

```
checkpoint/
  image_tower.dat                 # TorchSharp weights (not ONNX)
  model_config.json               # tower shape and saved model options
  tag_vocabulary.json
  tag_embeddings.bin
  logit_scale_bias.dat            # learned logit scale/bias
  training_progress.json          # epoch/step, ValidationSplitSeed, early stopping (+ optimizer state)
  logit_scale_bias_optimizer.dat  # separate optimizer state for the scale/bias
  exclusive_tags.txt              # exclusive groups (copied into the model bundle)
  tag_parentage.json              # with --tag-parentage
  tag_thresholds.json             # per-tag thresholds from the latest evaluation (copied into the bundle)
  evaluation_report.json          # latest held-out report
  loss_history.json               # the dashboard's loss chart
  history/epoch-N/                # each epoch's reports (and, offloaded, its snapshot and markers)
  snapshots/step-N/               # --snapshot-evaluation-steps reports
```

Checkpoints, datasets and model bundles are not committed: `.gitignore` excludes `*.onnx`, `*.pt` and `*.bin`
outside test fixtures.

## Experimental options (off by default)

Every option below was built for an A/B and **not adopted**. They remain in the code, off by default. The results
are in [research/](../research/README.md).

| Theme | Options | Research page |
|---|---|---|
| Context shortcut | `--context-prior-weight`, `--focus-share`, `--mosaic-classification-weight`, `--mosaic-full-size`, `--out-of-context-share` / `-max-prior` / `-weight` / `-all`, `--excluder-share`, `--excluded-negative-weight` | [context shortcut](../research/experiments-context-shortcut.md) |
| Classification pooling | `--classification-pooling-temperature`, `-start-temperature`, `-anneal-steps`, `-mined-negatives`, `-top-k`, `-bottom-weight` | [context shortcut](../research/experiments-context-shortcut.md) |
| Erased views | `--erasure-pairs-per-step`, `--erasure-loss hinge\|negative\|negative-only`, `--erasure-weight`, `--erasure-margin`, `--erasure-fill`, `--erasure-every`, `--erasure-anchor-weight` | [erased views](../research/experiments-erased-views.md) |
| Glimpses | `--glimpse-pool-zooms`, `--glimpse-pool-min-zoom`, `--glimpse-label-framing-floor`, `--glimpse-framed-positive-pooling`, `--glimpse-proposal-detach-tags`, `--separate-glimpse-tower`, `--glimpse-tower-lr-scale`, `--context-pairs-per-batch`, `--context-pair-weight`, `--context-pair-margin`, `--glimpse-feature-split`, `-weight`, `-negatives`, `--glimpse-exclusive-per-batch` | [glimpses](../research/experiments-glimpses.md) |
| Locality negatives | `--locality-negative-weight`, `--locality-per-pair`, `--locality-confidence-gate`, `--locality-detach-scaling`, `--locality-min-observations`, `--locality-prior` | [glimpses](../research/experiments-glimpses.md) |
| Window labels | `--mosaic-glimpse-weight`, `--twin-every`, `--twin-weight`, `--twin-detach-scaling`, `--twin-paste`, `--self-map-weight`, `--self-map-detach-scaling`, `--self-map-negatives`, `--self-map-negative-weight`, `--mosaic-large-grid`, `--mosaic-large-grid-every` | [window labels](../research/experiments-window-labels.md) |
| Built, never A/B tested | `--negative-loss hill`, `--hard-negative-source-weight` | |

## Measuring speed

- **`benchmark-tower`** times the tower's forward and backward pass on random input. It can measure:
  - each memory layout or precision (`--layouts nchw,bf16-mixed,...`);
  - each op (`--components`);
  - the glimpse path (`--glimpses N [--vary] [--no-pad]`);
  - uploads (`--uploads`).

  Weights and inputs are created on the CPU, because `manual_seed` doesn't seed the XPU generator.
- **`UNBOORU_PROFILE_STEPS=N`** makes `train` time each phase of N steps, after 50 warm-up steps, and write
  `step_profile.txt` to the checkpoint directory.
- **`soak-locality`** repeats the locality term's step on random input and prints device memory, to find a leak.

## Tests

```sh
dotnet test unbooru-tagger-tests
```

Tests run sequentially: TorchSharp's RNG is process-global native state. They include ONNX-export equivalence
against TorchSharp's forward pass, loss and gradient equivalences (chunked, sparse and pooled versus their
reference implementations), and the sampler and target logic.
