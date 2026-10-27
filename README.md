# unbooru-tagger

A multi-label anime/booru-style image tagger built on a dual-encoder,
sigmoid-contrastive architecture (JoyTag / SigLIP-style), implemented in
C# with [TorchSharp](https://github.com/dotnet/TorchSharp) for training
and ONNX Runtime for inference. Unlike a fixed-head classifier (WD14
tagger, DeepDanbooru), tags live in a learned embedding table, so new
tags can be added later without retraining the whole model. See
[`CLAUDE.md`](CLAUDE.md) for the full architecture rationale, including an
explicit rundown of where the implementation has ended up differing from
the original plan (entirely C#/.NET rather than Python/PyTorch being the
biggest one — TorchSharp for training, ONNX Runtime for inference; there
is no Python code in this repo).

## Features

- **Tagging** — score an image against the full tag vocabulary and get
  back confidence values per tag.
- **Localization** — a rough heatmap showing where in the image a given
  tag's evidence concentrates, with no bounding-box training data needed.
- **Bounding-box detection** — approximate boxes from connected components
  over the localization heatmap, tightened without any bounding-box
  training data via per-tag percentile thresholding and a bilateral filter
  that snaps the heatmap to the source image's own edges. Newest feature,
  least battle-tested.
- **Open-ended vocabulary** — add a brand-new tag and fine-tune just that
  row, without touching the rest of the model. The row can warm-start from
  a frozen text embedding of the tag string (`--warm-start-model`/
  `--warm-start-vocab`, see below) instead of small random noise.
- **Rare-tag aware training** — oversampling and a minimum-image
  threshold so long-tail tags aren't drowned out by common ones.
- **Label-free localization & representation training** — an auxiliary
  MIL/log-sum-exp pooled loss that directly rewards sharp per-location
  responses, plus a SimSiam-style self-supervised consistency loss
  between random-crop views of the same image. Both use only the tag
  labels/images already in the corpus — no bounding-box data needed.
- **Letterboxed, aspect-ratio-preserving preprocessing** — non-square
  images are resized to fit and padded to a square canvas rather than
  squashed/stretched, so the model never sees distorted geometry. Training
  and inference both mask the padding bars out of pooling, the
  localization loss, and heatmap/box coordinates, so padding never dilutes
  gradient or gets mistaken for image content.

## Repository layout

| Project | Type | Purpose |
|---|---|---|
| `unbooru-tagger-core` | library | Shared model-bundle loading, ONNX image encoding, tag vocabulary/embedding storage, SigLIP-style scoring, heatmap/box generation, dataset manifest & cache formats. |
| `unbooru-tagger-data-unbooru-import` | CLI | Pulls images/tags from the [unbooru](#prerequisites) database and builds training datasets/caches. |
| `unbooru-tagger-data-booru-downloader` | CLI | Downloads images/tags directly from Danbooru/Gelbooru into a trainable dataset directory — no `unbooru`/SQL Server needed. |
| `unbooru-tagger-training` | CLI | Trains the dual-encoder model with TorchSharp, adds new tags, exports to ONNX. |
| `unbooru-tagger-inference` | CLI | Loads an exported model bundle and tags/localizes/detects on images. |
| `unbooru-tagger-tests` | xUnit | Test suite covering all of the above. |

## Prerequisites

- **.NET 10 SDK** (see [`global.json`](global.json); prerelease SDKs are
  allowed via `rollForward: latestMajor`).
- To build/run **`unbooru-tagger-data-unbooru-import`** (and to open the
  full `.sln`, which references it): a sibling checkout of the
  [`unbooru`](https://github.com/da3dsoul/unbooru) repo at `../unbooru`
  relative to this repo, e.g.:
  ```
  GitHub/
    unbooru/           <- sibling repo, provides Core.csproj + Abstractions.csproj
    unbooru-tagger/     <- this repo
  ```
  `unbooru-tagger-core`, `unbooru-tagger-training`,
  `unbooru-tagger-inference`, and `unbooru-tagger-data-booru-downloader`
  do **not** need `unbooru` and can be built standalone.
- A SQL Server instance with an `unbooru` database, if you intend to
  build datasets from `unbooru-tagger-data-unbooru-import`.
  `unbooru-tagger-data-booru-downloader` is the alternative that needs
  neither — just network access to Danbooru and/or Gelbooru.
- To train with GPU acceleration: **Linux with an NVIDIA GPU** — the
  training project pulls in `TorchSharp-cuda-linux` (bundles its own CUDA
  12.8 runtime, just needs a compatible driver) only on Linux. On Windows
  (and any other non-Linux OS) it falls back to `TorchSharp-cpu`, so
  training on Windows is CPU-only.
- Or **an Intel Arc GPU** (XPU, tested on an Arc Pro B70): build with
  `-p:TorchBackend=xpu` (see [Training on an Intel GPU](#training-on-an-intel-gpu-xpu)).

## Getting started (end users — tagging images)

You need a trained model bundle: either train your own (see below) or
grab one from this repo's
[Releases](https://github.com/da3dsoul/unbooru-tagger/releases) page. A
model bundle is a directory containing:

```
model/
  image_encoder.onnx
  tag_vocabulary.json
  tag_embeddings.bin
  logit_scaling.json   # optional -- missing means Scale=1, Bias=0 (an older export)
```

Build and run the inference CLI:

```sh
dotnet build unbooru-tagger-inference -c Release
```

**Tag a single image:**

```sh
dotnet run --project unbooru-tagger-inference -- tag ./cat.png --model-dir ./model --threshold 0.4
```

Prints `confidence<TAB>tag` lines to stdout, most confident first.

**Tag a directory of images (batch):**

```sh
dotnet run --project unbooru-tagger-inference -- tag-batch ./images --model-dir ./model > tags.json
```

Prints a single JSON object: `{ "filename.png": { "tag": confidence, ... }, ... }`.

**Render a heatmap for one tag on one image:**

```sh
dotnet run --project unbooru-tagger-inference -- heatmap ./cat.png solo --model-dir ./model --out solo_heatmap.png
```

**Detect approximate bounding boxes:**

```sh
dotnet run --project unbooru-tagger-inference -- detect ./cat.png --model-dir ./model --box-threshold 0.5 --box-percentile 0.6 --out detected.png
```

Prints JSON detections to stdout; `--out` additionally writes an
annotated PNG with colored boxes and tag/confidence labels. `--min-heatmap-range`
(default `0.05`) suppresses boxes entirely for a tag whose heatmap is too flat to
carry real spatial signal, instead of reporting a canvas-spanning or fragmented
box built from noise. `--min-box-fraction` (default `0.002`) drops regions covering
less than that fraction of the heatmap grid (isolated specks), and
`--max-boxes-per-tag` (default `3`, `0` = unlimited) keeps only the strongest. Each
box is labeled with its own peak confidence, not the tag's whole-image score.

**Compare against a gold set by eye:**

```sh
dotnet run --project unbooru-tagger-inference -- render-gold ./gold/gold.json --out-dir ./gold/renders --model-dir ./model
```

Writes one PNG per gold image. It shows the hand-checked boxes drawn
exactly as `detect` draws, with the same per-tag colors, beside that
model's `detect` output. The model panel shows only the tags the gold set
boxes on that image (`--all-tags` shows everything). Omit `--model-dir` to
render the gold boxes alone. Takes the same box options as `detect`.

Every command accepts `--model-dir` (default `./model`). `tag`, `tag-batch`
and `detect` also accept `--threshold` for the minimum tag confidence to report.
Omit it to use each tag's calibrated threshold from the bundle's
`tag_thresholds.json` (see `evaluate` below), with 0.5 for uncalibrated tags or
bundles without that file. If the bundle has `exclusive_tags.txt`, only the
stronger tag of any mutually exclusive pair (e.g. `solo` vs `multiple_girls`) is
reported.

## Building from source (developers)

Clone this repo, and — if you need `unbooru-tagger-data-unbooru-import`
or the full solution — clone `unbooru` alongside it:

```sh
git clone https://github.com/da3dsoul/unbooru-tagger.git
git clone https://github.com/da3dsoul/unbooru.git   # sibling, only if you need unbooru-tagger-data-unbooru-import
```

Build everything:

```sh
dotnet build unbooru-tagger.sln
```

Or build a single project (no sibling repo needed for these):

```sh
dotnet build unbooru-tagger-core
dotnet build unbooru-tagger-training
dotnet build unbooru-tagger-inference
dotnet build unbooru-tagger-data-booru-downloader
```

Run the test suite:

```sh
dotnet test unbooru-tagger-tests
```

Tests run sequentially, not in parallel — TorchSharp's RNG is
process-global native state, so parallel test execution would make
TorchSharp-seeded tests non-deterministic
(`[assembly: CollectionBehavior(DisableTestParallelization = true)]`).

There is no configuration file anywhere in the solution — connection
strings, model paths, and hyperparameters are all passed as CLI flags,
never read from `appsettings.json` or environment variables.

## Building a dataset (`unbooru-tagger-data-unbooru-import`)

Requires the `unbooru` sibling repo and a reachable `unbooru` SQL Server
database.

**Small dataset** — pulls images matching any of the given tags plus an
equal-sized sample of non-matching images, for quick iteration:

```sh
dotnet run --project unbooru-tagger-data-unbooru-import -- build-small-dataset \
  --tags hatsune_miku twintails \
  --connection-string "Server=...;Database=unbooru;..." \
  --out ./data/miku-small \
  --max-images 500
```

Writes `./data/miku-small/images/<ImageId>` files and a
`./data/miku-small/manifest.json`.

**Large cache** — streams the whole corpus (or a capped subset) into a
preprocessed, memory-mappable cache for full training runs. Interrupted
runs resume automatically if you re-run against the same `--out`:

```sh
dotnet run --project unbooru-tagger-data-unbooru-import -- build-large-cache \
  --connection-string "Server=...;Database=unbooru;..." \
  --out ./data/large-cache \
  --input-size 224 \
  --min-tag-images 100
```

Writes `images.bin` (preprocessed pixel data) and `tag_rows.sqlite` (a
SQLite database of packed per-image tag-row indices — an older
`tag_rows.jsonl` from a pre-existing `--out` is migrated to this
automatically, once, the first time it's opened) under `--out`.
`--min-images-per-tag` (default 15) reserves the rarest-first images per
tag when `--max-images` caps the corpus; `--min-tag-images` (default 100)
is the corpus-wide occurrence count a tag needs to get a vocabulary row
at all.

## Crawling a dataset from Danbooru/Gelbooru (`unbooru-tagger-data-booru-downloader`)

An alternative to `unbooru-tagger-data-unbooru-import` that needs no
`unbooru` repo/SQL Server at all — it downloads directly from the Danbooru and
Gelbooru public APIs into the same trainable dataset directory shape
`build-large-cache` produces (`images.bin`, `tag_rows.sqlite`,
`tag_vocabulary.json`), plus its own `crawl.sqlite` for survey results,
per-tag/site resumability, and cross-site image dedup, and a few more
small files it manages itself under `--output-dir`: `excluded_tags.txt`
(tag-exclusion rules, see below), `tag_aliases.json` (a cached copy of
Danbooru's active tag-alias table, used to merge a raw spelling one site
still uses into its canonical name on the other), and `crawl-errors.log`
(a durable record of transient site failures `crawl` hit and retried).

**1. Survey tags** — records each tag's per-site post count and which
tags are worth crawling (`--min-images`, default 500). Gelbooru requires
`--gelbooru-api-key`/`--gelbooru-user-id` (from your Gelbooru account's
API settings page) even for this read-only listing — omitting them fails
with a 401:

```sh
dotnet run --project unbooru-tagger-data-booru-downloader -- survey-tags \
  --output-dir ./data/crawled \
  --min-images 500 \
  --gelbooru-api-key <key> \
  --gelbooru-user-id <user-id>
```

Prints how many tags are eligible and an upper-bound estimate of the
image slots a crawl would need (before cross-tag/cross-site dedup, which
only shows up once the crawl actually runs), plus how many otherwise-
eligible tags were excluded via `excluded_tags.txt`.

Before eligibility is computed, Danbooru's active tag-alias table
(antecedent → consequent, e.g. `head_pat` → `headpat`) is fetched and
merged in, so a raw spelling one site still uses doesn't get surveyed as
a separate tag from its canonical name elsewhere. This fetch is required,
not best-effort: if it fails (Danbooru unreachable, rate-limited, ...),
`survey-tags` exits with a non-zero code and makes no changes rather than
risk a survey that's silently wrong. `crawl` never fetches this table
itself — it only reads whatever `survey-tags`/`refresh-tags` already
cached to `tag_aliases.json`.

Every tag's identity — in the vocabulary, `tag_rows.sqlite`,
`excluded_tags.txt`, progress output — is its raw booru name prefixed
with its category (Danbooru's `tags.json` `category`/Gelbooru's `type`):
`white_hair`, `elf`, `character:frieren`, `series:sousou_no_frieren`.
General tags (the overwhelming majority) are left bare; artist/copyright/
character/meta tags get an `artist:`/`series:`/`character:`/`meta:`
prefix. This makes metadata tags trivially easy to spot (they all start
with `meta:`) for the exclusion file below.

Every `meta:` tag is excluded from the vocabulary by default — never
surveyed as eligible, never searched as a crawl target, and stripped
from every image's tag list even if another target tag's search happens
to pull in a post carrying one. This is a blanket rule, not a
hand-maintained list: a real survey of Danbooru's meta category (~800
tags, `tags.json?search[category]=5`) is overwhelmingly upload/link
bookkeeping (`bad_pixiv_id`, `md5_mismatch`...), per-language commentary
(`french_commentary`, `translation_request`...), and tagging-workflow
placeholders (`character_request`, `check_copyright`...) — none of it
recoverable from the pixels. General/artist/series/character tags are
never touched by this rule.

Two carve-outs pull back out of that blanket, since some meta tags
genuinely describe visual production technique: any tag ending in
`_(medium)` (`pen_(medium)`, `oil_painting_(medium)`,
`photoshop_(medium)`...) is automatically included — Danbooru's own
convention for "how this was made" — plus a short curated list
(`scan`, `ai-generated`, `traditional_media`, `vector_art`...) for
technique tags that don't happen to use that suffix. The first
`survey-tags`/`crawl`/`refresh-tags` run against a `--output-dir` seeds
these as `!`-prefixed lines in `excluded_tags.txt`. Edit the file by hand
to adjust either direction — a bare line excludes that identity outright
(any category, not just meta), a `!`-prefixed line includes it despite
the meta blanket — it's read fresh on every run, so edits take effect
immediately without re-running `survey-tags`.

**2. Crawl** — pulls a target of `--max-images` (default 1000) images per
eligible tag, rarest tag first, then automatically tops up negative
(non-tagged) examples per tag:

```sh
dotnet run --project unbooru-tagger-data-booru-downloader -- crawl \
  --output-dir ./data/crawled \
  --min-images 500 \
  --max-images 1000 \
  --input-size 224 \
  --gelbooru-api-key <key> \
  --gelbooru-user-id <user-id>
```

Every option this command was run with (sites, `--min-images`/`--max-images`/
`--input-size`, rates, API credentials, negative-mining settings) is saved to
`last-crawl-command.json` under `--output-dir`. A later re-run just needs:

```sh
dotnet run --project unbooru-tagger-data-booru-downloader -- crawl \
  --output-dir ./data/crawled \
  --resume
```

`--resume` reuses everything from that file and ignores any of those same
options if you also pass them alongside it — only `--output-dir` matters
together with `--resume`. This is separate from (and doesn't replace) the
per-tag/site progress checkpointing described below, which is what actually
lets a re-run pick back up mid-corpus instead of starting over — `--resume`
only saves you from having to reconstruct the command itself.

Every configured site runs its own worker concurrently — each doing a
full positive-then-negative pass over every eligible tag at its own rate
limit — rather than the crawler round-robining pages between whichever
site has made fewer requests so far. Within a site, `--max-images` is a
target, not a hard cap: each site keeps searching a tag until it
personally accounts for its own even share
(`ceil(--max-images / site count)`) even after the combined count across
sites already looks met, so a faster/bigger site can't starve a slower
one out of ever contributing anything of its own — a tag's actual
combined image count can end up slightly over `--max-images` as a
result. Downloads and processing (decode/hash/resize) within a page also
run concurrently, gated by separate per-site semaphores, so one site's
network waits don't queue up behind the other's CPU work. Each site's
live progress shows a running duplicate count alongside its image tally
(e.g. `smelling_penis (750/500/143 dupes)`) — a high count there means
most of that overshoot came from re-discovering already-known images
while chasing its own floor, not from genuinely new ones.

Re-running the same command resumes efficiently, even against a
multi-million-image corpus: per-tag/site pagination cursors, plus each
(tag, site) pair's own fairness-floor and duplicate counts, are
checkpointed durably every page, so a restart never has to re-derive a
partially-crawled tag's progress from scratch. The pixel/tag-row cache
resumes the same way — a small sidecar (`images.bin.resume`) caches
where to pick back up instead of re-walking every already-cached record.
On startup, `crawl` also checks that `crawl.sqlite`'s row indices still
agree with the cache files' actual image count, and refuses to continue
with an explicit error if they've drifted apart (e.g.
`images.bin`/`tag_rows.sqlite` restored or moved independently of
`crawl.sqlite`) rather than silently reassigning an already-claimed row
to the wrong image. An image satisfying several
eligible tags at once is only ever downloaded once — both sites return a
post's full tag list in the same response used to find it, so there's
never a need to re-fetch tags for an already-cached image. Dedup happens
in two layers: an exact md5 check before ever downloading, plus a 64-bit
DCT perceptual hash (`PerceptualHash`) checked (Hamming distance <= 2,
calibrated against measured cross-site recompression noise) against a
banded, near-O(1) `PerceptualHashIndex` of every previously-cached
image — this is what catches the same source image cross-posted to both
sites after a re-encode/re-compress, which changes the md5 but not the
content. Either way, the post is recorded as an additional source of the
existing cache row rather than appended as a new image, and if the
duplicate carries eligible tags the first-seen copy didn't have, those
tags are merged into the existing row rather than discarded.
`--gelbooru-api-key`/`--gelbooru-user-id` are required, not optional —
Gelbooru rejects unauthenticated API requests with a 401. Optional
`--danbooru-login`/`--danbooru-api-key` raises Danbooru's rate-limit
tier; `--rate-danbooru`/`--rate-gelbooru` (defaults 4 and 2 requests/sec)
cap request rate. `--negative-target` defaults to `2 * --min-images` —
deliberately more negatives than positives, since an image with many tags
(ordinary on boorus) becomes a positive for all of them at once, which
would otherwise skew the surviving negative pool toward sparsely-tagged
images.

Once a tag's negative pool is actually short (checked against the whole
corpus first — a tag whose negatives are already covered organically,
just as a side effect of other tags' positive crawls, costs zero extra
requests here), the negative phase prefers hard negatives over plain
random background: it scans the corpus's own already-recorded tag
co-occurrence (`TagCooccurrenceIndex`) for tags that commonly show up
alongside the target AND have enough images carrying them *without* the
target to trust as a real negative source (`--negative-cooccurrence-ratio`,
default `0.5`; `--negative-cooccurrence-min-examples`, default `15`), then
queries up to `--max-hard-negative-sources` (default `3`) of those —
e.g. `vocaloid -hatsune_miku` — before falling back to the plain
`-{tag}` query every tag has always used. The counter-example floor is
what keeps a near-subset pair (`large_breasts` almost always implies
`breasts`) from being mined in the direction that has no real
counter-examples to draw from, while still allowing the useful reverse
direction and genuinely spread-out pairs (a character vs. the series
it's from) through. Set `--max-hard-negative-sources 0` to disable this
and always use the plain query, matching pre-existing behavior exactly.

A transient failure (network reset, DNS, TLS, timeout, HTTP 429, or a
5xx) is retried automatically with capped exponential backoff; a
permanently-gone download (404/410) is skipped immediately instead of
being treated as a site failure. If a site's retries are exhausted
anyway, that site's worker doesn't give up — it logs to
`crawl-errors.log`, shows an "ERROR ... retrying at HH:mm:ss" status on
its own progress row, waits 20 minutes, and retries the same page,
indefinitely — meant for unattended multi-hour/day runs to survive a
transient outage without a human needing to notice and restart. `crawl`'s
final output prints a note pointing at `crawl-errors.log` if any site hit
an error during the run.

**3. Refresh tags** — re-fetches previously-crawled posts by id (not by
re-listing tags) to catch tag edits/removals made on the site since
`crawl` last saw them:

```sh
dotnet run --project unbooru-tagger-data-booru-downloader -- refresh-tags \
  --output-dir ./data/crawled \
  --min-images 500 \
  --gelbooru-api-key <key> \
  --gelbooru-user-id <user-id>
```

Reconciles each affected image's tags as the union of every known
source's current tags — unlike `crawl`'s duplicate-merge path, this can
both add and remove a tag. A tag is only ever removed once every known
source of an image has a real (non-`null`) tag snapshot, so a dataset
that never captured snapshots before this feature existed is never at
risk of a premature drop. Resumable per site (`--reset` restarts a
site's sweep from the beginning instead of resuming after the last post
it checked); `--only-tags <identity...>` skips the full sweep and only
re-checks images currently holding one of the given tag identities — for
a scoped correction (e.g. reconciling images a tag-alias merge just
orphaned) where sweeping the whole corpus would be far more work than
the problem needs. Like `survey-tags`, this depends on successfully
fetching Danbooru's tag-alias table and exits without making changes if
that fails. If a site goes unavailable partway through, it's dropped for
the rest of the run (picked back up on the next invocation) rather than
retried forever; if every configured site fails, the command exits
non-zero without touching anything.

## Training a model (`unbooru-tagger-training`)

**Train** (from a manifest or a preprocessed cache — pick one):

```sh
dotnet run --project unbooru-tagger-training -- train \
  --manifest ./data/miku-small/manifest.json \
  --checkpoint-dir ./checkpoint \
  --epochs 10 \
  --batch-size 32
```

```sh
dotnet run --project unbooru-tagger-training -- train \
  --cache-dir ./data/large-cache \
  --checkpoint-dir ./checkpoint
```

Re-running against the same `--checkpoint-dir` automatically resumes
(model weights, optimizer state, epoch count, step within the current
epoch, validation split, and early-stopping history if present). A
checkpoint is saved after every epoch, and also every
`--checkpoint-interval-minutes` (default 15) *within* an epoch — an
epoch can take multiple days, so a crash only loses up to that interval
of work, not the whole epoch. Set it to `0` to disable mid-epoch
checkpointing. Uses CUDA if available, CPU otherwise. Key flags:
`--embedding-dim` (default 512), `--lr` (default 1e-4),
`--validation-fraction` (default 0.1), `--early-stopping-patience`
(default 3).

**Tower size (fresh runs only; a resume uses the checkpoint's
`model_config.json`).** `--stem-channels` (default 64), `--stage-channels`
(default `64,128,256`; each width change downsamples 2x) and
`--blocks-per-stage` (default `2,2,2`) set the ConvNeXt tower.
`--fused-stages` (default 0) adds a top-down path that fuses earlier stages
into the spatial map, each one halving the heatmap stride: 4 stages + 2 fused
is stride 8, a 64x64 map at 512px.

**Deliberate batches and targets (default).** The corpus was crawled per
tag, with positives and co-occurrence-chosen hard negatives for each one,
and training uses that structure directly:

- `--sampling tag-centric` (default; `rare-tag` is the old random-image
  sampler) builds each batch around a few *anchor* tags drawn uniformly
  over the vocabulary: `--positives-per-anchor` (default 2) images with
  the tag plus `--hard-negatives-per-anchor` (default 2) images with a
  commonly co-occurring tag but *not* this one (same series, other
  character; `breasts` without `large_breasts`).
- `--targets deliberate` (default; `legacy` = every untagged tag is an
  equally certain negative) weights each (image, tag) pair by how sure the
  corpus is about it:
  - **Full-strength pairs:** positives and verified negatives — the anchor's
    hard negatives, and tags mutually exclusive with one the image carries
    (`--certain-negative-weight`, default 1).
  - **Ignored pairs:** a tag usually implied by one the image carries (on ≥
    `--implied-threshold`, default 0.6, of that tag's images) is treated as
    a likely-missing label.
  - **Damped pairs:** all other negatives use the asymmetric loss —
    anything predicted below `--negative-clip` (default 0.05) costs nothing,
    and the rest is damped by `p^--negative-gamma` (default 4). This keeps
    each image's ~40 positives from being drowned out by ~25K unverified
    negatives.
- `--exclusive-tags <file>` — mutually exclusive groups, one
  comma-separated group per line, every pair on a line exclusive (e.g.
  `solo, multiple_girls`). Defaults to the checkpoint's saved copy, else
  built-in people-count and rating groups. Saved into the checkpoint and
  shipped with the exported model, where inference keeps only the stronger
  tag of any contradictory pair.

**Which tags get localized.** Whole-image tags have no region to point at:
artist/series/meta/rating tags, backgrounds, mediums, styles, color
themes, lighting, weather, framing/angle, people counts, and image format.
The per-location loss never trains on them, and `detect` reports them
without boxes. The rules live in `localizable_tags.txt`, one per line:
- `tag` — exact match.
- `prefix*` or `*suffix` — anchored match. Anchoring keeps `* background`
  from catching unrelated tags, and avoids substrings: "shadow" would also
  hit `eyeshadow`.
- `!rule` — exception (e.g. `!signature`, a small real region).

Underscores and spaces are interchangeable. `--localizable-tags <file>`
overrides the built-in defaults. The rules are saved into the checkpoint
and shipped in the exported model bundle.

**Validation measures ranking, not mean loss.** Mean sigmoid loss over
every (image, tag) pair is dominated ~600:1 by easy negatives. It can sit
near its floor while positives are ranked badly. Each epoch now scores the
held-out split per tag: AP of the tag's held-out positives against its
hard negatives plus random negatives, with likely-missing images left out.
Early stopping tracks mean AP. Each epoch also writes into the checkpoint:
- `tag_thresholds.json` — each tag's best-F1 cutoff.
- `evaluation_report.json` — per-tag AP, hard-negative AP, F1 and
  threshold, worst first, plus image-level precision/recall at those
  thresholds and the rate at which mutually exclusive tags fire together.

**Content ratings.** Run the crawler's `export-ratings --output-dir <dataset>`
once (and again after anything that changes the cache's rows). It writes
`ratings.bin`, one rating per image, taken as the *strictest* across the
image's sources so an explicit image is never learned as safe. `train`
then learns three cumulative labels:
- `rating:at_least_sensitive`
- `rating:at_least_questionable`
- `rating:is_explicit`

These add three vocabulary rows, so a checkpoint from before ratings can
only be resumed with `--no-ratings`. Validation calibrates each label's
cutoff for recall (90% / 97% / 97%) rather than F1, since missing explicit
content is the costly error. It reports a rating confusion matrix with the
rate of questionable/explicit images called general (`rating_report.json`).
`tag` prints the derived overall rating first.

**Label corrections from the crawl data.** Three more crawler exports, each
run once against the dataset directory (`--output-dir <dataset>`); training and
`evaluate` pick the files up automatically when present:
- `export-implications` writes `tag_implications.json`, Danbooru's active tag
  implications (witch_hat → hat). Danbooru applies them to every post but
  Gelbooru doesn't, so each image's implied tags are filled in when the tags are
  loaded. On run 2 this added 523,867 missing labels to 507,666 images.
- `export-image-sites` writes `image_sites.bin` (which sites each image came
  from) and `export-tag-agreement` writes `tag_agreement.json` (per tag, how
  often each site lists it on images both sites have). Together they mark the
  ~950 tags only one site's tag system uses (Gelbooru's `silver_hair`,
  `black_legwear`; Danbooru's `covered_nipples`). An image from only the other
  site can never carry those, so its absence isn't trained or evaluated as a
  negative. The agreement counts aren't a tagging-recall measure: Gelbooru
  largely copies Danbooru's tags.

**Context negatives.** If tag t is on 60–90% of tag s's images
(`--implied-threshold` 0.6, `--missing-threshold` 0.9), s is a context source
for t: a character who usually wears a hat. Images with s but not t are
trained as unverified negatives and drawn per anchor tag
(`--context-negatives-per-anchor`, default 1); at 90% and above an absence is
still treated as a missing label. `evaluate` reports a per-tag context AP.

`--context-negative-weight` (default 0 = off) trains every such pair in a batch,
not just the sampled ones, as a full-strength negative with this weight instead
of a damped one, so "this character, therefore gloves" is no longer nearly free.
It applies to localizable tags only: whole-image tags like `1girl` are often just
untagged.

**Localization options.**
- `--localization-map fused` trains localization (and draws heatmaps) on the
  fused high-resolution laterals only, without the last stage. Needs
  `--fused-stages > 0`; can be switched on when resuming. `local` also leaves
  out the coarsest (stride-16) lateral, so each location sees about 100px
  (needs `--fused-stages >= 2`); `full` is the whole map classification pools.
  Omitted, a resume keeps the checkpoint's setting (a fresh run uses `full`).
- `--mosaics-per-step` (default 0 = off): each step, this many canvases of four
  of the batch's images, each shrunk 2x into one tile of a 2x2 grid. Every tile
  keeps all its tags, and its tags are negatives in the other tiles, so the
  localization loss learns each tag in its own tile. Costs a forward/backward
  pass per mosaic (2 mosaics: about +28% step time). Needs deliberate targets,
  `--localization-weight > 0` and at least 4 images per batch.
  `--mosaic-weight` (default 1) scales it relative to `--localization-weight`.
- `--crop-consistency-weight` (default 0): the classifier's answer on a random
  crop teaches the localization map where a tag is. Gated to tags with held-out
  AP ≥ `--crop-teacher-min-ap` (0.8), ≥ `--crop-teacher-min-positives` (20) and
  context AP ≥ 0.8 in the latest epoch evaluation, and to confident answers
  (`--crop-teacher-confidence` 0.9). Off in run 2 until enough tags qualify.
- `--localization-background-weight` (default 0): background suppression.
  Tested; it shrank heatmaps without moving peaks.
- `--gold <gold.json>` scores the gold set and the crop teacher at every epoch
  evaluation; every epoch's reports are archived in `history/epoch-N/`.
- `--max-steps N` stops after N steps and saves (for short comparison runs).

**Status panel.** In a terminal, `train` shows a full-width panel: phase,
epoch and whole-run progress with time left and finish times, speed, loss
(this step and a 1,000-step mean), learning rate, checkpoints, the last
evaluation and recent events. Below that are line charts: the loss over the
whole run (2,000-step means, saved to `loss_history.json` with each checkpoint
so a resume keeps it), and, once evaluations exist, held-out mAP, gold
pointing, "blanked object still called", the two context gaps and
exclusive-tag violations at every epoch and snapshot evaluation (see
`--snapshot-evaluation-steps` below), each with its latest change. With output
redirected it prints a one-line summary every 5 minutes.

`--validation-max-images` caps how many held-out images that uses
(default 0 = all). To measure any existing checkpoint the same way, and
calibrate its thresholds without training:

```sh
dotnet run --project unbooru-tagger-training -c Release -- evaluate \
  --checkpoint-dir ./checkpoint --cache-dir ./data/large-cache --max-images 20000
```

`evaluate` also measures localization (`localization_report.json`).
Label-free checks on `--localization-samples` (default 300) held-out
images:
- **Deletion:** does blanking a tag's top heatmap cells lower its
  confidence more than blanking random cells?
- **Two characters:** on images tagged with exactly two characters, do
  their heatmaps peak in different places?
- **Crop consistency:** does a crop's heatmap match the same crop of the
  full heatmap?
- **Solo counts:** on `solo` images, do once-per-person tags yield exactly
  one box?

With `--gold <gold.json>`, it also scores against hand-checked boxes:
pointing accuracy, heatmap energy inside the boxes, and box
precision/recall/count error. It also runs an **occlusion** check: blank a
found tag's gold boxes and see whether the classifier still calls it,
against a control that blanks the same sizes elsewhere.

Two more checks of whether tags are found from the object or from context
go into `evaluation_report.json` and are printed by `evaluate`:
- **Context reliance:** each held-out image's prior for a tag is the highest
  P(tag | its character/artist/series) in training. Recall and
  false-positive rate are compared across prior bands; gaps near 0 mean the
  object is what's seen.
- **Exclusive violations:** which mutually exclusive pairs fired together,
  and which side the image actually carries. `--embeddings-cache <file>` saves held-out
embeddings so re-evaluating the same checkpoint skips the slow embedding
pass.

**Data loading.** With `--cache-dir`, batches are sampled and loaded ahead
of the training loop on a background thread, so disk reads and decoding
overlap GPU compute. Each window of `--prefetch-batches` (default 8)
batches is read in one burst: images are deduplicated, sorted by file
offset, and read with `--io-parallelism` (default 32) reads in flight,
then decoded in parallel straight into the batch buffers. A striped
spinning-disk array only delivers its throughput when every disk has work
queued. On the 6-disk RAID5 this was measured at ~48 images/s for the old
one-read-at-a-time path vs. ~450–590 images/s for this loader. Validation
passes and `evaluate` use the same loader. The status panel reports
ms/step and how much of it was spent waiting on data. If that wait isn't
near zero, raise `--io-parallelism` or `--prefetch-batches`.

`--gradient-accumulation-steps` (default `1`, disabled) — accumulates
gradients over this many `--batch-size` micro-batches before each
optimizer step, for an effective batch of `batchSize * this` without
needing enough memory to actually fit that many images at once. Useful
when memory caps `--batch-size` lower than you'd like: a larger effective
batch means more positive rows per optimizer step in the sigmoid-
contrastive label matrix (see the loss weighting note below), which
matters more the smaller `--batch-size` is forced to be.

`--localization-tag-chunk-size` (default `4096`) — the localization loss
(below) scores every spatial location against the whole tag vocabulary at
once by default, which is the single largest memory user at a
large-vocabulary scale (a `[batch, vocabSize, locations]` tensor). This
processes the vocabulary in chunks of this size instead, backpropagating
each chunk immediately so its memory is freed before the next chunk's
forward pass — exact, not an approximation (each tag's pooling only ever
depends on its own locations, never other tags, so this is just a
different order of the same sum). Lower uses less memory, at some
per-chunk overhead cost from smaller matmuls; raising it toward or above
the vocabulary size approaches the unchunked memory profile.

Two auxiliary, label-free training objectives are on by default, each
individually disableable by setting its weight to `0` (the
self-supervised term additionally skips its extra forward passes
entirely when disabled):

- `--localization-weight` (default `0.1`) / `--localization-temperature`
  (default `0.35`) — an MIL/log-sum-exp pooled loss over every spatial
  location instead of one pooled embedding, so sharp localization is
  directly rewarded rather than left as a side effect of pooling. Lower
  temperature concentrates gradient on a tag's single best-matching
  location (more max-like); higher temperature smooths back toward the
  main loss's plain average-pooling behavior (mathematically identical
  in the limit).
- `--self-supervised-weight` (default `0.1`) — a SimSiam-style
  consistency loss between two random-crop-augmented views of the same
  image, predicting each other's pooled embedding through a small
  predictor head with stop-gradient on the target side. Uses no tag
  labels, just images already in the corpus.

**Add a new tag** without retraining the whole model — creates the new
tag's embedding row and fine-tunes only that row with the image encoder
frozen:

```sh
dotnet run --project unbooru-tagger-training -- add-tag hatsune_miku_nurse_costume \
  --checkpoint-dir ./checkpoint \
  --images ./new-tag-manifest.json \
  --steps 300
```

By default the new row starts from small random noise. Pass
`--warm-start-model <path-to-onnx>` and `--warm-start-vocab
<path-to-vocab.txt>` (a standard BERT-family WordPiece vocab file) to
instead start it from a frozen text-embedding model's encoding of the
tag string — underscores are treated as spaces
(`hatsune_miku_nurse_costume` → `hatsune miku nurse costume`) before
tokenizing. The model's output dimension must exactly match
`--embedding-dim`/the checkpoint's own embedding dimension; there's no
projection step, so a mismatch fails immediately with a clear error
rather than silently truncating/padding. The same two flags also apply
to `train`'s fresh-vocabulary path (a brand-new `--checkpoint-dir` with
no existing checkpoint), for every tag's initial row instead of just one.

The tag is promoted from warm-start-only to fully trained once it has
seen `--min-image-threshold` images (default 15).

**Export to ONNX** for inference:

```sh
dotnet run --project unbooru-tagger-training -- export-onnx \
  --checkpoint-dir ./checkpoint \
  --model-dir ./model
```

Produces the `model/` bundle described above, ready for
`unbooru-tagger-inference`. If the checkpoint has `exclusive_tags.txt` or
`tag_thresholds.json` (from `train`/`evaluate`), they're copied into the
bundle. With them, inference uses each tag's calibrated threshold unless
`--threshold` is passed explicitly, and drops the weaker of any two
mutually exclusive tags.

### Training on an Intel GPU (XPU)

The stock `TorchSharp` package runs on Intel Arc GPUs against the libtorch shipped in PyTorch's
`torch==2.10.0+xpu` pip wheel. 2.10.0 is the libtorch TorchSharp 0.107.0 is built against, and
the C++ ABI matches, so no native rebuild is needed.

1. Download that wheel and its Intel runtime dependencies from
   `https://download.pytorch.org/whl/xpu`, unzip them, and copy every
   `torch/lib/*.so*` and `*.data/data/lib/*.so*` into one folder. No oneAPI install is needed;
   the system needs the `xe` driver, `libze-intel-gpu1` and `libze1`.
2. Build with `dotnet build unbooru-tagger-training -c Release -p:TorchBackend=xpu`. This
   references `TorchSharp` alone, with no bundled libtorch.
3. Run with that folder on `LD_LIBRARY_PATH`, and with these variables set, so the Intel
   runtime can't silently migrate allocations that don't fit in VRAM into host RAM. Without
   them, over-allocating thrashes over PCIe instead of failing:
   ```sh
   export NEOReadDebugKeys=1 EnableSharedSystemUsmSupport=0 EnableImplicitMigrationOnFaultableHardware=0
   ```

`DeviceSelector` picks CUDA, then XPU, then CPU. On XPU it enables oneDNN's TF32 mode for
convolutions (`OneDnnPrecision`), matching cuDNN's default on CUDA. That took the ConvNeXt-T
tower from 203 to 138 ms/step. An XPU out-of-memory aborts the process rather than throwing a
.NET exception, so you resume from the last checkpoint.

### Offloading epoch evaluation (`--evaluation-command`)

`train --evaluation-command <exe>` runs each epoch's evaluation in another process instead
of inline, so training keeps going. A typical use is `evaluate` on a second GPU; it has to be
a separate process because a CUDA and an XPU libtorch can't share one. At each epoch end the
trainer writes a model-only snapshot to `history/epoch-N/checkpoint/` and starts
`<exe> <snapshot dir> <history/epoch-N>`. The command must write the reports there and exit 0:

```sh
#!/usr/bin/env bash
exec dotnet unbooru-tagger-training.dll evaluate --checkpoint-dir "$1" --out-dir "$2" \
  --cache-dir ./dataset --max-images 20000 --localization-samples 0 --gold gold.json
```

`--max-images` and `--validation-fraction` must match the training run's
`--validation-max-images`/`--validation-fraction`.

Results are applied in epoch order at the next checkpoint save: early stopping, plus the
top-level `evaluation_report.json`/`tag_thresholds.json` copies that `export-onnx` reads. That
makes early stopping lag by up to an epoch. The best epoch's weights stay in its snapshot.
The final epoch waits for its evaluation. Each `history/epoch-N/` also keeps
`evaluation.log` and `evaluation.done`/`.failed`/`.applied` markers. A restarted trainer
applies a finished evaluation it hadn't applied yet, and reruns one that never finished.

`--snapshot-evaluation-steps N` (default 0 = epoch ends only; needs `--evaluation-command`)
also snapshots the model every N global steps and runs the same command on it in the
background, for the status panel's trend charts. Reports go to `snapshots/step-N/`, and the
snapshot's weights are deleted once it's evaluated. They never affect early stopping. On a
resume, finished ones are reloaded onto the charts, unfinished ones restart, and ones past
the resumed step are deleted.

### Checkpoint directory contents

```
checkpoint/
  image_tower.dat       # TorchSharp native weights (not ONNX)
  model_config.json
  tag_vocabulary.json
  tag_embeddings.bin
  logit_scale_bias.dat  # learned logit scale/bias -- missing on an older checkpoint, starts fresh on resume
  training_progress.json (+ optimizer state, for resume)
  logit_scale_bias_optimizer.dat  # separate optimizer state, so resuming an older checkpoint can't disturb the main one's momentum
  exclusive_tags.txt     # mutually exclusive tag groups training used (copied into the model bundle)
  tag_thresholds.json    # per-tag calibrated thresholds from the latest evaluation (copied into the model bundle)
  evaluation_report.json # latest held-out per-tag report, worst tags first
  loss_history.json      # loss means for the status panel's chart, reloaded on resume
  history/epoch-N/       # each epoch's reports (and, with --evaluation-command, its snapshot and markers)
  snapshots/step-N/      # --snapshot-evaluation-steps reports
```

## Model artifacts are not checked into git

`.gitignore` excludes `*.onnx`, `*.pt`, and `*.bin` (except test
fixtures). Checkpoints and exported model bundles are expected to be
trained locally or downloaded from
[Releases](https://github.com/da3dsoul/unbooru-tagger/releases) — never
committed.

## License

No license file is currently included in this repository.
