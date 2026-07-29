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
  row, without touching the rest of the model. The row is meant to
  warm-start from a frozen text embedding of the tag string (see
  `ITagTextEmbedder` in `unbooru-tagger-core`), but no CLI flag wires a
  concrete model into that path yet — new tags currently start from small
  random noise instead.
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
annotated PNG with colored boxes and tag/confidence labels.

Every command accepts `--model-dir` (default `./model`) and (except
`detect`, which has its own `--box-threshold`) `--threshold` (default
`0.5`) for the minimum tag confidence to report.

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
(model weights, optimizer state, epoch count, and early-stopping history
if present). A checkpoint is saved after every epoch. Uses CUDA if
available, CPU otherwise. Key flags: `--embedding-dim` (default 512),
`--lr` (default 1e-4), `--validation-fraction` (default 0.1),
`--early-stopping-patience` (default 3).

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
tag's embedding row (random-init today, see the Features note above) and
fine-tunes only that row with the image encoder frozen:

```sh
dotnet run --project unbooru-tagger-training -- add-tag hatsune_miku_nurse_costume \
  --checkpoint-dir ./checkpoint \
  --images ./new-tag-manifest.json \
  --steps 300
```

The tag is promoted from warm-start-only to fully trained once it has
seen `--min-image-threshold` images (default 15).

**Export to ONNX** for inference:

```sh
dotnet run --project unbooru-tagger-training -- export-onnx \
  --checkpoint-dir ./checkpoint \
  --model-dir ./model
```

Produces the `model/` bundle described above, ready for
`unbooru-tagger-inference`.

### Checkpoint directory contents

```
checkpoint/
  image_tower.dat       # TorchSharp native weights (not ONNX)
  model_config.json
  tag_vocabulary.json
  tag_embeddings.bin
  training_progress.json (+ optimizer state, for resume)
```

## Model artifacts are not checked into git

`.gitignore` excludes `*.onnx`, `*.pt`, and `*.bin` (except test
fixtures). Checkpoints and exported model bundles are expected to be
trained locally or downloaded from
[Releases](https://github.com/da3dsoul/unbooru-tagger/releases) — never
committed.

## License

No license file is currently included in this repository.
