# Mistakes and lessons

This page records what went wrong, so it isn't repeated. It is grouped by kind. Each entry gives what happened,
what it cost (where recorded), the fix, and the lesson.

## Data incidents

| What happened | Cost | Fix | Lesson |
|---|---|---|---|
| **Vocabulary corruption** (2026-08-04). A crash mid-crawl truncated `tag_vocabulary.json` to 0 bytes: `TagVocabulary.Save()` was a bare `File.WriteAllText`. | 263 of 2,611,746 images lost their tags. The vocabulary rebuilt to 15,299 tags against the 6,341 an earlier checkpoint (`checkpoint-medium`) was trained on, so that checkpoint could no longer be resumed as is. Whether to extend it with `add-tag` or start fresh was left open in the notes. | Atomic save (temp file and rename); a `rebuild-vocabulary` command that replays `crawl.sqlite`, seeded from a known-good older vocabulary (row indices are append-only, so an older vocabulary is a correct prefix). | Write every state file atomically. |
| **Gelbooru silent search redirect** (2026-08-06). `tags=water_fountain` returned exactly `fountain`'s posts, in no alias listing anywhere. | A tag gets many pages fetched for near-zero counted progress. | Trust the site's own search: fold the searched tag into every post it returned. Later, `detect-deprecated-redirects` finds such pairs by exact post-id sequence matches. | An alias table can be permanently incomplete; verify with the search itself. |
| **Damaged pixel cache** (2026-08-24). A `prune-unsupported-media` compaction reported success at the right row count but wrote empty 0x0 records for part of the corpus. | "Millions of rows" of pixels lost (exact count not recorded). | `recover-corrupted-images`: copy what is valid, re-download the rest, MD5-verified against the recorded identity, in a staging directory; write-side verification added. | A row count is not a verification. |
| **Crawler checkpoint ordering** (2026-07-21). The dedup index could commit ahead of an unflushed cache file. | A crash would leave the index pointing at rows that no longer existed: silent, permanent image loss. | Commit the cache flush, the vocabulary delta, the dedup writes and the cursor together once per page. | Commit related state together. |
| **Gelbooru survey order and pagination.** `orderby` without `order=DESC` gave an unspecified order; paging by raw offset is unsafe on a growing site. | Broken early exit; skipped or repeated posts. | `order=DESC`; anchor pages on post id. | |
| **GIFs read as artwork.** SkiaSharp silently decodes a GIF's first frame. | Bad images in the corpus. | Skip video and GIF URLs; `prune-unsupported-media` for old ones. | |

## Model and training bugs

| What happened | Fix |
|---|---|
| The pooled embedding was projected into the embedding space, but the spatial map wasn't, so heatmap dot products were meaningless (2026-07-17). | `Projection` is a 1x1 conv applied before pooling; the pooled output is identical because a linear map commutes with average pooling. |
| NaN loss at step 1: residual blocks compounded a near-zero GroupNorm variance. | LayerScale, a larger eps and pre-norms. The remaining rare cliff was handled by failing fast on NaN, after a probabilistic regression test proved flaky and was removed. |
| Training ran only on the CPU: no device was ever passed to the layers. | A `Device` parameter threaded through; `DeviceSelector`. |
| Checkpoints were saved only at the end of a run, and the first resume didn't read them back. | Per-epoch and then 15-minute checkpoints; full resume, including optimizer state. |
| GroupNorm affine parameters were left uninitialized by TorchSharp (NaN on the CPU, zeros on CUDA). Run 2 was created before the fix. | Initialized to 1/0; run 2 was judged recoverable and kept. |
| Inference assumed a 224px input. | Read the size from the ONNX graph. |
| **Ratings never reached training**, and the first version's likely-missing rule zeroed most "at least sensitive" negatives, so the label fired on every image. | Ratings are treated as exhaustively known: an absent rating label is a certain negative. |
| **A wrong default exclusive rule:** `solo, no_humans`. At least 54 held-out images carry both. | Removed. |
| **Full-strength context negatives on people counts** doubled 1girl + 2girls contradictions. | Limited to localizable tags. |
| **Batch 7's "exclusive" rule was wrong.** `TagStatistics` keeps only a tag's 8 strongest context sources (`.Take(8)`), so a gloves image of glove-wearing character #9 counted as exclusive. | Exclusive = a known character/artist/series with prior < 0.2. The mis-specified arm (split3) is recorded as such. |

## Evaluation bugs

Details are in [experiments-measurement.md](experiments-measurement.md).

- **`evaluate` fell back to validation seed 0** when `optimizer.dat` was missing. That seed overlaps training
  images. Fixed 2026-10-24.
- **The validation split wasn't persisted** before 2026-09-01, so a resume could leak validation images into
  training.
- **Non-reproducible seeds** (`HashCode.Combine`): two evaluations of one checkpoint differed by about 0.001 mAP.
- **Box specks** in the gold checks: box precision 0.1-0.3% and count errors of 100-190 in every run until
  2026-11-01.
- **Whole-canvas pooling** in the gold checks until 2026-10-29: the object effect is about 0.15 lower than under
  the content-only protocol.
- **Gold images are training images.** Only 30 of the 289 main gold images are in run 4's held-out split. This was
  recognized while building gold-small (2026-10-28).
- **Evaluation memory:** a forward held all intermediates, so gold scoring used 13.2 GB and ran the A4000 out of
  memory.
- **Context AP as a context metric:** its chance level was about 0.98 on a 20K sample.
- **Thresholded occlusion rate:** 7 points of noise between identical controls.
- **The gold-auto ignore boxes** excused real misses. Dropped from the recipe.

## Artifacts mistaken for gains

| Apparent gain | Real cause | How it was caught |
|---|---|---|
| Locality negatives doubled "best glimpse on the object" (31.9% → 60.6% medium). | The term only penalized close windows, so the evidence moved into the 256px window, which overlaps most objects. The funnel's chance level was over all windows, not over the winning zoom. | Breaking the winning glimpse down by zoom: 256px wins went 5-12% → about 50-100%. |
| Zero-fill erasure: object effect 0.87. | The model learned "a grey patch lowers a score". | Re-scored with an unseen noise fill: 0.37, below the controls. |
| hinge5's control drop looked fixed at 6K (0.12). | The patch cue came back with training (0.67 at 12K, 0.31 at 18K). | Extending the arm. |
| Rising context false-positive gap in run 4 (20.4% → 24.6%). | Mostly untagged objects the model finds. | Hand-checking 180 high-prior negatives; the labelled context check. |
| Channels-last "gave different outputs" (2026-10-28). | TorchSharp's `manual_seed` doesn't seed the XPU generator, so the variants got different random weights. | Creating weights on the CPU. |
| The pooled evaluation path was suspected of the mAP drop. | It was correct (T = 1000 equals the average); the drop came from training. | A direct check on 2,000 images. |

## Method mistakes

- **The first glimpse run.** The labels picked its training windows, and evaluation scored a different 378-window
  grid. The model never learned where to look, and a day of B70 time was lost. The run was deleted.
  - Lesson: training and evaluation must run the same forward, and design choices must be measured before a
    multi-day run depends on them.
- **Calling a VRAM leak fixed too early.** The locality term was declared fixed after 11 flat minutes; it crashed
  at 47.
  - Lesson: soak-test a change for more than an hour.
- **Reading a comparison script backwards.** `compare.py` prints arrows as "other → first argument", and a
  regression was misread (halo).
  - Lesson: list the run under review first.

## Operational mistakes

These are recorded in the memory notes; they cost training time rather than correctness.

- **Killing a trainer mid-step on the Arc** left a copy-engine fault, and the next load hung for 7 hours. Ctrl+C
  now finishes the step and saves.
- **Unattended A/B scripts:**
  - An overnight A/B script stopped run 5 and paused the watchdog for its whole length. Its waiter matched only
    success strings, so a device out-of-memory and a copy-engine hang went unnoticed for 7 hours.
  - Lesson: never leave the watchdog paused; every waiter must fire on failure and on stalls; any script that
    stops the main run must restart it on every exit path. The watchdog now has a callback that reports each
    restart for diagnosis.
- **Editing a running bash script** killed it: bash reads scripts as it goes, so ab9's comparison step never ran.
- **Unverified launches:**
  - a dump directory that didn't exist yet;
  - a script without `+x`;
  - a `dotnet build` under `flock` without `--disable-build-servers`, where the compiler server inherited the lock
    and an evaluation waited about 5 minutes.

  Lesson: check within a minute that a launched job is progressing, and smoke-test new code paths before an A/B.
- **Killing processes by name pattern** nearly hit an unrelated server. Kill by exact PID.
- **Scratch builds in `/tmp`** (a tmpfs, so RAM) added to the host memory pressure that slowed run 4 to 3 s/step.
- **Closing the maintainer's tmux windows** on restarts detached them from what they were watching. Restart in
  place with `respawn-pane`.
- **Tests that hang forever:** a fake HTTP handler that returns the same non-empty page makes any page-until-empty
  client loop forever. This recurred once in the same session.
