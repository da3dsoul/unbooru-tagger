# Data and evaluation sets

This page covers what data exists and how each piece was gathered. None of the data is in git. Everything below
lives in the maintainer's data directory, outside the repository, and is referred to by
its relative name (`dataset/`, `gold/`, ...). The code that reads or writes each file *is* in git, and is named
where it matters.

## The dataset and crawler

### Sources

The training corpus was crawled from **Danbooru and Gelbooru** by `unbooru-tagger-data-booru-downloader`. An
older importer, `unbooru-tagger-data-unbooru-import`, reads the maintainer's own `unbooru` SQL Server database
instead; it is not what the current corpus came from.

### How the crawl works

The command-level reference is in [docs/TRAINING.md](../docs/TRAINING.md#building-a-dataset).

1. **`survey-tags`** records each tag's post count on each site, and which tags are eligible.
2. **`crawl`** fetches up to `--max-images` posts per eligible tag, rarest tag first, across both sites. Each site
   runs its own worker and must contribute its fair share of each tag. A negative phase then tops up images
   without each tag.
   - The negative phase prefers **hard negatives**: images with a tag that commonly co-occurs with the target but
     lacks the target (for example `vocaloid -hatsune_miku`), mined from the corpus's own co-occurrence counts.
     Commit `56a9929` added this.
   - **Dedup** is an exact MD5 check, plus a 64-bit DCT perceptual hash (Hamming distance ≤ 2, calibrated against
     measured cross-site recompression noise). A duplicate post becomes an extra *source* of the existing row, and
     its extra tags are merged in.
3. **Tag identities are category-prefixed** (`character:frieren`, `series:...`, `artist:...`, `meta:...`;
   general tags bare).
   - Every `meta:` tag is excluded, except `*_(medium)` tags and a short curated list.
   - Danbooru's tag-alias table is merged in during survey.
   - `detect-deprecated-redirects` finds tags a site silently redirects. This is the Gelbooru search-redirect
     incident: see [mistakes-and-lessons.md](mistakes-and-lessons.md).

### The parameters of the last crawl

From `dataset/last-crawl-command.json`; credentials are omitted here:

| Parameter | Value |
|---|---|
| Sites | Danbooru, Gelbooru |
| Minimum posts for eligibility | 500 |
| Target images per tag | 1,000 |
| Input size | 512 |
| Rate limits | Danbooru 4 req/s, Gelbooru 2 req/s |
| Negative target | 1,000 |
| Hard negatives | co-occurrence ratio 0.5, at least 15 counter-examples, up to 3 sources |

### What's on disk

From a directory listing on 2026-11-03:

| File | Size / count | Content |
|---|---|---|
| `dataset/images.bin` | 3.10 TB | Preprocessed pixels, uint8, content region only. The storage notes give about 5.39M records at about 576 KB average. |
| `dataset/crawl.sqlite` | 8.5 GB | Survey results, per-tag/site progress, per-source tag snapshots, dedup index. |
| `dataset/tag_rows.sqlite` | 0.88 GB | Packed per-image tag-row indices. |
| `dataset/tag_vocabulary.json` | 24,899 rows | The vocabulary. A fresh run drops the rows no image carries; run 2 went 24,902 → 18,434, the 24,902 including 3 rating rows. |
| `dataset/ratings.bin`, `image_sites.bin` | 5,388,027 bytes each | One byte per cache row. |
| `dataset/tag_implications.json`, `tag_agreement.json`, `tag_aliases.json`, `detected_tag_redirects.json`, `excluded_tags.txt` | small | See below. |

### Storage throughput

`images.bin` lives on a 6-HDD RAID5 behind bcache. Measured 2026-10-21 with parallel `pread`:

| Access pattern | Throughput |
|---|---|
| random, 1 in flight | ~85 img/s (~50 MB/s) |
| random, 32 / 64 in flight | ~590 / ~690 img/s |
| random sorted by offset, 32 in flight | ~940 img/s |
| sequential 16 MB reads x8 parallel | ~800 MB/s (~1,370 img/s) |

- The old one-read-at-a-time loader managed about 48 img/s.
- `BatchPrefetcher` with offset-sorted reads, 32 in flight, gets about 450-590 img/s, read plus decode.
- With prefetching, a training step waits about 1 ms on data.

### Corrections derived from the crawl data

These are what made the labels usable.

- **Content ratings** (`export-ratings` → `ratings.bin`):
  - Each image gets the strictest rating across its sources.
  - Training learns three cumulative labels (at least sensitive / at least questionable / explicit). They are
    treated as exhaustively known: an absent rating label is a certain negative.
  - Their cutoffs are calibrated for recall (90% / 97% / 97%), not F1.
  - Before this, ratings never reached training. In the first version, the likely-missing rule zeroed most "at
    least sensitive" negatives, and it fired on every image.
- **Danbooru implications** (`export-implications` → `tag_implications.json`):
  - 46,895 active rules. Usable ones (both tags in the vocabulary, matched on raw names with the category prefix
    stripped): 6,625 in run 2's vocabulary (witch_hat → hat), 6,627 in run 4's, 6,630 in the dataset's full
    24,899-tag vocabulary. The two figures in the notes (6,625 and 6,630) are these counts against different
    vocabularies, rechecked 2026-11-03.
  - Danbooru applies implications to every post; Gelbooru doesn't. So training completes each image's implied
    tags. On run 2 this added 523,867 labels on 507,666 images.
- **Site coverage** (`export-image-sites` → `image_sites.bin`; `export-tag-agreement` → `tag_agreement.json`):
  - Images by site: Danbooru-only 1.18M, Gelbooru-only 1.33M, both 2.87M.
  - Single-site tags, by the rule training uses (`TagAgreement.LoadSiteCoverage`: at least 200 uses on the
    images both sites have, and under 2% of them on one site): 925 Gelbooru-only (silver_hair, black_legwear,
    `*_footwear`, ...) and 25 Danbooru-only, 950 in all (`tag_agreement.json`, recounted 2026-11-03; the run-2
    notes give "~950" and "949"). `CLAUDE.md` gave "1,136: 993 Gelbooru-only and 130 Danbooru-only"; that
    figure doesn't add up and doesn't reproduce under this rule or under "never listed by one site" (1,790 +
    60), so it is treated as an error.
  - An image from the other site can never carry such a tag, so those pairs get weight 0, are never sampled as
    negatives, and are left out of evaluation.
  - **The agreement counts are not tagging recall.** Gelbooru largely copies Danbooru, so agreement is 96-99% for
    nearly every tag.
- **Missing labels, measured from `crawl.sqlite`** (2026-10-27/28, images on both sites):
  - Gelbooru leaves gloves, hat, earrings and similar tags off about 3% of the time Danbooru has them (shoes 14%).
    That is about 1% of those tags' negatives.
  - Two-tag conditionals did not predict which labels are missing (AUC about 0.66, the same as single-tag).
  - This supported lowering the focal gamma from 4 to 2.

### Tag parentage

- **Why:** Danbooru's meanings differ from the maintainer's. On Danbooru, `shoes` means non-boot shoes: only 9%
  of boot images carry it, and 17% of geta images. There is no `footwear` tag, and `hair_ornament` excludes
  ribbons, bows and hairbands. So boot images trained as weak `shoes` negatives.
- **How the rules were made** (`tag-parentage/`): 346 rules were drafted from tag-name suffixes, hand-written
  families and Danbooru tag groups, each with evidence from a 10% sample. The maintainer reviewed them on a web
  page. Accepted: 266 is-a, 12 implies, 17 excludes. Left out as unsure: mittens, kote and oven_mitts > gloves,
  choker > jewelry, tabi > shoes.
- **The `no_X` tags:** the maintainer pointed out that `no_X` means the character usually has X and this image
  shows them without it. That makes these images ideal certain negatives for X: the context says X, and the
  object is absent.
- **Training** (`--tag-parentage`, run 4 from step about 502K):
  - Kinds complete their parents: boots → shoes.
  - A parent with no kind named leaves the kinds unknown (weight 0).
  - Excludes rules give certain negatives, which are sampled first as hard negatives.
  - The rules were switched on directly, without an A/B, at the maintainer's choice.

## The gold set (`gold/`)

- **Size:** 289 validation images (164 clean, 125 messy) with **certain, group and ignore** boxes. Ignore and
  group regions use COCO crowd/ignore semantics.
- **Rules:** `gold/tools/LABELING.md`. Every visible instance of every target tag is boxed, and physically
  separate objects are separate instances.
  - `certain` means sure, to within a few pixels.
  - `group` is a cluster that can't be boxed one by one.
  - `ignore` means "might be", and counts neither way. "A wrong certain box is worse than an ignore box."
- **Contents of `gold.json` now:** 1,333 targets; 1,717 certain, 44 group and 783 ignore boxes. No image is
  marked verified.
- **Who labelled it:** not recorded in the sources read. `CLAUDE.md` calls it "hand-checked". Its labelling
  procedure (grid images with pixel labels, a JSON per image) is the same one gold-small's Claude subagents
  followed.
- **Caveat:** picked from run 1's validation split. Only 30 of the 289 images are in run 4's held-out split, so
  for run 4 and run 5 most gold images are training images (their tags, not their boxes).
- **Uses:** `evaluate --gold` scores it, and `render-gold` plus the generated `gold/compare.html` show model
  boxes over the images.

## Gold-small (`gold-small/`)

- **Size:** 126 images from run 4's held-out split (validation seed 2069560488). General rating, clean, ≥ 320px,
  none from the main set. 18 small-object tags x 7 (4 full-body or wide, 3 cowboy shots).
- **Labelling:** Claude subagents under `gold-small/tools/LABELING.md` (a pilot of 5, then 8 batches): 366
  certain, 15 group and 99 ignore boxes at labelling time.
  - Median longest side of a certain box: earrings 12px, ring 10px, hairclip 17px, gloves 62px, shoes 48px.
- **Review:** the maintainer reviewed every image on a LAN review page on 2026-10-28: 114 verified, 8 excluded,
  4 neither. The non-excluded images hold 515 certain / 12 group / 2 ignore boxes (the figures in the memory
  notes); counting the 8 excluded images too, the file holds 520 / 12 / 26 (both counts from `gold.json`,
  2026-11-03). They are the same file counted two ways, not an edit.
- **Decision:** the maintainer kept it a **separate** set, scored automatically in every evaluation
  (`evaluate --gold-small` → `localization_report_small.json`).
- **First results:** gold-small pointing went 7.8% (run 4 epoch 1) → 23.2% (550K), and small objects (< 32px)
  4.1% → 10.6%. At 550K, earrings, hairclip, hair_bow, halo, bracelet and wristwatch had zero pointing.

## Gold-auto: SAM 3 boxes (`gold-auto/`)

- **Purpose:** the maintainer asked for better localization evaluation data that doesn't depend on hand
  labelling for its bulk.
- **Images:** 1,020 from run 4's held-out split, in neither gold set: 34 object tags x 30 (40% full body, 30%
  cowboy shot, 30% upper body or portrait; one in five with two people). That is 3,212 (image, tag) targets.

### How the box source was chosen

Every option below was calibrated blind against gold-small's verified boxes. "Found" is instance recall and
"right" is precision, both at IoU ≥ 0.5:

| Box source | Found | Right | Note |
|---|---|---|---|
| OWLv2 base + Grounding DINO base, score ≥ 0.2 | 66% | 27% | ≥ 0.3: 55/46; ≥ 0.4: 42/69; matched-box IoU 0.83 |
| OWLv2 large, ≥ 0.05 | 80% | 25% | |
| OWLv2 large + DINO, ≥ 0.05 | 83% | 18% | |
| Qwen2.5-VL 3B as a detector | 37% | 51% | loose boxes |
| Qwen 3B as a yes/no verifier of proposals | 10% | 33% | at P(yes) ≥ 0.7; can't judge box fit |
| Claude check of proposals, Opus (20 images) | 82% | 91% | about 15-18K tokens per image |
| Claude check, Sonnet | 67% | 86% | |
| Claude check, Haiku | 49% | 78% | |
| **SAM 3 (facebook/sam3), ≥ 0.5** | 67% | 74% | IoU 0.84-0.88 |
| **SAM 3, ≥ 0.7** | 55% | 88% | |
| **SAM 3, ≥ 0.8** | 43% | 95% | |

- The Claude check would have cost about 18M tokens for the full set, which the maintainer judged too expensive.
  SAM 3 was "best by far" of the local options.
- **Recipe:** SAM 3 boxes with score ≥ 0.7 as `certain`, and no ignore boxes (`tools/make_gold.py ... --certain
  0.7 --ignore 0.7`). The resulting `gold.json` holds 2,824 certain boxes; the scored pairs number 2,718.
- **Agreement check** (`agree/`, run 4's ctl4 scored on gold-small's images):

  | Box source | Pointing | Object effect | Pairs |
  |---|---|---|---|
  | The maintainer's boxes (34 tags) | 52.3% | 0.75 | 516 |
  | SAM ≥ 0.7, with 0.1-0.7 boxes as ignore | 66.5% | 1.10 | 315 |
  | The maintainer's boxes on just those pairs | 59.4% | 0.93 | |
  | SAM ≥ 0.7, no ignore boxes | 60.4% | 0.93 | |

  - SAM's boxes score like the maintainer's on the same pairs.
  - The ignore boxes excused real misses, so they are not used.
  - What remains is selection: SAM keeps the easier pairs, which every checkpoint shares.
  - The ranking across very different checkpoints held.
- **Status:** used as "the least noisy localization readout" in the 2026-11-03 A/Bs.

Checking the boxes with Claude subagents was the original plan (`gold-auto/tools/CHECK.md`). It was dropped once
the cost was measured (about 15-18K tokens an image, about 18M for the set), and the README in `gold-auto/`
records the switch to SAM 3.

## Hand-checked context negatives and the labelled context check

### The question

The context-reliance check counts as a false positive any call on a "context negative": an image whose
character, artist or series carries the tag 60-90% of the time, but which isn't tagged with it. Is such a call a
real false positive, or an untagged object?

### The maintainer's set (`context-negatives/`, 2026-10-30)

- 180 held-out images: 36 each for gloves, hat, glasses, wings and tail.
- Labelled by eye: 27 present (visible, a missing label), 50 unclear (crowds, comics, tiny figures), 103 absent.
- So at least 15%, and up to about 40%, of these "negatives" carry the object.

### Grown with a Sonnet labeller (`ctxfp/`, `context-negatives-2/`, `-3/`)

- **Calibration** on the maintainer's 180:
  - Of the maintainer's 103 absent, the Sonnet subagent marked 37 absent, 66 unclear, 0 present.
  - Of the 27 present, it marked 23 present and 4 unclear.

  So its absent and present labels are reliable but conservative.
- **Labelled:** 360 more pairs from a 100K pool (68 absent / 61 present / 231 unclear). Then all the remaining
  high-prior negatives inside the first 20K held-out images for gloves, hat and glasses (86 absent / 54 present /
  141 unclear).

### The combined set (`context-check/context-check.json`)

- 257 absent and 142 present pairs. The maintainer's labels win on overlap.
- 197 absent and 88 present fall in the first 20K held-out images, which evaluations score.
- `evaluate --context-check` reports:
  - how often a tag is called where the object is absent (true context false positives);
  - how often it is called where the object is visible but untagged;
  - the mean log-odds past the threshold, which is what the charts show.
- Older reports were backfilled from their `scores.jsonl`.

## Missing labels with SAM 3 (`missing-labels/`)

- **Set-up** (2026-11-03): the first 3,000 held-out images in evaluate's order, plus the labelled context check's
  images. That is 3,243 exported images.
- **SAM 3 pass:** all 34 object tags per image (`tools/sam3_all.py`: image encoded once, text once; bf16 1.5
  s/image against 4.4 fp32).
- **Planned analysis** (`analyse.py`):
  1. SAM 3 against the labelled context check;
  2. SAM 3's recall on labelled objects;
  3. the untagged-but-there rate per tag (SAM ≥ 0.8);
  4. the model's calls on SAM-confirmed untagged objects;
  5. precision and AP against raw labels versus labels completed with SAM's sure finds.
- **Results** (2026-11-03, ctl5 = run 4 at about 1.519M; 2,417 clean images x 34 tags; SAM "sure" = score >= 0.8):
  1. Against the labelled context check (285 pairs): 31 of SAM's 32 sure finds are labelled visible (96.9%), and
     it is sure on 0.5% of the 197 absent pairs. It is conservative: sure on 35% of the 88 visible-but-untagged
     pairs.
  2. Sure on 40% of the labelled objects (3,900). Some phrases fail outright: thighhighs 0%, ear_piercing 0%,
     hair_ribbon 4%, scarf 9%; those tags' rows say little.
  3. Untagged but there: 2,451 of 78,278 untagged pairs (3.1%), 0.63 per labelled object. Highest: horns 9.5%,
     animal_ears 8.8%, bracelet 8.7%, shoes 8.6%, bowtie 8.3%, hairband 8.0%. SAM's precision was only checked on
     gloves, hat, glasses, wings and tail, so the horns and animal_ears rates may include false finds (in the
     example image a hair ribbon scored 0.58 as horns, below "sure").
  4. The model calls 23% of SAM-confirmed untagged objects (560 / 2,451) against 1.5% of pairs where SAM finds
     nothing (< 0.3). By tag: shoes 70%, hat 58%, gloves 45%, boots 45%, choker 38%; horns 7%, animal_ears 11%,
     bracelet 13%. Mostly it follows the labels' conventions, not the pixels.
  5. Precision / recall / mean AP over the 34 tags at calibrated thresholds: raw labels 48.1% / 58.7% / 0.437;
     labels completed with SAM's sure finds 59.9% / 44.9% / 0.461. A fifth of what held-out precision counts as
     false positives here are real objects.
- **Example figure:** `missing-labels/example/example-3699786.png` (tagged objects green, sure untagged orange,
  unsure grey).

## The validation split

- **How it is drawn:** each run holds out `--validation-fraction` (0.1) of the cache by a seeded split.
  Evaluations score the first 20,000 held-out images.
- **Fixes:**
  - Before 2026-09-01, the split wasn't persisted, so a resume could leak validation images into training.
  - Before 2026-10-24, `evaluate` fell back to seed 0 when `optimizer.dat` was missing, which overlaps training
    images.
- **Run 5 reuses run 4's split** (`--validation-split-seed 2069560488`), so their charts compare at the same
  steps, and gold-small, gold-auto and the context check are held out for both.
