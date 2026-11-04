using System.CommandLine;
using UnbooruTagger.Training.Commands;
using UnbooruTagger.Training.Training;

var checkpointDirOption = new Option<string>("--checkpoint-dir")
{
    Description = "Directory holding the TorchSharp training checkpoint (image_tower.dat, model_config.json, tag_vocabulary.json, tag_embeddings.bin)",
    DefaultValueFactory = _ => "./checkpoint"
};

var warmStartModelOption = new Option<string?>("--warm-start-model")
{
    Description = "Path to a frozen ONNX text-embedding model used to warm-start a brand-new tag's row from its name instead of random noise. Must be given together with --warm-start-vocab; omit both for the old random-init behavior."
};
var warmStartVocabOption = new Option<string?>("--warm-start-vocab")
{
    Description = "Path to the WordPiece vocab.txt (one token per line, BERT-family format) matching --warm-start-model's tokenizer."
};

var manifestOption = new Option<string>("--manifest") { Description = "Path to the dataset manifest JSON (mutually exclusive with --cache-dir)" };
var cacheDirOption = new Option<string>("--cache-dir") { Description = "Path to a PreprocessedDatasetCache built by unbooru-tagger-data's build-large-cache (mutually exclusive with --manifest)" };
var embeddingDimOption = new Option<int>("--embedding-dim") { Description = "Tag/image embedding dimension", DefaultValueFactory = _ => 512 };
var inputSizeOption = new Option<int>("--input-size") { Description = "Square input resolution fed to the image tower (ignored for --cache-dir, which has its own baked-in size)", DefaultValueFactory = _ => 224 };
var epochsOption = new Option<int>("--epochs") { Description = "Maximum number of passes over the dataset (early stopping may end training sooner)", DefaultValueFactory = _ => 5 };
var batchSizeOption = new Option<int>("--batch-size") { DefaultValueFactory = _ => 16 };
var learningRateOption = new Option<double>("--lr") { DefaultValueFactory = _ => 1e-4 };
var validationFractionOption = new Option<double>("--validation-fraction") { Description = "Fraction of the dataset held out for early-stopping evaluation", DefaultValueFactory = _ => 0.1 };
var earlyStoppingPatienceOption = new Option<int>("--early-stopping-patience") { Description = "Epoch evaluations without an improvement in held-out mean per-tag AP (1 - mAP is what is tracked) before stopping", DefaultValueFactory = _ => 3 };
var localizationWeightOption = new Option<double>("--localization-weight")
{
    Description = "Weight of the per-location (MIL/log-sum-exp) auxiliary loss that directly rewards sharp spatial responses instead of leaving localization as a side effect of pooling. 0 disables it.",
    DefaultValueFactory = _ => 0.1
};
var localizationTemperatureOption = new Option<double>("--localization-temperature")
{
    Description = "Log-sum-exp pooling temperature for the localization loss: lower sharpens toward a max over locations, higher smooths toward plain average pooling",
    DefaultValueFactory = _ => 0.35
};
var selfSupervisedWeightOption = new Option<double>("--self-supervised-weight")
{
    Description = "Weight of the label-free SimSiam-style consistency loss between two augmented crops of the same image. 0 disables it (skips the extra forward passes entirely).",
    DefaultValueFactory = _ => 0.1
};
var checkpointIntervalMinutesOption = new Option<double>("--checkpoint-interval-minutes")
{
    Description = "How often to save a checkpoint mid-epoch, in addition to the checkpoint already saved at every epoch boundary -- an epoch can take multiple days, so this bounds how much training a crash can lose. 0 disables mid-epoch checkpointing.",
    DefaultValueFactory = _ => 15.0
};
var gradientAccumulationStepsOption = new Option<int>("--gradient-accumulation-steps")
{
    Description = "Accumulate gradients over this many micro-batches of --batch-size before each optimizer step, giving an effective batch of batchSize * this without needing more memory for a larger --batch-size. 1 (default) disables accumulation -- steps every micro-batch, exactly the old behavior.",
    DefaultValueFactory = _ => 1
};
var localizationTagChunkSizeOption = new Option<int>("--localization-tag-chunk-size")
{
    Description = "How many tags the localization loss (--localization-weight) scores against every spatial location at once. Scoring the whole vocabulary at once is the single largest memory user at this vocabulary's scale; processing it in chunks bounds that peak without changing the result (each chunk is backpropagated immediately, exact, not an approximation). Lower uses less memory at some per-chunk overhead cost.",
    DefaultValueFactory = _ => 4096
};

var samplingOption = new Option<string>("--sampling")
{
    Description = "How batches are built: 'tag-centric' anchors each batch on a few tags' positives AND hard negatives (images with a commonly co-occurring tag but not the anchor), drawing anchors uniformly over tags; 'rare-tag' is the original rare-tag-weighted random-image sampling.",
    DefaultValueFactory = _ => "tag-centric"
};
samplingOption.AcceptOnlyFromAmong("tag-centric", "rare-tag");
var positivesPerAnchorOption = new Option<int>("--positives-per-anchor") { Description = "Tag-centric sampling: positives drawn per anchor tag", DefaultValueFactory = _ => 2 };
var hardNegativesPerAnchorOption = new Option<int>("--hard-negatives-per-anchor") { Description = "Tag-centric sampling: hard negatives drawn per anchor tag", DefaultValueFactory = _ => 2 };
var targetsOption = new Option<string>("--targets")
{
    Description = "'deliberate' weights every (image, tag) pair by how certain the corpus is about it -- positives and verified negatives (anchor hard negatives, mutually-exclusive tags) at full strength, likely-missing tags ignored, other negatives damped by the asymmetric loss; 'legacy' treats every untagged tag as an equally certain negative.",
    DefaultValueFactory = _ => "deliberate"
};
targetsOption.AcceptOnlyFromAmong("deliberate", "legacy");
var negativeGammaOption = new Option<float>("--negative-gamma") { Description = "Deliberate targets: focal exponent damping unverified negatives (asymmetric loss gamma-)", DefaultValueFactory = _ => 4f };
var negativeClipOption = new Option<float>("--negative-clip") { Description = "Deliberate targets: an unverified negative predicted below this probability costs nothing (asymmetric loss margin)", DefaultValueFactory = _ => 0.05f };
var certainNegativeWeightOption = new Option<float>("--certain-negative-weight") { Description = "Deliberate targets: weight of verified negatives (anchor hard negatives, mutually-exclusive tags)", DefaultValueFactory = _ => 1f };
var impliedThresholdOption = new Option<double>("--implied-threshold")
{
    Description = "Deliberate targets: if tag t appears on at least this fraction of tag s's images (but below --missing-threshold), s is a context source for t: an image with s but without t is trained as an unverified negative and sampled for t as a context negative",
    DefaultValueFactory = _ => 0.6
};
var missingThresholdOption = new Option<double>("--missing-threshold")
{
    Description = "Deliberate targets: if tag t appears on at least this fraction of tag s's images, an image with s but without t is treated as a likely-missing label (ignored) rather than a negative. Set equal to --implied-threshold for the original behavior (no context band).",
    DefaultValueFactory = _ => 0.9
};
var contextNegativesPerAnchorOption = new Option<int>("--context-negatives-per-anchor")
{
    Description = "Tag-centric sampling: context negatives drawn per anchor tag (images with a tag that usually comes with the anchor, but without the anchor), for tags that have any.",
    DefaultValueFactory = _ => 1
};
var contextNegativeWeightOption = new Option<float>("--context-negative-weight")
{
    Description = "Deliberate targets: train context negatives (an image with a tag that usually comes with tag t, but without t) as full-strength negatives with this weight, instead of focal-damped unverified ones. Applies to every such pair in a batch, not just sampled ones, for localizable tags only (whole-image tags like 1girl are often just untagged). 0 = unverified (the original behavior).",
    DefaultValueFactory = _ => 0f
};
var contextPriorWeightOption = new Option<float>("--context-prior-weight")
{
    Description = "Deliberate targets: context-prior debiasing. During training only, add this times logit(prior) - logit(base rate) to localizable general tags' whole-image logits, where the prior is the highest P(tag | character/artist/series) over the image's own tags. Context then explains what it can, and the image tower learns the tag from the object. 0 = off.",
    DefaultValueFactory = _ => 0f
};
var exclusiveTagsOption = new Option<string?>("--exclusive-tags")
{
    Description = "File of mutually exclusive tag groups (one comma-separated group per line, e.g. 'solo, multiple_girls'). Omit to reuse the checkpoint's saved copy, or the built-in people-count/rating groups."
};
var tagParentageOption = new Option<string?>("--tag-parentage")
{
    Description = "Reviewed tag parentage rules (the tag-parentage review page's rules.json): kinds complete their parents (boots -> shoes), a parent with no kind named leaves its kinds unknown instead of negative, no_X rules X out. Saved into the checkpoint; omit to reuse the checkpoint's copy, 'none' to turn it off."
};
var validationMaxImagesOption = new Option<int>("--validation-max-images")
{
    Description = "Evaluate at most this many held-out images per epoch (the same fixed subset every epoch). 0 = the whole validation split.",
    DefaultValueFactory = _ => 0
};

var noRatingsOption = new Option<bool>("--no-ratings")
{
    Description = "Ignore the dataset's ratings.bin. By default content ratings are learned as three cumulative labels (rating:at_least_sensitive, rating:at_least_questionable, rating:is_explicit), which adds three vocabulary rows -- so a checkpoint trained before ratings existed can only be resumed with this flag."
};
var localizationMinedNegativesOption = new Option<int>("--localization-mined-negatives")
{
    Description = "With deliberate targets, the per-location loss trains only on localizable positives, certain negatives, and this many of each image's highest-scoring wrong tags -- not the whole vocabulary.",
    DefaultValueFactory = _ => 32
};
var localizationBackgroundWeightOption = new Option<float>("--localization-background-weight")
{
    Description = "Background suppression: for each tag an image has, its least tag-like locations (--localization-background-fraction) are trained as a negative, so a tag's heatmap can't stay high across the whole image. Weight relative to the localization loss; 0 = off.",
    DefaultValueFactory = _ => 0f
};
var localizationBackgroundFractionOption = new Option<float>("--localization-background-fraction")
{
    Description = "Fraction of an image's locations, lowest-scoring for the tag first, treated as background by --localization-background-weight.",
    DefaultValueFactory = _ => 0.4f
};
var localizationMapOption = new Option<string?>("--localization-map")
{
    Description = "Which map localization trains on and heatmaps are drawn from: 'local' = the fused laterals without the coarsest one (stride 16), so each location sees about 100px (needs --fused-stages >= 2); 'fused' = all fused high-resolution laterals (needs --fused-stages > 0); 'full' = the whole spatial map classification pools. Applies on resume too; omitted keeps the checkpoint's setting (full for a fresh run)."
};
localizationMapOption.AcceptOnlyFromAmong("full", "fused", "local");
var cropConsistencyWeightOption = new Option<float>("--crop-consistency-weight")
{
    Description = "Crop consistency: each step, the classifier scores a random square crop of every image (an extra forward pass without gradients), and the localization map pooled over the same window is trained to agree, which pulls heatmap peaks onto the objects. Weight added to the loss; 0 = off.",
    DefaultValueFactory = _ => 0f
};
var cropTeacherMinApOption = new Option<double>("--crop-teacher-min-ap")
{
    Description = "Crop consistency only trains tags whose held-out AP in the latest epoch evaluation is at least this, so it waits for the classifier to be a trustworthy teacher; the trusted set is refreshed after every epoch.",
    DefaultValueFactory = _ => 0.8
};
var cropTeacherMinPositivesOption = new Option<int>("--crop-teacher-min-positives")
{
    Description = "...and at least this many held-out positives in that evaluation, so a rare tag's noisy AP can't qualify it.",
    DefaultValueFactory = _ => 20
};
var cropTeacherConfidenceOption = new Option<float>("--crop-teacher-confidence")
{
    Description = "Crop consistency only trains on the classifier's crop answers with p above this or below 1 minus it.",
    DefaultValueFactory = _ => 0.9f
};
var mosaicsPerStepOption = new Option<int>("--mosaics-per-step")
{
    Description = "Mosaic localization training: each step, this many canvases of four of the batch's images, each shrunk 2x into one tile. Every tile keeps all its tags (nothing is cropped away) and its location is known, so the localization loss learns each tag in its own tile and not in the others, at half scale. Costs a forward/backward pass per mosaic; needs deliberate targets and --localization-weight > 0. 0 = off.",
    DefaultValueFactory = _ => 0
};
var mosaicWeightOption = new Option<float>("--mosaic-weight")
{
    Description = "Mosaic loss weight, relative to --localization-weight.",
    DefaultValueFactory = _ => 1f
};
var mosaicClassificationWeightOption = new Option<float>("--mosaic-classification-weight")
{
    Description = "Also classify each mosaic tile: the canvas's full feature map pooled over the tile, trained on that image's own tags. The canvas's deeper layers see the other tiles, so a tile is right only if it answers from its own pixels. Weight relative to the whole-image tag loss; 0 = mosaics train localization only.",
    DefaultValueFactory = _ => 0f
};
var mosaicFullSizeOption = new Option<bool>("--mosaic-full-size")
{
    Description = "Mosaic tiles at the input size on a canvas twice as wide, instead of shrunk 2x (about 4x a mosaic's cost; small objects stay their real size)."
};
var erasurePairsOption = new Option<int>("--erasure-pairs-per-step")
{
    Description = "Erased-view training (Shetty et al. 2019, GAIN): each step, this many localizable positives the model already calls get the region their own localization map points at blanked, and a same-size region elsewhere blanked; the tag must drop more for the object than for the control. Two extra images through the tower per pair. 0 = off.",
    DefaultValueFactory = _ => 0
};
var erasureLossOption = new Option<string>("--erasure-loss")
{
    Description = "hinge: the object-blanked score must fall --erasure-margin below the control-blanked one. negative: the object-blanked image is a certain negative and the control-blanked one a positive. negative-only: the object-blanked image is a certain negative; the control-blanked one is trained only through --erasure-anchor-weight.",
    DefaultValueFactory = _ => "hinge"
};
erasureLossOption.AcceptOnlyFromAmong("hinge", "negative", "negative-only");
var erasureWeightOption = new Option<float>("--erasure-weight") { Description = "Erased-view loss weight, relative to the whole-image tag loss.", DefaultValueFactory = _ => 1f };
var erasureMarginOption = new Option<float>("--erasure-margin") { Description = "The erased-view hinge's margin, in log-odds.", DefaultValueFactory = _ => 1f };
var erasureAnchorOption = new Option<float>("--erasure-anchor-weight") { Description = "Also hold the control-blanked score to the unblanked one, at this weight (0 = off): the hinge alone lets any filled patch lower every score.", DefaultValueFactory = _ => 0f };
var validationSplitSeedOption = new Option<int?>("--validation-split-seed") { Description = "A fresh run's validation split seed (default: random), to hold out the same images as another run: its training_progress.json's ValidationSplitSeed. A resume keeps its own." };
var erasureEveryOption = new Option<int>("--erasure-every") { Description = "Erased views on every Nth step only (1 = every step).", DefaultValueFactory = _ => 1 };
var erasureFillOption = new Option<string[]>("--erasure-fill")
{
    Description = "Fills for erased-view blanks, one drawn per pair (comma-separated or repeated): zero, mean, noise. Default mean,noise; zero alone taught the model that a grey patch lowers a score.",
    DefaultValueFactory = _ => [],
    AllowMultipleArgumentsPerToken = true
};
var focusShareOption = new Option<double>("--focus-share")
{
    Description = "Fraction of batch anchors drawn from tags the latest epoch evaluation found answered from context instead of the object (context gaps >= 20 pts, or still called with the object blanked >= 90%), each with as many context negatives as positives and grouped on one mosaic. Re-picked after every epoch evaluation; needs tag-centric sampling. 0 = off.",
    DefaultValueFactory = _ => 0
};
var hardNegativeSourceWeightOption = new Option<float>("--hard-negative-source-weight")
{
    Description = "Deliberate targets: train every (image, tag) pair where the image carries one of the tag's hard-negative sources (a tag that often comes with it, whose images often lack it) but not the tag as a full-strength negative with this weight, not only the anchors' sampled hard negatives. Unverified negatives are focal-damped, so a wrong \"yes\" there is nearly free. Localizable tags only; 0 = off.",
    DefaultValueFactory = _ => 0f
};
var classificationPoolingOption = new Option<float?>("--classification-pooling-temperature")
{
    Description = "Score localizable tags by a log-sum-exp over a 16x16 grid of the full map's cells at this temperature (dot-product units, like --localization-temperature), instead of the dot product with the averaged embedding, so one clear region can carry a small object's score. Saved in model_config.json and used by evaluate; 0 = averaged; omitted keeps the checkpoint's setting. Not exported to ONNX yet."
};
var classificationPoolingStartOption = new Option<float?>("--classification-pooling-start-temperature")
{
    Description = "With --classification-pooling-anneal-steps, anneal the pooling temperature log-linearly from this down to --classification-pooling-temperature, from the step pooling was turned on. A high temperature is the averaged embedding, so a run switched to pooling starts where it was. Saved in model_config.json; a resume continues it."
};
var classificationPoolingAnnealOption = new Option<int>("--classification-pooling-anneal-steps")
{
    Description = "Global steps to anneal the pooling temperature over (see --classification-pooling-start-temperature).",
    DefaultValueFactory = _ => 0
};
var classificationPoolingTopKOption = new Option<int?>("--classification-pooling-top-k")
{
    Description = "Pool localizable tags' window (or cell) scores as the mean of the k highest instead of a log-sum-exp; only those windows train. Saved in model_config.json and used by evaluate; 0 = the log-sum-exp; omitted keeps the checkpoint's setting. Needs --classification-pooling-temperature above 0 to turn pooling on."
};
var classificationPoolingBottomWeightOption = new Option<float?>("--classification-pooling-bottom-weight")
{
    Description = "With --classification-pooling-top-k: add this weight times the mean of the k lowest windows' scores (WILDCAT's negative evidence, alpha 0.7 in the paper). Saved in model_config.json; 0 = off; omitted keeps the checkpoint's setting."
};
var glimpsePoolZoomsOption = new Option<int?>("--glimpse-pool-zooms")
{
    Description = "With --classification-glimpses: only windows of the first N zooms (32/64/128/256px) score localizable tags; 3 leaves out the 256px window. The rest are still chosen. Saved in model_config.json; omitted keeps the checkpoint's setting (4 = all)."
};
var bf16Option = new Option<bool>("--bf16")
{
    Description = "Run the image tower in bfloat16: weights, gradients, the optimizer and checkpoints stay fp32 (ImageTower.ComputeType). Not saved with the checkpoint: pass it on every resume.",
};
var classificationPoolingMinedOption = new Option<int>("--classification-pooling-mined-negatives")
{
    Description = "Per image, the highest-scoring negatives trained on their pooled score each step, besides positives and full-strength negatives; the rest train on the averaged score.",
    DefaultValueFactory = _ => 64
};
var classificationGlimpsesOption = new Option<bool>("--classification-glimpses")
{
    Description = "Fresh runs only: score localizable tags from 64px zoomed glimpses of the image (32-256px windows; a log-sum-exp at --classification-pooling-temperature) instead of the whole image, so an object tag can't be read off the character. Saved in model_config.json and used by evaluate. Not exported to ONNX."
};
var glimpsesPerImageOption = new Option<int>("--glimpses-per-image")
{
    Description = "With --classification-glimpses, the windows the model chooses to look at per image, from its own localization map (Glimpses.Propose): in training, and in evaluation (saved with the checkpoint).",
    DefaultValueFactory = _ => 16
};
var glimpseUnframedPositiveWeightOption = new Option<float>("--glimpse-unframed-positive-weight")
{
    Description = "With --classification-glimpses, the glimpse loss's weight for a labelled positive none of the chosen windows frames (its map peak is in none of them). The classifier never saw it; 0 leaves it to the localization loss to teach the map where it is.",
    DefaultValueFactory = _ => 0f
};
var trainGoldOption = new Option<string?>("--gold")
{
    Description = "Hand-checked gold.json to score localization and the crop teacher against at every epoch evaluation (localization_report.json in the checkpoint directory); needs --cache-dir."
};
var maxStepsOption = new Option<int>("--max-steps")
{
    Description = "Stop after this many training steps in this process and save a checkpoint (for short comparison runs). 0 = no limit.",
    DefaultValueFactory = _ => 0
};
var evaluationCommandOption = new Option<string?>("--evaluation-command")
{
    Description = "Executable that runs each epoch's evaluation in the background, called as <command> <snapshot dir> <report dir> (e.g. a script running `evaluate` on another GPU). Training continues meanwhile; results are applied as they come back. Unset = evaluate inline."
};
var snapshotEvaluationStepsOption = new Option<int>("--snapshot-evaluation-steps")
{
    Description = "With --evaluation-command: also evaluate a snapshot of the model every this many steps (counted over the whole run) in the background, for the dashboard's trend charts. Reports go to <checkpoint>/snapshots/step-N/; the snapshot's weights are deleted once evaluated. Doesn't affect early stopping. 0 = only at epoch ends.",
    DefaultValueFactory = _ => 0
};
var chartReferenceOption = new Option<string[]>("--chart-reference")
{
    Description = "Another run's evaluation to compare against on the dashboard's graphical charts, as label=report directory (e.g. v3-e1=context-check/v3-e1). Repeatable.",
    DefaultValueFactory = _ => []
};
var chartCompareOption = new Option<string[]>("--chart-compare")
{
    Description = "Another run to draw as curves over this run's steps on the dashboard's graphical charts, as label=checkpoint directory (e.g. \"run 4=checkpoint-v4\"), the way charts --compare does. Re-read every 10 minutes. Repeatable.",
    DefaultValueFactory = _ => []
};
var textDashboardOption = new Option<bool>("--text-dashboard") { Description = "Keep the all-text dashboard even in a terminal that can show graphical charts (Sixel)" };
var localizableTagsOption = new Option<string?>("--localizable-tags")
{
    Description = "Rules file for which tags have no region to localize (backgrounds, mediums, styles, lighting, framing, people counts, artist:/series:/...). One rule per line: 'tag', 'prefix*', '*suffix', '!' prefix for exceptions. Omit to reuse the checkpoint's saved copy, or the built-in defaults."
};
var ioParallelismOption = new Option<int>("--io-parallelism")
{
    Description = "Concurrent image reads in flight when loading from --cache-dir. A striped array needs many outstanding reads to keep every disk busy: measured on the 6-disk RAID5, 1 read at a time ~85 images/s, 32 in flight ~600-940/s.",
    DefaultValueFactory = _ => 32
};
var prefetchBatchesOption = new Option<int>("--prefetch-batches")
{
    Description = "Batches sampled and loaded ahead of training on a background thread (each window's images are read in one offset-sorted burst). Up to twice this many loaded batches are kept in memory.",
    DefaultValueFactory = _ => 8
};

var stemChannelsOption = new Option<int>("--stem-channels") { Description = "Fresh runs: image tower stem width", DefaultValueFactory = _ => 64 };
var stageChannelsOption = new Option<string>("--stage-channels") { Description = "Fresh runs: comma-separated channels per stage (each width change downsamples 2x)", DefaultValueFactory = _ => "64,128,256" };
var blocksPerStageOption = new Option<string>("--blocks-per-stage") { Description = "Fresh runs: comma-separated ConvNeXt blocks per stage", DefaultValueFactory = _ => "2,2,2" };
var fusedStagesOption = new Option<int>("--fused-stages")
{
    Description = "Fresh runs: earlier stages fused into the spatial map via a top-down path; each halves the heatmap stride (e.g. 4 stages + 2 fused = stride 8, a 64x64 map at 512px)",
    DefaultValueFactory = _ => 0
};
var lrScheduleOption = new Option<string>("--lr-schedule")
{
    Description = "'cosine': linear warmup over --warmup-steps then cosine decay to 5% across the run's --epochs; 'constant': fixed --lr",
    DefaultValueFactory = _ => "cosine"
};
lrScheduleOption.AcceptOnlyFromAmong("cosine", "constant");
var warmupStepsOption = new Option<int>("--warmup-steps") { Description = "Optimizer steps of linear learning-rate warmup (cosine schedule)", DefaultValueFactory = _ => 2000 };
var keepVocabularyOption = new Option<bool>("--keep-full-vocabulary") { Description = "Fresh runs: keep vocabulary rows no image carries instead of dropping them" };

var trainCommand = new Command("train", "Full/periodic fine-tune pass over a dataset manifest or preprocessed cache");
trainCommand.Options.Add(manifestOption);
trainCommand.Options.Add(cacheDirOption);
trainCommand.Options.Add(checkpointDirOption);
trainCommand.Options.Add(embeddingDimOption);
trainCommand.Options.Add(inputSizeOption);
trainCommand.Options.Add(epochsOption);
trainCommand.Options.Add(batchSizeOption);
trainCommand.Options.Add(learningRateOption);
trainCommand.Options.Add(validationFractionOption);
trainCommand.Options.Add(earlyStoppingPatienceOption);
trainCommand.Options.Add(localizationWeightOption);
trainCommand.Options.Add(localizationTemperatureOption);
trainCommand.Options.Add(selfSupervisedWeightOption);
trainCommand.Options.Add(checkpointIntervalMinutesOption);
trainCommand.Options.Add(gradientAccumulationStepsOption);
trainCommand.Options.Add(warmStartModelOption);
trainCommand.Options.Add(warmStartVocabOption);
trainCommand.Options.Add(localizationTagChunkSizeOption);
trainCommand.Options.Add(samplingOption);
trainCommand.Options.Add(positivesPerAnchorOption);
trainCommand.Options.Add(hardNegativesPerAnchorOption);
trainCommand.Options.Add(targetsOption);
trainCommand.Options.Add(negativeGammaOption);
trainCommand.Options.Add(negativeClipOption);
trainCommand.Options.Add(certainNegativeWeightOption);
trainCommand.Options.Add(impliedThresholdOption);
trainCommand.Options.Add(missingThresholdOption);
trainCommand.Options.Add(contextNegativesPerAnchorOption);
trainCommand.Options.Add(contextNegativeWeightOption);
trainCommand.Options.Add(contextPriorWeightOption);
trainCommand.Options.Add(exclusiveTagsOption);
trainCommand.Options.Add(tagParentageOption);
trainCommand.Options.Add(validationMaxImagesOption);
trainCommand.Options.Add(ioParallelismOption);
trainCommand.Options.Add(noRatingsOption);
trainCommand.Options.Add(stemChannelsOption);
trainCommand.Options.Add(stageChannelsOption);
trainCommand.Options.Add(blocksPerStageOption);
trainCommand.Options.Add(fusedStagesOption);
trainCommand.Options.Add(lrScheduleOption);
trainCommand.Options.Add(warmupStepsOption);
trainCommand.Options.Add(keepVocabularyOption);
trainCommand.Options.Add(localizationMinedNegativesOption);
trainCommand.Options.Add(localizationBackgroundWeightOption);
trainCommand.Options.Add(localizationBackgroundFractionOption);
trainCommand.Options.Add(maxStepsOption);
trainCommand.Options.Add(evaluationCommandOption);
trainCommand.Options.Add(snapshotEvaluationStepsOption);
trainCommand.Options.Add(chartReferenceOption);
trainCommand.Options.Add(chartCompareOption);
trainCommand.Options.Add(textDashboardOption);
trainCommand.Options.Add(trainGoldOption);
trainCommand.Options.Add(cropConsistencyWeightOption);
trainCommand.Options.Add(cropTeacherMinApOption);
trainCommand.Options.Add(cropTeacherConfidenceOption);
trainCommand.Options.Add(mosaicsPerStepOption);
trainCommand.Options.Add(mosaicWeightOption);
trainCommand.Options.Add(mosaicClassificationWeightOption);
trainCommand.Options.Add(mosaicFullSizeOption);
trainCommand.Options.Add(erasurePairsOption);
trainCommand.Options.Add(erasureLossOption);
trainCommand.Options.Add(erasureWeightOption);
trainCommand.Options.Add(erasureMarginOption);
trainCommand.Options.Add(erasureFillOption);
trainCommand.Options.Add(erasureEveryOption);
trainCommand.Options.Add(erasureAnchorOption);
trainCommand.Options.Add(validationSplitSeedOption);
trainCommand.Options.Add(focusShareOption);
trainCommand.Options.Add(hardNegativeSourceWeightOption);
var outOfContextShareOption = new Option<double>("--out-of-context-share")
{
    Description = "Fraction of a localizable general anchor tag's positives drawn for a low context prior (the highest P(tag | character/artist/series) of the image): the lowest of 16 random positives, if below --out-of-context-max-prior. Gloves on a character who rarely wears them can only be learned from the gloves. 0 = off.",
    DefaultValueFactory = _ => 0
};
var outOfContextMaxPriorOption = new Option<double>("--out-of-context-max-prior") { Description = "Highest context prior an out-of-context positive may have", DefaultValueFactory = _ => 0.3 };
var outOfContextWeightOption = new Option<float>("--out-of-context-weight") { Description = "Weight of an out-of-context positive's pair with its anchor tag (0 = the usual 1)", DefaultValueFactory = _ => 0f };
trainCommand.Options.Add(outOfContextShareOption);
trainCommand.Options.Add(outOfContextMaxPriorOption);
trainCommand.Options.Add(outOfContextWeightOption);
var outOfContextAllOption = new Option<bool>("--out-of-context-all") { Description = "Give --out-of-context-weight to every localizable general positive in a batch whose context prior is below --out-of-context-max-prior, not only to sampled anchors' (the way context negatives are weighted)" };
trainCommand.Options.Add(outOfContextAllOption);
var excluderShareOption = new Option<double>("--excluder-share")
{
    Description = "Fraction of anchors drawn from the tags an excludes rule (--tag-parentage) rules out, e.g. gloves for no_gloves, each with two of the excluders' images as certain negatives: they're rare, so otherwise they reach training only by chance. 0 = off.",
    DefaultValueFactory = _ => 0
};
var excludedNegativeWeightOption = new Option<float>("--excluded-negative-weight") { Description = "Weight of a tag an excludes rule rules out on an image (no_gloves: gloves), if above --certain-negative-weight. 0 = that weight.", DefaultValueFactory = _ => 0f };
trainCommand.Options.Add(excluderShareOption);
trainCommand.Options.Add(excludedNegativeWeightOption);
trainCommand.Options.Add(classificationPoolingOption);
trainCommand.Options.Add(classificationPoolingStartOption);
trainCommand.Options.Add(classificationPoolingAnnealOption);
trainCommand.Options.Add(classificationPoolingTopKOption);
trainCommand.Options.Add(classificationPoolingBottomWeightOption);
trainCommand.Options.Add(glimpsePoolZoomsOption);
trainCommand.Options.Add(classificationPoolingMinedOption);
trainCommand.Options.Add(classificationGlimpsesOption);
trainCommand.Options.Add(glimpsesPerImageOption);
trainCommand.Options.Add(glimpseUnframedPositiveWeightOption);
var glimpseLabelFramingStepsOption = new Option<int>("--glimpse-label-framing-steps")
{
    Description = "With --classification-glimpses, a fresh run's warm-up: a share of each image's windows (half at step 0, none from this global step on) frames labelled positives, by the same rule as the model's own choice, while the localization map can't point yet. 0 = the model's choice from the start.",
    DefaultValueFactory = _ => 0
};
trainCommand.Options.Add(glimpseLabelFramingStepsOption);
var glimpseLabelFramingFloorOption = new Option<int>("--glimpse-label-framing-floor")
{
    Description = "With --classification-glimpses: at least this many of each image's windows always frame labelled positives (as --glimpse-label-framing-steps does early on), so a tag the proposals rarely suspect still gets positive glimpse gradient. 0 = off.",
    DefaultValueFactory = _ => 0
};
trainCommand.Options.Add(glimpseLabelFramingFloorOption);
var glimpseFramedPositivePoolingOption = new Option<bool>("--glimpse-framed-positive-pooling")
{
    Description = "With label framing (--glimpse-label-framing-floor / -steps): a labelled positive the framing looked at is scored, for its loss, from the close windows (below 256px) that frame it only, so they get its \"yes\"; negatives and every other tag pool every window. Training only."
};
trainCommand.Options.Add(glimpseFramedPositivePoolingOption);
var glimpsePoolMinZoomOption = new Option<int?>("--glimpse-pool-min-zoom")
{
    Description = "With --classification-glimpses: windows of zooms below this index (0 = 32px) don't score localizable tags; 1 leaves out the 32px window. Saved in model_config.json; omitted keeps the checkpoint's setting."
};
trainCommand.Options.Add(glimpsePoolMinZoomOption);
var separateGlimpseTowerOption = new Option<bool>("--separate-glimpse-tower")
{
    Description = "With --classification-glimpses: embed glimpses with a separate copy of the image tower (made from the current weights the first time, its own optimizer, saved as glimpse_tower.dat and used by evaluate) instead of the tower the whole image goes through."
};
trainCommand.Options.Add(separateGlimpseTowerOption);
var fineLocalizationOption = new Option<bool>("--fine-localization")
{
    Description = "Add a stride-2 level under the localization map (a small branch from the pixels and the stem's features, its output added to the 2x-upsampled map; starts at zero, so the model scores as before). Saved as fine_localization.dat with its own optimizer and kept on resume; ONNX export doesn't support it yet."
};
trainCommand.Options.Add(fineLocalizationOption);
var trainInputSizeOption = new Option<int?>("--train-input-size")
{
    Description = "With --cache-dir: train at this input size instead of the cache's (e.g. 1024 to fine-tune at a higher resolution). Cached images are rescaled (Mitchell cubic to enlarge); rows in --hires-store come from it."
};
trainCommand.Options.Add(trainInputSizeOption);
var hiResStoreOption = new Option<string?>("--hires-store")
{
    Description = "Hi-res side stores, comma-separated (HiResImageStore: JPEGs of some cache rows at a larger long side, e.g. one at 1024 and one at 2048). Their rows are read from them instead of the cache, fitted to the input size: from the smallest store that reaches it, else the largest. Rows a pull appends while training runs are picked up every few minutes."
};
trainCommand.Options.Add(hiResStoreOption);
var trainScalesOption = new Option<string?>("--train-scales")
{
    Description = "Mixed-scale training: comma-separated size:batch:share[:stored], e.g. 512:10:0.3,1024:4:0.5:0.25,2048:1:0.2:1. Each micro-batch is drawn at one size with its own batch size; 'stored' is the share of its images drawn from rows the hi-res stores hold at that size or larger. Each image counts the same in the loss whatever its batch size (relative to --batch-size, which still sets the epoch length). The checkpoint keeps --train-input-size (or the cache's) as its input size."
};
trainCommand.Options.Add(trainScalesOption);
var focusTagsOption = new Option<string[]>("--focus-tags")
{
    Description = "With --focus-share: draw the focus anchors from these tags (comma-separated) instead of the ones the latest epoch evaluation picks.",
    AllowMultipleArgumentsPerToken = true,
    CustomParser = r => r.Tokens.SelectMany(t => t.Value.Split(',', StringSplitOptions.RemoveEmptyEntries)).ToArray()
};
trainCommand.Options.Add(focusTagsOption);
var glimpseTowerLearningRateScaleOption = new Option<float>("--glimpse-tower-lr-scale")
{
    Description = "With --separate-glimpse-tower: its learning rate as a multiple of the run's (it has to leave the 512px solution; the main tower doesn't).",
    DefaultValueFactory = _ => 1f
};
trainCommand.Options.Add(glimpseTowerLearningRateScaleOption);
var contextPairsPerBatchOption = new Option<int>("--context-pairs-per-batch")
{
    Description = "Tag-centric sampling: each batch starts with this many pairs of images of one source (a character, artist or series), one with a localizable tag and one without, the tag drawn uniformly from those with context sources. 0 = off.",
    DefaultValueFactory = _ => 0
};
trainCommand.Options.Add(contextPairsPerBatchOption);
var contextPairWeightOption = new Option<float>("--context-pair-weight")
{
    Description = "With --context-pairs-per-batch and --classification-glimpses: weight of a hinge making the image with the tag score --context-pair-margin logits above its pair without it. 0 = off.",
    DefaultValueFactory = _ => 0f
};
trainCommand.Options.Add(contextPairWeightOption);
var contextPairMarginOption = new Option<float>("--context-pair-margin")
{
    Description = "The --context-pair-weight hinge's margin, in logits.",
    DefaultValueFactory = _ => 1f
};
trainCommand.Options.Add(contextPairMarginOption);
var glimpseFeatureSplitOption = new Option<bool>("--glimpse-feature-split")
{
    Description = "Singh et al. 2020's feature split on the glimpse embedding: half of it is the object half. A localizable general tag's positive on an image whose character/artist/series rarely carries it (prior below 0.2: an exclusive image) is scored with the other half held at its running mean, without gradient there, so the object half carries those calls. Saved in the model config."
};
trainCommand.Options.Add(glimpseFeatureSplitOption);
var glimpseFeatureSplitWeightOption = new Option<float>("--glimpse-feature-split-weight")
{
    Description = "With --glimpse-feature-split: the glimpse loss's weight on exclusive positives (Singh et al. weight them by how much rarer they are, at least 3).",
    DefaultValueFactory = _ => 1f
};
trainCommand.Options.Add(glimpseFeatureSplitWeightOption);
var glimpseFeatureSplitNegativesOption = new Option<bool>("--glimpse-feature-split-negatives")
{
    Description = "With --glimpse-feature-split: context negatives too (the tag a \"no\" on an image whose character/artist/series usually carries it: prior at least 0.6) score from the object half, so it learns both sides: the object without the character, and the character without the object. Singh et al. split only the exclusive positives."
};
trainCommand.Options.Add(glimpseFeatureSplitNegativesOption);
var glimpseExclusivePerBatchOption = new Option<int>("--glimpse-exclusive-per-batch")
{
    Description = "Each batch starts with this many images for the feature split: alternately a context negative and an exclusive positive, anchored on localizable tags with context sources. A larger share of the batch instead of a heavier weight.",
    DefaultValueFactory = _ => 0
};
trainCommand.Options.Add(glimpseExclusivePerBatchOption);
var mosaicGlimpseWeightOption = new Option<float>("--mosaic-glimpse-weight")
{
    Description = "With --classification-glimpses and --mosaics-per-step: the mosaic canvases get glimpses too, chosen by the model, and a tag one tile carries is a \"no\" in every window on tiles whose image says no, window by window, and a \"yes\" pooled over the windows inside its own tile. Window-level labels from the tags alone. 0 = off.",
    DefaultValueFactory = _ => 0f
};
trainCommand.Options.Add(mosaicGlimpseWeightOption);
var twinEveryOption = new Option<int>("--twin-every")
{
    Description = "Matched twins: every Nth batch is one image with a localizable general tag and images without it whose tags otherwise overlap it most (at most one character each). With --classification-glimpses, the half of its glimpses most like the twins' glimpses are a \"no\" for the tag (--twin-weight). 0 = off.",
    DefaultValueFactory = _ => 0
};
trainCommand.Options.Add(twinEveryOption);
var twinWeightOption = new Option<float>("--twin-weight") { Description = "Weight of the matched twins' window term.", DefaultValueFactory = _ => 1f };
trainCommand.Options.Add(twinWeightOption);
var twinDetachScalingOption = new Option<bool>("--twin-detach-scaling") { Description = "The twins' window term sends no gradient into the shared logit scale and bias." };
trainCommand.Options.Add(twinDetachScalingOption);
var twinPasteOption = new Option<int>("--twin-paste")
{
    Description = "With --twin-every: in each twin batch, this many twins get the positive's region at its tag's localization-map peak pasted in at the same place (and carry the tag), and as many others a same-size region from elsewhere in the positive (certain negatives). Needs a map that points. 0 = off.",
    DefaultValueFactory = _ => 0
};
trainCommand.Options.Add(twinPasteOption);
var selfMapWeightOption = new Option<float>("--self-map-weight")
{
    Description = "Window labels from the model's own localization map (SelfMapWindows; review/open-vocab-research PLAN #1): per labelled localizable positive, the 64/128px windows at the tag's map peak are a \"yes\" and two of its bottom-40% 64px windows a \"no\", each window embedded as a 64px glimpse by the image tower and trained by a per-window BCE at this weight. Image-level losses unchanged. 0 = off.",
    DefaultValueFactory = _ => 0f
};
trainCommand.Options.Add(selfMapWeightOption);
var selfMapDetachScalingOption = new Option<bool>("--self-map-detach-scaling") { Description = "The self-map window term sends no gradient into the shared logit scale and bias." };
trainCommand.Options.Add(selfMapDetachScalingOption);
var selfMapNegativesOption = new Option<string>("--self-map-negatives")
{
    Description = "With --self-map-weight, also windows on images WITHOUT the tag (SAM 3's supervised-negatives row): per image, up to 4 localizable tags it is a certain negative for, the 64/128px windows at that tag's map peak (where the model would look for it) are a \"no\". certain = hard negatives, exclusive groups and excludes rules; all = those plus context negatives (--context-negative-weight); none = off.",
    DefaultValueFactory = _ => "none"
};
selfMapNegativesOption.AcceptOnlyFromAmong("none", "certain", "all");
trainCommand.Options.Add(selfMapNegativesOption);
var selfMapNegativeWeightOption = new Option<float>("--self-map-negative-weight") { Description = "With --self-map-negatives: the negative windows' term weight.", DefaultValueFactory = _ => 1f };
trainCommand.Options.Add(selfMapNegativeWeightOption);
var mosaicLargeGridOption = new Option<int>("--mosaic-large-grid")
{
    Description = "On every --mosaic-large-grid-every'th step, one canvas of N x N batch images shrunk 4x instead of the 2x2 mosaics (review/open-vocab-research PLAN #3, after OWLv2's mixed grids): smaller objects, more tiles to say \"not here\". Needs --batch-size >= N². 0 = off.",
    DefaultValueFactory = _ => 0
};
trainCommand.Options.Add(mosaicLargeGridOption);
var mosaicLargeGridEveryOption = new Option<int>("--mosaic-large-grid-every") { Description = "With --mosaic-large-grid: the large canvas on every Nth step, the 2x2 mosaics on the rest.", DefaultValueFactory = _ => 2 };
trainCommand.Options.Add(mosaicLargeGridEveryOption);
var negativeLossOption = new Option<string>("--negative-loss")
{
    Description = "Deliberate targets: the unverified negatives' loss: asl (the asymmetric loss, --negative-gamma / --negative-clip) or hill ((1.5 - p) p^2, robust to missing labels).",
    DefaultValueFactory = _ => "asl"
};
negativeLossOption.AcceptOnlyFromAmong("asl", "hill");
trainCommand.Options.Add(negativeLossOption);
var glimpseProposalWeightOption = new Option<float>("--glimpse-proposal-weight")
{
    Description = "With --classification-glimpses, the weight of the proposal loss: the whole-image proposal scores, which pick where the glimpses look, trained on every localizable tag from the labels.",
    DefaultValueFactory = _ => 1f
};
trainCommand.Options.Add(glimpseProposalWeightOption);
var glimpseProposalDetachTagsOption = new Option<bool>("--glimpse-proposal-detach-tags")
{
    Description = "With --classification-glimpses, the proposal loss trains only the proposal head and the tower, not the tag embeddings, so the whole-image context it learns can't pull the embeddings the localization map points with."
};
trainCommand.Options.Add(glimpseProposalDetachTagsOption);
var localityNegativeWeightOption = new Option<float>("--locality-negative-weight")
{
    Description = "With --classification-glimpses: a close window where one labelled tag is confidently found trains each other labelled tag \"no\" there, weighted by how rarely the two are together (glasses and shoes, not gloves and a face), learned as training runs. This weight times that term; 0 = off.",
    DefaultValueFactory = _ => 0f
};
trainCommand.Options.Add(localityNegativeWeightOption);
var localityPriorOption = new Option<string?>("--locality-prior")
{
    Description = "With --locality-negative-weight: start the which-tags-are-together table from this file (evaluate --locality-prior-out on a trained map) instead of from nothing."
};
trainCommand.Options.Add(localityPriorOption);
var localityMinObservationsOption = new Option<float>("--locality-min-observations")
{
    Description = "With --locality-negative-weight: weighted observations a tag pair needs before it gives negatives.",
    DefaultValueFactory = _ => 20f
};
trainCommand.Options.Add(localityMinObservationsOption);
var localityConfidenceGateOption = new Option<float>("--locality-confidence-gate")
{
    Description = "With --locality-negative-weight: only windows where the found tag's call is at least this sure (e.g. 0.9) give negatives; 0 = any.",
    DefaultValueFactory = _ => 0f
};
trainCommand.Options.Add(localityConfidenceGateOption);
var localityPerPairOption = new Option<bool>("--locality-per-pair")
{
    Description = "With --locality-negative-weight: each image's term is the mean over its tag pairs instead of the sum, so images with many tags don't dominate."
};
trainCommand.Options.Add(localityPerPairOption);
var localityDetachScalingOption = new Option<bool>("--locality-detach-scaling")
{
    Description = "With --locality-negative-weight: the term trains no gradient into the shared logit scale and bias, only the tower and the tag embeddings."
};
trainCommand.Options.Add(localityDetachScalingOption);
trainCommand.Options.Add(bf16Option);
trainCommand.Options.Add(cropTeacherMinPositivesOption);
trainCommand.Options.Add(localizationMapOption);
trainCommand.Options.Add(localizableTagsOption);
trainCommand.Options.Add(prefetchBatchesOption);
trainCommand.SetAction(parseResult => TrainCommandHandler.Run(
    parseResult.GetValue(manifestOption),
    parseResult.GetValue(cacheDirOption),
    parseResult.GetRequiredValue(checkpointDirOption),
    parseResult.GetRequiredValue(embeddingDimOption),
    parseResult.GetRequiredValue(inputSizeOption),
    parseResult.GetRequiredValue(epochsOption),
    parseResult.GetRequiredValue(batchSizeOption),
    parseResult.GetRequiredValue(learningRateOption),
    parseResult.GetRequiredValue(validationFractionOption),
    parseResult.GetRequiredValue(earlyStoppingPatienceOption),
    parseResult.GetRequiredValue(localizationWeightOption),
    parseResult.GetRequiredValue(localizationTemperatureOption),
    parseResult.GetRequiredValue(selfSupervisedWeightOption),
    parseResult.GetRequiredValue(checkpointIntervalMinutesOption),
    parseResult.GetRequiredValue(gradientAccumulationStepsOption),
    parseResult.GetValue(warmStartModelOption),
    parseResult.GetValue(warmStartVocabOption),
    parseResult.GetRequiredValue(localizationTagChunkSizeOption),
    new TrainingTargetOptions(
        TagCentricSampling: parseResult.GetRequiredValue(samplingOption) == "tag-centric",
        PositivesPerAnchor: parseResult.GetRequiredValue(positivesPerAnchorOption),
        HardNegativesPerAnchor: parseResult.GetRequiredValue(hardNegativesPerAnchorOption),
        DeliberateTargets: parseResult.GetRequiredValue(targetsOption) == "deliberate",
        NegativeGamma: parseResult.GetRequiredValue(negativeGammaOption),
        NegativeClip: parseResult.GetRequiredValue(negativeClipOption),
        CertainNegativeWeight: parseResult.GetRequiredValue(certainNegativeWeightOption),
        ImpliedThreshold: parseResult.GetRequiredValue(impliedThresholdOption),
        MissingThreshold: parseResult.GetRequiredValue(missingThresholdOption),
        ContextNegativesPerAnchor: parseResult.GetRequiredValue(contextNegativesPerAnchorOption),
        ContextNegativeWeight: parseResult.GetRequiredValue(contextNegativeWeightOption),
        ContextPriorWeight: parseResult.GetRequiredValue(contextPriorWeightOption),
        ExclusiveTagsPath: parseResult.GetValue(exclusiveTagsOption),
        TagParentagePath: parseResult.GetValue(tagParentageOption),
        ValidationMaxImages: parseResult.GetRequiredValue(validationMaxImagesOption),
        Ratings: !parseResult.GetValue(noRatingsOption),
        LocalizationMinedNegatives: parseResult.GetRequiredValue(localizationMinedNegativesOption),
        LocalizationBackgroundWeight: parseResult.GetRequiredValue(localizationBackgroundWeightOption),
        LocalizationBackgroundFraction: parseResult.GetRequiredValue(localizationBackgroundFractionOption),
        MaxSteps: parseResult.GetRequiredValue(maxStepsOption),
        GoldPath: parseResult.GetValue(trainGoldOption),
        CropConsistencyWeight: parseResult.GetRequiredValue(cropConsistencyWeightOption),
        CropTeacherMinAveragePrecision: parseResult.GetRequiredValue(cropTeacherMinApOption),
        CropTeacherConfidence: parseResult.GetRequiredValue(cropTeacherConfidenceOption),
        CropTeacherMinPositives: parseResult.GetRequiredValue(cropTeacherMinPositivesOption),
        MosaicsPerStep: parseResult.GetRequiredValue(mosaicsPerStepOption),
        MosaicWeight: parseResult.GetRequiredValue(mosaicWeightOption),
        MosaicClassificationWeight: parseResult.GetRequiredValue(mosaicClassificationWeightOption),
        MosaicFullSizeTiles: parseResult.GetValue(mosaicFullSizeOption),
        ErasurePairsPerStep: parseResult.GetRequiredValue(erasurePairsOption),
        ErasureLoss: parseResult.GetRequiredValue(erasureLossOption) switch { "negative" => ErasureLoss.Negative, "negative-only" => ErasureLoss.NegativeOnly, _ => ErasureLoss.Hinge },
        ErasureWeight: parseResult.GetRequiredValue(erasureWeightOption),
        ErasureMargin: parseResult.GetRequiredValue(erasureMarginOption),
        ErasureFillNames: parseResult.GetRequiredValue(erasureFillOption).SelectMany(v => v.Split(',', StringSplitOptions.RemoveEmptyEntries)).Select(OcclusionFills.Parse).ToList(),
        ErasureEvery: parseResult.GetRequiredValue(erasureEveryOption),
        ErasureAnchorWeight: parseResult.GetRequiredValue(erasureAnchorOption),
        ValidationSplitSeed: parseResult.GetValue(validationSplitSeedOption),
        FocusShare: parseResult.GetRequiredValue(focusShareOption),
        HardNegativeSourceWeight: parseResult.GetRequiredValue(hardNegativeSourceWeightOption),
        OutOfContextShare: parseResult.GetRequiredValue(outOfContextShareOption),
        OutOfContextMaxPrior: parseResult.GetRequiredValue(outOfContextMaxPriorOption),
        OutOfContextWeight: parseResult.GetRequiredValue(outOfContextWeightOption),
        OutOfContextAll: parseResult.GetValue(outOfContextAllOption),
        ExcluderShare: parseResult.GetRequiredValue(excluderShareOption),
        ExcludedNegativeWeight: parseResult.GetRequiredValue(excludedNegativeWeightOption),
        ClassificationPoolingTemperature: parseResult.GetValue(classificationPoolingOption),
        ClassificationPoolingStartTemperature: parseResult.GetValue(classificationPoolingStartOption),
        ClassificationPoolingAnnealSteps: parseResult.GetValue(classificationPoolingAnnealOption),
        ClassificationPoolingTopK: parseResult.GetValue(classificationPoolingTopKOption),
        ClassificationPoolingBottomWeight: parseResult.GetValue(classificationPoolingBottomWeightOption),
        GlimpsePoolZooms: parseResult.GetValue(glimpsePoolZoomsOption),
        ClassificationPoolingMinedNegatives: parseResult.GetValue(classificationPoolingMinedOption),
        ClassificationGlimpses: parseResult.GetValue(classificationGlimpsesOption),
        GlimpsesPerImage: parseResult.GetValue(glimpsesPerImageOption),
        GlimpseUnframedPositiveWeight: parseResult.GetValue(glimpseUnframedPositiveWeightOption),
        GlimpseLabelFramingSteps: parseResult.GetValue(glimpseLabelFramingStepsOption),
        GlimpseLabelFramingFloor: parseResult.GetValue(glimpseLabelFramingFloorOption),
        GlimpseFramedPositivePooling: parseResult.GetValue(glimpseFramedPositivePoolingOption),
        SeparateGlimpseTower: parseResult.GetValue(separateGlimpseTowerOption),
        GlimpseTowerLearningRateScale: parseResult.GetValue(glimpseTowerLearningRateScaleOption),
        FineLocalization: parseResult.GetValue(fineLocalizationOption),
        ContextPairsPerBatch: parseResult.GetValue(contextPairsPerBatchOption),
        ContextPairMargin: parseResult.GetValue(contextPairMarginOption),
        ContextPairWeight: parseResult.GetValue(contextPairWeightOption),
        GlimpseFeatureSplit: parseResult.GetValue(glimpseFeatureSplitOption),
        GlimpseFeatureSplitWeight: parseResult.GetValue(glimpseFeatureSplitWeightOption),
        GlimpseFeatureSplitNegatives: parseResult.GetValue(glimpseFeatureSplitNegativesOption),
        GlimpseExclusivePerBatch: parseResult.GetValue(glimpseExclusivePerBatchOption),
        MosaicGlimpseWeight: parseResult.GetValue(mosaicGlimpseWeightOption),
        TwinEvery: parseResult.GetValue(twinEveryOption),
        TwinWeight: parseResult.GetValue(twinWeightOption),
        TwinDetachScaling: parseResult.GetValue(twinDetachScalingOption),
        TwinPaste: parseResult.GetValue(twinPasteOption),
        SelfMapWeight: parseResult.GetValue(selfMapWeightOption),
        SelfMapDetachScaling: parseResult.GetValue(selfMapDetachScalingOption),
        SelfMapNegatives: parseResult.GetRequiredValue(selfMapNegativesOption),
        SelfMapNegativeWeight: parseResult.GetValue(selfMapNegativeWeightOption),
        MosaicLargeGrid: parseResult.GetValue(mosaicLargeGridOption),
        MosaicLargeGridEvery: parseResult.GetValue(mosaicLargeGridEveryOption),
        GlimpsePoolMinZoom: parseResult.GetValue(glimpsePoolMinZoomOption),
        HillNegatives: parseResult.GetValue(negativeLossOption) == "hill",
        GlimpseProposalWeight: parseResult.GetValue(glimpseProposalWeightOption),
        GlimpseProposalDetachTags: parseResult.GetValue(glimpseProposalDetachTagsOption),
        LocalityNegativeWeight: parseResult.GetRequiredValue(localityNegativeWeightOption),
        LocalityPriorPath: parseResult.GetValue(localityPriorOption),
        LocalityMinObservations: parseResult.GetRequiredValue(localityMinObservationsOption),
        LocalityConfidenceGate: parseResult.GetRequiredValue(localityConfidenceGateOption),
        LocalityPerPair: parseResult.GetValue(localityPerPairOption),
        LocalityDetachScaling: parseResult.GetValue(localityDetachScalingOption),
        Bf16Compute: parseResult.GetValue(bf16Option),
        LocalizationFromFusedOnly: parseResult.GetValue(localizationMapOption) switch { "fused" or "local" => true, "full" => false, _ => null },
        LocalizationSkippedLaterals: parseResult.GetValue(localizationMapOption) switch { "local" => 1, "fused" or "full" => 0, _ => null },
        LocalizableTagsPath: parseResult.GetValue(localizableTagsOption),
        CompactVocabulary: !parseResult.GetValue(keepVocabularyOption),
        CosineSchedule: parseResult.GetRequiredValue(lrScheduleOption) == "cosine",
        WarmupSteps: parseResult.GetRequiredValue(warmupStepsOption),
        EvaluationCommand: parseResult.GetValue(evaluationCommandOption),
        SnapshotEvaluationSteps: parseResult.GetRequiredValue(snapshotEvaluationStepsOption),
        ChartReferences: parseResult.GetValue(chartReferenceOption) ?? [],
        TextDashboard: parseResult.GetValue(textDashboardOption),
        ChartComparisons: parseResult.GetValue(chartCompareOption) ?? [],
        TrainInputSize: parseResult.GetValue(trainInputSizeOption),
        HiResStore: parseResult.GetValue(hiResStoreOption),
        FocusTagNames: parseResult.GetValue(focusTagsOption),
        TrainScales: parseResult.GetValue(trainScalesOption)),
    parseResult.GetRequiredValue(ioParallelismOption),
    parseResult.GetRequiredValue(prefetchBatchesOption),
    new TowerOptions(
        parseResult.GetRequiredValue(stemChannelsOption),
        parseResult.GetRequiredValue(stageChannelsOption).Split(',').Select(int.Parse).ToArray(),
        parseResult.GetRequiredValue(blocksPerStageOption).Split(',').Select(int.Parse).ToArray(),
        parseResult.GetRequiredValue(fusedStagesOption))));

var tagArgument = new Argument<string>("tag") { Description = "The new tag to add" };
var imagesOption = new Option<string>("--images") { Description = "Dataset manifest of the newly tagged images", Required = true };
var stepsOption = new Option<int>("--steps") { Description = "Gradient steps to fine-tune the new row for", DefaultValueFactory = _ => 200 };
var addTagLearningRateOption = new Option<double>("--lr") { DefaultValueFactory = _ => 1e-3 };
var minImageThresholdOption = new Option<int>("--min-image-threshold")
{
    Description = "Images needed before the tag is promoted from warm-start-only to trained",
    DefaultValueFactory = _ => 15
};
var addTagEarlyStoppingPatienceOption = new Option<int>("--early-stopping-patience")
{
    Description = "Steps without validation-loss improvement before stopping (only applies once there are enough images for a validation split)",
    DefaultValueFactory = _ => 5
};

var addTagCacheDirOption = new Option<string?>("--cache-dir")
{
    Description = "The corpus (PreprocessedDatasetCache) the checkpoint was trained on. When given, the new tag also trains against hard negatives from it -- images carrying the tags that usually accompany it in --images, but not the new tag -- plus random corpus images. Strongly recommended: without negatives the new tag learns to fire on everything similar."
};
var negativesPerPositiveOption = new Option<int>("--negatives-per-positive") { Description = "With --cache-dir: corpus negatives drawn per manifest image (half hard, half random)", DefaultValueFactory = _ => 4 };

var addTagCommand = new Command("add-tag", "Warm-start and fine-tune a single new tag row, image encoder frozen");
addTagCommand.Arguments.Add(tagArgument);
addTagCommand.Options.Add(checkpointDirOption);
addTagCommand.Options.Add(imagesOption);
addTagCommand.Options.Add(stepsOption);
addTagCommand.Options.Add(addTagLearningRateOption);
addTagCommand.Options.Add(minImageThresholdOption);
addTagCommand.Options.Add(addTagEarlyStoppingPatienceOption);
addTagCommand.Options.Add(warmStartModelOption);
addTagCommand.Options.Add(warmStartVocabOption);
addTagCommand.Options.Add(addTagCacheDirOption);
addTagCommand.Options.Add(negativesPerPositiveOption);
addTagCommand.SetAction(parseResult => AddTagCommandHandler.Run(
    parseResult.GetRequiredValue(checkpointDirOption),
    parseResult.GetRequiredValue(tagArgument),
    parseResult.GetRequiredValue(imagesOption),
    parseResult.GetRequiredValue(stepsOption),
    parseResult.GetRequiredValue(addTagLearningRateOption),
    parseResult.GetRequiredValue(minImageThresholdOption),
    parseResult.GetRequiredValue(addTagEarlyStoppingPatienceOption),
    parseResult.GetValue(warmStartModelOption),
    parseResult.GetValue(warmStartVocabOption),
    parseResult.GetValue(addTagCacheDirOption),
    parseResult.GetRequiredValue(negativesPerPositiveOption)));

var modelDirOption = new Option<string>("--model-dir")
{
    Description = "Directory to write image_encoder.onnx, tag_vocabulary.json and tag_embeddings.bin",
    DefaultValueFactory = _ => "./model"
};

var exportOnnxCommand = new Command("export-onnx", "Export a training checkpoint to the ONNX bundle unbooru-tagger-inference reads");
exportOnnxCommand.Options.Add(checkpointDirOption);
exportOnnxCommand.Options.Add(modelDirOption);
var ignorePoolingOption = new Option<bool>("--ignore-pooling")
{
    Description = "Export a checkpoint that scores localizable tags per tag (classification pooling, glimpses) anyway: its boxes are right, its scores for those tags aren't (render-gold --scores supplies them)"
};
exportOnnxCommand.Options.Add(ignorePoolingOption);
exportOnnxCommand.SetAction(parseResult => ExportOnnxCommandHandler.Run(
    parseResult.GetRequiredValue(checkpointDirOption),
    parseResult.GetRequiredValue(modelDirOption),
    parseResult.GetValue(ignorePoolingOption)));

var evaluateCacheDirOption = new Option<string>("--cache-dir") { Description = "The PreprocessedDatasetCache the checkpoint was trained on", Required = true };
var evaluateMaxImagesOption = new Option<int>("--max-images")
{
    Description = "Evaluate at most this many held-out images (a fixed prefix of the validation split). 0 = the whole split.",
    DefaultValueFactory = _ => 20000
};
var evaluateBatchSizeOption = new Option<int>("--batch-size") { Description = "Held-out images per forward pass. 32 fits a 16 GB GPU (A4000) for run 4's tower since the forwards have their own dispose scopes; before that, 32 ran out of memory.", DefaultValueFactory = _ => 10 };
var evaluateOutDirOption = new Option<string?>("--out-dir") { Description = "Where to write tag_thresholds.json and evaluation_report.json. Defaults to --checkpoint-dir, where export-onnx picks the thresholds up." };
var embeddingsCacheOption = new Option<string?>("--embeddings-cache")
{
    Description = "File to save the held-out image embeddings to, and reuse them from on later runs (ignored if older than the checkpoint's image tower or the held-out set changed size) -- skips the slow embedding pass when re-evaluating the same checkpoint."
};
var goldOption = new Option<string?>("--gold") { Description = "Hand-checked box set (gold.json) to score localization against" };
var smallGoldOption = new Option<string?>("--gold-small")
{
    Description = "A second hand-checked box set (gold.json), scored on its own into localization_report_small.json (gold scoring, crop teacher, occlusion) and charted separately"
};
var localizationSamplesOption = new Option<int>("--localization-samples")
{
    Description = "Held-out images used for the label-free localization checks (deletion, crop consistency, two-character, solo counts). 0 skips them.",
    DefaultValueFactory = _ => 300
};
var mapScoreTemperatureOption = new Option<float>("--map-score-temperature")
{
    Description = "With --gold: also run the occlusion check with localizable tags scored from the localization map, pooled by a log-sum-exp at this temperature (training's --localization-temperature, 0.25, is the natural choice). 0 (the default) skips it: it repeats the whole occlusion check, one pair at a time.",
    DefaultValueFactory = _ => 0f
};
var worstTagsOption = new Option<int>("--show") { Description = "How many worst tags to list", DefaultValueFactory = _ => 25 };

var evaluateCommand = new Command("evaluate", "Measure a checkpoint's held-out per-tag ranking quality (AP vs. each tag's hard negatives, image-level precision/recall) and calibrate per-tag thresholds");
evaluateCommand.Options.Add(checkpointDirOption);
evaluateCommand.Options.Add(evaluateCacheDirOption);
evaluateCommand.Options.Add(validationFractionOption);
evaluateCommand.Options.Add(evaluateMaxImagesOption);
evaluateCommand.Options.Add(evaluateBatchSizeOption);
evaluateCommand.Options.Add(exclusiveTagsOption);
evaluateCommand.Options.Add(tagParentageOption);
evaluateCommand.Options.Add(evaluateOutDirOption);
evaluateCommand.Options.Add(worstTagsOption);
evaluateCommand.Options.Add(embeddingsCacheOption);
evaluateCommand.Options.Add(goldOption);
evaluateCommand.Options.Add(smallGoldOption);
var dumpOcclusionOption = new Option<bool>("--dump-occlusion")
{
    Description = "Also write occlusion_pairs.jsonl (and occlusion_pairs_small.jsonl): every gold pair of the occlusion check with its full, blanked-object and blanked-control confidence"
};
evaluateCommand.Options.Add(dumpOcclusionOption);
var occlusionFillOption = new Option<string>("--occlusion-fill")
{
    Description = "What the occlusion check fills blanked boxes with: zero (the padding value), mean (the image's mean color) or noise (with the image's colors). Score an erased-view model with a fill it didn't train on: whether its effect holds there says whether it learned the object or the fill.",
    DefaultValueFactory = _ => "zero"
};
occlusionFillOption.AcceptOnlyFromAmong("zero", "mean", "noise");
evaluateCommand.Options.Add(occlusionFillOption);
evaluateCommand.Options.Add(localizationSamplesOption);
evaluateCommand.Options.Add(mapScoreTemperatureOption);
var dumpScoresOption = new Option<string[]>("--dump-scores")
{
    Description = "Also write scores.jsonl: every held-out image's tags and its probability for each of these tags (comma-separated or repeated), to look at one tag's false positives and misses",
    DefaultValueFactory = _ => [],
    AllowMultipleArgumentsPerToken = true
};
evaluateCommand.Options.Add(dumpScoresOption);
var contextCheckOption = new Option<string?>("--context-check")
{
    Description = "Hand-labelled context negatives (JSON list of {cacheIndex, tag, label: present|absent}): reports how often a tag is called on the held-out ones where the object is absent, and where it's visible but untagged. The context-reliance bands can't tell those apart."
};
evaluateCommand.Options.Add(contextCheckOption);
var evaluatePrecisionOption = new Option<string>("--precision")
{
    Description = "The image tower's arithmetic: fp32 (cuDNN's TF32 convolutions, as on CUDA by default), tf32 (also TF32 matrix multiplies, which the 1x1 convolutions run as) or bf16 (bfloat16 compute from the fp32 weights, as train --bf16). A glimpse checkpoint's evaluation scores ~6x the pixels of a plain one, so this is most of its time.",
    DefaultValueFactory = _ => "fp32"
};
evaluatePrecisionOption.AcceptOnlyFromAmong("fp32", "tf32", "bf16");
evaluateCommand.Options.Add(evaluatePrecisionOption);
var glimpseGridOption = new Option<bool>("--glimpse-grid")
{
    Description = "A glimpse checkpoint: score every window of the grid (378 per image at 512px) instead of the ones the model chooses from its localization map, as evaluations did before 2026-10-31. To compare with those; ~6x the cost of a whole-image pass."
};
evaluateCommand.Options.Add(glimpseGridOption);
var glimpseObjectOnlyOption = new Option<bool>("--glimpse-object-only")
{
    Description = "A checkpoint trained with --glimpse-feature-split: score glimpses from the object half of their embedding alone (the context half held at its training mean), as training scores exclusive images. Without it the whole embedding scores, as Singh et al. test."
};
evaluateCommand.Options.Add(glimpseObjectOnlyOption);
var twinCheckOption = new Option<string?>("--twin-check")
{
    Description = "Only check the matched twins' \"no\" windows (--twin-every) against the gold boxes (--gold, --gold-small): how often they hold the object. Writes each pair to this JSONL file."
};
evaluateCommand.Options.Add(twinCheckOption);
var mapWindowCheckOption = new Option<string?>("--map-window-check")
{
    Description = "Only check window labels read off this checkpoint's own localization map against the gold boxes (--gold, --gold-small): the peak window's and the top 15%'s precision, the bottom 40%'s contamination, the map's window AUC. Writes each pair to this JSONL file."
};
evaluateCommand.Options.Add(mapWindowCheckOption);
var peakWindowScoresOption = new Option<string?>("--peak-window-scores")
{
    Description = "Only write, for every held-out image and each --dump-scores tag, the whole-image score and the glimpse scores of the 64px and 128px windows at the tag's map peak (as --self-map-weight picks them), with the label and the --context-check label, to this JSONL file."
};
evaluateCommand.Options.Add(peakWindowScoresOption);
var noCropTeacherOption = new Option<bool>("--no-crop-teacher")
{
    Description = "Leave the crop-teacher check out of the gold checks. It only tells crop consistency whether to trust the classifier, and on a glimpse checkpoint it is ~40% of the gold checks' time."
};
evaluateCommand.Options.Add(noCropTeacherOption);
var localityPriorOutOption = new Option<string?>("--locality-prior-out")
{
    Description = "Instead of evaluating: measure on this checkpoint's localization map how often each pair of labelled localizable tags is found within 64px of each other in training images, and write it here for train --locality-prior."
};
evaluateCommand.Options.Add(localityPriorOutOption);
var localityPriorImagesOption = new Option<int>("--locality-prior-images")
{
    Description = "With --locality-prior-out: how many training images to measure.",
    DefaultValueFactory = _ => 20000
};
evaluateCommand.Options.Add(localityPriorImagesOption);
var goldImagesOption = new Option<string?>("--gold-images")
{
    Description = "Read the gold sets' images from this directory (<cache index>.png, e.g. upscaled copies) instead of the cache, letterboxed at --gold-input-size. The gold.json boxes must be in those images' pixels."
};
evaluateCommand.Options.Add(goldImagesOption);
var goldInputSizeOption = new Option<int?>("--gold-input-size") { Description = "With --gold-images: the input size the gold images are scored at (default: the cache's)." };
evaluateCommand.Options.Add(goldInputSizeOption);
var evaluateInputSizeOption = new Option<int?>("--input-size") { Description = "Score the held-out images (and the gold sets, with --gold-images) at this input size instead of the cache's: cached images rescaled, --hires-store rows from it (as train --train-input-size)." };
evaluateCommand.Options.Add(evaluateInputSizeOption);
var evaluateHiResOption = new Option<string?>("--hires-store") { Description = "A hi-res side store whose rows are read instead of the cache's (as train --hires-store)." };
evaluateCommand.Options.Add(evaluateHiResOption);
evaluateCommand.SetAction(parseResult => EvaluateCommandHandler.Run(
    parseResult.GetRequiredValue(checkpointDirOption),
    parseResult.GetRequiredValue(evaluateCacheDirOption),
    parseResult.GetRequiredValue(validationFractionOption),
    parseResult.GetRequiredValue(evaluateMaxImagesOption),
    parseResult.GetRequiredValue(evaluateBatchSizeOption),
    parseResult.GetValue(exclusiveTagsOption),
    parseResult.GetValue(evaluateOutDirOption),
    parseResult.GetRequiredValue(worstTagsOption),
    parseResult.GetValue(embeddingsCacheOption),
    parseResult.GetValue(goldOption),
    parseResult.GetRequiredValue(localizationSamplesOption),
    parseResult.GetRequiredValue(mapScoreTemperatureOption),
    parseResult.GetValue(dumpScoresOption)?.SelectMany(v => v.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)).ToList(),
    parseResult.GetValue(tagParentageOption),
    parseResult.GetValue(smallGoldOption),
    parseResult.GetValue(dumpOcclusionOption),
    OcclusionFills.Parse(parseResult.GetRequiredValue(occlusionFillOption)),
    parseResult.GetValue(contextCheckOption),
    parseResult.GetRequiredValue(evaluatePrecisionOption),
    parseResult.GetValue(glimpseGridOption),
    !parseResult.GetValue(noCropTeacherOption),
    parseResult.GetValue(localityPriorOutOption),
    parseResult.GetRequiredValue(localityPriorImagesOption),
    parseResult.GetValue(glimpseObjectOnlyOption),
    parseResult.GetValue(twinCheckOption),
    parseResult.GetValue(mapWindowCheckOption),
    parseResult.GetValue(peakWindowScoresOption),
    parseResult.GetValue(goldImagesOption),
    parseResult.GetValue(goldInputSizeOption),
    parseResult.GetValue(evaluateInputSizeOption),
    parseResult.GetValue(evaluateHiResOption)));

var chartsEpochsOption = new Option<int>("--epochs") { Description = "The run's --epochs, for the step axis and epoch markers", DefaultValueFactory = _ => 6 };
var chartsReferenceOption = new Option<string[]>("--reference")
{
    Description = "Another run's evaluation to compare against, as label=report directory (e.g. v3-e1=context-check/v3-e1); drawn as a dashed line in each panel. Repeatable.",
    DefaultValueFactory = _ => [],
    AllowMultipleArgumentsPerToken = false
};
var chartsCompareOption = new Option<string[]>("--compare")
{
    Description = "Another run to draw as a curve over this run's steps, as label=checkpoint directory (e.g. run4=checkpoint-v4): its evaluations in each panel and its smoothed loss. Repeatable.",
    DefaultValueFactory = _ => [],
    AllowMultipleArgumentsPerToken = false
};
var chartsPngOption = new Option<string?>("--png") { Description = "Also (or, when output is redirected, only) save the charts to this PNG" };
var chartsWatchOption = new Option<bool>("--watch") { Description = "Keep the charts up and redraw them whenever a checkpoint save or a finished evaluation changes the data" };
var chartsIntervalOption = new Option<int>("--interval") { Description = "With --watch: seconds between checks for new data", DefaultValueFactory = _ => 30 };
var chartsWidthOption = new Option<int>("--width") { Description = "Image width in pixels (default: the terminal's)", DefaultValueFactory = _ => 0 };
var chartsHeightOption = new Option<int>("--height") { Description = "Image height in pixels (default: the terminal's)", DefaultValueFactory = _ => 0 };
var chartsCommand = new Command("charts", "Draw a run's loss and evaluation trends as graphics: in the terminal as Sixel (Konsole, xterm, WezTerm, tmux 3.4+), or to a PNG. Reads only the checkpoint directory, so it runs alongside training.");
chartsCommand.Options.Add(checkpointDirOption);
chartsCommand.Options.Add(chartsEpochsOption);
chartsCommand.Options.Add(chartsReferenceOption);
chartsCommand.Options.Add(chartsCompareOption);
chartsCommand.Options.Add(chartsPngOption);
chartsCommand.Options.Add(chartsWatchOption);
chartsCommand.Options.Add(chartsIntervalOption);
chartsCommand.Options.Add(chartsWidthOption);
chartsCommand.Options.Add(chartsHeightOption);
chartsCommand.SetAction(parseResult => ChartsCommandHandler.Run(
    parseResult.GetRequiredValue(checkpointDirOption),
    parseResult.GetRequiredValue(chartsEpochsOption),
    parseResult.GetValue(chartsReferenceOption) ?? [],
    parseResult.GetValue(chartsCompareOption) ?? [],
    parseResult.GetValue(chartsPngOption),
    parseResult.GetValue(chartsWatchOption),
    parseResult.GetRequiredValue(chartsIntervalOption),
    parseResult.GetRequiredValue(chartsWidthOption),
    parseResult.GetRequiredValue(chartsHeightOption)));

var benchmarkBatchOption = new Option<int>("--batch-size") { DefaultValueFactory = _ => 12 };
var benchmarkInputOption = new Option<int>("--input-size") { DefaultValueFactory = _ => 512 };
var benchmarkIterationsOption = new Option<int>("--iterations") { DefaultValueFactory = _ => 30 };
var benchmarkLayoutsOption = new Option<string[]>("--layouts")
{
    Description = "Memory layouts to time: nchw, nhwc-input (channels-last input), nhwc-all (input and conv weights), bf16 (whole tower in bfloat16), bf16-mixed (fp32 weights, bfloat16 compute: train --bf16), bf16-mixed-nhwc (the same on channels-last input).",
    DefaultValueFactory = _ => ["nchw", "nhwc-input", "nhwc-all"],
    AllowMultipleArgumentsPerToken = true,
    CustomParser = r => r.Tokens.SelectMany(t => t.Value.Split(',', StringSplitOptions.RemoveEmptyEntries)).ToArray()
};
var soakStepsOption = new Option<int>("--steps") { DefaultValueFactory = _ => 20000 };
var soakPadOption = new Option<int>("--pad-to") { DefaultValueFactory = _ => 1 };
var soakModeOption = new Option<string>("--mode") { Description = "full, no-backward or no-upload.", DefaultValueFactory = _ => "full" };
var soakBatchOption = new Option<int>("--batch-size") { DefaultValueFactory = _ => 10 };
var soakVocabularyOption = new Option<int>("--vocabulary") { DefaultValueFactory = _ => 25000 };
var soakPositivesOption = new Option<int>("--max-positives") { DefaultValueFactory = _ => 60 };
var soakReportOption = new Option<int>("--report-every") { DefaultValueFactory = _ => 500 };
var soakMaxVramOption = new Option<long>("--max-vram-mb") { Description = "Stop once this process's device memory passes this (it may share the device with a training run).", DefaultValueFactory = _ => 4000 };
var soakLocalityCommand = new Command("soak-locality", "Run the locality negatives' step and backward over and over on random training-shaped input and print this process's device memory (finding a leak).");
foreach (var option in new Option[] { soakStepsOption, soakPadOption, soakModeOption, soakBatchOption, soakVocabularyOption, soakPositivesOption, soakReportOption, soakMaxVramOption })
    soakLocalityCommand.Options.Add(option);
soakLocalityCommand.SetAction(parseResult => SoakLocalityCommandHandler.Run(parseResult.GetRequiredValue(soakStepsOption), parseResult.GetRequiredValue(soakPadOption),
    parseResult.GetRequiredValue(soakModeOption), parseResult.GetRequiredValue(soakBatchOption), parseResult.GetRequiredValue(soakVocabularyOption),
    parseResult.GetRequiredValue(soakPositivesOption), parseResult.GetRequiredValue(soakReportOption), parseResult.GetRequiredValue(soakMaxVramOption)));
var benchmarkTowerCommand = new Command("benchmark-tower", "Time run 4's image tower forward+backward on random input per memory layout (for tuning the training step).");
benchmarkTowerCommand.Options.Add(benchmarkBatchOption);
benchmarkTowerCommand.Options.Add(benchmarkInputOption);
benchmarkTowerCommand.Options.Add(benchmarkIterationsOption);
benchmarkTowerCommand.Options.Add(benchmarkLayoutsOption);
var benchmarkComponentsOption = new Option<bool>("--components") { Description = "Time each op of one block per stage (forward + backward) instead of the whole tower, as ConvNeXtBlock runs them; with --layouts bf16-mixed, in bfloat16 as train --bf16 does." };
benchmarkTowerCommand.Options.Add(benchmarkComponentsOption);
var benchmarkGlimpsesOption = new Option<int>("--glimpses") { Description = "Time the glimpse path instead, with this many glimpses per image (--batch-size images at --input-size)." };
var benchmarkVaryOption = new Option<bool>("--vary") { Description = "With --glimpses, drop up to 3 windows per image at random each iteration, as training's sampling does." };
benchmarkTowerCommand.Options.Add(benchmarkGlimpsesOption);
benchmarkTowerCommand.Options.Add(benchmarkVaryOption);
var benchmarkNoPadOption = new Option<bool>("--no-pad") { Description = "With --glimpses, don't pad the glimpse batch to a fixed size." };
benchmarkTowerCommand.Options.Add(benchmarkNoPadOption);
var benchmarkGlimpseChunkOption = new Option<int>("--glimpse-chunk") { Description = "With --glimpses, glimpses per tower pass (Glimpses.Embed's chunk).", DefaultValueFactory = _ => UnbooruTagger.Training.Model.Glimpses.DefaultChunk };
benchmarkTowerCommand.Options.Add(benchmarkGlimpseChunkOption);
var benchmarkUploadsOption = new Option<bool>("--uploads") { Description = "Time how long a batch's upload holds the CPU while the device is busy: float vs uint8 pixels, ordinary vs pinned memory." };
benchmarkTowerCommand.Options.Add(benchmarkUploadsOption);
benchmarkTowerCommand.SetAction(parseResult => (parseResult.GetValue(benchmarkLayoutsOption) ?? []).Contains("probe")
    ? BenchmarkTowerCommandHandler.ProbeLayout()
    : parseResult.GetValue(benchmarkUploadsOption)
    ? BenchmarkTowerCommandHandler.RunUploads(parseResult.GetRequiredValue(benchmarkBatchOption), parseResult.GetRequiredValue(benchmarkInputOption),
        parseResult.GetRequiredValue(benchmarkIterationsOption))
    : parseResult.GetValue(benchmarkGlimpsesOption) > 0
    ? BenchmarkTowerCommandHandler.RunGlimpses(
        parseResult.GetRequiredValue(benchmarkBatchOption),
        parseResult.GetRequiredValue(benchmarkInputOption),
        parseResult.GetRequiredValue(benchmarkIterationsOption),
        parseResult.GetValue(benchmarkGlimpsesOption),
        parseResult.GetValue(benchmarkVaryOption),
        !parseResult.GetValue(benchmarkNoPadOption),
        parseResult.GetRequiredValue(benchmarkGlimpseChunkOption))
    : parseResult.GetValue(benchmarkComponentsOption)
    ? BenchmarkTowerCommandHandler.RunComponents(
        parseResult.GetRequiredValue(benchmarkBatchOption),
        parseResult.GetRequiredValue(benchmarkInputOption),
        parseResult.GetRequiredValue(benchmarkIterationsOption),
        (parseResult.GetValue(benchmarkLayoutsOption) ?? []).Contains("bf16-mixed"))
    : BenchmarkTowerCommandHandler.Run(
        parseResult.GetRequiredValue(benchmarkBatchOption),
        parseResult.GetRequiredValue(benchmarkInputOption),
        parseResult.GetRequiredValue(benchmarkIterationsOption),
        parseResult.GetRequiredValue(benchmarkLayoutsOption)));

var glimpseFramesCommand = new Command("glimpse-frames", "How well the glimpse choice (Glimpses.Propose, Sensor) fits the gold set's boxes. Reads any checkpoint with a localization map.");
var glimpseFramesGoldOption = new Option<string>("--gold") { Description = "The gold set's gold.json", Required = true };
var glimpseFramesCountOption = new Option<int[]>("--counts") { Description = "Window budgets per image to try Propose with", DefaultValueFactory = _ => [16, 32, 48], AllowMultipleArgumentsPerToken = true };
glimpseFramesCommand.Options.Add(checkpointDirOption);
glimpseFramesCommand.Options.Add(evaluateCacheDirOption);
glimpseFramesCommand.Options.Add(glimpseFramesGoldOption);
glimpseFramesCommand.Options.Add(glimpseFramesCountOption);
glimpseFramesCommand.SetAction(parseResult => GlimpseFramesCommandHandler.Run(
    parseResult.GetRequiredValue(checkpointDirOption),
    parseResult.GetRequiredValue(evaluateCacheDirOption),
    parseResult.GetRequiredValue(glimpseFramesGoldOption),
    parseResult.GetRequiredValue(glimpseFramesCountOption),
    16));

var windowProbeCommand = new Command("window-probe", "The glimpse sensor's ceiling: fine-tune the glimpse path on the gold boxes' windows, k-fold by image, and dump the held-out images' windows as evaluate --dump-occlusion does.");
var windowProbeGoldOption = new Option<string[]>("--gold") { Description = "gold.json files (the first's dumps are glimpse_*.jsonl, the rest glimpse_*_small.jsonl)", Required = true, AllowMultipleArgumentsPerToken = true };
var windowProbeWindowsOption = new Option<string[]>("--windows") { Description = "Per gold file, the evaluate --dump-occlusion directory whose glimpse_embeddings*.jsonl give each image's windows", Required = true, AllowMultipleArgumentsPerToken = true };
var windowProbeOutOption = new Option<string>("--out-dir") { Required = true };
var windowProbeFoldsOption = new Option<int>("--folds") { DefaultValueFactory = _ => 5 };
var windowProbeStepsOption = new Option<int>("--steps") { Description = "Fine-tuning steps per fold (0 = the untrained baseline)", DefaultValueFactory = _ => 400 };
var windowProbeImagesOption = new Option<int>("--images-per-step") { DefaultValueFactory = _ => 8 };
var windowProbeWindowsPerImageOption = new Option<int>("--windows-per-image") { DefaultValueFactory = _ => 48 };
var windowProbeLrOption = new Option<float>("--lr") { DefaultValueFactory = _ => 1e-4f };
var windowProbeBf16Option = new Option<bool>("--bf16");
var windowProbeSeedOption = new Option<int>("--seed") { DefaultValueFactory = _ => 0 };
windowProbeCommand.Options.Add(checkpointDirOption);
windowProbeCommand.Options.Add(evaluateCacheDirOption);
foreach (var option in new Option[] { windowProbeGoldOption, windowProbeWindowsOption, windowProbeOutOption, windowProbeFoldsOption, windowProbeStepsOption,
             windowProbeImagesOption, windowProbeWindowsPerImageOption, windowProbeLrOption, windowProbeBf16Option, windowProbeSeedOption })
    windowProbeCommand.Options.Add(option);
windowProbeCommand.SetAction(parseResult => WindowProbeCommandHandler.Run(
    parseResult.GetRequiredValue(checkpointDirOption),
    parseResult.GetRequiredValue(evaluateCacheDirOption),
    parseResult.GetRequiredValue(windowProbeGoldOption),
    parseResult.GetRequiredValue(windowProbeWindowsOption),
    parseResult.GetRequiredValue(windowProbeOutOption),
    parseResult.GetValue(windowProbeFoldsOption),
    parseResult.GetValue(windowProbeStepsOption),
    parseResult.GetValue(windowProbeImagesOption),
    parseResult.GetValue(windowProbeWindowsPerImageOption),
    parseResult.GetValue(windowProbeLrOption),
    parseResult.GetValue(windowProbeBf16Option),
    parseResult.GetValue(windowProbeSeedOption)));

var rootCommand = new RootCommand("unbooru-tagger training CLI")
{
    windowProbeCommand,
    glimpseFramesCommand,
    trainCommand,
    benchmarkTowerCommand,
    soakLocalityCommand,
    addTagCommand,
    exportOnnxCommand,
    evaluateCommand,
    chartsCommand
};

return rootCommand.Parse(args).Invoke();
