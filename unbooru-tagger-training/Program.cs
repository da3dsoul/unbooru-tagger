using System.CommandLine;
using UnbooruTagger.Training.Commands;

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
var earlyStoppingPatienceOption = new Option<int>("--early-stopping-patience") { Description = "Epochs without validation-loss improvement before stopping", DefaultValueFactory = _ => 3 };
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
var exclusiveTagsOption = new Option<string?>("--exclusive-tags")
{
    Description = "File of mutually exclusive tag groups (one comma-separated group per line, e.g. 'solo, multiple_girls'). Omit to reuse the checkpoint's saved copy, or the built-in people-count/rating groups."
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
trainCommand.Options.Add(exclusiveTagsOption);
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
trainCommand.Options.Add(textDashboardOption);
trainCommand.Options.Add(trainGoldOption);
trainCommand.Options.Add(cropConsistencyWeightOption);
trainCommand.Options.Add(cropTeacherMinApOption);
trainCommand.Options.Add(cropTeacherConfidenceOption);
trainCommand.Options.Add(mosaicsPerStepOption);
trainCommand.Options.Add(mosaicWeightOption);
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
        ExclusiveTagsPath: parseResult.GetValue(exclusiveTagsOption),
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
        LocalizationFromFusedOnly: parseResult.GetValue(localizationMapOption) switch { "fused" or "local" => true, "full" => false, _ => null },
        LocalizationSkippedLaterals: parseResult.GetValue(localizationMapOption) switch { "local" => 1, "fused" or "full" => 0, _ => null },
        LocalizableTagsPath: parseResult.GetValue(localizableTagsOption),
        CompactVocabulary: !parseResult.GetValue(keepVocabularyOption),
        CosineSchedule: parseResult.GetRequiredValue(lrScheduleOption) == "cosine",
        WarmupSteps: parseResult.GetRequiredValue(warmupStepsOption),
        EvaluationCommand: parseResult.GetValue(evaluationCommandOption),
        SnapshotEvaluationSteps: parseResult.GetRequiredValue(snapshotEvaluationStepsOption),
        ChartReferences: parseResult.GetValue(chartReferenceOption) ?? [],
        TextDashboard: parseResult.GetValue(textDashboardOption)),
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
exportOnnxCommand.SetAction(parseResult => ExportOnnxCommandHandler.Run(
    parseResult.GetRequiredValue(checkpointDirOption),
    parseResult.GetRequiredValue(modelDirOption)));

var evaluateCacheDirOption = new Option<string>("--cache-dir") { Description = "The PreprocessedDatasetCache the checkpoint was trained on", Required = true };
var evaluateMaxImagesOption = new Option<int>("--max-images")
{
    Description = "Evaluate at most this many held-out images (a fixed prefix of the validation split). 0 = the whole split.",
    DefaultValueFactory = _ => 20000
};
var evaluateBatchSizeOption = new Option<int>("--batch-size") { Description = "Held-out images per forward pass; 32 runs out of memory on a 16 GB GPU with the ConvNeXt-T tower.", DefaultValueFactory = _ => 10 };
var evaluateOutDirOption = new Option<string?>("--out-dir") { Description = "Where to write tag_thresholds.json and evaluation_report.json. Defaults to --checkpoint-dir, where export-onnx picks the thresholds up." };
var embeddingsCacheOption = new Option<string?>("--embeddings-cache")
{
    Description = "File to save the held-out image embeddings to, and reuse them from on later runs (ignored if older than the checkpoint's image tower or the held-out set changed size) -- skips the slow embedding pass when re-evaluating the same checkpoint."
};
var goldOption = new Option<string?>("--gold") { Description = "Hand-checked box set (gold.json) to score localization against" };
var localizationSamplesOption = new Option<int>("--localization-samples")
{
    Description = "Held-out images used for the label-free localization checks (deletion, crop consistency, two-character, solo counts). 0 skips them.",
    DefaultValueFactory = _ => 300
};
var worstTagsOption = new Option<int>("--show") { Description = "How many worst tags to list", DefaultValueFactory = _ => 25 };

var evaluateCommand = new Command("evaluate", "Measure a checkpoint's held-out per-tag ranking quality (AP vs. each tag's hard negatives, image-level precision/recall) and calibrate per-tag thresholds");
evaluateCommand.Options.Add(checkpointDirOption);
evaluateCommand.Options.Add(evaluateCacheDirOption);
evaluateCommand.Options.Add(validationFractionOption);
evaluateCommand.Options.Add(evaluateMaxImagesOption);
evaluateCommand.Options.Add(evaluateBatchSizeOption);
evaluateCommand.Options.Add(exclusiveTagsOption);
evaluateCommand.Options.Add(evaluateOutDirOption);
evaluateCommand.Options.Add(worstTagsOption);
evaluateCommand.Options.Add(embeddingsCacheOption);
evaluateCommand.Options.Add(goldOption);
evaluateCommand.Options.Add(localizationSamplesOption);
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
    parseResult.GetRequiredValue(localizationSamplesOption)));

var chartsEpochsOption = new Option<int>("--epochs") { Description = "The run's --epochs, for the step axis and epoch markers", DefaultValueFactory = _ => 6 };
var chartsReferenceOption = new Option<string[]>("--reference")
{
    Description = "Another run's evaluation to compare against, as label=report directory (e.g. v3-e1=context-check/v3-e1); drawn as a dashed line in each panel. Repeatable.",
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
chartsCommand.Options.Add(chartsPngOption);
chartsCommand.Options.Add(chartsWatchOption);
chartsCommand.Options.Add(chartsIntervalOption);
chartsCommand.Options.Add(chartsWidthOption);
chartsCommand.Options.Add(chartsHeightOption);
chartsCommand.SetAction(parseResult => ChartsCommandHandler.Run(
    parseResult.GetRequiredValue(checkpointDirOption),
    parseResult.GetRequiredValue(chartsEpochsOption),
    parseResult.GetValue(chartsReferenceOption) ?? [],
    parseResult.GetValue(chartsPngOption),
    parseResult.GetValue(chartsWatchOption),
    parseResult.GetRequiredValue(chartsIntervalOption),
    parseResult.GetRequiredValue(chartsWidthOption),
    parseResult.GetRequiredValue(chartsHeightOption)));

var rootCommand = new RootCommand("unbooru-tagger training CLI")
{
    trainCommand,
    addTagCommand,
    exportOnnxCommand,
    evaluateCommand,
    chartsCommand
};

return rootCommand.Parse(args).Invoke();
