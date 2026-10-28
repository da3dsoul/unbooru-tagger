namespace UnbooruTagger.Training.Commands;

/// <summary>
/// How <see cref="TrainCommandHandler"/> builds batches and per-pair targets. The defaults
/// are the deliberate mode; <see cref="Legacy"/> reproduces the original random-image
/// batches with every untagged tag treated as an equally certain negative.
/// </summary>
/// <param name="TagCentricSampling">Anchor each batch on a few tags' positives and hard negatives (<see cref="Training.TagCentricBatchSampler"/>) instead of rare-tag-weighted random images.</param>
/// <param name="PositivesPerAnchor">Positives drawn per anchor tag.</param>
/// <param name="HardNegativesPerAnchor">Hard negatives drawn per anchor tag.</param>
/// <param name="DeliberateTargets">Weight each (image, tag) pair by how certain the corpus is about it (<see cref="Training.PairWeightBuilder"/>) and use the asymmetric loss for unverified negatives.</param>
/// <param name="NegativeGamma">Focal exponent damping unverified negatives (ASL's gamma-).</param>
/// <param name="NegativeClip">Probability margin below which an unverified negative costs nothing (ASL's m).</param>
/// <param name="CertainNegativeWeight">Weight of anchor hard negatives and mutually-exclusive tags.</param>
/// <param name="ImpliedThreshold">P(t|s) at or above which t is "usually implied" by s. Between this and <paramref name="MissingThreshold"/>, s is a context source for t (TagStatistics.ContextSources): an image with s but not t is an unverified negative and is sampled for t as a context negative.</param>
/// <param name="MissingThreshold">P(t|s) at or above which an image with s but not t is a likely-missing label (weight 0) rather than a negative.</param>
/// <param name="ContextNegativesPerAnchor">Context negatives drawn per anchor tag (for tags that have context sources).</param>
/// <param name="ContextNegativeWeight">Train context negatives (an image with one of a tag's context sources but not the tag) at full strength with this weight instead of as focal-damped unverified negatives (PairWeightBuilder). Localizable tags only. 0 = unverified, as before.</param>
/// <param name="HardNegativeSourceWeight">Train every pair where an image carries one of a tag's hard-negative sources but not the tag at full strength with this weight (PairWeightBuilder), not only the anchors' sampled hard negatives. Localizable tags only. 0 = off.</param>
/// <param name="ClassificationPoolingTemperature">Score localizable tags by a log-sum-exp over the full map's cells at this temperature instead of the averaged embedding (Model.ClassificationPooling); 0 = averaged; null keeps the checkpoint's setting.</param>
/// <param name="ClassificationPoolingStartTemperature">With <paramref name="ClassificationPoolingAnnealSteps"/>, the temperature annealed down from (ClassificationPooling.Schedule); null = no anneal.</param>
/// <param name="ClassificationPoolingAnnealSteps">Global steps to anneal from <paramref name="ClassificationPoolingStartTemperature"/> to <paramref name="ClassificationPoolingTemperature"/>, counted from when pooling was turned on.</param>
/// <param name="ClassificationPoolingMinedNegatives">Per image, the highest-scoring pooled negatives trained on their pooled score each step (ClassificationPooling.TrainedRows), besides positives and full-strength negatives.</param>
/// <param name="ContextPriorWeight">Deliberate targets: add this times the context prior's log-odds over the base rate to localizable general tags' training logits (<see cref="Training.ContextPrior"/>), so the image tower isn't rewarded for re-learning "this character, therefore gloves". 0 = off.</param>
/// <param name="ExclusiveTagsPath">Mutually-exclusive tag groups file; null uses the checkpoint's saved copy, else the built-in defaults.</param>
/// <param name="LocalizationMinedNegatives">Deliberate targets: per image, how many of the highest-scoring wrong (localizable) tags the per-location loss also trains on, besides positives and certain negatives.</param>
/// <param name="LocalizationBackgroundWeight">Weight of the background-suppression term (SigmoidContrastiveLoss.ComputeLocalizedPairs): each positive tag's least tag-like locations must not score as the tag. 0 = off.</param>
/// <param name="LocalizationBackgroundFraction">Fraction of an image's locations, lowest-scoring first, treated as background for that tag.</param>
/// <param name="Bf16Compute">Run the image tower in bfloat16 with fp32 weights (ImageTower.ComputeType). Not saved: pass it on every resume.</param>
/// <param name="LocalizationFromFusedOnly">Train localization (and draw heatmaps) on the fused laterals only, without the last stage (ImageTower.LocalizationFromFusedOnly). Null keeps the checkpoint's setting.</param>
/// <param name="LocalizationSkippedLaterals">With a fused-only map, how many of the coarsest laterals it leaves out (ImageTower.LocalizationSkippedLaterals; 1 = <c>--localization-map local</c>). Null keeps the checkpoint's setting.</param>
/// <param name="CropConsistencyWeight">Weight of the crop-consistency loss (SigmoidContrastiveLoss.ComputeWindowConsistency): the classifier's verdict on a random crop teaches the localization map where each tag is. Costs an extra no-gradient forward pass per step. 0 = off.</param>
/// <param name="CropTeacherMinAveragePrecision">Crop consistency only trains tags whose held-out AP in the latest epoch evaluation is at least this (CropTeacherGate), so it waits for the classifier to be a trustworthy teacher, and the trusted set grows as training improves it.</param>
/// <param name="CropTeacherMinPositives">...and at least this many held-out positives, so a rare tag's noisy AP can't qualify it.</param>
/// <param name="CropTeacherConfidence">Crop consistency only trains on teacher answers with p above this or below 1 minus it.</param>
/// <param name="MosaicsPerStep">Mosaics built per step from the batch's own images (Mosaic): four at half size per canvas, each tile a localization bag with the other tiles' tags as negatives. Costs a forward/backward pass per mosaic. 0 = off.</param>
/// <param name="MosaicWeight">Mosaic loss weight, relative to the localization weight.</param>
/// <param name="MosaicClassificationWeight">Also classify each mosaic tile: the canvas's full map pooled over the tile, trained on the tile image's whole-image targets (Mosaic.TileTargets). 0 = mosaics train the localization map only.</param>
/// <param name="MosaicFullSizeTiles">Mosaic tiles at the input size on a canvas twice as wide, instead of shrunk 2x (about 4x a mosaic's cost).</param>
/// <param name="FocusShare">Fraction of anchors drawn from the tags the latest epoch evaluation found answered from context (FocusTags), each with as many context negatives as positives and grouped on one mosaic. 0 = off.</param>
/// <param name="GoldPath">Hand-checked gold.json: scored (localization + crop teacher, LocalizationEvaluator) at every epoch evaluation and written to localization_report.json. Null = skip.</param>
/// <param name="MaxSteps">Stop after this many steps in this process, saving a checkpoint (for short comparison runs). 0 = no limit.</param>
/// <param name="LocalizableTagsPath">Rules for which tags have a region to localize (see <c>TagLocalizability</c>); null uses the checkpoint's saved copy, else the built-in defaults.</param>
/// <param name="CompactVocabulary">Fresh runs only: drop vocabulary rows no image carries.</param>
/// <param name="CosineSchedule">Linear warmup over <paramref name="WarmupSteps"/>, then cosine decay of the learning rate across the run (see LearningRateSchedule). Off = constant rate.</param>
/// <param name="Ratings">Learn content ratings from the dataset's ratings.bin, if present, as three cumulative labels.</param>
/// <param name="EvaluationCommand">Run each epoch's evaluation as <c>&lt;command&gt; &lt;snapshot dir&gt; &lt;report dir&gt;</c> in the background instead of inline, e.g. `evaluate` on another GPU (see OffloadedEvaluations). Null = inline.</param>
/// <param name="SnapshotEvaluationSteps">With <paramref name="EvaluationCommand"/>: also snapshot and evaluate the model every this many steps (counted over the whole run), alongside training, for the dashboard's trend charts (see SnapshotEvaluations). Doesn't affect early stopping. 0 = epochs only.</param>
/// <param name="ChartReferences">Other runs' evaluation report directories to draw as reference lines on the dashboard's graphical charts, as label=directory.</param>
/// <param name="TextDashboard">Keep the all-text dashboard even when the terminal can show the graphical charts (DashboardScreen).</param>
/// <param name="ValidationMaxImages">Cap on held-out images evaluated per epoch (a fixed prefix of the split, so it's the same images every epoch). 0 = the whole split.</param>
/// <summary>Image tower shape for a fresh run (ignored on resume, where the checkpoint's config wins).</summary>
/// <param name="FusedStages">Earlier stages fused into the spatial map (ImageTower.FusedStages): each one halves the heatmap stride.</param>
public sealed record TowerOptions(int StemChannels, int[] StageChannels, int[] BlocksPerStage, int FusedStages)
{
    /// <summary>The original 1.7M-parameter tower: 64/128/256 channels, 2/2/2 blocks, stride 16, no fusion.</summary>
    public static TowerOptions Original { get; } = new(64, [64, 128, 256], [2, 2, 2], 0);
}

public sealed record TrainingTargetOptions(
    bool TagCentricSampling = true,
    int PositivesPerAnchor = 2,
    int HardNegativesPerAnchor = 2,
    bool DeliberateTargets = true,
    float NegativeGamma = 4f,
    float NegativeClip = 0.05f,
    float CertainNegativeWeight = 1f,
    double ImpliedThreshold = 0.6,
    double MissingThreshold = 0.9,
    int ContextNegativesPerAnchor = 1,
    float ContextNegativeWeight = 0f,
    float ContextPriorWeight = 0f,
    float? ClassificationPoolingTemperature = null,
    float? ClassificationPoolingStartTemperature = null,
    int ClassificationPoolingAnnealSteps = 0,
    int ClassificationPoolingMinedNegatives = 64,
    float HardNegativeSourceWeight = 0f,
    string? ExclusiveTagsPath = null,
    int ValidationMaxImages = 0,
    bool Ratings = true,
    int LocalizationMinedNegatives = 32,
    float LocalizationBackgroundWeight = 0f,
    float LocalizationBackgroundFraction = 0.4f,
    int MaxSteps = 0,
    string? GoldPath = null,
    float CropConsistencyWeight = 0f,
    double CropTeacherMinAveragePrecision = 0.8,
    float CropTeacherConfidence = 0.9f,
    int CropTeacherMinPositives = 20,
    int MosaicsPerStep = 0,
    float MosaicWeight = 1f,
    float MosaicClassificationWeight = 0f,
    bool MosaicFullSizeTiles = false,
    double FocusShare = 0,
    bool? LocalizationFromFusedOnly = null,
    bool Bf16Compute = false,
    int? LocalizationSkippedLaterals = null,
    string? LocalizableTagsPath = null,
    bool CompactVocabulary = true,
    bool CosineSchedule = false,
    int WarmupSteps = 0,
    string? EvaluationCommand = null,
    int SnapshotEvaluationSteps = 0,
    IReadOnlyList<string>? ChartReferences = null,
    bool TextDashboard = false)
{
    public static TrainingTargetOptions Default { get; } = new();

    public static TrainingTargetOptions Legacy { get; } = new(TagCentricSampling: false, DeliberateTargets: false);
}
