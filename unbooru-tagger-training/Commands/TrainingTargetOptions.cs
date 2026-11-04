using UnbooruTagger.Training.Training;

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
/// <param name="OutOfContextShare">Fraction of a localizable general anchor's positives drawn for a low context prior: the lowest-prior of 16 random positives, if below <paramref name="OutOfContextMaxPrior"/> (<see cref="Training.ContextPriorTable"/>). 0 = off.</param>
/// <param name="OutOfContextMaxPrior">Highest context prior an out-of-context positive may have.</param>
/// <param name="OutOfContextAll">Give <paramref name="OutOfContextWeight"/> to every localizable general positive in a batch whose context prior is below <paramref name="OutOfContextMaxPrior"/>, not only sampled anchors'.</param>
/// <param name="OutOfContextWeight">Weight of an out-of-context positive's anchor pair (PairWeightBuilder); 0 = the usual 1.</param>
/// <param name="ExcluderShare">Fraction of anchors drawn from the tags an excludes rule rules out (gloves for no_gloves), each with two excluder images as certain negatives. 0 = off.</param>
/// <param name="ExcludedNegativeWeight">Weight of a tag ruled out by an excludes rule, if above --certain-negative-weight. 0 = that weight.</param>
/// <param name="HardNegativeSourceWeight">Train every pair where an image carries one of a tag's hard-negative sources but not the tag at full strength with this weight (PairWeightBuilder), not only the anchors' sampled hard negatives. Localizable tags only. 0 = off.</param>
/// <param name="ClassificationPoolingTemperature">Score localizable tags by a log-sum-exp over the full map's cells at this temperature instead of the averaged embedding (Model.ClassificationPooling); 0 = averaged; null keeps the checkpoint's setting.</param>
/// <param name="ClassificationPoolingStartTemperature">With <paramref name="ClassificationPoolingAnnealSteps"/>, the temperature annealed down from (ClassificationPooling.Schedule); null = no anneal.</param>
/// <param name="ClassificationPoolingAnnealSteps">Global steps to anneal from <paramref name="ClassificationPoolingStartTemperature"/> to <paramref name="ClassificationPoolingTemperature"/>, counted from when pooling was turned on.</param>
/// <param name="ClassificationGlimpses">A fresh run's localizable tags are scored from zoomed glimpses (Model.Glimpses) instead of the map's cells, at ClassificationPoolingTemperature.</param>
/// <param name="GlimpsesPerImage">With ClassificationGlimpses, the windows the model chooses to look at per image (Model.Glimpses.Propose), in training and (saved with the checkpoint) evaluation.</param>
/// <param name="GlimpseUnframedPositiveWeight">With ClassificationGlimpses, the glimpse loss's weight for a labelled positive whose map peak none of the chosen windows holds: the classifier never saw it, so training it "yes" would teach the tag from whatever the windows did show. 0 leaves it to the localization loss, which teaches the map to find it.</param>
/// <param name="GlimpseLabelFramingSteps">With ClassificationGlimpses, a fresh run's warm-up: a share of each image's windows (half at step 0, falling linearly to none at this global step) frames labelled positives, by the same rule as the model's own choice, while the localization map can't point yet. 0 = the model's choice from the start.</param>
/// <param name="GlimpseProposalWeight">With ClassificationGlimpses, the weight of the proposal loss: the whole-image proposal scores (Model.ImageTower.Proposals), which pick where the glimpses look, trained on every localizable tag from the labels.</param>
/// <param name="GlimpseProposalDetachTags">With ClassificationGlimpses, the proposal loss trains only the proposal head and the tower, not the tag embeddings: the whole-image context it learns ("this character, so gloves") then can't pull the embeddings the localization map points with.</param>
/// <param name="LocalityNegativeWeight">With ClassificationGlimpses, the weight of the locality negatives (<see cref="Training.LocalityNegatives"/>): a close window where one labelled tag is confidently found trains each other labelled tag "no" there, weighted by how rarely the two are together (learned as training runs). 0 = off.</param>
/// <param name="LocalityPriorPath">The locality negatives' starting table (<c>evaluate --locality-prior-out</c>), or null to learn it from nothing.</param>
/// <param name="LocalityMinObservations">Weighted observations a pair needs before it gives locality negatives.</param>
/// <param name="LocalityConfidenceGate">Only windows where the found tag's call is at least this sure give locality negatives; 0 = any.</param>
/// <param name="LocalityPerPair">Each image's locality term is the mean over its pairs, not the sum.</param>
/// <param name="LocalityDetachScaling">The locality term trains no gradient into the shared logit scale and bias.</param>
/// <param name="ClassificationPoolingMinedNegatives">Per image, the highest-scoring pooled negatives trained on their pooled score each step (ClassificationPooling.TrainedRows), besides positives and full-strength negatives.</param>
/// <param name="ContextPriorWeight">Deliberate targets: add this times the context prior's log-odds over the base rate to localizable general tags' training logits (<see cref="Training.ContextPrior"/>), so the image tower isn't rewarded for re-learning "this character, therefore gloves". 0 = off.</param>
/// <param name="ExclusiveTagsPath">Mutually-exclusive tag groups file; null uses the checkpoint's saved copy, else the built-in defaults.</param>
/// <param name="TagParentagePath">Reviewed tag parentage rules (TagParentage); null uses the checkpoint's saved copy, "none" turns them off.</param>
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
/// <param name="ErasurePairsPerStep">Erased-view pairs per step (ErasureContrast): a called localizable positive with the region its map points at blanked, and a same-size control blanked. Costs a forward/backward pass over two images per pair. 0 = off.</param>
/// <param name="ErasureLoss">Hinge (object-blanked score a margin below control-blanked), Negative (object-blanked a negative, control-blanked a positive) or NegativeOnly (object-blanked a negative; the control only through the anchor).</param>
/// <param name="ErasureWeight">Erased-view loss weight, relative to the whole-image tag loss.</param>
/// <param name="ErasureMargin">The hinge's margin, in log-odds.</param>
/// <param name="ErasureAnchorWeight">Also hold each pair's control-blanked score to its unblanked score, at this weight (ErasureContrast.Compute). 0 = off.</param>
/// <param name="ValidationSplitSeed">A fresh run's validation split seed (null = random), so it can hold out the same images as another run
/// (gold-small and the labelled context check are drawn from run 4's held-out images). A resume keeps its own and refuses a different one.</param>
/// <param name="ErasureEvery">Erased views on every Nth step only (1 = every step), for a lower dose at a lower cost.</param>
/// <param name="ErasureFillNames">Fills blanked regions are drawn from, one per pair (OcclusionFill); null = OcclusionFills.Training.</param>
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
/// <param name="ChartComparisons">Other runs' checkpoint directories to draw as curves over this run's steps on the dashboard's graphical charts, as label=directory (what <c>charts --compare</c> draws).</param>
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
    int? ClassificationPoolingTopK = null,
    float? ClassificationPoolingBottomWeight = null,
    int? GlimpsePoolZooms = null,
    int ClassificationPoolingMinedNegatives = 64,
    bool ClassificationGlimpses = false,
    int GlimpsesPerImage = 16,
    float GlimpseUnframedPositiveWeight = 0f,
    int GlimpseLabelFramingSteps = 0,
    int GlimpseLabelFramingFloor = 0,
    bool GlimpseFramedPositivePooling = false,
    bool SeparateGlimpseTower = false,
    float GlimpseTowerLearningRateScale = 1f,
    bool FineLocalization = false,
    int ContextPairsPerBatch = 0,
    float ContextPairMargin = 1f,
    float ContextPairWeight = 0f,
    bool GlimpseFeatureSplit = false,
    float GlimpseFeatureSplitWeight = 1f,
    bool GlimpseFeatureSplitNegatives = false,
    int GlimpseExclusivePerBatch = 0,
    float MosaicGlimpseWeight = 0f,
    int TwinEvery = 0,
    float TwinWeight = 1f,
    bool TwinDetachScaling = false,
    int TwinPaste = 0,
    float SelfMapWeight = 0f,
    bool SelfMapDetachScaling = false,
    string SelfMapNegatives = "none",
    float SelfMapNegativeWeight = 1f,
    int MosaicLargeGrid = 0,
    int MosaicLargeGridEvery = 2,
    int? GlimpsePoolMinZoom = null,
    bool HillNegatives = false,
    float GlimpseProposalWeight = 1f,
    bool GlimpseProposalDetachTags = false,
    float LocalityNegativeWeight = 0f,
    string? LocalityPriorPath = null,
    float LocalityMinObservations = 20f,
    float LocalityConfidenceGate = 0f,
    bool LocalityPerPair = false,
    bool LocalityDetachScaling = false,
    float HardNegativeSourceWeight = 0f,
    double OutOfContextShare = 0,
    double OutOfContextMaxPrior = 0.3,
    float OutOfContextWeight = 0f,
    bool OutOfContextAll = false,
    double ExcluderShare = 0,
    float ExcludedNegativeWeight = 0f,
    string? ExclusiveTagsPath = null,
    string? TagParentagePath = null,
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
    int ErasurePairsPerStep = 0,
    ErasureLoss ErasureLoss = ErasureLoss.Hinge,
    float ErasureWeight = 1f,
    float ErasureMargin = 1f,
    IReadOnlyList<OcclusionFill>? ErasureFillNames = null,
    int ErasureEvery = 1,
    float ErasureAnchorWeight = 0f,
    int? ValidationSplitSeed = null,
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
    bool TextDashboard = false,
    IReadOnlyList<string>? ChartComparisons = null,
    int? TrainInputSize = null,
    string? HiResStore = null,
    IReadOnlyList<string>? FocusTagNames = null,
    string? TrainScales = null)
{
    public IReadOnlyList<OcclusionFill> ErasureFills => ErasureFillNames is { Count: > 0 } fills ? fills : OcclusionFills.Training;

    public static TrainingTargetOptions Default { get; } = new();

    public static TrainingTargetOptions Legacy { get; } = new(TagCentricSampling: false, DeliberateTargets: false);
}
