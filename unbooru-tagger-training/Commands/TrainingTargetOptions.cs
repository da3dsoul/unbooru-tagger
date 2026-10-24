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
/// <param name="ExclusiveTagsPath">Mutually-exclusive tag groups file; null uses the checkpoint's saved copy, else the built-in defaults.</param>
/// <param name="LocalizationMinedNegatives">Deliberate targets: per image, how many of the highest-scoring wrong (localizable) tags the per-location loss also trains on, besides positives and certain negatives.</param>
/// <param name="LocalizationBackgroundWeight">Weight of the background-suppression term (SigmoidContrastiveLoss.ComputeLocalizedPairs): each positive tag's least tag-like locations must not score as the tag. 0 = off.</param>
/// <param name="LocalizationBackgroundFraction">Fraction of an image's locations, lowest-scoring first, treated as background for that tag.</param>
/// <param name="LocalizationFromFusedOnly">Train localization (and draw heatmaps) on the fused laterals only, without the last stage (ImageTower.LocalizationFromFusedOnly). Null keeps the checkpoint's setting.</param>
/// <param name="CropConsistencyWeight">Weight of the crop-consistency loss (SigmoidContrastiveLoss.ComputeWindowConsistency): the classifier's verdict on a random crop teaches the localization map where each tag is. Costs an extra no-gradient forward pass per step. 0 = off.</param>
/// <param name="CropTeacherMinAveragePrecision">Crop consistency only trains tags whose held-out AP in the latest epoch evaluation is at least this (CropTeacherGate), so it waits for the classifier to be a trustworthy teacher, and the trusted set grows as training improves it.</param>
/// <param name="CropTeacherMinPositives">...and at least this many held-out positives, so a rare tag's noisy AP can't qualify it.</param>
/// <param name="CropTeacherConfidence">Crop consistency only trains on teacher answers with p above this or below 1 minus it.</param>
/// <param name="GoldPath">Hand-checked gold.json: scored (localization + crop teacher, LocalizationEvaluator) at every epoch evaluation and written to localization_report.json. Null = skip.</param>
/// <param name="MaxSteps">Stop after this many steps in this process, saving a checkpoint (for short comparison runs). 0 = no limit.</param>
/// <param name="LocalizableTagsPath">Rules for which tags have a region to localize (see <c>TagLocalizability</c>); null uses the checkpoint's saved copy, else the built-in defaults.</param>
/// <param name="CompactVocabulary">Fresh runs only: drop vocabulary rows no image carries.</param>
/// <param name="CosineSchedule">Linear warmup over <paramref name="WarmupSteps"/>, then cosine decay of the learning rate across the run (see LearningRateSchedule). Off = constant rate.</param>
/// <param name="Ratings">Learn content ratings from the dataset's ratings.bin, if present, as three cumulative labels.</param>
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
    bool? LocalizationFromFusedOnly = null,
    string? LocalizableTagsPath = null,
    bool CompactVocabulary = true,
    bool CosineSchedule = false,
    int WarmupSteps = 0)
{
    public static TrainingTargetOptions Default { get; } = new();

    public static TrainingTargetOptions Legacy { get; } = new(TagCentricSampling: false, DeliberateTargets: false);
}
