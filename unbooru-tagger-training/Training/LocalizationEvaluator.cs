using System.Text.Json;
using UnbooruTagger.Core.Encoding;
using UnbooruTagger.Core.Scoring;
using UnbooruTagger.Core.Vocabulary;
using UnbooruTagger.Training.Model;
using static TorchSharp.torch;

namespace UnbooruTagger.Training.Training;

/// <summary>How sure a gold box is (see the gold set's LABELING.md).</summary>
public enum GoldKind
{
    /// <summary>A definite instance: missing it costs recall; matching it is a true positive.</summary>
    Certain,
    /// <summary>One box around a cluster too dense to box one by one: finding it (or anything in it) is a hit, nothing inside counts against the model, and counts aren't scored.</summary>
    Group,
    /// <summary>Where the tag might be but the labeler isn't sure: predictions there count neither for nor against.</summary>
    Ignore,
}

public sealed record GoldInstance(string Tag, int[] Box, GoldKind Kind);

/// <summary>One hand-checked image from the gold set (see the gold directory's <c>gold.json</c>): boxes in content-region pixels.</summary>
/// <param name="Stratum">"clean" or "messy" (comics, collages, crowds, low resolution, ...) — scored separately so neither hides the other.</param>
public sealed record GoldImage(int CacheIndex, LetterboxBox Content, IReadOnlyList<string> Targets, IReadOnlyList<GoldInstance> Instances, IReadOnlySet<string> NotVisible, string Stratum);

/// <summary>Gold-set results for one tag or one stratum.</summary>
/// <param name="Pairs">(image, tag) pairs with at least one certain or group box.</param>
/// <param name="PointingAccuracy">Fraction of pairs whose heatmap peak falls in a certain or group box (pairs whose peak lands in an ignore region don't count either way).</param>
/// <param name="EnergyInBox">Mean share of the (min-max normalized) heatmap mass inside certain/group boxes, ignore regions left out of both sides.</param>
/// <param name="InstancePrecision">Extracted boxes that match a certain box (IoU ≥ 0.3), over extracted boxes not excused by a group or ignore region.</param>
/// <param name="InstanceRecall">Certain boxes matched plus group boxes touched, over certain plus group boxes.</param>
/// <param name="MeanCountError">Mean |extracted − certain| box count, over pairs with only certain boxes.</param>
/// <param name="TolerantPointing">Pointing with the pointing game's tolerance (Zhang et al. 2016, 15px): the peak within
/// <see cref="LocalizationEvaluator.PointingTolerance"/> canvas pixels of a certain or group box (4 map cells at stride 4), so a peak
/// one cell off a small box isn't a miss. The gap to <paramref name="PointingAccuracy"/> is near-misses.</param>
/// <param name="RegionPrecision">The heatmap's region (cells within <see cref="TagScorer.DefaultBoxBelowPeak"/> log-odds of its peak, the cut
/// detect's boxes come from): mean share of its cells inside certain or group boxes, ignore regions left out. 2026-11-04: judges the
/// map itself, without the box extraction's merging and filtering.</param>
/// <param name="RegionRecall">Mean share of the certain and group boxes' cells inside that region.</param>
public sealed record GoldResult(string Name, int Pairs, int GoldInstances, double PointingAccuracy, double EnergyInBox, double InstancePrecision, double InstanceRecall, double MeanCountError,
    double TolerantPointing = double.NaN, double RegionPrecision = double.NaN, double RegionRecall = double.NaN);

/// <param name="DeletionGap">Mean drop in a tag's whole-image confidence when its top-heatmap cells are blanked, minus the drop from blanking the same number of random cells. Larger = the heatmap points at what the prediction actually depends on.</param>
/// <param name="DeletionWinRate">Fraction of pairs where blanking the heatmap's top cells hurt more than blanking random cells (0.5 = chance).</param>
/// <param name="TwoCharacterCorrelation">Mean Pearson correlation between the two characters' heatmaps on images tagged with exactly two characters. Lower = they point at different places.</param>
/// <param name="TwoCharacterDistinctPeaks">Fraction of those images where each character's peak lies where its own heatmap beats the other's.</param>
/// <param name="CropConsistency">Mean correlation between a crop's own heatmap and the same crop of the full image's heatmap. 1 = heatmaps move with the content.</param>
/// <param name="SoloSingleInstanceRate">On images tagged solo, the fraction where a once-per-person tag (blush, necktie, hat, ...) yields exactly one extracted box.</param>
/// <param name="Teacher">Crop-teacher check on the gold set, overall and per stratum (<see cref="TeacherResult"/>); empty without a gold set.</param>
/// <param name="TeacherTags">The same per tag.</param>
/// <param name="Occlusion">Occlusion check on the gold set, overall, per stratum and per size bucket (<see cref="OcclusionResult"/>); empty without a gold set.</param>
/// <param name="OcclusionTags">The same per tag.</param>
/// <param name="MapOcclusion">
/// The occlusion check again with localizable tags scored from the localization map instead of the pooled
/// embedding (<see cref="LocalizationEvaluator.MapScoreTemperature"/>): the tag's per-location logits pooled
/// by a temperature log-sum-exp over the image's content, as the localization loss pools them. Each location
/// sees only part of the image, so this asks whether a map-scored classifier would answer from the object.
/// </param>
/// <param name="MapOcclusionTags">The same per tag.</param>
/// <param name="Gold">Gold-set results overall ("all") and per stratum ("clean", "messy").</param>
/// <param name="GoldTags">Gold-set results per tag, worst pointing first.</param>
public sealed record LocalizationReport(
    IReadOnlyList<GoldResult> Gold,
    int DeletionPairs, double DeletionGap, double DeletionWinRate,
    int TwoCharacterImages, double TwoCharacterCorrelation, double TwoCharacterDistinctPeaks,
    int CropPairs, double CropConsistency,
    int SoloPairs, double SoloSingleInstanceRate,
    IReadOnlyList<GoldResult> GoldTags,
    IReadOnlyList<TeacherResult>? Teacher = null,
    IReadOnlyList<TeacherResult>? TeacherTags = null,
    IReadOnlyList<OcclusionResult>? Occlusion = null,
    IReadOnlyList<OcclusionResult>? OcclusionTags = null,
    IReadOnlyList<OcclusionResult>? MapOcclusion = null,
    IReadOnlyList<OcclusionResult>? MapOcclusionTags = null,
    IReadOnlyList<GlimpseFunnelResult>? GlimpseFunnel = null);

/// <summary>
/// The glimpse funnel on the gold set, for a glimpse checkpoint (<see cref="Glimpses"/>): where a labelled object is lost
/// between the model suspecting it and calling it. Over the localizable (image, tag) pairs with a certain or group box.
/// </summary>
/// <param name="Suspected">The tag is among the proposal head's top <see cref="ImageTower.GlimpseCount"/>, the tags
/// <see cref="Glimpses.Propose"/> places windows for (in rank order, so only the first dozen or so get any).</param>
/// <param name="PeakFramed">A chosen close window (128px or smaller) is a look at the tag's own map peak
/// (<see cref="Glimpses.Frames"/>): what training's "framed" share counts, which says nothing of where the object is.</param>
/// <param name="Covered">A chosen close window, chosen for any tag, holds at least half of one of the tag's gold boxes, or
/// lies at least half inside one (<see cref="LocalizationEvaluator.Covers"/>): the model looked at the object.</param>
/// <param name="BestOnObject">The tag's highest-scoring glimpse, at any zoom, covers a gold box the same way: the score
/// was decided from the object.</param>
/// <param name="WindowsOnObject">The mean share of an image's glimpses that cover one of the tag's gold boxes: what
/// <paramref name="BestOnObject"/> would be if the best glimpse were a random one.</param>
/// <param name="BestZoom">Which zoom the tag's highest-scoring glimpse is, as shares in <see cref="Glimpses.Zooms"/> order
/// (32, 64, 128, 256px windows).</param>
/// <param name="Called">The tag's glimpse score is at least 0.5.</param>
/// <param name="CalledWhenCovered">Called, over the covered pairs.</param>
/// <param name="CalledWhenNotCovered">Called, over the pairs no close window covered.</param>
public sealed record GlimpseFunnelResult(
    string Name, int Pairs, double Suspected, double PeakFramed, double Covered, double BestOnObject, double Called,
    double CalledWhenCovered, double CalledWhenNotCovered, double WindowsOnObject = double.NaN, double[]? BestZoom = null);

/// <summary>
/// Does a whole-image call depend on the object itself? For each gold pair the classifier finds
/// on the full image, the tag's boxes are blanked to the neutral padding value and the image is
/// scored again, with everything else (the character, the artist's style, the rest of the outfit)
/// still there. A control blanks rectangles of the same sizes elsewhere in the image, touching no
/// box of the tag, so a drop from merely blanking something is subtracted out.
/// </summary>
/// <param name="Pairs">Pairs found on the full image (p &gt;= 0.5) that both blanks fit in.</param>
/// <param name="StillYesRate">Fraction still called (p &gt;= 0.5) with the object blanked: answered from context. Lower is better.</param>
/// <param name="ObjectDrop">Mean drop in confidence from blanking the object.</param>
/// <param name="ControlDrop">Mean drop from blanking the same area elsewhere.</param>
/// <param name="WinRate">Fraction of pairs where blanking the object dropped confidence more than the control (0.5 = the object doesn't matter).</param>
/// <param name="ScoredPairs">Every gold pair both blanks fit in, found on the full image or not: a fixed set, where <paramref name="Pairs"/> moves with the model's calibration.</param>
/// <param name="ObjectEffect">
/// Mean over <paramref name="ScoredPairs"/> of logit(control blanked) − logit(object blanked): how much more the object counts than
/// the same area elsewhere, in log-odds. Threshold-free, so unlike <paramref name="StillYesRate"/> it doesn't jump when scores near
/// 0.5 move a little (a quarter of the found pairs change between checkpoints 4,500 steps apart). Higher is better; 0 = the object
/// doesn't matter.
/// </param>
/// <param name="ObjectEffectMedian">Its median.</param>
public sealed record OcclusionResult(
    string Name, int Pairs, double StillYesRate, double ObjectDrop, double ControlDrop, double WinRate,
    int ScoredPairs = 0, double ObjectEffect = double.NaN, double ObjectEffectMedian = double.NaN);

/// <summary>
/// One gold pair's pointing (<c>evaluate --dump-occlusion</c>, pointing_pairs.jsonl): where the tag's map peaks (content pixels), its
/// distance to the nearest certain/group box (0 inside), and how high the map gets inside the boxes: the best in-box cell's rank among
/// the content's cells (1 = it is the peak; near 1 = the map responds to the object but another spot wins; -1 = no cell centre in a box), and
/// <paramref name="PeaksAbove"/>: how many separate peaks (local maxima of a 9x9-cell window, outside the boxes) are higher than that
/// best in-box cell (0 = the object holds the map's highest peak; 1 = one other spot wins); <paramref name="TopPeaks"/>: the map's 5
/// highest such local maxima anywhere, [x, y] in content pixels, highest first.
/// </summary>
public sealed record PointingPair(int CacheIndex, string Tag, string Stratum, string Size, float PeakX, float PeakY, double Distance, bool? Hit, bool? NearHit,
    double BoxBestRank, int[][] Boxes, int PeaksAbove = -1, int[][]? TopPeaks = null);

/// <summary>One gold pair of the (pooled) occlusion check: the confidence on the full image, with the object blanked, and with the control blanked.</summary>
public sealed record OcclusionPair(int CacheIndex, string Tag, string Stratum, string Size, float Full, float Occluded, float Control);

/// <summary>
/// One labelled localizable gold pair as the glimpse funnel saw it: its gold boxes and every chosen window ([zoom, x, y, side]
/// on the canvas) with the tag's logit there, the pooled probability, and how many localizable tags the proposal head ranks above it.
/// </summary>
/// <summary>
/// One gold image's chosen windows ([zoom, x, y, side]) with their glimpse embeddings (zoom code included, as the tags see them;
/// base64 little-endian float32, window-major, <c>Dim</c> per window) and the image's pooled whole-image embedding: for probing
/// whether a window's embedding holds the object at all, independently of the tag rows.
/// </summary>
public sealed record GlimpseEmbeddings(int CacheIndex, int Dim, int[][] Windows, string Embeddings, string Pooled);

public sealed record GlimpsePair(int CacheIndex, string Tag, string Stratum, string Size, int[][] Boxes, int[][] Windows, float[] Logits, float Probability, int ProposalRank);

/// <summary>
/// How reliable the whole-image classifier is as a teacher for crop consistency
/// (<see cref="SigmoidContrastiveLoss.ComputeWindowConsistency"/>): for each visible gold
/// target, one square crop around a certain instance (the tag is in it) and one that overlaps
/// no box of that tag at all (it isn't), each scored by the classifier alone.
/// </summary>
/// <param name="FullImageYesRate">Fraction of pairs the classifier finds on the uncropped image (its ordinary recall here).</param>
/// <param name="IncludeYesRate">Fraction of include-crops it says contain the tag (p &gt;= 0.5).</param>
/// <param name="ExcludeNoRate">Fraction of exclude-crops it says don't (p &lt; 0.5). Low = it answers from context, not the object.</param>
/// <param name="BalancedAccuracy">Mean of the two.</param>
/// <param name="ConfidentCoverage">Fraction of crop answers with p &gt; 0.9 or p &lt; 0.1, the ones a gated loss would train on.</param>
/// <param name="ConfidentAccuracy">How often those confident answers are right.</param>
public sealed record TeacherResult(
    string Name, int IncludePairs, int ExcludePairs, double FullImageYesRate, double IncludeYesRate, double ExcludeNoRate,
    double BalancedAccuracy, double ConfidentCoverage, double ConfidentAccuracy);

/// <summary>
/// Measures localization quality — the part of the model the tag-ranking metrics can't see.
/// Two kinds of evidence:
/// <list type="bullet">
/// <item><b>Gold boxes</b> (optional, hand-checked): pointing accuracy, heatmap energy inside the
/// boxes, and instance-level precision/recall/count error for boxes extracted the same way
/// <c>detect</c> extracts them (<see cref="TagScorer.DetectBoxes"/>, before edge refinement).</item>
/// <item><b>Label-free checks</b> on held-out images: deletion (does blanking the heatmap's
/// peak actually lower the prediction more than blanking random cells?), two-character
/// separation, crop consistency, and once-per-person instance counts on solo images.</item>
/// </list>
/// Heatmaps are scored exactly as inference scores them: <c>sigmoid(scale·dot + bias)</c> per
/// location with the model's learned whole-image scaling.
/// </summary>
public sealed class LocalizationEvaluator(
    ImageTower imageTower,
    float[] tagEmbeddings,
    int dim,
    LogitScaling scaling,
    Func<IReadOnlyList<int>, EncodedImage[]> readImages,
    int inputSize,
    Device device)
{
    /// <summary>
    /// With a value, the occlusion check is repeated with localizable tags scored from the localization map at
    /// this log-sum-exp temperature (<see cref="LocalizationReport.MapOcclusion"/>); training's
    /// --localization-temperature is the natural choice.
    /// </summary>
    public float? MapScoreTemperature { get; init; }

    /// <summary>With a list, every pair of the pooled occlusion check is added to it (<c>evaluate --dump-occlusion</c>).</summary>
    public List<OcclusionPair>? OcclusionPairs { get; init; }

    /// <summary>With a list, every scored gold pair's pointing is added to it (<c>evaluate --dump-occlusion</c>).</summary>
    public List<PointingPair>? PointingPairs { get; init; }

    /// <summary>When set, the glimpse funnel adds each gold pair's windows and logits here (<c>evaluate --dump-occlusion</c>).</summary>
    public List<GlimpsePair>? GlimpsePairs { get; init; }

    /// <summary>With <see cref="GlimpsePairs"/>, each gold image's window embeddings (<c>evaluate --dump-occlusion</c>).</summary>
    public List<GlimpseEmbeddings>? GlimpseEmbeddingDump { get; init; }

    /// <summary>
    /// Whether the gold checks include the crop teacher (<see cref="LocalizationReport.Teacher"/>). It only says whether
    /// crop consistency (<c>--crop-teacher-*</c>) may trust the classifier, and costs ~40% of a glimpse checkpoint's gold
    /// checks (each crop is a full pass with its own glimpses); <c>evaluate --no-crop-teacher</c> leaves it out.
    /// </summary>
    public bool CropTeacher { get; init; } = true;

    /// <summary>What the occlusion check fills blanked rectangles with (<see cref="Training.OcclusionFill"/>).</summary>
    public OcclusionFill OcclusionFill { get; init; } = OcclusionFill.Zero;

    // One instance per person — on a solo image, exactly one box is right.
    public static readonly string[] OncePerPersonTags = ["blush", "necktie", "hat", "choker", "scarf", "glasses", "headphones", "halo"];

    public static List<GoldImage> LoadGold(string goldJsonPath)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(goldJsonPath));
        var images = new List<GoldImage>();
        foreach (var image in document.RootElement.GetProperty("images").EnumerateArray())
        {
            if (image.TryGetProperty("excluded", out var excluded) && excluded.ValueKind == JsonValueKind.True)
                continue;

            var box = image.GetProperty("contentBox").EnumerateArray().Select(v => v.GetInt32()).ToArray();
            images.Add(new GoldImage(
                image.GetProperty("cacheIndex").GetInt32(),
                new LetterboxBox(box[0], box[1], box[2], box[3]),
                image.GetProperty("targets").EnumerateArray().Select(v => v.GetString()!).ToList(),
                image.GetProperty("instances").EnumerateArray()
                    .Select(i => new GoldInstance(
                        i.GetProperty("tag").GetString()!,
                        i.GetProperty("box").EnumerateArray().Select(v => (int)Math.Round(v.GetDouble())).ToArray(),
                        i.TryGetProperty("kind", out var kind) ? ParseKind(kind.GetString()) : GoldKind.Certain))
                    .ToList(),
                image.GetProperty("notVisible").EnumerateArray().Select(v => v.GetString()!).ToHashSet(),
                image.TryGetProperty("stratum", out var stratum) ? stratum.GetString() ?? "clean" : "clean"));
        }

        return images;
    }

    /// <summary>
    /// The gold set read through tag families (<see cref="TagFamilies"/>), as training reads labels:
    /// <list type="bullet">
    /// <item>a box of a kind is also a box of everything it is a kind of (a boots box is a shoes box), and that
    /// parent becomes a target unless the labeler marked it not visible; a parent box already there in the same
    /// place (IoU ≥ 0.5) isn't doubled;</item>
    /// <item>a box the labeler drew for the parent alone (shoes, kind not named) is an ignore region for every
    /// targeted kind of it: a boots call there is neither right nor wrong.</item>
    /// </list>
    /// </summary>
    public static List<GoldImage> ExpandFamilies(IReadOnlyList<GoldImage> gold, TagFamilies families, TagVocabulary vocabulary)
    {
        var rowOf = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var record in vocabulary.Records)
            rowOf[record.Tag] = record.RowIndex;
        string Name(int row) => vocabulary.GetByRowIndex(row).Tag;

        var expanded = new List<GoldImage>(gold.Count);
        foreach (var image in gold)
        {
            var instances = image.Instances.ToList();
            var targets = image.Targets.ToList();
            foreach (var instance in image.Instances)
            {
                if (!rowOf.TryGetValue(instance.Tag, out var row))
                    continue;
                foreach (var parent in families.KindOf(row).Select(Name))
                {
                    if (image.NotVisible.Contains(parent))
                        continue;
                    if (!instances.Any(x => x.Tag == parent && IoU(x.Box, instance.Box) >= 0.5))
                        instances.Add(instance with { Tag = parent });
                    if (instance.Kind != GoldKind.Ignore && !targets.Contains(parent))
                        targets.Add(parent);
                }
            }

            foreach (var target in image.Targets)
            {
                if (!rowOf.TryGetValue(target, out var row))
                    continue;
                foreach (var parent in families.FamilyAncestors(row).Select(Name))
                    foreach (var parentOnly in image.Instances.Where(x => x.Tag == parent))
                        if (!instances.Any(x => x.Tag == target && IoU(x.Box, parentOnly.Box) >= 0.3))
                            instances.Add(new GoldInstance(target, parentOnly.Box, GoldKind.Ignore));
            }

            expanded.Add(image with { Instances = instances, Targets = targets });
        }

        return expanded;
    }

    private static GoldKind ParseKind(string? kind) => kind?.ToLowerInvariant() switch
    {
        "group" => GoldKind.Group,
        "ignore" => GoldKind.Ignore,
        _ => GoldKind.Certain,
    };

    public LocalizationReport Evaluate(
        TagVocabulary vocabulary,
        IReadOnlyList<IReadOnlyList<int>> imageTagRows,
        IReadOnlyList<int> heldOutImages,
        IReadOnlyList<GoldImage>? gold,
        TagLocalizability localizability,
        int sampleImages = 300,
        int seed = 0,
        bool labelFreeChecks = true)
    {
        var random = new Random(seed);
        var rowOf = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var record in vocabulary.Records)
            rowOf[record.Tag] = record.RowIndex;
        var localizable = vocabulary.Records.Select(r => localizability.IsLocalizable(r.Tag)).ToArray();

        var (goldSummary, goldTags) = gold is { Count: > 0 } ? EvaluateGold(gold, rowOf) : ([], []);
        var (teacherSummary, teacherTags) = gold is { Count: > 0 } && CropTeacher ? EvaluateTeacher(gold, rowOf, random) : ([], []);
        var (occlusionSummary, occlusionTags) = gold is { Count: > 0 } ? EvaluateOcclusion(gold, rowOf, random, (pixels, boxes) => Score(pixels, boxes), _ => true, OcclusionPairs) : ([], []);
        // Its own random source: the pooled check and everything after it see the same draws as before.
        var (mapOcclusionSummary, mapOcclusionTags) = gold is { Count: > 0 } && MapScoreTemperature is { } temperature
            ? EvaluateOcclusion(gold, rowOf, new Random(seed + 1), (pixels, boxes) => MapScorer(pixels, boxes, temperature), t => localizable[t])
            : ([], []);
        var funnel = gold is { Count: > 0 } ? EvaluateGlimpseFunnel(gold, rowOf) : [];

        // The label-free checks cost thousands of extra forward passes; training's per-epoch
        // evaluation skips them and scores only the gold set (labelFreeChecks: false).
        var (deletionPairs, deletionGap, deletionWin, cropPairs, cropConsistency) = (0, double.NaN, double.NaN, 0, double.NaN);
        var (twoImages, twoCorrelation, twoDistinct) = (0, double.NaN, double.NaN);
        var (soloPairs, soloSingle) = (0, double.NaN);
        if (labelFreeChecks)
        {
            var sample = heldOutImages.OrderBy(_ => random.Next()).Take(sampleImages).ToArray();
            (deletionPairs, deletionGap, deletionWin, cropPairs, cropConsistency) = EvaluateDeletionAndCrops(sample, imageTagRows, localizable, random);

            var characterRows = vocabulary.Records.Where(r => r.Tag.StartsWith("character:", StringComparison.Ordinal)).Select(r => r.RowIndex).ToHashSet();
            var twoCharacter = heldOutImages.Where(i => imageTagRows[i].Count(characterRows.Contains) == 2).OrderBy(_ => random.Next()).Take(sampleImages / 2).ToArray();
            (twoImages, twoCorrelation, twoDistinct) = EvaluateTwoCharacters(twoCharacter, imageTagRows, characterRows);

            (soloPairs, soloSingle) = EvaluateSoloCounts(heldOutImages, imageTagRows, rowOf, random, sampleImages);
        }

        return new LocalizationReport(
            goldSummary,
            deletionPairs, deletionGap, deletionWin,
            twoImages, twoCorrelation, twoDistinct,
            cropPairs, cropConsistency,
            soloPairs, soloSingle,
            goldTags,
            teacherSummary, teacherTags,
            occlusionSummary, occlusionTags,
            mapOcclusionSummary, mapOcclusionTags,
            funnel);
    }

    /// <summary>Running totals for one tag or stratum.</summary>
    private sealed class GoldTally
    {
        public int Pairs, GoldInstances, PointingCounted, PointingHits, TolerantCounted, TolerantHits, ScoredExtracted, MatchedExtracted, RecallItems, RecalledItems, CountPairs, RegionPairs;
        public double Energy, CountError, RegionPrecision, RegionRecall;

        public void Add(PairScore score)
        {
            Pairs++;
            GoldInstances += score.RecallItems;
            if (score.PeakHit is bool hit)
            {
                PointingCounted++;
                if (hit)
                    PointingHits++;
            }
            if (score.NearHit is bool near)
            {
                TolerantCounted++;
                if (near)
                    TolerantHits++;
            }
            Energy += score.Energy;
            if (score.RegionPrecision is double regionPrecision && score.RegionRecall is double regionRecall)
            {
                RegionPairs++;
                RegionPrecision += regionPrecision;
                RegionRecall += regionRecall;
            }
            ScoredExtracted += score.ScoredExtracted;
            MatchedExtracted += score.MatchedExtracted;
            RecallItems += score.RecallItems;
            RecalledItems += score.RecalledItems;
            if (score.CountError is int error)
            {
                CountPairs++;
                CountError += error;
            }
        }

        public GoldResult ToResult(string name) => new(
            name, Pairs, GoldInstances,
            PointingCounted > 0 ? (double)PointingHits / PointingCounted : 0,
            Pairs > 0 ? Energy / Pairs : 0,
            ScoredExtracted > 0 ? (double)MatchedExtracted / ScoredExtracted : 0,
            RecallItems > 0 ? (double)RecalledItems / RecallItems : 0,
            CountPairs > 0 ? CountError / CountPairs : 0,
            TolerantCounted > 0 ? (double)TolerantHits / TolerantCounted : double.NaN,
            RegionPairs > 0 ? RegionPrecision / RegionPairs : double.NaN,
            RegionPairs > 0 ? RegionRecall / RegionPairs : double.NaN);
    }

    /// <summary>The tolerant pointing's reach (<see cref="GoldResult.TolerantPointing"/>), in canvas pixels.</summary>
    public const int PointingTolerance = 16;

    /// <summary>One (image, tag) pair's gold scoring. <see cref="PeakHit"/> is null when the peak fell in an ignore region; <see cref="CountError"/> is null when the pair has group or ignore boxes.
    /// <see cref="NearHit"/> is the same with <see cref="PointingTolerance"/>.</summary>
    public readonly record struct PairScore(bool? PeakHit, double Energy, int ScoredExtracted, int MatchedExtracted, int RecallItems, int RecalledItems, int? CountError, bool? NearHit = null,
        double? RegionPrecision = null, double? RegionRecall = null);

    /// <summary>
    /// Scores one (image, tag) pair against its gold boxes with group/ignore semantics:
    /// an extracted box matching a certain box (IoU ≥ 0.3) is a true positive; one at least
    /// half inside a group or ignore box is excused (neither TP nor FP; touching a group marks
    /// it found); anything else is a false positive.
    /// </summary>
    public static PairScore ScorePair(IReadOnlyList<GoldInstance> instances, (float X, float Y) peak, double energy, IReadOnlyList<int[]> extracted)
    {
        var certain = instances.Where(x => x.Kind == GoldKind.Certain).Select(x => x.Box).ToList();
        var groups = instances.Where(x => x.Kind == GoldKind.Group).Select(x => x.Box).ToList();
        var ignore = instances.Where(x => x.Kind == GoldKind.Ignore).Select(x => x.Box).ToList();

        bool? peakHit = certain.Concat(groups).Any(b => Contains(b, peak.X, peak.Y)) ? true
            : ignore.Any(b => Contains(b, peak.X, peak.Y)) ? null
            : false;
        bool? nearHit = certain.Concat(groups).Any(b => Distance(b, peak.X, peak.Y) <= PointingTolerance) ? true
            : ignore.Any(b => Contains(b, peak.X, peak.Y)) ? null
            : false;

        int scored = 0, matched = 0;
        var groupsFound = new bool[groups.Count];
        foreach (var e in extracted)
        {
            if (certain.Any(g => IoU(e, g) >= 0.3))
            {
                scored++;
                matched++;
                continue;
            }

            var excused = false;
            for (var g = 0; g < groups.Count; g++)
                if (FractionInside(e, groups[g]) >= 0.5 || IoU(e, groups[g]) >= 0.3)
                {
                    groupsFound[g] = true;
                    excused = true;
                }
            if (excused || ignore.Any(g => FractionInside(e, g) >= 0.5 || IoU(e, g) >= 0.3))
                continue;
            scored++;
        }

        var recalled = certain.Count(g => extracted.Any(e => IoU(e, g) >= 0.3)) + groupsFound.Count(f => f);
        int? countError = groups.Count == 0 && ignore.Count == 0 ? Math.Abs(extracted.Count - certain.Count) : null;
        return new PairScore(peakHit, energy, scored, matched, certain.Count + groups.Count, recalled, countError, nearHit);
    }

    private static bool Contains(int[] b, float x, float y) => x >= b[0] && x < b[0] + b[2] && y >= b[1] && y < b[1] + b[3];

    /// <summary>A point's distance to a box (0 inside it).</summary>
    private static double Distance(int[] b, float x, float y)
    {
        var dx = Math.Max(0, Math.Max(b[0] - x, x - (b[0] + b[2])));
        var dy = Math.Max(0, Math.Max(b[1] - y, y - (b[1] + b[3])));
        return Math.Sqrt(dx * dx + dy * dy);
    }

    private static double FractionInside(int[] inner, int[] outer)
    {
        var area = (double)inner[2] * inner[3];
        if (area <= 0)
            return 0;
        var w = Math.Max(0, Math.Min(inner[0] + inner[2], outer[0] + outer[2]) - Math.Max(inner[0], outer[0]));
        var h = Math.Max(0, Math.Min(inner[1] + inner[3], outer[1] + outer[3]) - Math.Max(inner[1], outer[1]));
        return w * (double)h / area;
    }

    private (List<GoldResult> Summary, List<GoldResult> Tags) EvaluateGold(IReadOnlyList<GoldImage> gold, IReadOnlyDictionary<string, int> rowOf)
    {
        var perTag = new Dictionary<string, GoldTally>();
        var perStratum = new Dictionary<string, GoldTally> { ["all"] = new() };
        foreach (var chunk in gold.Chunk(ImagesPerPass))
        {
            var tagsPerImage = chunk.Select(g => g.Targets
                .Where(t => !g.NotVisible.Contains(t) && rowOf.ContainsKey(t)
                    && g.Instances.Any(x => x.Tag == t && x.Kind != GoldKind.Ignore))
                .ToArray()).ToArray();
            var heatmaps = Heatmaps(chunk.Select(g => g.CacheIndex).ToArray(), tagsPerImage.Select(ts => ts.Select(t => rowOf[t]).ToArray()).ToArray(), out var boxes);

            for (var i = 0; i < chunk.Length; i++)
            {
                var image = chunk[i];
                for (var k = 0; k < tagsPerImage[i].Length; k++)
                {
                    var tag = tagsPerImage[i][k];
                    var instances = image.Instances.Where(x => x.Tag == tag).ToList();
                    var map = heatmaps[i][k];
                    var extracted = DetectBoxes(map, boxes[i])
                        .Select(b => new[] { b.X, b.Y, b.Width, b.Height }).ToList();
                    var peak = PeakCellCenterInContent(map, boxes[i]);
                    var boxedInstances = instances.Where(x => x.Kind != GoldKind.Ignore).Select(x => x.Box).ToList();
                    var ignoredInstances = instances.Where(x => x.Kind == GoldKind.Ignore).Select(x => x.Box).ToList();
                    var region = RegionInBoxes(map, boxes[i], boxedInstances, ignoredInstances);
                    var score = ScorePair(instances, peak, EnergyInBoxes(map, boxes[i], boxedInstances, ignoredInstances), extracted)
                        with { RegionPrecision = region?.Precision, RegionRecall = region?.Recall };

                    if (PointingPairs is not null)
                    {
                        var boxed = instances.Where(x => x.Kind != GoldKind.Ignore).Select(x => x.Box).ToArray();
                        PointingPairs.Add(new PointingPair(image.CacheIndex, tag, image.Stratum, SizeBucket(tag, instances), peak.X, peak.Y,
                            boxed.Min(b => Distance(b, peak.X, peak.Y)), score.PeakHit, score.NearHit, BoxBestRank(map, boxes[i], boxed), boxed,
                            PeaksAbove(map, boxes[i], boxed), TopPeaks(map, boxes[i], 5)));
                    }
                    (perTag.TryGetValue(tag, out var tagTally) ? tagTally : perTag[tag] = new()).Add(score);
                    perStratum["all"].Add(score);
                    (perStratum.TryGetValue(image.Stratum, out var stratumTally) ? stratumTally : perStratum[image.Stratum] = new()).Add(score);
                    var size = SizeBucket(tag, instances);
                    (perStratum.TryGetValue(size, out var sizeTally) ? sizeTally : perStratum[size] = new()).Add(score);
                }
            }
        }

        return (
            perStratum.Select(kv => kv.Value.ToResult(kv.Key)).OrderBy(r => r.Name == "all" ? 0 : 1).ThenBy(r => r.Name).ToList(),
            perTag.Select(kv => kv.Value.ToResult(kv.Key)).OrderBy(r => r.PointingAccuracy).ToList());
    }

    /// <summary>
    /// Size bucket of one (image, tag) pair, by its largest certain or group box (sqrt of area, in
    /// the 512px input): the same tag can be 10px in a full-body shot and fill a close-up, and a
    /// localization map has to handle both. Characters get their own bucket.
    /// </summary>
    public static string SizeBucket(string tag, IReadOnlyList<GoldInstance> instances)
    {
        if (tag.StartsWith("character:", StringComparison.Ordinal))
            return "size: characters";
        var largest = instances.Where(x => x.Kind != GoldKind.Ignore).Select(x => Math.Sqrt((double)x.Box[2] * x.Box[3])).DefaultIfEmpty(0).Max();
        return largest < 32 ? "size: small <32px" : largest < 96 ? "size: medium 32-96px" : "size: large >=96px";
    }

    private sealed class TeacherTally
    {
        public int Include, IncludeYes, Exclude, ExcludeNo, Full, FullYes, Confident, ConfidentRight;

        public void Add(bool tagInCrop, float p)
        {
            if (tagInCrop) { Include++; if (p >= 0.5f) IncludeYes++; }
            else { Exclude++; if (p < 0.5f) ExcludeNo++; }
            if (p > 0.9f || p < 0.1f)
            {
                Confident++;
                if ((p > 0.9f) == tagInCrop) ConfidentRight++;
            }
        }

        public TeacherResult ToResult(string name)
        {
            double Rate(int n, int d) => d > 0 ? (double)n / d : double.NaN;
            var includeYes = Rate(IncludeYes, Include);
            var excludeNo = Rate(ExcludeNo, Exclude);
            return new TeacherResult(name, Include, Exclude, Rate(FullYes, Full), includeYes, excludeNo,
                (includeYes + excludeNo) / 2, Rate(Confident, Include + Exclude), Rate(ConfidentRight, Confident));
        }
    }

    private (List<TeacherResult> Summary, List<TeacherResult> Tags) EvaluateTeacher(
        IReadOnlyList<GoldImage> gold, IReadOnlyDictionary<string, int> rowOf, Random random)
    {
        var cell = imageTower.OutputStride;
        var grid = inputSize / cell;
        var perTag = new Dictionary<string, TeacherTally>();
        var perStratum = new Dictionary<string, TeacherTally> { ["all"] = new() };
        foreach (var chunk in gold.Chunk(ImagesPerCheck))
        {
            using var scope = NewDisposeScope();
            var (pixels, boxes) = Load(chunk.Select(g => g.CacheIndex).ToArray());
            var fullScore = Score(pixels, boxes);

            // (image, tag, window, tag-in-crop) for every crop to score, batched below.
            var crops = new List<(int Image, string Tag, CellWindow Window, bool Contains)>();
            for (var i = 0; i < chunk.Length; i++)
            {
                var image = chunk[i];
                var (minY, minX, height, width) = ContentCells(boxes[i], grid);
                var shortSide = Math.Min(height, width);
                foreach (var tag in image.Targets.Where(t => !image.NotVisible.Contains(t) && rowOf.ContainsKey(t)))
                {
                    var all = image.Instances.Where(x => x.Tag == tag).Select(x => ToCanvas(x.Box, boxes[i])).ToList();
                    var certain = image.Instances.Where(x => x.Tag == tag && x.Kind == GoldKind.Certain).Select(x => ToCanvas(x.Box, boxes[i])).ToList();
                    if (certain.Count == 0)
                        continue;

                    var full = fullScore(i, rowOf[tag]);
                    foreach (var tally in Tallies(perTag, perStratum, tag, image.Stratum))
                    {
                        tally.Full++;
                        if (full >= 0.5f) tally.FullYes++;
                    }

                    // Include: the smallest training-sized square around one certain instance.
                    var target = certain[random.Next(certain.Count)];
                    var needed = (int)Math.Ceiling(Math.Max(target[2], target[3]) / (double)cell);
                    var side = Math.Max(needed, (int)Math.Round(0.35 * shortSide));
                    if (side <= shortSide)
                    {
                        var centerY = (target[1] + target[3] / 2.0) / cell;
                        var centerX = (target[0] + target[2] / 2.0) / cell;
                        var y = Math.Clamp((int)Math.Round(centerY - side / 2.0), minY, minY + height - side);
                        var x = Math.Clamp((int)Math.Round(centerX - side / 2.0), minX, minX + width - side);
                        crops.Add((i, tag, new CellWindow(y, x, side), true));
                    }

                    // Exclude: a training-sized square touching no box of this tag (of any kind).
                    foreach (var fraction in new[] { 0.5, 0.35 })
                    {
                        var size = Math.Max(1, (int)Math.Round(fraction * shortSide));
                        var free = new List<CellWindow>();
                        for (var y = minY; y + size <= minY + height; y++)
                            for (var x = minX; x + size <= minX + width; x++)
                            {
                                var window = new[] { x * cell, y * cell, size * cell, size * cell };
                                if (all.All(b => !Overlaps(window, b)))
                                    free.Add(new CellWindow(y, x, size));
                            }

                        if (free.Count == 0)
                            continue;
                        crops.Add((i, tag, free[random.Next(free.Count)], false));
                        break;
                    }
                }
            }

            foreach (var batch in crops.Chunk(ScoreBatch))
            {
                using var batchScope = NewDisposeScope();
                var cropPixels = cat(batch.Select(c => UnbooruTagger.Training.Training.CropConsistency.CropPixels(pixels[c.Image..(c.Image + 1)], [c.Window], grid)).ToArray(), 0);
                var cropScore = Score(cropPixels);
                for (var k = 0; k < batch.Length; k++)
                {
                    var p = cropScore(k, rowOf[batch[k].Tag]);
                    foreach (var tally in Tallies(perTag, perStratum, batch[k].Tag, chunk[batch[k].Image].Stratum))
                        tally.Add(batch[k].Contains, p);
                }
            }
        }

        return (
            perStratum.Select(kv => kv.Value.ToResult(kv.Key)).OrderBy(r => r.Name == "all" ? 0 : 1).ThenBy(r => r.Name).ToList(),
            perTag.Select(kv => kv.Value.ToResult(kv.Key)).OrderBy(r => r.BalancedAccuracy).ToList());

        static IEnumerable<TeacherTally> Tallies(Dictionary<string, TeacherTally> perTag, Dictionary<string, TeacherTally> perStratum, string tag, string stratum)
        {
            yield return perTag.TryGetValue(tag, out var t) ? t : perTag[tag] = new();
            yield return perStratum["all"];
            yield return perStratum.TryGetValue(stratum, out var s) ? s : perStratum[stratum] = new();
        }
    }

    private sealed class FunnelTally
    {
        private int _pairs, _suspected, _peakFramed, _covered, _best, _called, _calledCovered, _calledUncovered;
        private double _windowsOnObject;
        private readonly int[] _bestZoom = new int[Glimpses.Zooms.Length];

        public void Add(bool suspected, bool peakFramed, bool covered, bool best, bool called, double windowsOnObject, int bestZoom)
        {
            _pairs++;
            _windowsOnObject += windowsOnObject;
            if (bestZoom >= 0) _bestZoom[bestZoom]++;
            if (suspected) _suspected++;
            if (peakFramed) _peakFramed++;
            if (covered) _covered++;
            if (best) _best++;
            if (called) _called++;
            if (called && covered) _calledCovered++;
            if (called && !covered) _calledUncovered++;
        }

        public GlimpseFunnelResult ToResult(string name)
        {
            double Rate(int n, int d) => d > 0 ? (double)n / d : double.NaN;
            return new GlimpseFunnelResult(name, _pairs, Rate(_suspected, _pairs), Rate(_peakFramed, _pairs), Rate(_covered, _pairs),
                Rate(_best, _pairs), Rate(_called, _pairs), Rate(_calledCovered, _covered), Rate(_calledUncovered, _pairs - _covered),
                _pairs > 0 ? _windowsOnObject / _pairs : double.NaN, _bestZoom.Select(z => Rate(z, _bestZoom.Sum())).ToArray());
        }
    }

    /// <summary>
    /// <see cref="GlimpseFunnelResult"/> per stratum and size bucket: each gold image's windows chosen as the model chooses
    /// them (<see cref="Glimpses.Propose"/>, as <see cref="GlimpseForward"/>), its tags' per-glimpse logits and map peaks.
    /// Empty for a checkpoint without glimpses, or when scoring the whole grid.
    /// </summary>
    private List<GlimpseFunnelResult> EvaluateGlimpseFunnel(IReadOnlyList<GoldImage> gold, IReadOnlyDictionary<string, int> rowOf)
    {
        if (!imageTower.ClassificationGlimpses || imageTower.GlimpseGrid || imageTower.ClassificationPoolingTemperature <= 0
            || ClassificationPooledTags is not { } perTag)
            return [];
        var temperature = imageTower.ClassificationPoolingTemperature;
        var close = Glimpses.Zooms.Length - 1;
        var localizableTags = LocalizableTags();
        var columnOf = new Dictionary<int, long>();
        for (var t = 0; t < perTag.Count; t++)
            if (perTag[t])
                columnOf[t] = columnOf.Count;
        var tallies = new Dictionary<string, FunnelTally> { ["all"] = new() };
        foreach (var chunk in gold.Chunk(ImagesPerPass))
        {
            var tags = chunk.Select(g => g.Targets
                .Where(t => !g.NotVisible.Contains(t) && rowOf.TryGetValue(t, out var row) && perTag[row]
                    && g.Instances.Any(x => x.Tag == t && x.Kind != GoldKind.Ignore))
                .ToArray()).ToArray();
            var most = tags.Max(t => t.Length);
            if (most == 0)
                continue;

            GlimpseWindow[][] windows;
            float[] dots, peakX, peakY, ranks;
            int slots;
            bool[] valid, pooledValid;
            using (no_grad())
            using (NewDisposeScope())
            {
                var (pixels, boxes) = Load(chunk.Select(g => g.CacheIndex).ToArray());
                var (pooled, localization) = imageTower.ForwardPooled(pixels, ContentMask(pixels, boxes));
                var proposals = imageTower.Proposals(pooled);
                var contentBoxes = Glimpses.ContentBoxes(boxes, device);
                var mapMask = SpatialMask.Build(boxes, inputSize, localization.shape[2], localization.shape[3], device);
                using var choice = Glimpses.Propose(proposals, localization, localizableTags, mapMask, contentBoxes, inputSize, imageTower.OutputStride, imageTower.GlimpseCount);
                windows = choice.ToWindows();
                var (cells, validCells) = Glimpses.Embed(imageTower, pixels, contentBoxes, choice); // [B, D, K]
                slots = (int)cells.shape[2];
                // Every chosen window (they fill an image's first slots in order), and which of them pool (ImageTower.GlimpsePoolZooms).
                valid = choice.Valid.cpu().data<bool>().ToArray();
                pooledValid = validCells.cpu().data<bool>().ToArray();
                if (GlimpseEmbeddingDump is not null)
                {
                    var cellValues = cells.cpu().data<float>().ToArray(); // [B, D, K]
                    var pooledValues = pooled.cpu().data<float>().ToArray();
                    for (var i = 0; i < chunk.Length; i++)
                    {
                        var windowEmbeddings = new float[windows[i].Length * dim];
                        for (int s = 0, n = 0; s < slots && n < windows[i].Length; s++)
                        {
                            if (!valid[i * slots + s])
                                continue;
                            for (var d = 0; d < dim; d++)
                                windowEmbeddings[n * dim + d] = cellValues[((long)i * dim + d) * slots + s];
                            n++;
                        }
                        GlimpseEmbeddingDump.Add(new GlimpseEmbeddings(chunk[i].CacheIndex, dim,
                            windows[i].Select(w => new[] { w.Zoom, w.X, w.Y, w.Side }).ToArray(),
                            Convert.ToBase64String(System.Runtime.InteropServices.MemoryMarshal.AsBytes(windowEmbeddings.AsSpan())),
                            Convert.ToBase64String(System.Runtime.InteropServices.MemoryMarshal.AsBytes(pooledValues.AsSpan(i * dim, dim)))));
                    }
                }

                // Each pair's tag, padded with an image's last one, or any pair's for an image with none (never read).
                var padded = tags.SelectMany(t => Enumerable.Range(0, most).Select(k => rowOf[t.Length > 0 ? t[Math.Min(k, t.Length - 1)] : tags.First(x => x.Length > 0)[0]])).ToArray();
                var rows = tensor(padded.Select(r => (long)r).ToArray(), device: device);
                _deviceTags ??= tensor(tagEmbeddings, [tagEmbeddings.Length / dim, dim]).to(device).DetachFromDisposeScope();
                var embeddings = _deviceTags.index_select(0, rows).reshape(chunk.Length, most, dim);
                dots = embeddings.bmm(cells).cpu().data<float>().ToArray(); // [B, T, K]
                var (x, y) = Glimpses.Peaks(localization, embeddings, mapMask, imageTower.OutputStride);
                (peakX, peakY) = (x.to_type(ScalarType.Float32).cpu().data<float>().ToArray(), y.to_type(ScalarType.Float32).cpu().data<float>().ToArray());
                // How many localizable tags the proposal head ranks above each pair's.
                var scores = proposals.matmul(localizableTags.t()); // [B, L]
                var own = scores.gather(1, tensor(padded.Select(r => columnOf[r]).ToArray(), device: device).reshape(chunk.Length, most)); // [B, T]
                ranks = scores.unsqueeze(1).gt(own.unsqueeze(2)).sum(2).to_type(ScalarType.Float32).cpu().data<float>().ToArray();
            }

            for (var i = 0; i < chunk.Length; i++)
            {
                var image = chunk[i];
                var content = image.Content;
                for (var k = 0; k < tags[i].Length; k++)
                {
                    var tag = tags[i][k];
                    var instances = image.Instances.Where(x => x.Tag == tag).ToList();
                    var targets = instances.Where(x => x.Kind != GoldKind.Ignore)
                        .Select(x => new[] { x.Box[0] + content.X, x.Box[1] + content.Y, x.Box[2], x.Box[3] }).ToList();

                    // The image's windows fill its first slots, in order (Propose's scatter).
                    var logits = new List<(double Logit, GlimpseWindow Window)>();
                    var pooledLogits = new List<(double Logit, GlimpseWindow Window)>();
                    for (var s = 0; s < slots && logits.Count < windows[i].Length; s++)
                        if (valid[i * slots + s])
                        {
                            logits.Add((dots[((long)i * most + k) * slots + s], windows[i][logits.Count]));
                            if (pooledValid[i * slots + s])
                                pooledLogits.Add(logits[^1]);
                        }
                    var called = false;
                    var probability = float.NaN;
                    var best = false;
                    var bestZoom = -1;
                    var onObject = logits.Count > 0 ? logits.Count(l => targets.Any(b => Covers(l.Window, b))) / (double)logits.Count : 0;
                    if (pooledLogits.Count > 0)
                    {
                        probability = Sigmoid(scaling.Apply((float)ClassificationPooling.Pool(pooledLogits.Select(l => l.Logit).ToList(), temperature, imageTower.ClassificationPoolingTopK, imageTower.ClassificationPoolingBottomWeight)));
                        called = probability >= 0.5f;
                        var top = pooledLogits.MaxBy(l => l.Logit).Window;
                        best = targets.Any(b => Covers(top, b));
                        bestZoom = top.Zoom;
                    }
                    var covered = windows[i].Any(w => w.Zoom < close && targets.Any(b => Covers(w, b)));
                    var sensor = Glimpses.Sensor((int)peakX[i * most + k], (int)peakY[i * most + k], content, inputSize);
                    var peakFramed = Glimpses.Frames(windows[i], sensor);
                    var suspected = ranks[i * most + k] < imageTower.GlimpseCount;
                    GlimpsePairs?.Add(new GlimpsePair(image.CacheIndex, tag, image.Stratum, SizeBucket(tag, instances), targets.ToArray(),
                        logits.Select(l => new[] { l.Window.Zoom, l.Window.X, l.Window.Y, l.Window.Side }).ToArray(),
                        logits.Select(l => (float)l.Logit).ToArray(), probability, (int)ranks[i * most + k]));

                    foreach (var name in new[] { "all", image.Stratum, SizeBucket(tag, instances) })
                        (tallies.TryGetValue(name, out var tally) ? tally : tallies[name] = new()).Add(suspected, peakFramed, covered, best, called, onObject, bestZoom);
                }
            }
        }

        return tallies.Select(kv => kv.Value.ToResult(kv.Key)).OrderBy(r => r.Name == "all" ? 0 : 1).ThenBy(r => r.Name).ToList();
    }

    /// <summary>
    /// Whether a glimpse window shows a gold box ([x, y, w, h] on the canvas): their overlap is at least half of the smaller of
    /// the two, so a small object wholly inside a large window counts, and so does a window wholly on a large object.
    /// </summary>
    public static bool Covers(GlimpseWindow window, int[] box)
    {
        var w = Math.Min(window.X + window.Side, box[0] + box[2]) - Math.Max(window.X, box[0]);
        var h = Math.Min(window.Y + window.Side, box[1] + box[3]) - Math.Max(window.Y, box[1]);
        if (w <= 0 || h <= 0)
            return false;
        var smaller = Math.Min((double)window.Side * window.Side, (double)box[2] * box[3]);
        return smaller > 0 && w * (double)h >= 0.5 * smaller;
    }

    private sealed class OcclusionTally
    {
        public int Pairs, StillYes, Wins;
        public double ObjectDrop, ControlDrop;
        private readonly List<double> _effects = [];

        /// <summary>Every scored pair counts toward the object effect; only found ones (full p &gt;= 0.5) toward the rest.</summary>
        public void Add(float full, float occluded, float control)
        {
            _effects.Add(Logit(control) - Logit(occluded));
            if (full < 0.5f)
                return;
            Pairs++;
            if (occluded >= 0.5f) StillYes++;
            ObjectDrop += full - occluded;
            ControlDrop += full - control;
            if (control > occluded) Wins++;
        }

        private static double Logit(float p)
        {
            var q = Math.Clamp((double)p, 1e-6, 1 - 1e-6);
            return Math.Log(q / (1 - q));
        }

        public OcclusionResult ToResult(string name)
        {
            var (effect, median) = _effects.Count == 0 ? (double.NaN, double.NaN) : (_effects.Average(), Median(_effects));
            return Pairs == 0
                ? new(name, 0, double.NaN, double.NaN, double.NaN, double.NaN, _effects.Count, effect, median)
                : new(name, Pairs, (double)StillYes / Pairs, ObjectDrop / Pairs, ControlDrop / Pairs, (double)Wins / Pairs, _effects.Count, effect, median);
        }

        private static double Median(List<double> values)
        {
            var sorted = values.Order().ToArray();
            return sorted.Length % 2 == 1 ? sorted[sorted.Length / 2] : (sorted[sorted.Length / 2 - 1] + sorted[sorted.Length / 2]) / 2;
        }
    }

    /// <summary>Margin (input pixels) added around each blanked gold box, so its edges and the outline around it go too.</summary>
    public const int OcclusionMargin = 4;

    /// <param name="score">Scores a batch of images (with their content boxes): (image, tag) → confidence.</param>
    /// <param name="eligible">Which tag rows to check.</param>
    private (List<OcclusionResult> Summary, List<OcclusionResult> Tags) EvaluateOcclusion(
        IReadOnlyList<GoldImage> gold, IReadOnlyDictionary<string, int> rowOf, Random random,
        Func<Tensor, LetterboxBox[], Func<int, int, float>> score, Func<int, bool> eligible, List<OcclusionPair>? sink = null)
    {
        var perTag = new Dictionary<string, OcclusionTally>();
        var perStratum = new Dictionary<string, OcclusionTally> { ["all"] = new() };
        var floats = 3 * inputSize * inputSize;
        foreach (var chunk in gold.Chunk(ImagesPerCheck))
        {
            using var scope = NewDisposeScope();
            var (pixels, boxes) = Load(chunk.Select(g => g.CacheIndex).ToArray());
            var fullScore = score(pixels, boxes);
            var pixelArray = pixels.cpu().data<float>().ToArray();

            // (image, tag, full-image confidence, object blank, control blank) for every pair to score.
            var pairs = new List<(int Image, string Tag, float Full, List<int[]> Object, List<int[]> Control)>();
            for (var i = 0; i < chunk.Length; i++)
            {
                var image = chunk[i];
                var content = new[] { boxes[i].X, boxes[i].Y, boxes[i].Width, boxes[i].Height };
                foreach (var tag in image.Targets.Where(t => !image.NotVisible.Contains(t) && rowOf.TryGetValue(t, out var row) && eligible(row)).Distinct())
                {
                    var instances = image.Instances.Where(x => x.Tag == tag).ToList();
                    if (!instances.Any(x => x.Kind != GoldKind.Ignore))
                        continue;
                    var full = fullScore(i, rowOf[tag]);

                    // Every box of the tag, ignore boxes too: whatever might be the object goes.
                    var blank = instances.Select(x => Grow(ToCanvas(x.Box, boxes[i]), OcclusionMargin, content)).Where(b => b[2] > 0 && b[3] > 0).ToList();
                    if (ControlRectangles(blank, content, random) is { } control)
                        pairs.Add((i, tag, full, blank, control));
                }
            }

            foreach (var batch in pairs.Chunk(ScoreBatch / 2))
            {
                using var batchScope = NewDisposeScope();
                var variants = new float[batch.Length * 2 * floats];
                for (var k = 0; k < batch.Length; k++)
                {
                    var source = pixelArray.AsSpan(batch[k].Image * floats, floats);
                    source.CopyTo(variants.AsSpan(2 * k * floats, floats));
                    source.CopyTo(variants.AsSpan((2 * k + 1) * floats, floats));
                    OcclusionFills.Apply(variants.AsSpan(2 * k * floats, floats), batch[k].Object, inputSize, OcclusionFill, random);
                    OcclusionFills.Apply(variants.AsSpan((2 * k + 1) * floats, floats), batch[k].Control, inputSize, OcclusionFill, random);
                }

                var variantBoxes = batch.SelectMany(p => new[] { boxes[p.Image], boxes[p.Image] }).ToArray();
                var variantScore = score(tensor(variants, [batch.Length * 2, 3, inputSize, inputSize]).to(device), variantBoxes);
                for (var k = 0; k < batch.Length; k++)
                {
                    var (imageIndex, tag, full, _, _) = batch[k];
                    var occluded = variantScore(2 * k, rowOf[tag]);
                    var control = variantScore(2 * k + 1, rowOf[tag]);
                    var image = chunk[imageIndex];
                    var size = SizeBucket(tag, image.Instances.Where(x => x.Tag == tag).ToList());
                    sink?.Add(new OcclusionPair(image.CacheIndex, tag, image.Stratum, size, full, occluded, control));
                    foreach (var key in new[] { image.Stratum, size })
                        (perStratum.TryGetValue(key, out var t) ? t : perStratum[key] = new()).Add(full, occluded, control);
                    perStratum["all"].Add(full, occluded, control);
                    (perTag.TryGetValue(tag, out var tagTally) ? tagTally : perTag[tag] = new()).Add(full, occluded, control);
                }
            }
        }

        return (
            perStratum.Select(kv => kv.Value.ToResult(kv.Key)).OrderBy(r => r.Name == "all" ? 0 : 1).ThenBy(r => r.Name).ToList(),
            perTag.Select(kv => kv.Value.ToResult(kv.Key)).OrderByDescending(r => r.StillYesRate).ToList());
    }

    /// <summary><paramref name="box"/> grown by <paramref name="margin"/> on every side and clipped to <paramref name="content"/> (all [x, y, w, h] on the canvas).</summary>
    public static int[] Grow(int[] box, int margin, int[] content)
    {
        var x0 = Math.Max(content[0], box[0] - margin);
        var y0 = Math.Max(content[1], box[1] - margin);
        var x1 = Math.Min(content[0] + content[2], box[0] + box[2] + margin);
        var y1 = Math.Min(content[1] + content[3], box[1] + box[3] + margin);
        return [x0, y0, Math.Max(0, x1 - x0), Math.Max(0, y1 - y0)];
    }

    /// <summary>
    /// One rectangle the size of each of <paramref name="objects"/>, placed at random inside
    /// <paramref name="content"/>, overlapping none of the objects nor each other; null if some
    /// don't fit (an object filling most of the image has nowhere else to put its control).
    /// </summary>
    public static List<int[]>? ControlRectangles(IReadOnlyList<int[]> objects, int[] content, Random random, int attempts = 200)
    {
        var placed = new List<int[]>();
        foreach (var size in objects.OrderByDescending(b => b[2] * b[3]))
        {
            if (size[2] > content[2] || size[3] > content[3])
                return null;
            int[]? spot = null;
            for (var attempt = 0; attempt < attempts && spot is null; attempt++)
            {
                var candidate = new[]
                {
                    content[0] + random.Next(content[2] - size[2] + 1),
                    content[1] + random.Next(content[3] - size[3] + 1),
                    size[2], size[3],
                };
                if (!objects.Any(o => Overlaps(candidate, o)) && !placed.Any(p => Overlaps(candidate, p)))
                    spot = candidate;
            }

            if (spot is null)
                return null;
            placed.Add(spot);
        }

        return placed;
    }

    /// <summary>Sets <paramref name="rectangles"/> (canvas [x, y, w, h]) of a [3, size, size] image to the normalized neutral padding value, 0.</summary>
    public static void Blank(Span<float> image, IEnumerable<int[]> rectangles, int size)
    {
        var channel = size * size;
        foreach (var r in rectangles)
            for (var ch = 0; ch < 3; ch++)
                for (var y = Math.Max(0, r[1]); y < Math.Min(size, r[1] + r[3]); y++)
                {
                    var start = Math.Max(0, r[0]);
                    var end = Math.Min(size, r[0] + r[2]);
                    if (end > start)
                        image.Slice(ch * channel + y * size + start, end - start).Clear();
                }
    }

    /// <summary>The content's cell range on a <paramref name="grid"/>-cell map: (first row, first column, rows, columns).</summary>
    private (int MinY, int MinX, int Height, int Width) ContentCells(LetterboxBox content, int grid)
    {
        var valid = ImagePreprocessing.ComputeSpatialValidity(content, inputSize, grid, grid);
        int minY = grid, maxY = -1, minX = grid, maxX = -1;
        for (var y = 0; y < grid; y++)
            for (var x = 0; x < grid; x++)
                if (valid[y, x])
                {
                    minY = Math.Min(minY, y); maxY = Math.Max(maxY, y);
                    minX = Math.Min(minX, x); maxX = Math.Max(maxX, x);
                }

        return (minY, minX, maxY - minY + 1, maxX - minX + 1);
    }

    /// <summary>A gold box (content coordinates) on the model canvas.</summary>
    private static int[] ToCanvas(int[] box, LetterboxBox content) => [box[0] + content.X, box[1] + content.Y, box[2], box[3]];

    private static bool Overlaps(int[] a, int[] b) =>
        a[0] < b[0] + b[2] && b[0] < a[0] + a[2] && a[1] < b[1] + b[3] && b[1] < a[1] + a[3];

    private (int Pairs, double Gap, double WinRate, int CropPairs, double CropConsistency) EvaluateDeletionAndCrops(
        int[] images, IReadOnlyList<IReadOnlyList<int>> imageTagRows, bool[] localizable, Random random)
    {
        double gapSum = 0, cropSum = 0;
        int pairs = 0, wins = 0, cropPairs = 0;
        foreach (var chunk in images.Chunk(ImagesPerPass))
        {
            using var scope = NewDisposeScope();
            var (pixels, boxes) = Load(chunk);

            // Up to 2 localizable positive tags per image: the ones the model is most sure of,
            // so the test asks "where does the evidence for this confident call sit?".
            var (pooledScores, spatial) = Forward(pixels, boxes);
            var chosen = new int[chunk.Length][];
            for (var i = 0; i < chunk.Length; i++)
                chosen[i] = imageTagRows[chunk[i]].Where(t => t < localizable.Length && localizable[t])
                    .OrderByDescending(t => pooledScores(i, t)).Take(2).ToArray();

            var gridH = (int)spatial.shape[2];
            var gridW = (int)spatial.shape[3];
            var cell = inputSize / gridW;
            var floats = 3 * inputSize * inputSize;
            var pixelArray = pixels.cpu().data<float>().ToArray();

            for (var i = 0; i < chunk.Length; i++)
            {
                if (chosen[i].Length == 0)
                    continue;
                var valid = ImagePreprocessing.ComputeSpatialValidity(boxes[i], inputSize, gridH, gridW);
                var validCells = Enumerable.Range(0, gridH * gridW).Where(c => valid[c / gridW, c % gridW]).ToArray();
                if (validCells.Length < 10)
                    continue;

                foreach (var tag in chosen[i])
                {
                    var map = HeatmapOf(spatial, i, tag);
                    var count = Math.Max(1, validCells.Length / 10);
                    var top = validCells.OrderByDescending(c => map[c / gridW, c % gridW]).Take(count).ToArray();
                    var rand = validCells.OrderBy(_ => random.Next()).Take(count).ToArray();

                    var original = pooledScores(i, tag);
                    var maskedTop = ScoreMasked(pixelArray.AsSpan(i * floats, floats), boxes[i], top, gridW, cell, tag);
                    var maskedRandom = ScoreMasked(pixelArray.AsSpan(i * floats, floats), boxes[i], rand, gridW, cell, tag);
                    var gap = (original - maskedTop) - (original - maskedRandom);
                    gapSum += gap;
                    if (gap > 0)
                        wins++;
                    pairs++;

                    var crop = CropConsistency(pixelArray.AsSpan(i * floats, floats), valid, map, tag, random);
                    if (crop is double c)
                    {
                        cropSum += c;
                        cropPairs++;
                    }
                }
            }
        }

        return (pairs, pairs > 0 ? gapSum / pairs : 0, pairs > 0 ? (double)wins / pairs : 0, cropPairs, cropPairs > 0 ? cropSum / cropPairs : 0);
    }

    private (int Images, double Correlation, double Distinct) EvaluateTwoCharacters(int[] images, IReadOnlyList<IReadOnlyList<int>> imageTagRows, HashSet<int> characterRows)
    {
        double correlationSum = 0;
        int count = 0, distinct = 0;
        foreach (var chunk in images.Chunk(ImagesPerPass))
        {
            var tags = chunk.Select(i => imageTagRows[i].Where(characterRows.Contains).Take(2).ToArray()).ToArray();
            var heatmaps = Heatmaps(chunk, tags, out var boxes);
            for (var i = 0; i < chunk.Length; i++)
            {
                var valid = ImagePreprocessing.ComputeSpatialValidity(boxes[i], inputSize, heatmaps[i][0].GetLength(0), heatmaps[i][0].GetLength(1));
                var a = heatmaps[i][0];
                var b = heatmaps[i][1];
                correlationSum += Pearson(a, b, valid);
                var (ay, ax) = Peak(a, valid);
                var (by, bx) = Peak(b, valid);
                if (a[ay, ax] > b[ay, ax] && b[by, bx] > a[by, bx])
                    distinct++;
                count++;
            }
        }

        return (count, count > 0 ? correlationSum / count : 0, count > 0 ? (double)distinct / count : 0);
    }

    private (int Pairs, double SingleRate) EvaluateSoloCounts(
        IReadOnlyList<int> heldOutImages, IReadOnlyList<IReadOnlyList<int>> imageTagRows, IReadOnlyDictionary<string, int> rowOf, Random random, int sampleImages)
    {
        if (!rowOf.TryGetValue("solo", out var solo))
            return (0, 0);
        var perPerson = OncePerPersonTags.Where(rowOf.ContainsKey).Select(t => rowOf[t]).ToHashSet();
        var candidates = heldOutImages
            .Where(i => imageTagRows[i].Contains(solo) && imageTagRows[i].Any(perPerson.Contains))
            .OrderBy(_ => random.Next()).Take(sampleImages).ToArray();

        int pairs = 0, single = 0;
        foreach (var chunk in candidates.Chunk(ImagesPerPass))
        {
            var tags = chunk.Select(i => imageTagRows[i].Where(perPerson.Contains).ToArray()).ToArray();
            var heatmaps = Heatmaps(chunk, tags, out var boxes);
            for (var i = 0; i < chunk.Length; i++)
                for (var k = 0; k < tags[i].Length; k++)
                {
                    var extracted = DetectBoxes(heatmaps[i][k], boxes[i]);
                    pairs++;
                    if (extracted.Count == 1)
                        single++;
                }
        }

        return (pairs, pairs > 0 ? (double)single / pairs : 0);
    }

    // ---- model plumbing ----

    /// <summary>Images per pass in the gold checks: 16 at 512px, fewer at a larger input (<c>evaluate --gold-input-size</c>).</summary>
    /// <summary>
    /// Images loaded at once by the crop-teacher and occlusion checks, whose whole-image scores run them in one pass: 8, as
    /// before, up to 512px, fewer above (8 at 2048px ran the A4000 out of memory).
    /// </summary>
    private int ImagesPerCheck => inputSize <= 512 ? 8 : Math.Max(1, 8 * 512 * 512 / inputSize / inputSize);

    private int ImagesPerPass => inputSize <= 512 ? 16 : Math.Max(1, 16 * 512 * 512 / inputSize / inputSize / 4);

    private (Tensor Pixels, LetterboxBox[] Boxes) Load(IReadOnlyList<int> indices)
    {
        var encoded = readImages(indices);
        var floats = 3 * inputSize * inputSize;
        var pixels = new float[indices.Count * floats];
        var boxes = new LetterboxBox[indices.Count];
        Parallel.For(0, indices.Count, i =>
        {
            boxes[i] = encoded[i].Content;
            ImagePreprocessing.ReconstructInto(encoded[i], inputSize, pixels.AsSpan(i * floats, floats));
        });
        return (tensor(pixels, [indices.Count, 3, inputSize, inputSize]).to(device), boxes);
    }

    /// <summary>
    /// Rows the classifier scores per tag (<see cref="ClassificationPooling"/>), when the tower's
    /// <see cref="ImageTower.ClassificationPoolingTemperature"/> is above 0; the rest use the averaged embedding.
    /// </summary>
    public IReadOnlyList<bool>? ClassificationPooledTags { get; init; }

    /// <summary>Images per forward pass in the checks that only need whole-image scores (<see cref="Score"/>).</summary>
    private int ScoreBatch => inputSize <= 512 ? 32 : Math.Max(2, 32 * 512 * 512 / inputSize / inputSize);

    /// <summary>
    /// A whole-image scorer alone: the pooled embedding without building the full map (<see cref="ImageTower.ForwardPooled"/>),
    /// unless per-tag classification pooling needs the cells.
    /// </summary>
    /// <param name="content">Each image's content box: pooled over the content only, as training and the held-out
    /// evaluation pool, not the letterbox bars. Null when the whole canvas is content (a resized crop).</param>
    private Func<int, int, float> Score(Tensor pixels, IReadOnlyList<LetterboxBox>? content = null)
    {
        if (imageTower.ClassificationPoolingTemperature > 0 && ClassificationPooledTags is not null)
            return Forward(pixels, content).PooledScore;
        using var _ = no_grad();
        using var scope = NewDisposeScope();
        var mask = ContentMask(pixels, content);
        // With a stride-2 localization level, skip its map (the 32-image batch's would be ~6 GB); without, as before.
        var pooled = (imageTower.Fine is not null ? imageTower.ForwardPooledOnly(pixels, mask) : imageTower.ForwardPooled(pixels, mask).Pooled).cpu().data<float>().ToArray();
        return (image, tag) => Sigmoid(scaling.Apply(Dot(pooled.AsSpan(image * dim, dim), tagEmbeddings.AsSpan(tag * dim, dim))));
    }

    /// <summary>[batch, 1, map, map]: 1 on each image's content cells (<paramref name="content"/>), or everywhere when null.</summary>
    private Tensor ContentMask(Tensor pixels, IReadOnlyList<LetterboxBox>? content)
    {
        var (height, width) = (pixels.shape[2] / imageTower.OutputStride, pixels.shape[3] / imageTower.OutputStride);
        return content is null
            ? ones(pixels.shape[0], 1, height, width, device: pixels.device)
            : SpatialMask.Build(content, inputSize, height, width, pixels.device);
    }

    /// <summary>
    /// The localization map (left on the device; <see cref="HeatmapOf"/> reads one tag's map off it) plus a whole-image
    /// scorer, pooled over <paramref name="content"/> as in <see cref="Score"/>.
    /// </summary>
    private (Func<int, int, float> PooledScore, Tensor Spatial) Forward(Tensor pixels, IReadOnlyList<LetterboxBox>? content = null)
    {
        using var _ = no_grad();
        if (imageTower.ClassificationGlimpses && imageTower.ClassificationPoolingTemperature > 0 && ClassificationPooledTags is not null)
            return GlimpseForward(pixels, content);
        var temperature = imageTower.ClassificationPoolingTemperature;
        if (imageTower.Fine is not null && (temperature <= 0 || ClassificationPooledTags is null))
        {
            // With a stride-2 localization level the full map would be upsampled to its resolution only to be pooled
            // (2 GB for 16 images): pool it without building it. Only then, so other towers' numbers stay identical.
            var (pooledTensor, fineLocalization) = imageTower.ForwardPooled(pixels, ContentMask(pixels, content));
            float[] finePooled;
            using (pooledTensor)
                finePooled = pooledTensor.cpu().data<float>().ToArray();
            return ((image, tag) => Sigmoid(scaling.Apply(Dot(finePooled.AsSpan(image * dim, dim), tagEmbeddings.AsSpan(tag * dim, dim)))), fineLocalization);
        }
        // Whole-image scores pool the full map; heatmaps come from the localization map.
        var (spatial, localization) = imageTower.ForwardMaps(pixels);
        float[] pooled;
        using (NewDisposeScope())
            pooled = ImageTower.MaskedPool(spatial, ContentMask(pixels, content)).cpu().data<float>().ToArray();
        if (temperature <= 0 || ClassificationPooledTags is not { } perTag)
            return ((image, tag) => Sigmoid(scaling.Apply(Dot(pooled.AsSpan(image * dim, dim), tagEmbeddings.AsSpan(tag * dim, dim)))), localization);

        // [B, D, cells], the whole canvas counted as content, as the averaged score here is, so a plain average per
        // cell (masking with ones would copy the full map, which ran the A4000 out of memory on gold batches).
        var cells = nn.functional.adaptive_avg_pool2d(spatial, [ClassificationPooling.Grid, ClassificationPooling.Grid]).flatten(2).cpu();
        var cellCount = (int)cells.shape[2];
        var cellValues = cells.data<float>().ToArray();
        return ((image, tag) =>
        {
            var embedding = tagEmbeddings.AsSpan(tag * dim, dim);
            if (!perTag[tag])
                return Sigmoid(scaling.Apply(Dot(pooled.AsSpan(image * dim, dim), embedding)));
            // Pooled over the cells as ClassificationPooling.RawScores pools them.
            var dots = new double[cellCount];
            for (var c = 0; c < cellCount; c++)
            {
                double sum = 0;
                for (var d = 0; d < dim; d++)
                    sum += cellValues[((long)image * dim + d) * cellCount + c] * embedding[d];
                dots[c] = sum;
            }
            return Sigmoid(scaling.Apply((float)ClassificationPooling.Pool(dots, temperature, imageTower.ClassificationPoolingTopK, imageTower.ClassificationPoolingBottomWeight)));
        }, localization);
    }

    /// <summary>
    /// <see cref="Forward"/> for a tower that scores localizable tags from glimpses (<see cref="Glimpses"/>): the windows the
    /// model chooses from its localization map (<see cref="Glimpses.Look"/>; the content is the whole canvas when
    /// <paramref name="content"/> is null), pooled as <see cref="ClassificationPooling.RawScores"/> pools them.
    /// </summary>
    private (Func<int, int, float> PooledScore, Tensor Spatial) GlimpseForward(Tensor pixels, IReadOnlyList<LetterboxBox>? content)
    {
        // The pooled embedding without building the full map, which glimpse batches can't afford next to their crops.
        var (pooledTensor, localization) = imageTower.ForwardPooled(pixels, ContentMask(pixels, content));
        var boxes = content ?? Enumerable.Repeat(new LetterboxBox(0, 0, (int)pixels.shape[3], (int)pixels.shape[2]), (int)pixels.shape[0]).ToList();
        float[] pooled, cellValues;
        bool[] valid;
        int cellCount;
        using (NewDisposeScope())
        {
            pooled = pooledTensor.cpu().data<float>().ToArray();
            // The windows the model chooses, as in training (or the whole grid: --glimpse-grid).
            var (cells, validCells) = Glimpses.Look(imageTower, pixels, boxes, pooledTensor, localization, LocalizableTags());
            pooledTensor.Dispose();
            cellCount = (int)cells.shape[2];
            cellValues = cells.cpu().data<float>().ToArray();
            valid = validCells.cpu().data<bool>().ToArray();
        }
        var temperature = imageTower.ClassificationPoolingTemperature;
        var perTag = ClassificationPooledTags!;
        return ((image, tag) =>
        {
            var embedding = tagEmbeddings.AsSpan(tag * dim, dim);
            if (!perTag[tag])
                return Sigmoid(scaling.Apply(Dot(pooled.AsSpan(image * dim, dim), embedding)));
            var dots = new List<double>();
            for (var c = 0; c < cellCount; c++)
            {
                if (!valid[image * cellCount + c])
                    continue;
                double sum = 0;
                for (var d = 0; d < dim; d++)
                    sum += cellValues[((long)image * dim + d) * cellCount + c] * embedding[d];
                dots.Add(sum);
            }
            if (dots.Count == 0)
                return Sigmoid(scaling.Apply(Dot(pooled.AsSpan(image * dim, dim), embedding)));
            return Sigmoid(scaling.Apply((float)ClassificationPooling.Pool(dots, temperature, imageTower.ClassificationPoolingTopK, imageTower.ClassificationPoolingBottomWeight)));
        }, localization);
    }

    private Tensor? _localizableTags;

    /// <summary>[L, D] the localizable tags' embeddings on the device (<see cref="ClassificationPooledTags"/>), which <see cref="Glimpses.Propose"/> ranks.</summary>
    private Tensor LocalizableTags()
    {
        if (_localizableTags is null)
        {
            _deviceTags ??= tensor(tagEmbeddings, [tagEmbeddings.Length / dim, dim]).to(device).DetachFromDisposeScope();
            var rows = ClassificationPooledTags!.Select((pooledTag, t) => (pooledTag, t)).Where(x => x.pooledTag).Select(x => (long)x.t).ToArray();
            using var index = tensor(rows, device: device);
            _localizableTags = _deviceTags.index_select(0, index).DetachFromDisposeScope();
        }
        return _localizableTags;
    }

    /// <summary>
    /// A scorer reading tags off the localization map: per-location logits over the image's content, pooled by a
    /// log-sum-exp at <paramref name="temperature"/> (as <see cref="SigmoidContrastiveLoss.ComputeLocalizedPairs"/>
    /// pools them), then the learned scaling. The map stays on the device; each call scores one pair.
    /// </summary>
    private Func<int, int, float> MapScorer(Tensor pixels, LetterboxBox[] boxes, float temperature)
    {
        Tensor localization, valid;
        using (no_grad())
        {
            localization = imageTower.ForwardMaps(pixels).Localization;
            var (height, width) = (localization.shape[2], localization.shape[3]);
            localization = localization.reshape([localization.shape[0], dim, height * width]);
            valid = SpatialMask.Build(boxes, inputSize, height, width, device).reshape([localization.shape[0], height * width]).gt(0);
        }

        _deviceTags ??= tensor(tagEmbeddings, [tagEmbeddings.Length / dim, dim]).to(device).DetachFromDisposeScope();
        return (image, tag) =>
        {
            using var scope = NewDisposeScope();
            using var _ = no_grad();
            var logits = (_deviceTags[tag].matmul(localization[image]) / temperature).masked_fill(valid[image].logical_not(), float.NegativeInfinity);
            var pooled = temperature * (logits.logsumexp(0) - valid[image].sum().to_type(ScalarType.Float32).log());
            return Sigmoid(scaling.Apply(pooled.item<float>()));
        };
    }

    private Tensor? _deviceTags;

    private float[][][,] Heatmaps(IReadOnlyList<int> indices, int[][] tagsPerImage, out LetterboxBox[] boxes)
    {
        using var scope = NewDisposeScope();
        var (pixels, loadedBoxes) = Load(indices);
        boxes = loadedBoxes;
        var (_, spatial) = Forward(pixels, loadedBoxes);
        var result = new float[indices.Count][][,];
        for (var i = 0; i < indices.Count; i++)
            result[i] = tagsPerImage[i].Select(t => HeatmapOf(spatial, i, t)).ToArray();
        return result;
    }

    /// <summary>One tag's sigmoid heatmap off a localization map from <see cref="Forward"/> (the dot products on its device).</summary>
    private float[,] HeatmapOf(Tensor localization, int image, int tag)
    {
        var gridH = (int)localization.shape[2];
        var gridW = (int)localization.shape[3];
        float[] values;
        using (no_grad())
        using (NewDisposeScope())
        {
            _deviceTags ??= tensor(tagEmbeddings, [tagEmbeddings.Length / dim, dim]).to(device).DetachFromDisposeScope();
            values = _deviceTags[tag].matmul(localization[image].reshape([dim, gridH * gridW])).cpu().data<float>().ToArray();
        }

        var map = new float[gridH, gridW];
        for (var y = 0; y < gridH; y++)
            for (var x = 0; x < gridW; x++)
                map[y, x] = Sigmoid(scaling.Apply(values[y * gridW + x]));
        return map;
    }

    private float ScoreMasked(ReadOnlySpan<float> imagePixels, LetterboxBox content, int[] cells, int gridW, int cell, int tag)
    {
        using var scope = NewDisposeScope();
        var copy = imagePixels.ToArray();
        var channel = inputSize * inputSize;
        foreach (var c in cells)
        {
            var (gy, gx) = (c / gridW, c % gridW);
            for (var y = gy * cell; y < (gy + 1) * cell; y++)
                for (var x = gx * cell; x < (gx + 1) * cell; x++)
                    for (var ch = 0; ch < 3; ch++)
                        copy[ch * channel + y * inputSize + x] = 0f; // the normalized "neutral" padding value
        }

        return Score(tensor(copy, [1, 3, inputSize, inputSize]).to(device), [content])(0, tag);
    }

    /// <summary>Heatmap of a random ~70% crop (cell-aligned, inside the content) vs. the same crop of the full heatmap, both on the crop's own grid.</summary>
    private double? CropConsistency(ReadOnlySpan<float> imagePixels, bool[,] valid, float[,] fullMap, int tag, Random random)
    {
        var gridH = valid.GetLength(0);
        var gridW = valid.GetLength(1);
        int minY = gridH, maxY = -1, minX = gridW, maxX = -1;
        for (var y = 0; y < gridH; y++)
            for (var x = 0; x < gridW; x++)
                if (valid[y, x])
                {
                    minY = Math.Min(minY, y); maxY = Math.Max(maxY, y);
                    minX = Math.Min(minX, x); maxX = Math.Max(maxX, x);
                }

        var contentH = maxY - minY + 1;
        var contentW = maxX - minX + 1;
        var cropH = Math.Max(4, (int)(contentH * 0.7));
        var cropW = Math.Max(4, (int)(contentW * 0.7));
        if (cropH >= contentH || cropW >= contentW)
            return null;
        var y0 = minY + random.Next(contentH - cropH + 1);
        var x0 = minX + random.Next(contentW - cropW + 1);

        using var scope = NewDisposeScope();
        var cell = inputSize / gridW;
        var image = tensor(imagePixels.ToArray(), [1, 3, inputSize, inputSize]);
        var cropped = image[.., .., TensorIndex.Slice(y0 * cell, (y0 + cropH) * cell), TensorIndex.Slice(x0 * cell, (x0 + cropW) * cell)];
        var resized = nn.functional.interpolate(cropped, [inputSize, inputSize], mode: InterpolationMode.Bilinear, align_corners: false);
        var (_, spatial) = Forward(resized.to(device));
        var cropMap = HeatmapOf(spatial, 0, tag);

        var fullCrop = new float[cropH * cropW];
        for (var y = 0; y < cropH; y++)
            for (var x = 0; x < cropW; x++)
                fullCrop[y * cropW + x] = fullMap[y0 + y, x0 + x];
        var upsampled = nn.functional.interpolate(tensor(fullCrop, [1, 1, cropH, cropW]), [gridH, gridW], mode: InterpolationMode.Bilinear, align_corners: false)
            .data<float>().ToArray();

        var all = new bool[gridH, gridW];
        for (var y = 0; y < gridH; y++)
            for (var x = 0; x < gridW; x++)
                all[y, x] = true;
        var reference = new float[gridH, gridW];
        for (var y = 0; y < gridH; y++)
            for (var x = 0; x < gridW; x++)
                reference[y, x] = upsampled[y * gridW + x];
        return Pearson(cropMap, reference, all);
    }

    // ---- geometry / statistics ----

    /// <summary>Boxes as <c>detect</c> extracts them (its defaults), in content pixels; without its edge-guided refinement, which needs the source image.</summary>
    private List<BoundingBox> DetectBoxes(float[,] map, LetterboxBox content) =>
        TagScorer.DetectBoxes(map, TagScorer.DefaultBoxThreshold, TagScorer.DefaultBoxPercentile, content, inputSize, content.Width, content.Height,
            TagScorer.DefaultMinHeatmapRange, TagScorer.DefaultMinBoxFraction, TagScorer.DefaultMaxBoxesPerTag, TagScorer.DefaultBoxBelowPeak);

    /// <summary>The best map cell inside <paramref name="boxes"/> (cell centres, content pixels), as the share of the content's cells at or below it.</summary>
    private double BoxBestRank(float[,] map, LetterboxBox content, int[][] boxes)
    {
        var valid = ImagePreprocessing.ComputeSpatialValidity(content, inputSize, map.GetLength(0), map.GetLength(1));
        var cellW = inputSize / (float)map.GetLength(1);
        var cellH = inputSize / (float)map.GetLength(0);
        var best = float.NegativeInfinity;
        for (var y = 0; y < map.GetLength(0); y++)
            for (var x = 0; x < map.GetLength(1); x++)
                if (valid[y, x] && boxes.Any(b => Contains(b, (x + 0.5f) * cellW - content.X, (y + 0.5f) * cellH - content.Y)))
                    best = Math.Max(best, map[y, x]);
        if (float.IsNegativeInfinity(best))
            return -1; // a box smaller than a cell, between cell centres (not NaN: JSON can't write it)
        int below = 0, total = 0;
        for (var y = 0; y < map.GetLength(0); y++)
            for (var x = 0; x < map.GetLength(1); x++)
                if (valid[y, x])
                {
                    total++;
                    if (map[y, x] <= best)
                        below++;
                }
        return (double)below / total;
    }

    /// <summary>The <paramref name="count"/> highest local maxima (9x9 cells) of the content, as [x, y] cell centres in content pixels.</summary>
    private int[][] TopPeaks(float[,] map, LetterboxBox content, int count)
    {
        var valid = ImagePreprocessing.ComputeSpatialValidity(content, inputSize, map.GetLength(0), map.GetLength(1));
        var (h, w) = (map.GetLength(0), map.GetLength(1));
        var cellW = inputSize / (float)w;
        var cellH = inputSize / (float)h;
        var maxima = new List<(float Value, int Y, int X)>();
        for (var y = 0; y < h; y++)
            for (var x = 0; x < w; x++)
            {
                if (!valid[y, x])
                    continue;
                var isMax = true;
                for (var dy = -4; dy <= 4 && isMax; dy++)
                    for (var dx = -4; dx <= 4; dx++)
                    {
                        var (yy, xx) = (y + dy, x + dx);
                        if ((dy != 0 || dx != 0) && yy >= 0 && yy < h && xx >= 0 && xx < w && valid[yy, xx] && map[yy, xx] > map[y, x])
                        {
                            isMax = false;
                            break;
                        }
                    }
                if (isMax)
                    maxima.Add((map[y, x], y, x));
            }
        return maxima.OrderByDescending(m => m.Value).Take(count)
            .Select(m => new[] { (int)((m.X + 0.5f) * cellW - content.X), (int)((m.Y + 0.5f) * cellH - content.Y) }).ToArray();
    }

    /// <summary>Local maxima (9x9 cells) outside <paramref name="boxes"/> higher than the best cell inside them; -1 when no cell centre is inside.</summary>
    private int PeaksAbove(float[,] map, LetterboxBox content, int[][] boxes)
    {
        var valid = ImagePreprocessing.ComputeSpatialValidity(content, inputSize, map.GetLength(0), map.GetLength(1));
        var (h, w) = (map.GetLength(0), map.GetLength(1));
        var cellW = inputSize / (float)w;
        var cellH = inputSize / (float)h;
        bool Inside(int y, int x) => boxes.Any(b => Contains(b, (x + 0.5f) * cellW - content.X, (y + 0.5f) * cellH - content.Y));
        var best = float.NegativeInfinity;
        for (var y = 0; y < h; y++)
            for (var x = 0; x < w; x++)
                if (valid[y, x] && Inside(y, x))
                    best = Math.Max(best, map[y, x]);
        if (float.IsNegativeInfinity(best))
            return -1;
        var count = 0;
        for (var y = 0; y < h; y++)
            for (var x = 0; x < w; x++)
            {
                if (!valid[y, x] || map[y, x] <= best || Inside(y, x))
                    continue;
                var isMax = true;
                for (var dy = -4; dy <= 4 && isMax; dy++)
                    for (var dx = -4; dx <= 4; dx++)
                    {
                        var (yy, xx) = (y + dy, x + dx);
                        if ((dy != 0 || dx != 0) && yy >= 0 && yy < h && xx >= 0 && xx < w && valid[yy, xx] && map[yy, xx] > map[y, x])
                        {
                            isMax = false;
                            break;
                        }
                    }
                if (isMax)
                    count++;
            }
        return count;
    }

    private (float X, float Y) PeakCellCenterInContent(float[,] map, LetterboxBox content)
    {
        var valid = ImagePreprocessing.ComputeSpatialValidity(content, inputSize, map.GetLength(0), map.GetLength(1));
        var (py, px) = Peak(map, valid);
        var cellW = inputSize / (float)map.GetLength(1);
        var cellH = inputSize / (float)map.GetLength(0);
        return ((px + 0.5f) * cellW - content.X, (py + 0.5f) * cellH - content.Y);
    }

    /// <summary>Share of the tag's normalized heatmap mass inside <paramref name="goldBoxes"/>, with cells inside <paramref name="ignoreBoxes"/> left out of both sides.</summary>
    /// <summary>
    /// <see cref="GoldResult.RegionPrecision"/> and <see cref="GoldResult.RegionRecall"/> for one pair: the content cells within
    /// <see cref="TagScorer.DefaultBoxBelowPeak"/> log-odds of the map's peak, against the boxes' cells (by cell centre). Null when no
    /// box covers a cell centre (a box smaller than a cell).
    /// </summary>
    private (double Precision, double Recall)? RegionInBoxes(float[,] map, LetterboxBox content, IReadOnlyList<int[]> goldBoxes, IReadOnlyList<int[]> ignoreBoxes)
    {
        var h = map.GetLength(0);
        var w = map.GetLength(1);
        var valid = ImagePreprocessing.ComputeSpatialValidity(content, inputSize, h, w);
        var max = float.MinValue;
        for (var y = 0; y < h; y++)
            for (var x = 0; x < w; x++)
                if (valid[y, x]) max = Math.Max(max, map[y, x]);
        var peak = Math.Clamp(max, 1e-7f, 1 - 1e-7f);
        var cut = 1 / (1 + MathF.Exp(-(MathF.Log(peak / (1 - peak)) - TagScorer.DefaultBoxBelowPeak)));
        int region = 0, regionInside = 0, boxCells = 0, boxCellsInRegion = 0;
        var cellW = inputSize / (float)w;
        var cellH = inputSize / (float)h;
        for (var y = 0; y < h; y++)
            for (var x = 0; x < w; x++)
            {
                if (!valid[y, x])
                    continue;
                var cx = (x + 0.5f) * cellW - content.X;
                var cy = (y + 0.5f) * cellH - content.Y;
                var inRegion = map[y, x] >= cut;
                var inBox = goldBoxes.Any(b => cx >= b[0] && cx < b[0] + b[2] && cy >= b[1] && cy < b[1] + b[3]);
                if (inBox)
                {
                    boxCells++;
                    if (inRegion) boxCellsInRegion++;
                }
                if (inRegion && (inBox || !ignoreBoxes.Any(b => cx >= b[0] && cx < b[0] + b[2] && cy >= b[1] && cy < b[1] + b[3])))
                {
                    region++;
                    if (inBox) regionInside++;
                }
            }

        return boxCells == 0 ? null : (region > 0 ? (double)regionInside / region : 0, (double)boxCellsInRegion / boxCells);
    }

    private double EnergyInBoxes(float[,] map, LetterboxBox content, IReadOnlyList<int[]> goldBoxes, IReadOnlyList<int[]> ignoreBoxes)
    {
        var h = map.GetLength(0);
        var w = map.GetLength(1);
        var valid = ImagePreprocessing.ComputeSpatialValidity(content, inputSize, h, w);
        float min = float.MaxValue, max = float.MinValue;
        for (var y = 0; y < h; y++)
            for (var x = 0; x < w; x++)
                if (valid[y, x]) { min = Math.Min(min, map[y, x]); max = Math.Max(max, map[y, x]); }
        if (max - min < 1e-6f)
            return 0;

        double inside = 0, total = 0;
        var cellW = inputSize / (float)w;
        var cellH = inputSize / (float)h;
        for (var y = 0; y < h; y++)
            for (var x = 0; x < w; x++)
            {
                if (!valid[y, x])
                    continue;
                var cx = (x + 0.5f) * cellW - content.X;
                var cy = (y + 0.5f) * cellH - content.Y;
                if (ignoreBoxes.Any(b => cx >= b[0] && cx < b[0] + b[2] && cy >= b[1] && cy < b[1] + b[3]))
                    continue;
                var v = (map[y, x] - min) / (max - min);
                total += v;
                if (goldBoxes.Any(b => cx >= b[0] && cx < b[0] + b[2] && cy >= b[1] && cy < b[1] + b[3]))
                    inside += v;
            }

        return total > 0 ? inside / total : 0;
    }

    private static (int Y, int X) Peak(float[,] map, bool[,] valid)
    {
        var best = (Y: 0, X: 0);
        var bestValue = float.MinValue;
        for (var y = 0; y < map.GetLength(0); y++)
            for (var x = 0; x < map.GetLength(1); x++)
                if (valid[y, x] && map[y, x] > bestValue)
                {
                    bestValue = map[y, x];
                    best = (y, x);
                }
        return best;
    }

    private static double Pearson(float[,] a, float[,] b, bool[,] valid)
    {
        double sa = 0, sb = 0, n = 0;
        for (var y = 0; y < a.GetLength(0); y++)
            for (var x = 0; x < a.GetLength(1); x++)
                if (valid[y, x]) { sa += a[y, x]; sb += b[y, x]; n++; }
        if (n < 2)
            return 0;
        double ma = sa / n, mb = sb / n, cov = 0, va = 0, vb = 0;
        for (var y = 0; y < a.GetLength(0); y++)
            for (var x = 0; x < a.GetLength(1); x++)
                if (valid[y, x])
                {
                    cov += (a[y, x] - ma) * (b[y, x] - mb);
                    va += (a[y, x] - ma) * (a[y, x] - ma);
                    vb += (b[y, x] - mb) * (b[y, x] - mb);
                }
        return va > 0 && vb > 0 ? cov / Math.Sqrt(va * vb) : 0;
    }

    public static double IoU(int[] a, int[] b)
    {
        var x1 = Math.Max(a[0], b[0]);
        var y1 = Math.Max(a[1], b[1]);
        var x2 = Math.Min(a[0] + a[2], b[0] + b[2]);
        var y2 = Math.Min(a[1] + a[3], b[1] + b[3]);
        var intersection = Math.Max(0, x2 - x1) * (double)Math.Max(0, y2 - y1);
        var union = (double)a[2] * a[3] + (double)b[2] * b[3] - intersection;
        return union > 0 ? intersection / union : 0;
    }

    private static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        var sum = 0f;
        for (var i = 0; i < a.Length; i++)
            sum += a[i] * b[i];
        return sum;
    }

    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));
}
