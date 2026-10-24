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
public sealed record GoldResult(string Name, int Pairs, int GoldInstances, double PointingAccuracy, double EnergyInBox, double InstancePrecision, double InstanceRecall, double MeanCountError);

/// <param name="DeletionGap">Mean drop in a tag's whole-image confidence when its top-heatmap cells are blanked, minus the drop from blanking the same number of random cells. Larger = the heatmap points at what the prediction actually depends on.</param>
/// <param name="DeletionWinRate">Fraction of pairs where blanking the heatmap's top cells hurt more than blanking random cells (0.5 = chance).</param>
/// <param name="TwoCharacterCorrelation">Mean Pearson correlation between the two characters' heatmaps on images tagged with exactly two characters. Lower = they point at different places.</param>
/// <param name="TwoCharacterDistinctPeaks">Fraction of those images where each character's peak lies where its own heatmap beats the other's.</param>
/// <param name="CropConsistency">Mean correlation between a crop's own heatmap and the same crop of the full image's heatmap. 1 = heatmaps move with the content.</param>
/// <param name="SoloSingleInstanceRate">On images tagged solo, the fraction where a once-per-person tag (blush, necktie, hat, ...) yields exactly one extracted box.</param>
/// <param name="Teacher">Crop-teacher check on the gold set, overall and per stratum (<see cref="TeacherResult"/>); empty without a gold set.</param>
/// <param name="TeacherTags">The same per tag.</param>
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
    IReadOnlyList<TeacherResult>? TeacherTags = null);

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
        var (teacherSummary, teacherTags) = gold is { Count: > 0 } ? EvaluateTeacher(gold, rowOf, random) : ([], []);

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
            teacherSummary, teacherTags);
    }

    /// <summary>Running totals for one tag or stratum.</summary>
    private sealed class GoldTally
    {
        public int Pairs, GoldInstances, PointingCounted, PointingHits, ScoredExtracted, MatchedExtracted, RecallItems, RecalledItems, CountPairs;
        public double Energy, CountError;

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
            Energy += score.Energy;
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
            CountPairs > 0 ? CountError / CountPairs : 0);
    }

    /// <summary>One (image, tag) pair's gold scoring. <see cref="PeakHit"/> is null when the peak fell in an ignore region; <see cref="CountError"/> is null when the pair has group or ignore boxes.</summary>
    public readonly record struct PairScore(bool? PeakHit, double Energy, int ScoredExtracted, int MatchedExtracted, int RecallItems, int RecalledItems, int? CountError);

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
        return new PairScore(peakHit, energy, scored, matched, certain.Count + groups.Count, recalled, countError);
    }

    private static bool Contains(int[] b, float x, float y) => x >= b[0] && x < b[0] + b[2] && y >= b[1] && y < b[1] + b[3];

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
        foreach (var chunk in gold.Chunk(16))
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
                    var extracted = TagScorer.DetectBoxes(map, 0.5f, 0.6f, boxes[i], inputSize, boxes[i].Width, boxes[i].Height, 0.05f)
                        .Select(b => new[] { b.X, b.Y, b.Width, b.Height }).ToList();
                    var score = ScorePair(
                        instances, PeakCellCenterInContent(map, boxes[i]),
                        EnergyInBoxes(map, boxes[i],
                            instances.Where(x => x.Kind != GoldKind.Ignore).Select(x => x.Box).ToList(),
                            instances.Where(x => x.Kind == GoldKind.Ignore).Select(x => x.Box).ToList()),
                        extracted);

                    (perTag.TryGetValue(tag, out var tagTally) ? tagTally : perTag[tag] = new()).Add(score);
                    perStratum["all"].Add(score);
                    (perStratum.TryGetValue(image.Stratum, out var stratumTally) ? stratumTally : perStratum[image.Stratum] = new()).Add(score);
                }
            }
        }

        return (
            perStratum.Select(kv => kv.Value.ToResult(kv.Key)).OrderBy(r => r.Name == "all" ? 0 : 1).ThenBy(r => r.Name).ToList(),
            perTag.Select(kv => kv.Value.ToResult(kv.Key)).OrderBy(r => r.PointingAccuracy).ToList());
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
        foreach (var chunk in gold.Chunk(8))
        {
            using var scope = NewDisposeScope();
            var (pixels, boxes) = Load(chunk.Select(g => g.CacheIndex).ToArray());
            var (fullScore, _) = Forward(pixels);

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

            foreach (var batch in crops.Chunk(8))
            {
                using var batchScope = NewDisposeScope();
                var cropPixels = cat(batch.Select(c => UnbooruTagger.Training.Training.CropConsistency.CropPixels(pixels[c.Image..(c.Image + 1)], [c.Window], grid)).ToArray(), 0);
                var (cropScore, _) = Forward(cropPixels);
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
        foreach (var chunk in images.Chunk(16))
        {
            using var scope = NewDisposeScope();
            var (pixels, boxes) = Load(chunk);

            // Up to 2 localizable positive tags per image: the ones the model is most sure of,
            // so the test asks "where does the evidence for this confident call sit?".
            var (pooledScores, spatial) = Forward(pixels);
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
                    var maskedTop = ScoreMasked(pixelArray.AsSpan(i * floats, floats), top, gridW, cell, tag);
                    var maskedRandom = ScoreMasked(pixelArray.AsSpan(i * floats, floats), rand, gridW, cell, tag);
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
        foreach (var chunk in images.Chunk(16))
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
        foreach (var chunk in candidates.Chunk(16))
        {
            var tags = chunk.Select(i => imageTagRows[i].Where(perPerson.Contains).ToArray()).ToArray();
            var heatmaps = Heatmaps(chunk, tags, out var boxes);
            for (var i = 0; i < chunk.Length; i++)
                for (var k = 0; k < tags[i].Length; k++)
                {
                    var extracted = TagScorer.DetectBoxes(heatmaps[i][k], 0.5f, 0.6f, boxes[i], inputSize, boxes[i].Width, boxes[i].Height, 0.05f);
                    pairs++;
                    if (extracted.Count == 1)
                        single++;
                }
        }

        return (pairs, pairs > 0 ? (double)single / pairs : 0);
    }

    // ---- model plumbing ----

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

    /// <summary>Spatial map plus a whole-image scorer (masked pooling, as training and inference pool).</summary>
    private (Func<int, int, float> PooledScore, Tensor Spatial) Forward(Tensor pixels)
    {
        using var _ = no_grad();
        // Whole-image scores pool the full map; heatmaps come from the localization map.
        var (spatial, localization) = imageTower.ForwardMaps(pixels);
        var spatialCpu = localization.cpu();
        var pooled = spatial.cpu().mean([2, 3]).data<float>().ToArray();
        return ((image, tag) => Sigmoid(scaling.Apply(Dot(pooled.AsSpan(image * dim, dim), tagEmbeddings.AsSpan(tag * dim, dim)))), spatialCpu);
    }

    private float[][][,] Heatmaps(IReadOnlyList<int> indices, int[][] tagsPerImage, out LetterboxBox[] boxes)
    {
        using var scope = NewDisposeScope();
        var (pixels, loadedBoxes) = Load(indices);
        boxes = loadedBoxes;
        var (_, spatial) = Forward(pixels);
        var result = new float[indices.Count][][,];
        for (var i = 0; i < indices.Count; i++)
            result[i] = tagsPerImage[i].Select(t => HeatmapOf(spatial, i, t)).ToArray();
        return result;
    }

    private float[,] HeatmapOf(Tensor spatialCpu, int image, int tag)
    {
        var gridH = (int)spatialCpu.shape[2];
        var gridW = (int)spatialCpu.shape[3];
        var values = spatialCpu[image].data<float>().ToArray(); // [dim, H, W]
        var map = new float[gridH, gridW];
        var embedding = tagEmbeddings.AsSpan(tag * dim, dim);
        for (var y = 0; y < gridH; y++)
            for (var x = 0; x < gridW; x++)
            {
                var sum = 0f;
                var at = y * gridW + x;
                for (var d = 0; d < dim; d++)
                    sum += values[d * gridH * gridW + at] * embedding[d];
                map[y, x] = Sigmoid(scaling.Apply(sum));
            }

        return map;
    }

    private float ScoreMasked(ReadOnlySpan<float> imagePixels, int[] cells, int gridW, int cell, int tag)
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

        var (score, _) = Forward(tensor(copy, [1, 3, inputSize, inputSize]).to(device));
        return score(0, tag);
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

    private (float X, float Y) PeakCellCenterInContent(float[,] map, LetterboxBox content)
    {
        var valid = ImagePreprocessing.ComputeSpatialValidity(content, inputSize, map.GetLength(0), map.GetLength(1));
        var (py, px) = Peak(map, valid);
        var cellW = inputSize / (float)map.GetLength(1);
        var cellH = inputSize / (float)map.GetLength(0);
        return ((px + 0.5f) * cellW - content.X, (py + 0.5f) * cellH - content.Y);
    }

    /// <summary>Share of the tag's normalized heatmap mass inside <paramref name="goldBoxes"/>, with cells inside <paramref name="ignoreBoxes"/> left out of both sides.</summary>
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
