using UnbooruTagger.Core.Dataset;
using System.Numerics;
using UnbooruTagger.Core.Scoring;
using UnbooruTagger.Core.Vocabulary;
using static TorchSharp.torch;

namespace UnbooruTagger.Training.Training;

/// <summary>
/// Held-out images where two mutually exclusive tags (<see cref="ExclusiveTagGroups"/>) both cleared their
/// thresholds. <paramref name="FirstTrue"/> / <paramref name="SecondTrue"/>: how many of those images
/// actually carry that tag (the rest of each pair's images carry neither).
/// </summary>
public sealed record ExclusiveViolation(string First, string Second, int Images, int FirstTrue, int SecondTrue);

/// <summary>One tag's held-out ranking quality.</summary>
/// <param name="AveragePrecision">AP over the tag's positives vs. its hard AND random negatives.</param>
/// <param name="HardAveragePrecision">AP over the tag's positives vs. only its hard negatives (NaN if it had none) — how well it separates from the tags it's actually confused with.</param>
/// <param name="BestF1">Best F1 over every possible cutoff on the same candidate set.</param>
/// <param name="Threshold">
/// The calibrated cutoff: the one maximizing F1 over EVERY held-out image, at the tag's real
/// prevalence — not over the candidate set, whose ~1:10 positive:negative ratio would pick
/// cutoffs far too loose for a tag that's really on 1 image in 1000.
/// </param>
/// <param name="HeldOutF1">F1 over every held-out image at <paramref name="Threshold"/>; NaN if the tag had too few held-out positives to calibrate (it then uses the fallback threshold).</param>
/// <param name="ContextNegatives">Held-out context negatives scored (<see cref="TagStatistics.ContextSources"/>: the tag's usual context without the tag).</param>
/// <param name="ContextAveragePrecision">AP of positives vs. only those (NaN if none): can the classifier tell the object from its context? Some are missing labels, so this is a lower bound. With few context negatives (a median of 2 per tag on a 20K sample) its chance level is close to 1, so it says little; <see cref="EvaluationReport.ContextReliance"/> measures the same question on every held-out image.</param>
public sealed record TagEvaluation(
    string Tag, int Positives, int HardNegatives, int RandomNegatives,
    double AveragePrecision, double HardAveragePrecision, double BestF1, float Threshold, double HeldOutF1,
    int ContextNegatives = 0, double ContextAveragePrecision = double.NaN);

/// <param name="MeanHeldOutF1">Macro mean of <see cref="TagEvaluation.HeldOutF1"/> — per-tag F1 at real prevalence.</param>
/// <param name="MeanAveragePrecision">Macro mean of <see cref="TagEvaluation.AveragePrecision"/> over every evaluated tag — the early-stopping metric.</param>
/// <param name="MicroPrecision">
/// Image-level: of every tag reported at its calibrated threshold, the fraction the image
/// was actually tagged with. A lower bound — a correct prediction of a tag the tagger left
/// off counts against it.
/// </param>
/// <param name="MicroRecall">Image-level: of every tag the images actually carry, the fraction reported at its calibrated threshold.</param>
/// <param name="RPrecision">Image-level: for an image with R true tags, the fraction of its top-R scored tags that are true, averaged over images — threshold-free.</param>
/// <param name="ExclusiveViolationRate">Fraction of images where two mutually exclusive tags both cleared their thresholds.</param>
/// <param name="MeanContextAveragePrecision">Macro mean of <see cref="TagEvaluation.ContextAveragePrecision"/> over tags that had context negatives.</param>
/// <param name="ExclusiveViolations">Which mutually exclusive pairs fired together, most frequent first, and which side the image actually carries.</param>
/// <param name="ContextReliance">Recall and false positives by how strongly each image's characters/artists/series predict the tag (<see cref="ContextRelianceEvaluator"/>).</param>
public sealed record EvaluationReport(
    int Images, int EvaluatedTags,
    double MeanAveragePrecision, double MeanHardAveragePrecision, double MeanBestF1, double MeanHeldOutF1,
    double MicroPrecision, double MicroRecall, double MicroF1, double RPrecision, double ExclusiveViolationRate,
    IReadOnlyList<TagEvaluation> Tags,
    double MeanContextAveragePrecision = double.NaN,
    ContextRelianceReport? ContextReliance = null,
    IReadOnlyList<ExclusiveViolation>? ExclusiveViolations = null)
{
    /// <summary>Lower-is-better form of <see cref="MeanAveragePrecision"/> for <see cref="EarlyStopping"/>.</summary>
    public double EarlyStoppingValue => 1 - MeanAveragePrecision;
}

/// <summary>
/// Measures what the tagger is actually for, instead of the mean sigmoid loss over every
/// (image, tag) pair — a number dominated ~600:1 by easy negatives, which can sit near its
/// floor while positives are ranked badly, the trap the first full training run's early
/// stopping fell into. Built on the same deliberate structure the corpus was crawled with:
/// <list type="bullet">
/// <item><b>Per tag</b>: rank the tag's held-out positives against its hard negatives
/// (held-out images with a commonly co-occurring tag but not this one — see
/// <see cref="TagStatistics.HardNegativeSources"/>) plus random negatives, and score AP and
/// best F1. Likely-missing images (carrying a tag that usually implies this one) are left
/// out of the negatives. Separately, each tag's threshold is calibrated as its best-F1
/// cutoff over every held-out image (real prevalence).</item>
/// <item><b>Per image</b>: with those thresholds applied, micro precision ("finds tags
/// that aren't there"), micro recall ("misses obvious ones"), R-precision, and how often
/// mutually exclusive tags fire together.</item>
/// </list>
/// </summary>
public static class TagRankingEvaluator
{
    /// <param name="pooled">[images × dim] held-out pooled image embeddings, row-major.</param>
    /// <param name="imageTags">Each held-out image's tags, same order as <paramref name="pooled"/>.</param>
    /// <param name="tagEmbeddings">[vocabulary × dim] tag embeddings, row-major.</param>
    /// <param name="trainingStatistics">Statistics from the TRAINING split — supplies implied tags and hard-negative sources, never held-out labels.</param>
    public static EvaluationReport Evaluate(
        float[] pooled,
        IReadOnlyList<IReadOnlyList<int>> imageTags,
        float[] tagEmbeddings,
        int dim,
        LogitScaling scaling,
        TagVocabulary vocabulary,
        TagStatistics trainingStatistics,
        ExclusiveTagGroups exclusiveGroups,
        Device device,
        int minPositives = 5,
        int hardNegativesPerPositive = 5,
        int randomNegativesPerPositive = 5,
        float fallbackThreshold = 0.5f,
        int minCalibrationPositives = 20,
        float minThreshold = 0.05f,
        int seed = 0,
        IReadOnlyList<BooruSites>? imageSites = null)
    {
        using var logits = ImageTagLogits.Linear(pooled, tagEmbeddings, dim, scaling);
        return Evaluate(logits, imageTags, vocabulary, trainingStatistics, exclusiveGroups, device, minPositives, hardNegativesPerPositive,
            randomNegativesPerPositive, fallbackThreshold, minCalibrationPositives, minThreshold, seed, imageSites);
    }

    /// <param name="logits">Every held-out image's logits for every tag (<see cref="ImageTagLogits"/>).</param>
    public static EvaluationReport Evaluate(
        ImageTagLogits logits,
        IReadOnlyList<IReadOnlyList<int>> imageTags,
        TagVocabulary vocabulary,
        TagStatistics trainingStatistics,
        ExclusiveTagGroups exclusiveGroups,
        Device device,
        int minPositives = 5,
        int hardNegativesPerPositive = 5,
        int randomNegativesPerPositive = 5,
        float fallbackThreshold = 0.5f,
        int minCalibrationPositives = 20,
        float minThreshold = 0.05f,
        int seed = 0,
        IReadOnlyList<BooruSites>? imageSites = null)
    {
        var imageCount = imageTags.Count;
        var vocabularySize = trainingStatistics.VocabularySize;

        var heldOut = new List<int>[vocabularySize];
        for (var t = 0; t < vocabularySize; t++)
            heldOut[t] = [];
        for (var image = 0; image < imageCount; image++)
            foreach (var t in imageTags[image])
                if ((uint)t < (uint)vocabularySize)
                    heldOut[t].Add(image);

        var tagSets = imageTags.Select(rows => rows.ToHashSet()).ToArray();

        // Hard and random negatives leave out the whole implied band (likely-missing AND
        // context), as before the band was split, so the early-stopping mAP stays comparable
        // across that change; context negatives are scored on their own below.
        // Site-blind images (no site they came from uses the tag) are no evidence either way.
        var tagSites = trainingStatistics.TagSites;
        bool IsSiteBlind(int image, int tag) =>
            imageSites is not null && tagSites is not null && (imageSites[image] & tagSites[tag]) == BooruSites.None;

        bool IsLikelyMissing(int image, int tag)
        {
            if (IsSiteBlind(image, tag))
                return true;
            foreach (var s in imageTags[image])
                if ((uint)s < (uint)vocabularySize
                    && (Array.BinarySearch(trainingStatistics.ImpliedBy[s], tag) >= 0 || Array.IndexOf(trainingStatistics.ContextSources[tag], s) >= 0))
                    return true;
            return false;
        }

        bool IsReallyLikelyMissing(int image, int tag)
        {
            if (IsSiteBlind(image, tag))
                return true;
            foreach (var s in imageTags[image])
                if ((uint)s < (uint)vocabularySize && Array.BinarySearch(trainingStatistics.ImpliedBy[s], tag) >= 0)
                    return true;
            return false;
        }

        var results = new TagEvaluation?[vocabularySize];
        Parallel.For(0, vocabularySize, t =>
        {
            var positives = heldOut[t];
            if (positives.Count < minPositives || imageCount <= positives.Count)
                return;

            var random = new Random(DeterministicSeed.For(seed, t));

            var hardPool = new HashSet<int>();
            foreach (var source in trainingStatistics.HardNegativeSources[t])
                foreach (var image in heldOut[source])
                    if (!tagSets[image].Contains(t) && !IsLikelyMissing(image, t))
                        hardPool.Add(image);
            var hard = hardPool.OrderBy(i => i).ToList();
            Shuffle(hard, random);
            if (hard.Count > hardNegativesPerPositive * positives.Count)
                hard.RemoveRange(hardNegativesPerPositive * positives.Count, hard.Count - hardNegativesPerPositive * positives.Count);

            var randomNegatives = new List<int>();
            var randomTarget = randomNegativesPerPositive * positives.Count;
            var hardSet = hard.ToHashSet();
            for (var attempt = 0; attempt < randomTarget * 4 && randomNegatives.Count < randomTarget; attempt++)
            {
                var image = random.Next(imageCount);
                if (!tagSets[image].Contains(t) && !hardSet.Contains(image) && !IsLikelyMissing(image, t))
                    randomNegatives.Add(image);
            }

            var contextPool = new HashSet<int>();
            foreach (var source in trainingStatistics.ContextSources[t])
                foreach (var image in heldOut[source])
                    if (!tagSets[image].Contains(t) && !IsReallyLikelyMissing(image, t))
                        contextPool.Add(image);
            var context = contextPool.OrderBy(i => i).ToList();
            Shuffle(context, random);
            if (context.Count > hardNegativesPerPositive * positives.Count)
                context.RemoveRange(hardNegativesPerPositive * positives.Count, context.Count - hardNegativesPerPositive * positives.Count);

            var positiveScores = Score(positives, t);
            var contextAp = context.Count > 0 ? RankingMetrics(positiveScores, Score(context, t)).AveragePrecision : double.NaN;
            var hardScores = Score(hard, t);
            var randomScores = Score(randomNegatives, t);

            var allNegatives = hardScores.Concat(randomScores).ToArray();
            var (ap, bestF1, threshold) = RankingMetrics(positiveScores, allNegatives);
            var hardAp = hardScores.Length > 0 ? RankingMetrics(positiveScores, hardScores).AveragePrecision : double.NaN;

            results[t] = new TagEvaluation(
                vocabulary.GetByRowIndex(t).Tag, positives.Count, hard.Count, randomNegatives.Count, ap, hardAp, bestF1, threshold, 0,
                context.Count, contextAp);
        });

        float[] Score(List<int> images, int tag)
        {
            var scores = new float[images.Count];
            for (var i = 0; i < images.Count; i++)
                scores[i] = logits.Probability(images[i], tag);
            return scores;
        }

        var (thresholds, heldOutF1) = CalibrateThresholds(logits, heldOut, minCalibrationPositives, minThreshold, fallbackThreshold, device, imageSites, tagSites);
        for (var t = 0; t < vocabularySize; t++)
            if (results[t] is { } result)
                results[t] = result with { Threshold = thresholds[t], HeldOutF1 = heldOutF1[t] };

        var evaluated = results.OfType<TagEvaluation>().ToList();

        var imageLevel = ImageLevelMetrics(logits, imageTags, thresholds, trainingStatistics, exclusiveGroups, device, imageSites);
        var contextReliance = ContextRelianceEvaluator.Evaluate(
            logits, imageTags, vocabulary, trainingStatistics, thresholds, device, seed: seed, imageSites: imageSites);

        var hardAps = evaluated.Select(e => e.HardAveragePrecision).Where(x => !double.IsNaN(x)).ToList();
        var contextAps = evaluated.Select(e => e.ContextAveragePrecision).Where(x => !double.IsNaN(x)).ToList();
        return new EvaluationReport(
            imageCount, evaluated.Count,
            evaluated.Count > 0 ? evaluated.Average(e => e.AveragePrecision) : 0,
            hardAps.Count > 0 ? hardAps.Average() : 0,
            evaluated.Count > 0 ? evaluated.Average(e => e.BestF1) : 0,
            evaluated.Any(e => !double.IsNaN(e.HeldOutF1)) ? evaluated.Where(e => !double.IsNaN(e.HeldOutF1)).Average(e => e.HeldOutF1) : 0,
            imageLevel.Precision, imageLevel.Recall, imageLevel.F1, imageLevel.RPrecision, imageLevel.ExclusiveViolationRate,
            evaluated.OrderBy(e => e.AveragePrecision).ToList(),
            contextAps.Count > 0 ? contextAps.Average() : double.NaN,
            contextReliance,
            imageLevel.Violations
                .Select(v => new ExclusiveViolation(vocabulary.Records[v.Key.First].Tag, vocabulary.Records[v.Key.Second].Tag, v.Value.Images, v.Value.FirstTrue, v.Value.SecondTrue))
                .OrderByDescending(v => v.Images).ToList());
    }

    /// <summary>Thresholds keyed by tag name for <see cref="TagThresholds"/>, for every tag that had enough held-out positives to calibrate — the rest fall back to inference's default.</summary>
    public static TagThresholds ToThresholds(EvaluationReport report) =>
        new(report.Tags.Where(t => !double.IsNaN(t.HeldOutF1)).ToDictionary(t => t.Tag, t => t.Threshold));

    /// <summary>AP, best F1, and the cutoff that achieves it, for positive vs. negative scores.</summary>
    public static (double AveragePrecision, double BestF1, float Threshold) RankingMetrics(float[] positiveScores, float[] negativeScores)
    {
        var ranked = positiveScores.Select(s => (Score: s, Positive: true))
            .Concat(negativeScores.Select(s => (Score: s, Positive: false)))
            .OrderByDescending(x => x.Score)
            .ThenBy(x => x.Positive) // pessimistic on ties: negatives rank first
            .ToArray();

        var totalPositives = positiveScores.Length;
        if (totalPositives == 0)
            return (0, 0, 1f);

        double truePositives = 0, apSum = 0, bestF1 = 0;
        var bestThreshold = ranked.Length > 0 ? ranked[0].Score : 1f;
        for (var i = 0; i < ranked.Length; i++)
        {
            if (ranked[i].Positive)
            {
                truePositives++;
                apSum += truePositives / (i + 1);
            }

            // A cutoff can only sit between distinct scores.
            if (i + 1 < ranked.Length && ranked[i + 1].Score == ranked[i].Score)
                continue;

            var precision = truePositives / (i + 1);
            var recall = truePositives / totalPositives;
            var f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0;
            if (f1 > bestF1)
            {
                bestF1 = f1;
                bestThreshold = i + 1 < ranked.Length ? (ranked[i].Score + ranked[i + 1].Score) / 2 : ranked[i].Score;
            }
        }

        return (apSum / totalPositives, bestF1, bestThreshold);
    }

    /// <summary>
    /// Per tag with at least <paramref name="minPositives"/> held-out positives: the cutoff
    /// maximizing F1 over every held-out image, and that F1. Runs on <paramref name="device"/>
    /// a block of tags at a time: score every image against the block, sort each column,
    /// and F1 at rank k is 2·TP(k) / (k + P). Other tags get <paramref name="fallbackThreshold"/>.
    /// Cutoffs are floored at <paramref name="minThreshold"/>: a best-F1 cutoff fitted to a
    /// handful of positives can land near zero, which on real images fires the tag on
    /// almost everything (seen on the epoch-12 model: rare tags calibrated from 5 positives
    /// reported at confidence 0.00-0.01).
    /// </summary>
    private static (float[] Thresholds, double[] F1) CalibrateThresholds(
        ImageTagLogits logits, List<int>[] heldOutPositives,
        int minPositives, float minThreshold, float fallbackThreshold, Device device,
        IReadOnlyList<BooruSites>? imageSites = null, IReadOnlyList<BooruSites>? tagSites = null)
    {
        var vocabularySize = heldOutPositives.Length;
        var imageCount = logits.ImageCount;
        var thresholds = new float[vocabularySize];
        var f1s = new double[vocabularySize];
        Array.Fill(thresholds, fallbackThreshold);
        Array.Fill(f1s, double.NaN);

        var calibrated = Enumerable.Range(0, vocabularySize).Where(t => heldOutPositives[t].Count >= minPositives).ToArray();
        if (calibrated.Length == 0 || imageCount == 0)
            return (thresholds, f1s);

        using var _ = no_grad();
        using var ranks = arange(1, imageCount + 1, dtype: ScalarType.Float32, device: device).unsqueeze(1);

        // Bounds the [images, block] score/sort/label tensors to a few hundred MB on a whole
        // validation split, not just a 20K-image sample.
        var block = Math.Max(1, 32_000_000 / imageCount);
        for (var start = 0; start < calibrated.Length; start += block)
        {
            using var scope = NewDisposeScope();
            var tags = calibrated.Skip(start).Take(block).ToArray();

            var labels = new bool[imageCount * tags.Length];
            for (var j = 0; j < tags.Length; j++)
                foreach (var image in heldOutPositives[tags[j]])
                    labels[image * tags.Length + j] = true;

            var probabilities = logits.Columns(tags, device).sigmoid();
            if (SiteBlindMask(imageSites, tagSites, tags, labels) is { } blind)
                probabilities = probabilities.masked_fill(blind.to(device), -1f); // sorts last: never a false positive
            var (sorted, order) = probabilities.sort(dim: 0, descending: true);
            var sortedLabels = tensor(labels, [imageCount, tags.Length]).to(device).gather(0, order).to(ScalarType.Float32);
            var truePositives = sortedLabels.cumsum(0);
            var positives = sortedLabels.sum(0, keepdim: true);
            var f1 = 2 * truePositives / (ranks + positives);
            var (bestF1, bestRank) = f1.max(0);
            var nextRank = (bestRank + 1).clamp_max(imageCount - 1);
            var cut = (sorted.gather(0, bestRank.unsqueeze(0)) + sorted.gather(0, nextRank.unsqueeze(0))) / 2;

            var cutValues = cut.squeeze(0).cpu().data<float>().ToArray();
            var f1Values = bestF1.cpu().data<float>().ToArray();
            for (var j = 0; j < tags.Length; j++)
            {
                thresholds[tags[j]] = Math.Max(cutValues[j], minThreshold);
                f1s[tags[j]] = f1Values[j];
            }
        }

        return (thresholds, f1s);
    }

    /// <summary>[images, tags] true where the image is site-blind for the tag and doesn't carry it; null if nothing is.</summary>
    private static Tensor? SiteBlindMask(IReadOnlyList<BooruSites>? imageSites, IReadOnlyList<BooruSites>? tagSites, int[] tags, bool[] labels)
    {
        if (imageSites is null || tagSites is null)
            return null;
        var mask = new bool[imageSites.Count * tags.Length];
        var any = false;
        for (var j = 0; j < tags.Length; j++)
        {
            var coverage = tagSites[tags[j]];
            if (coverage == (BooruSites.Danbooru | BooruSites.Gelbooru))
                continue;
            for (var image = 0; image < imageSites.Count; image++)
                if ((imageSites[image] & coverage) == BooruSites.None && !labels[image * tags.Length + j])
                    mask[image * tags.Length + j] = any = true;
        }

        return any ? tensor(mask, [imageSites.Count, tags.Length]) : null;
    }

    private static (double Precision, double Recall, double F1, double RPrecision, double ExclusiveViolationRate,
        Dictionary<(int First, int Second), (int Images, int FirstTrue, int SecondTrue)> Violations) ImageLevelMetrics(
        ImageTagLogits logits, IReadOnlyList<IReadOnlyList<int>> imageTags,
        float[] thresholds, TagStatistics trainingStatistics, ExclusiveTagGroups exclusiveGroups, Device device,
        IReadOnlyList<BooruSites>? imageSites = null)
    {
        var imageCount = imageTags.Count;
        var vocabularySize = thresholds.Length;
        if (imageCount == 0)
            return (0, 0, 0, 0, 0, []);

        var restricted = trainingStatistics.TagSites is { } coverage
            ? Enumerable.Range(0, vocabularySize).Where(t => coverage[t] != (BooruSites.Danbooru | BooruSites.Gelbooru)).ToArray()
            : [];

        // Stale vocabulary rows no training image carries are never scored as predictions.
        var trainable = Enumerable.Range(0, vocabularySize).Select(t => trainingStatistics.Positives[t].Length > 0).ToArray();

        using var _ = no_grad();
        using var thresholdTensor = tensor(thresholds).to(device);
        using var trainableTensor = tensor(trainable.Select(b => b ? 1f : 0f).ToArray()).to(device);

        var exclusiveColumns = exclusiveGroups.Rows.Where(t => t < vocabularySize).Order().ToArray();
        using var exclusiveColumnTensor = tensor(exclusiveColumns.Select(t => (long)t).ToArray()).to(device);

        long truePositives = 0, predicted = 0, actual = 0, violations = 0;
        var pairs = new Dictionary<(int First, int Second), (int Images, int FirstTrue, int SecondTrue)>();
        double rPrecisionSum = 0;
        const int chunk = 256;
        for (var start = 0; start < imageCount; start += chunk)
        {
            using var scope = NewDisposeScope();
            var count = Math.Min(chunk, imageCount - start);
            var probabilities = logits.Rows(start, count, device).sigmoid() * trainableTensor;
            var reported = probabilities >= thresholdTensor;
            if (imageSites is not null && trainingStatistics.TagSites is { } siteCoverage)
            {
                // A site-blind (image, tag) pair is neither a hit nor a false positive.
                var blindMask = new bool[count * vocabularySize];
                var any = false;
                foreach (var t in restricted)
                    for (var i = 0; i < count; i++)
                        if ((imageSites[start + i] & siteCoverage[t]) == BooruSites.None)
                            blindMask[i * vocabularySize + t] = any = true;
                if (any)
                    reported = reported.logical_and(tensor(blindMask, [count, vocabularySize]).to(device).logical_not());
            }

            var truthRows = new List<int>[count];
            var truthMask = new bool[count * vocabularySize];
            var maxTrue = 0;
            for (var i = 0; i < count; i++)
            {
                truthRows[i] = imageTags[start + i].Where(t => (uint)t < (uint)vocabularySize && trainable[t]).Distinct().ToList();
                foreach (var t in truthRows[i])
                    truthMask[i * vocabularySize + t] = true;
                actual += truthRows[i].Count;
                maxTrue = Math.Max(maxTrue, truthRows[i].Count);
            }

            var truth = tensor(truthMask, [count, vocabularySize]).to(device);
            truePositives += reported.logical_and(truth).sum().item<long>();
            predicted += reported.sum().item<long>();

            if (exclusiveColumns.Length > 0)
            {
                var exclusiveReported = reported.index_select(1, exclusiveColumnTensor).cpu().data<bool>().ToArray();
                for (var i = 0; i < count; i++)
                {
                    var fired = new List<int>();
                    for (var c = 0; c < exclusiveColumns.Length; c++)
                        if (exclusiveReported[i * exclusiveColumns.Length + c])
                            fired.Add(exclusiveColumns[c]);
                    var violated = false;
                    foreach (var a in fired)
                        foreach (var b in exclusiveGroups.ExcludedBy(a))
                            if (a < b && fired.Contains(b))
                            {
                                violated = true;
                                var (pairImages, firstTrue, secondTrue) = pairs.GetValueOrDefault((a, b));
                                pairs[(a, b)] = (pairImages + 1, firstTrue + (truthMask[i * vocabularySize + a] ? 1 : 0), secondTrue + (truthMask[i * vocabularySize + b] ? 1 : 0));
                            }
                    if (violated)
                        violations++;
                }
            }

            if (maxTrue > 0)
            {
                var topIndices = probabilities.topk(Math.Min(maxTrue, vocabularySize), dim: 1).indexes.cpu().data<long>().ToArray();
                for (var i = 0; i < count; i++)
                {
                    var truthSet = truthRows[i].ToHashSet();
                    if (truthSet.Count == 0)
                        continue;
                    var hits = 0;
                    for (var k = 0; k < truthSet.Count; k++)
                        if (truthSet.Contains((int)topIndices[i * maxTrue + k]))
                            hits++;
                    rPrecisionSum += hits / (double)truthSet.Count;
                }
            }
        }

        var precision = predicted > 0 ? truePositives / (double)predicted : 0;
        var recall = actual > 0 ? truePositives / (double)actual : 0;
        var f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0;
        return (precision, recall, f1, rPrecisionSum / imageCount, violations / (double)imageCount, pairs);
    }

    private static void Shuffle(List<int> list, Random random)
    {
        for (var i = list.Count - 1; i > 0; i--)
        {
            var j = random.Next(i + 1);
            (list[i], list[j]) = (list[j], list[i]);
        }
    }

    private static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        var sum = 0f;
        var i = 0;
        if (Vector.IsHardwareAccelerated)
        {
            var acc = Vector<float>.Zero;
            for (; i <= a.Length - Vector<float>.Count; i += Vector<float>.Count)
                acc += new Vector<float>(a[i..]) * new Vector<float>(b[i..]);
            sum = Vector.Sum(acc);
        }

        for (; i < a.Length; i++)
            sum += a[i] * b[i];
        return sum;
    }

    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));
}
