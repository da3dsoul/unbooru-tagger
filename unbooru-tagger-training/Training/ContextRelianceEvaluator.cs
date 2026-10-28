using UnbooruTagger.Core.Dataset;
using UnbooruTagger.Core.Scoring;
using UnbooruTagger.Core.Vocabulary;
using static TorchSharp.torch;

namespace UnbooruTagger.Training.Training;

/// <summary>Whole-image calls on held-out images whose context gives a tag one prior band (<see cref="ContextRelianceEvaluator"/>).</summary>
/// <param name="MinPrior">The band's lower edge: images whose characters/artists/series carry the tag at least this often in training.</param>
/// <param name="Recall">Of positives in the band, the fraction reported at the tag's calibrated threshold.</param>
/// <param name="FalsePositiveRate">Of negatives in the band, the fraction reported. In the top band (≥ 90%) many are missing labels, not real absences.</param>
public sealed record ContextBand(string Name, double MinPrior, int Positives, double Recall, int Negatives, double FalsePositiveRate);

public sealed record ContextTagResult(string Tag, double RecallGap, double FalsePositiveGap, IReadOnlyList<ContextBand> Bands);

/// <param name="Bands">Every evaluated general tag pooled, per prior band.</param>
/// <param name="MacroRecallGap">
/// Mean over tags of recall where context predicts the tag (prior ≥ 60%) minus recall where it
/// doesn't (&lt; 5%). Near 0 = the tag is found the same either way; large = it's found mostly
/// on the characters/artists that usually have it.
/// </param>
/// <param name="MacroFalsePositiveGap">
/// Mean over tags of the false-positive rate where context usually (60-90%) has the tag minus where
/// it doesn't (&lt; 5%). Large = "this character, therefore gloves" on images without them.
/// </param>
/// <param name="Tags">Per tag with enough images on both sides of a gap, largest false-positive gap first.</param>
public sealed record ContextRelianceReport(
    IReadOnlyList<ContextBand> Bands,
    double MacroRecallGap, int RecallGapTags,
    double MacroFalsePositiveGap, int FalsePositiveGapTags,
    IReadOnlyList<ContextTagResult> Tags);

/// <summary>
/// Does the classifier find a tag from the object, or from who drew it and who's in it? For every
/// held-out image and general tag t, the image's <b>context prior</b> is the highest P(t | s) over
/// its character, artist and series tags s, estimated from the training split. A classifier that
/// sees the object has the same recall and false-positive rate at every prior; one that learned
/// "this character wears gloves" misses gloves on characters who rarely wear them and reports them
/// on the usual characters' images without them.
/// <para>
/// This replaces <see cref="TagEvaluation.ContextAveragePrecision"/> as the thing to watch: that
/// only counts context sources in the 60-90% band and only their held-out images, which on a 20K
/// sample leaves a median of 2 negatives per tag, where AP's chance level is already ~0.98.
/// </para>
/// </summary>
public static class ContextRelianceEvaluator
{
    public static readonly (string Name, double MinPrior)[] BandEdges =
        [("prior <5%", 0), ("prior 5-30%", 0.05), ("prior 30-60%", 0.3), ("prior 60-90%", 0.6), ("prior >=90%", 0.9)];

    private const double HighPrior = 0.6;
    private const double LowPrior = 0.05;
    private const double MissingPrior = 0.9;

    /// <summary>A context source: a tag naming who is in the image or who made it.</summary>
    public static bool IsContextSource(string tag) =>
        tag.StartsWith("character:", StringComparison.Ordinal) || tag.StartsWith("artist:", StringComparison.Ordinal) || tag.StartsWith("series:", StringComparison.Ordinal);

    /// <summary>A tag measured here: anything without a category prefix (objects, clothing, features, ...).</summary>
    public static bool IsGeneralTag(string tag) => !tag.Contains(':');

    /// <param name="pooled">[images × dim] held-out pooled embeddings.</param>
    /// <param name="thresholds">Per-tag calibrated thresholds (the operating point inference uses).</param>
    /// <param name="minPerSide">Images needed on both sides of a gap for a tag to count toward it.</param>
    /// <param name="samplesPerSource">Training images sampled per context source to estimate P(t | s).</param>
    public static ContextRelianceReport Evaluate(
        float[] pooled, IReadOnlyList<IReadOnlyList<int>> imageTags, float[] tagEmbeddings, int dim, LogitScaling scaling,
        TagVocabulary vocabulary, TagStatistics trainingStatistics, float[] thresholds, Device device,
        int minPerSide = 10, int samplesPerSource = 512, int seed = 0, IReadOnlyList<BooruSites>? imageSites = null)
    {
        using var logits = ImageTagLogits.Linear(pooled, tagEmbeddings, dim, scaling);
        return Evaluate(logits, imageTags, vocabulary, trainingStatistics, thresholds, device, minPerSide, samplesPerSource, seed, imageSites);
    }

    /// <param name="logits">Every held-out image's logits for every tag (<see cref="ImageTagLogits"/>).</param>
    public static ContextRelianceReport Evaluate(
        ImageTagLogits logits, IReadOnlyList<IReadOnlyList<int>> imageTags,
        TagVocabulary vocabulary, TagStatistics trainingStatistics, float[] thresholds, Device device,
        int minPerSide = 10, int samplesPerSource = 512, int seed = 0, IReadOnlyList<BooruSites>? imageSites = null)
    {
        var imageCount = imageTags.Count;
        var vocabularySize = trainingStatistics.VocabularySize;
        var isSource = new bool[vocabularySize];
        var measured = new List<int>();
        foreach (var record in vocabulary.Records)
        {
            if (record.RowIndex >= vocabularySize)
                continue;
            isSource[record.RowIndex] = IsContextSource(record.Tag);
            if (IsGeneralTag(record.Tag) && trainingStatistics.Positives[record.RowIndex].Length > 0)
                measured.Add(record.RowIndex);
        }

        var priors = ImagePriors(imageTags, isSource, trainingStatistics, samplesPerSource, seed);
        var tagSets = imageTags.Select(rows => rows.ToHashSet()).ToArray();
        var tagSites = trainingStatistics.TagSites;

        // [tag][band] counts: positives, positives reported, negatives, negatives reported.
        var counts = new int[vocabularySize][];
        using (no_grad())
        {
            var block = Math.Max(1, 32_000_000 / Math.Max(1, imageCount));
            for (var start = 0; start < measured.Count; start += block)
            {
                using var scope = NewDisposeScope();
                var tags = measured.Skip(start).Take(block).ToArray();
                var cut = new float[tags.Length];
                for (var j = 0; j < tags.Length; j++)
                    cut[j] = thresholds[tags[j]];

                var probabilities = logits.Columns(tags, device).sigmoid();
                var reported = (probabilities >= tensor(cut).to(device)).cpu().data<bool>().ToArray();
                Parallel.For(0, tags.Length, j =>
                {
                    var t = tags[j];
                    var c = counts[t] = new int[BandEdges.Length * 4];
                    for (var image = 0; image < imageCount; image++)
                    {
                        if (imageSites is not null && tagSites is not null && (imageSites[image] & tagSites[t]) == BooruSites.None)
                            continue;
                        var band = Band(priors[image].GetValueOrDefault(t));
                        var hit = reported[image * tags.Length + j] ? 1 : 0;
                        if (tagSets[image].Contains(t))
                        {
                            c[band * 4]++;
                            c[band * 4 + 1] += hit;
                        }
                        else
                        {
                            c[band * 4 + 2]++;
                            c[band * 4 + 3] += hit;
                        }
                    }
                });
            }
        }

        return Summarize(measured.Select(t => (vocabulary.GetByRowIndex(t).Tag, counts[t])).ToList(), minPerSide);
    }

    /// <summary>
    /// The report from per-tag band counts (positives, positives reported, negatives, negatives
    /// reported, for each band of <see cref="BandEdges"/> in turn).
    /// </summary>
    public static ContextRelianceReport Summarize(IReadOnlyList<(string Tag, int[] Counts)> perTag, int minPerSide)
    {
        var pooledCounts = new int[BandEdges.Length * 4];
        var results = new List<ContextTagResult>();
        var recallGaps = new List<double>();
        var falsePositiveGaps = new List<double>();
        foreach (var (tag, c) in perTag)
        {
            for (var k = 0; k < c.Length; k++)
                pooledCounts[k] += c[k];

            var low = Side(c, b => b == 0);
            var highPositives = Side(c, b => BandEdges[b].MinPrior >= HighPrior);
            var usualNegatives = Side(c, b => BandEdges[b].MinPrior >= HighPrior && BandEdges[b].MinPrior < MissingPrior);
            var recallGap = low.Positives >= minPerSide && highPositives.Positives >= minPerSide
                ? Rate(highPositives.PositiveHits, highPositives.Positives) - Rate(low.PositiveHits, low.Positives) : double.NaN;
            var falsePositiveGap = low.Negatives >= minPerSide && usualNegatives.Negatives >= minPerSide
                ? Rate(usualNegatives.NegativeHits, usualNegatives.Negatives) - Rate(low.NegativeHits, low.Negatives) : double.NaN;
            if (!double.IsNaN(recallGap))
                recallGaps.Add(recallGap);
            if (!double.IsNaN(falsePositiveGap))
                falsePositiveGaps.Add(falsePositiveGap);
            if (!double.IsNaN(recallGap) || !double.IsNaN(falsePositiveGap))
                results.Add(new ContextTagResult(tag, recallGap, falsePositiveGap, Bands(c)));
        }

        return new ContextRelianceReport(
            Bands(pooledCounts),
            recallGaps.Count > 0 ? recallGaps.Average() : double.NaN, recallGaps.Count,
            falsePositiveGaps.Count > 0 ? falsePositiveGaps.Average() : double.NaN, falsePositiveGaps.Count,
            results.OrderByDescending(r => double.IsNaN(r.FalsePositiveGap) ? double.MinValue : r.FalsePositiveGap).ToList());
    }

    public static int Band(double prior)
    {
        for (var b = BandEdges.Length - 1; b > 0; b--)
            if (prior >= BandEdges[b].MinPrior)
                return b;
        return 0;
    }

    /// <summary>
    /// Per held-out image: tag → its context prior (the highest P(tag | s) over the image's
    /// context sources s), for priors ≥ 5%; absent means below that.
    /// </summary>
    private static Dictionary<int, float>[] ImagePriors(
        IReadOnlyList<IReadOnlyList<int>> imageTags, bool[] isSource, TagStatistics trainingStatistics, int samplesPerSource, int seed)
    {
        var sources = imageTags.SelectMany(rows => rows).Where(s => (uint)s < (uint)isSource.Length && isSource[s]).Distinct().ToArray();
        var conditional = new Dictionary<int, Dictionary<int, float>>();
        var lockObject = new object();
        Parallel.ForEach(sources, s =>
        {
            var images = trainingStatistics.Positives[s];
            if (images.Length == 0)
                return;
            var random = new Random(DeterministicSeed.For(seed, s));
            var sample = images.Length <= samplesPerSource ? images : images.OrderBy(_ => random.Next()).Take(samplesPerSource).ToArray();
            var tally = new Dictionary<int, int>();
            foreach (var image in sample)
                foreach (var t in trainingStatistics.ImageTagRows[image])
                    if (t != s)
                        tally[t] = tally.GetValueOrDefault(t) + 1;
            var probabilities = tally.Where(kv => kv.Value >= LowPrior * sample.Length)
                .ToDictionary(kv => kv.Key, kv => kv.Value / (float)sample.Length);
            lock (lockObject)
                conditional[s] = probabilities;
        });

        var priors = new Dictionary<int, float>[imageTags.Count];
        Parallel.For(0, imageTags.Count, i =>
        {
            var prior = new Dictionary<int, float>();
            foreach (var s in imageTags[i])
                if (conditional.TryGetValue(s, out var probabilities))
                    foreach (var (t, p) in probabilities)
                        if (p > prior.GetValueOrDefault(t))
                            prior[t] = p;
            priors[i] = prior;
        });
        return priors;
    }

    private static (int Positives, int PositiveHits, int Negatives, int NegativeHits) Side(int[] c, Func<int, bool> inSide)
    {
        int positives = 0, positiveHits = 0, negatives = 0, negativeHits = 0;
        for (var b = 0; b < BandEdges.Length; b++)
            if (inSide(b))
            {
                positives += c[b * 4];
                positiveHits += c[b * 4 + 1];
                negatives += c[b * 4 + 2];
                negativeHits += c[b * 4 + 3];
            }
        return (positives, positiveHits, negatives, negativeHits);
    }

    private static List<ContextBand> Bands(int[] c) =>
        BandEdges.Select((edge, b) => new ContextBand(edge.Name, edge.MinPrior,
            c[b * 4], Rate(c[b * 4 + 1], c[b * 4]), c[b * 4 + 2], Rate(c[b * 4 + 3], c[b * 4 + 2]))).ToList();

    private static double Rate(int n, int d) => d > 0 ? (double)n / d : double.NaN;
}
