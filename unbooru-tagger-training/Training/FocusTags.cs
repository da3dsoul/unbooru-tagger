using System.Text.Json;
using System.Text.Json.Serialization;
using UnbooruTagger.Core.Vocabulary;

namespace UnbooruTagger.Training.Training;

/// <summary>
/// Tags the classifier finds from context instead of the object, picked from the latest epoch
/// evaluation alone, so a tag added later is picked (or dropped) the same way. Training draws a
/// share of its anchors from them (<see cref="TagCentricBatchSampler.Focus"/>) with as many
/// context negatives as positives, and puts each such anchor's images on the same mosaic
/// (<see cref="Mosaic.PickTiles"/>): gloves next to the character who usually wears them, without.
/// With anchors drawn uniformly over ~25K tags, gloves was an anchor about once every 12K steps.
/// <para>
/// A localizable general tag is picked when either check says so:
/// <list type="bullet">
/// <item>Context reliance (held-out, <see cref="ContextRelianceReport.Tags"/>): a false-positive
/// gap (prior 60-90% vs. &lt; 5%) of at least <see cref="MinGap"/> over at least
/// <see cref="MinBandImages"/> negatives, or the same recall gap over that many positives.</item>
/// <item>Occlusion (gold, <see cref="LocalizationReport.OcclusionTags"/>): still called with the
/// object blanked at least <see cref="MaxStillCalled"/> of the time and an object drop under
/// <see cref="MinObjectDrop"/>, over at least <see cref="MinOcclusionPairs"/> pairs.</item>
/// </list>
/// </para>
/// </summary>
public sealed class FocusTags
{
    public const double MinGap = 0.2;
    public const int MinBandImages = 20;
    public const double MaxStillCalled = 0.9;
    public const double MinObjectDrop = 0.03;
    public const int MinOcclusionPairs = 10;

    /// <summary>Picked tag rows, ascending.</summary>
    public int[] Rows { get; }

    private FocusTags(int[] rows) => Rows = rows;

    public static FocusTags None { get; } = new([]);

    /// <param name="eligible">Per tag row: may be picked (training passes localizable general tags).</param>
    public static FocusTags From(EvaluationReport? report, LocalizationReport? localization, TagVocabulary vocabulary, IReadOnlyList<bool> eligible)
    {
        var picked = new SortedSet<int>();
        void Add(string tag)
        {
            if (vocabulary.TryGet(tag, out var record) && record.RowIndex < eligible.Count && eligible[record.RowIndex])
                picked.Add(record.RowIndex);
        }

        foreach (var tag in report?.ContextReliance?.Tags ?? [])
        {
            var top = tag.Bands.Where(b => b.MinPrior >= 0.6).ToList();
            var usual = top.Where(b => b.MinPrior < 0.9).Sum(b => b.Negatives);
            var positives = top.Sum(b => b.Positives);
            if ((tag.FalsePositiveGap >= MinGap && usual >= MinBandImages) || (tag.RecallGap >= MinGap && positives >= MinBandImages))
                Add(tag.Tag);
        }

        foreach (var tag in localization?.OcclusionTags ?? [])
            if (tag.Pairs >= MinOcclusionPairs && tag.StillYesRate >= MaxStillCalled && tag.ObjectDrop < MinObjectDrop)
                Add(tag.Name);

        return new FocusTags(picked.ToArray());
    }

    /// <summary>From the evaluation and localization reports in <paramref name="directory"/>; none if neither is there yet.</summary>
    public static FocusTags Load(string directory, string reportFileName, string localizationFileName, TagVocabulary vocabulary, IReadOnlyList<bool> eligible) =>
        From(Read<EvaluationReport>(Path.Combine(directory, reportFileName)), Read<LocalizationReport>(Path.Combine(directory, localizationFileName)), vocabulary, eligible);

    private static T? Read<T>(string path) where T : class =>
        File.Exists(path)
            ? JsonSerializer.Deserialize<T>(File.ReadAllText(path), new JsonSerializerOptions { NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals })
            : null;
}
