using System.Text.Json;
using Spectre.Console;
using UnbooruTagger.Core.Dataset;
using UnbooruTagger.Core.Encoding;
using UnbooruTagger.Core.Vocabulary;
using UnbooruTagger.Training.Commands;
using UnbooruTagger.Training.Model;
using static TorchSharp.torch;

namespace UnbooruTagger.Training.Training;

/// <summary>
/// How often the matched twins' "no" windows (<see cref="TwinWindows"/>) actually hold the object (<c>evaluate --twin-check</c>,
/// 2026-11-02): for each gold pair of a tag twin batches anchor on, the twins training would find for that image, the model's
/// own windows on all of them, and the "no" windows the rule picks on the gold image, scored against its gold boxes with the
/// ceiling test's rule (<see cref="WindowProbeCommandHandler.Label"/>). If "no" windows hold the object about as often as windows
/// in general, the rule labels the object "no" as often as anything else.
/// </summary>
public static class TwinCheck
{
    private sealed class Tally
    {
        public int Pairs, Windows, No, OnObject, OffObject, NoOnObject, NoOffObject, Suppressed, WithObjectWindow;
        public double Jaccard;

        public string Row(string name)
        {
            static string Rate(int n, int d) => d > 0 ? $"{(double)n / d:P0}" : "-";
            return $"{name,-22} pairs {Pairs,4}  Jaccard {Jaccard / Math.Max(1, Pairs):F2}  windows on object {Rate(OnObject, Windows),4}  " +
                   $"of 'no' windows on object {Rate(NoOnObject, No),4}  on-object windows 'no' {Rate(NoOnObject, OnObject),4}  " +
                   $"off-object windows 'no' {Rate(NoOffObject, OffObject),4}  every on-object window 'no' {Rate(Suppressed, WithObjectWindow),4}";
        }
    }

    public static void Run(
        ImageTower tower, float[] tagEmbeddings, int dim, TagStatistics statistics, IReadOnlyList<int> trainingIndices, IReadOnlyList<IReadOnlyList<int>> imageTagRows,
        IReadOnlyList<GoldImage> gold, TagVocabulary vocabulary, TagLocalizability localizability, PreprocessedDatasetCacheReader cache, Device device,
        string? dumpPath, int seed = 0)
    {
        var size = cache.InputSize;
        var rowOf = vocabulary.Records.ToDictionary(r => r.Tag, r => r.RowIndex);
        var eligible = vocabulary.Records.OrderBy(r => r.RowIndex)
            .Select(r => ContextRelianceEvaluator.IsGeneralTag(r.Tag) && localizability.IsLocalizable(r.Tag) && statistics.Positives[r.RowIndex].Length >= 50).ToArray();
        var sampler = new TagCentricBatchSampler(statistics, 1, 1, random: new Random(seed))
        {
            CharacterTags = vocabulary.Records.OrderBy(r => r.RowIndex).Select(r => r.Tag.StartsWith("character:", StringComparison.Ordinal)).ToArray(),
        };
        using var tags = tensor(tagEmbeddings, [tagEmbeddings.Length / dim, dim]).to(device);
        var localizableRows = vocabulary.Records.Where(r => localizability.IsLocalizable(r.Tag)).Select(r => (long)r.RowIndex).Order().ToArray();
        using var localizableTags = tags.index_select(0, tensor(localizableRows, device: device));

        var tallies = new Dictionary<string, Tally> { ["all"] = new() };
        var dump = new List<string>();
        var skipped = 0;
        tower.eval();
        foreach (var image in gold)
            foreach (var tag in image.Targets.Where(t => !image.NotVisible.Contains(t)).Distinct())
            {
                if (!rowOf.TryGetValue(tag, out var row) || !eligible[row])
                    continue;
                var boxes = image.Instances.Where(x => x.Tag == tag && x.Kind != GoldKind.Ignore)
                    .Select(x => new[] { x.Box[0] + image.Content.X, x.Box[1] + image.Content.Y, x.Box[2], x.Box[3] }).ToArray();
                var ignore = image.Instances.Where(x => x.Tag == tag && x.Kind == GoldKind.Ignore)
                    .Select(x => new[] { x.Box[0] + image.Content.X, x.Box[1] + image.Content.Y, x.Box[2], x.Box[3] }).ToArray();
                if (boxes.Length == 0)
                    continue;
                var found = sampler.FindTwins(imageTagRows[image.CacheIndex], row, 9)
                    .Select(t => (Image: trainingIndices[t.Image], t.Jaccard)).Where(t => t.Image != image.CacheIndex).ToArray();
                if (found.Length < TwinWindows.Matches)
                {
                    skipped++;
                    continue;
                }

                bool[] no;
                GlimpseWindow[] windows;
                using (no_grad())
                using (NewDisposeScope())
                {
                    var indices = found.Select(t => t.Image).Prepend(image.CacheIndex).ToArray();
                    var encoded = cache.ReadEncoded(indices);
                    var floats = 3 * size * size;
                    var pixelValues = new float[indices.Length * floats];
                    Parallel.For(0, indices.Length, i => ImagePreprocessing.ReconstructInto(encoded[i], size, pixelValues.AsSpan(i * floats, floats)));
                    var pixels = tensor(pixelValues, [indices.Length, 3, size, size]).to(device);
                    var content = encoded.Select(e => e.Content).ToArray();
                    var mapMask = SpatialMask.Build(content, size, size / tower.OutputStride, size / tower.OutputStride, device);
                    var (pooled, map) = tower.ForwardPooled(pixels, mapMask);
                    var contentBoxes = Glimpses.ContentBoxes(content, device);
                    using var choice = Glimpses.Propose(tower.Proposals(pooled), map, localizableTags, mapMask, contentBoxes, size, tower.OutputStride, tower.GlimpseCount);
                    var (cells, _) = Glimpses.Embed(tower, pixels, contentBoxes, choice);
                    var (peakX, peakY) = Glimpses.Peaks(map.narrow(0, 0, 1), tags[row].view(1, 1, -1), mapMask.narrow(0, 0, 1), tower.OutputStride);
                    var twins = tensor(Enumerable.Range(1, found.Length).Select(i => (long)i).ToArray(), device: device);
                    var mask = TwinWindows.NoWindows(cells, choice.Valid, 0, twins, ones(found.Length, ScalarType.Bool, device: device), choice, peakX, peakY);
                    windows = choice.ToWindows()[0];
                    no = mask.cpu().data<bool>().ToArray();
                }

                var labels = windows.Select(w => WindowProbeCommandHandler.Label(w, boxes, ignore)).ToArray();
                var size_ = LocalizationEvaluator.SizeBucket(tag, image.Instances.Where(x => x.Tag == tag).ToList());
                foreach (var name in new[] { "all", size_ })
                {
                    var t = tallies.TryGetValue(name, out var existing) ? existing : tallies[name] = new();
                    t.Pairs++;
                    t.Jaccard += found.Average(f => f.Jaccard);
                    t.Windows += windows.Length;
                    for (var k = 0; k < windows.Length; k++)
                    {
                        if (no[k]) t.No++;
                        if (labels[k] == 1f) { t.OnObject++; if (no[k]) t.NoOnObject++; }
                        if (labels[k] == 0f) { t.OffObject++; if (no[k]) t.NoOffObject++; }
                    }
                    if (labels.Any(l => l == 1f))
                    {
                        t.WithObjectWindow++;
                        if (Enumerable.Range(0, windows.Length).Where(k => labels[k] == 1f).All(k => no[k]))
                            t.Suppressed++;
                    }
                }
                dump.Add(JsonSerializer.Serialize(new
                {
                    image.CacheIndex, Tag = tag, Size = size_, Twins = found.Select(f => f.Image).ToArray(), Jaccard = found.Select(f => f.Jaccard).ToArray(),
                    Boxes = boxes, Windows = windows.Select(w => new[] { w.Zoom, w.X, w.Y, w.Side }).ToArray(), No = no.Take(windows.Length).ToArray(),
                }));
            }

        AnsiConsole.MarkupLineInterpolated($"Twin check: {tallies["all"].Pairs} gold pairs of twin-anchored tags ({skipped} without {TwinWindows.Matches} twins).");
        foreach (var (name, tally) in tallies.OrderBy(kv => kv.Key == "all" ? 0 : 1).ThenBy(kv => kv.Key))
            AnsiConsole.WriteLine(tally.Row(name.Replace("size: ", "")));
        if (dumpPath is not null)
            File.WriteAllLines(dumpPath, dump);
    }
}
