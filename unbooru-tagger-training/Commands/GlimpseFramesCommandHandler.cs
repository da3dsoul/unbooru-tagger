using Spectre.Console;
using UnbooruTagger.Core.Dataset;
using UnbooruTagger.Core.Encoding;
using UnbooruTagger.Core.Vocabulary;
using UnbooruTagger.Training.Checkpoints;
using UnbooruTagger.Training.Model;
using UnbooruTagger.Training.Training;
using static TorchSharp.torch;

namespace UnbooruTagger.Training.Commands;

/// <summary>
/// How well the glimpse choice fits the gold set's hand-drawn boxes, on any checkpoint with a localization map (so it can be
/// judged on a well-trained map before a run depends on it). Per gold pair (a localizable tag with a certain box):
/// <list type="bullet">
/// <item>whether the proposals (<see cref="ImageTower.Proposals"/>; the whole-image embedding for a checkpoint without the
/// head) put the tag among an image's top 8/16/32/64;</item>
/// <item>whether the glimpse sensor at the tag's own map peak (<see cref="Glimpses.Sensor"/>) has a window holding at least
/// half of the box, and one where the box is also at least a tenth of the window (big enough to see);</item>
/// <item>for each window budget, whether <see cref="Glimpses.Propose"/> frames the tag (<see cref="Glimpses.Frames"/>) and
/// whether a chosen close look (at most 128px) holds at least half of the box.</item>
/// </list>
/// </summary>
public static class GlimpseFramesCommandHandler
{
    private static readonly int[] RankDepths = [8, 16, 32, 64];

    public static int Run(string checkpointDir, string cacheDir, string goldPath, IReadOnlyList<int> counts, int batchSize)
    {
        var device = DeviceSelector.Best();
        using var gradients = no_grad();
        var (tower, _, vocabulary, embeddings, _) = Checkpoint.Load(checkpointDir, device);
        tower.eval();
        using var cache = new PreprocessedDatasetCacheReader(cacheDir);
        var localizability = TagLocalizability.Load(Path.Combine(checkpointDir, TagLocalizability.FileName));
        var dim = tower.EmbeddingDim;
        var rowCount = embeddings.RowCount;
        using var allTags = tensor(Enumerable.Range(0, rowCount).SelectMany(i => embeddings.GetRow(i).ToArray()).ToArray(), [rowCount, dim]).to(device);
        var localizableRows = Enumerable.Range(0, vocabulary.Records.Count).Where(r => localizability.IsLocalizable(vocabulary.GetByRowIndex(r).Tag)).ToArray();
        var localizableIndex = localizableRows.Select((row, i) => (row, i)).ToDictionary(x => x.row, x => x.i);
        using var localizableTags = allTags.index_select(0, tensor(localizableRows.Select(r => (long)r).ToArray(), device: device));

        var gold = LocalizationEvaluator.LoadGold(goldPath);
        int pairs = 0, sensorHolds = 0, sensorSees = 0, closeHolds = 0;
        var rankedAt = new int[RankDepths.Length];
        var framedAt = new int[counts.Count];
        var heldAt = new int[counts.Count];
        foreach (var chunk in gold.Chunk(batchSize))
        {
            using var scope = NewDisposeScope();
            var encoded = cache.ReadEncoded(chunk.Select(g => g.CacheIndex).ToArray());
            var size = cache.InputSize;
            var pixels = new float[chunk.Length * 3 * size * size];
            for (var i = 0; i < chunk.Length; i++)
                ImagePreprocessing.ReconstructInto(encoded[i], size, pixels.AsSpan(i * 3 * size * size, 3 * size * size));
            var content = encoded.Select(e => e.Content).ToArray();
            var input = tensor(pixels, [chunk.Length, 3, size, size]).to(device);
            var mapMask = SpatialMask.Build(content, size, size / tower.OutputStride, size / tower.OutputStride, device);
            var (pooled, map) = tower.ForwardPooled(input, mapMask);
            var contentBoxes = Glimpses.ContentBoxes(content, device);

            // Each image's gold pairs: localizable tags with a certain box, and those boxes on the canvas.
            var imagePairs = chunk.Select((g, i) => g.Instances.Where(x => x.Kind == GoldKind.Certain && vocabulary.TryGet(x.Tag, out var r) && localizableIndex.ContainsKey(r.RowIndex))
                .GroupBy(x => x.Tag)
                .Select(t => (Row: vocabulary.TryGet(t.Key, out var r) ? r.RowIndex : -1,
                    Boxes: t.Select(x => new[] { x.Box[0] + content[i].X, x.Box[1] + content[i].Y, x.Box[2], x.Box[3] }).ToList()))
                .ToList()).ToArray();
            var most = imagePairs.Max(p => p.Count);
            if (most == 0)
                continue;
            var rows = tensor(imagePairs.SelectMany(p => p.Select(x => (long)x.Row).Concat(Enumerable.Repeat(0L, most - p.Count))).ToArray(), device: device);
            var (peakX, peakY) = Glimpses.Peaks(map, allTags.index_select(0, rows).reshape(chunk.Length, most, dim), mapMask, tower.OutputStride);
            var (px, py) = (peakX.cpu().data<long>().ToArray(), peakY.cpu().data<long>().ToArray());
            // The proposals (a checkpoint without a proposal head: the whole-image classifier's embedding).
            var proposals = tower.ProposalWeights is null ? pooled : tower.Proposals(pooled);
            var top = proposals.matmul(localizableTags.t()).topk(RankDepths[^1], 1).indexes.cpu().data<long>().ToArray();
            var chosenFor = counts.Select(c =>
            {
                using var choice = Glimpses.Propose(proposals, map, localizableTags, mapMask, contentBoxes, size, tower.OutputStride, c);
                return choice.ToWindows();
            }).ToArray();

            for (var i = 0; i < chunk.Length; i++)
                for (var j = 0; j < imagePairs[i].Count; j++)
                {
                    var (row, boxes) = imagePairs[i][j];
                    pairs++;
                    var position = Array.IndexOf(top, (long)localizableIndex[row], i * RankDepths[^1], RankDepths[^1]) - i * RankDepths[^1];
                    for (var d = 0; d < RankDepths.Length; d++)
                        if (position >= 0 && position < RankDepths[d])
                            rankedAt[d]++;

                    bool Holds(GlimpseWindow w) => boxes.Any(b => Intersection(b, w) >= 0.5 * b[2] * b[3]);
                    bool Sees(GlimpseWindow w) => boxes.Any(b => Intersection(b, w) >= 0.5 * b[2] * b[3] && Intersection(b, w) >= 0.1 * w.Side * w.Side);
                    var sensor = Glimpses.Sensor((int)px[i * most + j], (int)py[i * most + j], content[i], size);
                    if (sensor.Any(Holds))
                        sensorHolds++;
                    if (sensor.Any(Sees))
                        sensorSees++;
                    if (sensor.Take(Glimpses.Zooms.Length - 1).Any(Holds))
                        closeHolds++;
                    for (var c = 0; c < counts.Count; c++)
                    {
                        var chosen = chosenFor[c][i];
                        if (Glimpses.Frames(chosen, sensor))
                            framedAt[c]++;
                        if (chosen.Any(w => w.Side <= 128 && Holds(w)))
                            heldAt[c]++;
                    }
                }
        }

        double Share(int n) => n / (double)pairs;
        Console.WriteLine($"{pairs} gold pairs (localizable tags with a certain box)");
        Console.WriteLine("ranked by the proposals among the top " + string.Join(", ", RankDepths.Select((d, k) => $"{d}: {Share(rankedAt[k]):P1}")));
        Console.WriteLine($"sensor at the tag's own peak: a window holds >=50% of the box {Share(sensorHolds):P1} (a close one, <=128px: {Share(closeHolds):P1}); " +
                          $"holds it with the box >=10% of the window {Share(sensorSees):P1}");
        for (var c = 0; c < counts.Count; c++)
            Console.WriteLine($"Propose, {counts[c]} windows: framed {Share(framedAt[c]):P1}; a chosen close look holds >=50% of the box {Share(heldAt[c]):P1}");
        return 0;
    }

    private static int Intersection(int[] box, GlimpseWindow window)
    {
        var w = Math.Min(box[0] + box[2], window.X + window.Side) - Math.Max(box[0], window.X);
        var h = Math.Min(box[1] + box[3], window.Y + window.Side) - Math.Max(box[1], window.Y);
        return w <= 0 || h <= 0 ? 0 : w * h;
    }
}
