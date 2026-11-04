using System.Text.Json;
using Spectre.Console;
using UnbooruTagger.Core.Dataset;
using UnbooruTagger.Core.Encoding;
using UnbooruTagger.Core.Scoring;
using UnbooruTagger.Core.Vocabulary;
using UnbooruTagger.Training.Model;
using static TorchSharp.torch;

namespace UnbooruTagger.Training.Training;

/// <summary>
/// The whole-image score and the peak windows' glimpse scores, side by side (<c>evaluate --peak-window-scores</c>, 2026-11-03;
/// the no-training check before plan #2 of review/open-vocab-research/PLAN.md, SAM 3's presence x window score). For every
/// held-out image and each <c>--dump-scores</c> tag, writes one JSON line: the label (1 = tagged), the labelled context set's
/// label where it has one, and the raw dot products (before the logit scale and bias, given on the first
/// line) of the pooled embedding and of the 64px and 128px windows centred on the tag's map peak, chosen as
/// <see cref="SelfMapWindows.Windows"/> chooses them in training.
/// </summary>
public static class PeakWindowScores
{
    public static void Run(ImageTower tower, float[] tagEmbeddings, int dim, IReadOnlyList<int> heldOut, IReadOnlyList<IReadOnlyList<int>> imageTagRows,
        IReadOnlyList<string> tags, TagVocabulary vocabulary, IReadOnlyList<LabelledContextPair>? contextCheck, LogitScaling scaling,
        PreprocessedDatasetCacheReader cache, Device device, int batchSize, string path)
    {
        var size = cache.InputSize;
        var rows = tags.Select(t => vocabulary.TryGet(t, out var r) ? r.RowIndex : throw new ArgumentException($"--dump-scores: '{t}' isn't in the vocabulary.")).ToArray();
        var verdicts = (contextCheck ?? []).GroupBy(p => (p.CacheIndex, p.Tag)).ToDictionary(g => g.Key, g => g.First().Label);
        var tagValues = rows.SelectMany(r => tagEmbeddings.AsSpan(r * dim, dim).ToArray()).ToArray();
        using var writer = new StreamWriter(path);
        writer.WriteLine(JsonSerializer.Serialize(new { scale = scaling.Scale, bias = scaling.Bias, tags }));
        tower.eval();
        var done = 0;
        foreach (var batch in heldOut.Chunk(batchSize))
        {
            float[] pooledValues, cellValues;
            bool[] valid;
            using (no_grad())
            using (NewDisposeScope())
            {
                var encoded = cache.ReadEncoded(batch);
                var floats = 3 * size * size;
                var pixelValues = new float[batch.Length * floats];
                Parallel.For(0, batch.Length, i => ImagePreprocessing.ReconstructInto(encoded[i], size, pixelValues.AsSpan(i * floats, floats)));
                var pixels = tensor(pixelValues, [batch.Length, 3, size, size]).to(device);
                var contents = encoded.Select(e => e.Content).ToArray();
                var mapMask = SpatialMask.Build(contents, size, size / tower.OutputStride, size / tower.OutputStride, device);
                var (pooled, map) = tower.ForwardPooled(pixels, mapMask);
                var tagTensor = tensor(tagValues, [1, rows.Length, dim]).to(device).expand(batch.Length, -1, -1);
                var contentBoxes = Glimpses.ContentBoxes(contents, device);
                var (choice, _) = SelfMapWindows.Windows(map, mapMask, tagTensor, ones([batch.Length, rows.Length], ScalarType.Bool, device: device),
                    contentBoxes, size, tower.OutputStride);
                var (cells, _) = Glimpses.Embed(tower, pixels, contentBoxes, choice, requireZoomCodes: false);
                pooledValues = pooled.cpu().data<float>().ToArray(); // [B, D]
                cellValues = cells.transpose(1, 2).contiguous().cpu().data<float>().ToArray(); // [B, P·PerPositive, D]
                valid = choice.Valid.cpu().data<bool>().ToArray();
            }
            var perImage = rows.Length * SelfMapWindows.PerPositive;
            for (var b = 0; b < batch.Length; b++)
            {
                var labelled = imageTagRows[batch[b]].ToHashSet();
                for (var p = 0; p < rows.Length; p++)
                {
                    double Dot(float[] values, long offset)
                    {
                        double v = 0;
                        for (var d = 0; d < dim; d++)
                            v += tagValues[p * dim + d] * values[offset + d];
                        return v;
                    }
                    double? Window(int k)
                    {
                        var index = b * perImage + p * SelfMapWindows.PerPositive + k;
                        return valid[index] ? Dot(cellValues, (long)index * dim) : null;
                    }
                    writer.WriteLine(JsonSerializer.Serialize(new
                    {
                        cacheIndex = batch[b], tag = tags[p], label = labelled.Contains(rows[p]) ? 1 : 0,
                        context = verdicts.GetValueOrDefault((batch[b], tags[p])),
                        whole = Dot(pooledValues, (long)b * dim), w64 = Window(0), w128 = Window(1),
                    }));
                }
            }
            done += batch.Length;
            if (done % (batchSize * 50) < batchSize)
                AnsiConsole.MarkupLineInterpolated($"[grey]peak window scores: {done:N0} of {heldOut.Count:N0} images[/]");
        }
    }
}
