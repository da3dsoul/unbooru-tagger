using System.Globalization;
using UnbooruTagger.Core.Encoding;
using UnbooruTagger.Core.Vocabulary;
using UnbooruTagger.Training.Model;
using static TorchSharp.torch;

namespace UnbooruTagger.Training.Training;

/// <summary>
/// The locality negatives' starting table (<see cref="LocalityNegatives.LoadPrior"/>), measured on a trained localization map:
/// for every training image and every pair of its labelled localizable tags, whether the two tags' map peaks are within
/// <see cref="LocalityNegatives.Distance"/> of each other on both axes. <c>evaluate --locality-prior-out</c>; run 4's map,
/// which points at medium objects about 60% of the time, gives a prior that doesn't share a glimpse run's early bias.
/// </summary>
public static class LocalityPrior
{
    public static Dictionary<(int A, int B), (double Together, double Seen)> Measure(
        ImageTower tower, float[] tagEmbeddings, int dim, Func<IReadOnlyList<int>, EncodedImage[]> readImages, int inputSize,
        IReadOnlyList<IReadOnlyList<int>> imageTagRows, IReadOnlyList<int> images, bool[] localizable, Device device, float distance,
        Action<int>? progress = null)
    {
        var pairs = new Dictionary<(int, int), (double, double)>();
        using var deviceTags = tensor(tagEmbeddings, [tagEmbeddings.Length / dim, dim]).to(device);
        var done = 0;
        foreach (var chunk in images.Chunk(16))
        {
            var tags = chunk.Select(i => imageTagRows[i].Where(t => localizable[t]).Distinct().ToArray()).ToArray();
            var most = tags.Max(t => t.Length);
            done += chunk.Length;
            if (most < 2)
                continue;
            float[] x, y;
            using (no_grad())
            using (NewDisposeScope())
            {
                var encoded = readImages(chunk);
                var floats = 3 * inputSize * inputSize;
                var pixels = new float[chunk.Length * floats];
                var boxes = new LetterboxBox[chunk.Length];
                Parallel.For(0, chunk.Length, i =>
                {
                    boxes[i] = encoded[i].Content;
                    ImagePreprocessing.ReconstructInto(encoded[i], inputSize, pixels.AsSpan(i * floats, floats));
                });
                var input = tensor(pixels, [chunk.Length, 3, inputSize, inputSize]).to(device);
                var (_, localization) = tower.ForwardMaps(input);
                var mapMask = SpatialMask.Build(boxes, inputSize, localization.shape[2], localization.shape[3], device);
                var rows = tensor(tags.SelectMany(t => Enumerable.Range(0, most).Select(k => (long)(t.Length > 0 ? t[Math.Min(k, t.Length - 1)] : 0))).ToArray(), device: device);
                var embedded = deviceTags.index_select(0, rows).reshape(chunk.Length, most, dim);
                var (px, py) = Glimpses.Peaks(localization, embedded, mapMask, tower.OutputStride);
                x = px.to_type(ScalarType.Float32).cpu().data<float>().ToArray();
                y = py.to_type(ScalarType.Float32).cpu().data<float>().ToArray();
            }

            for (var i = 0; i < chunk.Length; i++)
                for (var a = 0; a < tags[i].Length; a++)
                    for (var b = a + 1; b < tags[i].Length; b++)
                    {
                        var together = Math.Abs(x[i * most + a] - x[i * most + b]) < distance && Math.Abs(y[i * most + a] - y[i * most + b]) < distance;
                        var key = (Math.Min(tags[i][a], tags[i][b]), Math.Max(tags[i][a], tags[i][b]));
                        var (t, n) = pairs.GetValueOrDefault(key);
                        pairs[key] = (t + (together ? 1 : 0), n + 1);
                    }
            progress?.Invoke(done);
        }
        return pairs;
    }

    /// <summary>Writes the pairs seen at least <paramref name="minSeen"/> times as "tagA \t tagB \t together \t seen".</summary>
    public static int Write(string path, Dictionary<(int A, int B), (double Together, double Seen)> pairs, TagVocabulary vocabulary, int minSeen)
    {
        var lines = pairs.Where(p => p.Value.Seen >= minSeen)
            .OrderByDescending(p => p.Value.Seen)
            .Select(p => string.Join('\t', vocabulary.GetByRowIndex(p.Key.A).Tag, vocabulary.GetByRowIndex(p.Key.B).Tag,
                p.Value.Together.ToString(CultureInfo.InvariantCulture), p.Value.Seen.ToString(CultureInfo.InvariantCulture)))
            .ToList();
        File.WriteAllLines(path, lines);
        return lines.Count;
    }
}
