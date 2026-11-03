using System.Text.Json;
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
/// The glimpse sensor's ceiling (2026-11-02): can a glimpse embedding tell a window holding an object from one beside it at all,
/// if it is asked directly? Every loss so far left a window the object fills near chance (64px within-pair AUC ~0.6), which
/// reads either as "the loss never asked" or as "a 64px glimpse through this tower can't show it". Here the glimpse path is
/// fine-tuned on the gold boxes themselves, k-fold by image: random windows at every zoom, a tag's window a "yes" when it holds
/// at least half of one of the tag's boxes or the box fills half of it, a "no" when it touches none of them (ignore boxes
/// included), each tag's row and the logit scale frozen, a balanced BCE. Each held-out image's windows are those the checkpoint
/// itself chose in an earlier <c>evaluate --dump-occlusion</c> (<c>--windows</c>), re-embedded by the fine-tuned path and written
/// in that dump's format (glimpse_pairs*.jsonl, glimpse_embeddings*.jsonl), so review/pooling/compare.py and
/// review/glimpse-dump/probe_pooled.py read them unchanged. <c>--steps 0</c> is the same pipeline untrained: the baseline.
/// Gold labels are read as written (no tag-family expansion), so pair counts differ a little from evaluate's.
/// </summary>
public static class WindowProbeCommandHandler
{
    public static int Run(
        string checkpointDir, string cacheDir, IReadOnlyList<string> goldPaths, IReadOnlyList<string> windowDirs, string outDir,
        int folds, int steps, int imagesPerStep, int windowsPerImage, float learningRate, bool bf16, int seed)
    {
        var device = DeviceSelector.Best();
        using var cache = new PreprocessedDatasetCacheReader(cacheDir);
        var size = cache.InputSize;
        Directory.CreateDirectory(outDir);
        File.Copy(Path.Combine(checkpointDir, "model_config.json"), Path.Combine(outDir, "model_config.json"), overwrite: true);

        // Each gold file's images, and the windows the checkpoint chose for them (the dump beside each file's glimpse pairs).
        var sets = goldPaths.Select((path, i) =>
        {
            var suffix = i == 0 ? "" : "_small";
            var windows = new Dictionary<int, GlimpseWindow[]>();
            foreach (var line in File.ReadLines(Path.Combine(windowDirs[i], $"glimpse_embeddings{suffix}.jsonl")))
            {
                var row = JsonSerializer.Deserialize<GlimpseEmbeddings>(line)!;
                windows[row.CacheIndex] = row.Windows.Select(w => new GlimpseWindow(w[0], w[1], w[2], w[3])).ToArray();
            }
            return (Suffix: suffix, Gold: LocalizationEvaluator.LoadGold(path), Windows: windows);
        }).ToList();

        var (probeTower, _, vocabulary, _, _) = Checkpoint.Load(checkpointDir, device);
        var localizability = TagLocalizability.Load(Path.Combine(checkpointDir, TagLocalizability.FileName));
        probeTower.Dispose();
        int? RowOf(string tag) => vocabulary.TryGet(tag, out var record) && localizability.IsLocalizable(tag) ? record.RowIndex : null;

        // Every gold image once (an image in both sets trains as one), with its labelled localizable tags.
        var images = sets.SelectMany(s => s.Gold).GroupBy(g => g.CacheIndex).Select(g => g.First())
            .Select(g => (Gold: g, Tags: Labelled(g, RowOf).ToArray()))
            .Where(x => x.Tags.Length > 0).ToArray();
        var random = new Random(seed);
        var order = images.Select(x => x.Gold.CacheIndex).OrderBy(_ => random.Next()).ToArray();
        var foldOf = order.Select((index, i) => (index, i)).ToDictionary(x => x.index, x => x.i % folds);
        AnsiConsole.MarkupLineInterpolated($"{images.Length} gold images with labelled localizable tags, {folds} folds, {steps} steps of {imagesPerStep} images x {windowsPerImage} windows per fold.");

        var pairs = sets.Select(_ => new List<string>()).ToList();
        var embeddingRows = sets.Select(_ => new List<string>()).ToList();
        for (var fold = 0; fold < folds; fold++)
        {
            var (tower, _, _, embeddings, scaleBias) = Checkpoint.Load(checkpointDir, device);
            if (bf16)
                tower.ComputeType = ScalarType.BFloat16;
            tower.ChannelsLast = device.type != TorchSharp.DeviceType.CPU;
            var dim = tower.EmbeddingDim;
            using var tags = tensor(Enumerable.Range(0, embeddings.RowCount).SelectMany(i => embeddings.GetRow(i).ToArray()).ToArray(), [embeddings.RowCount, dim]).to(device);
            var training = images.Where(x => foldOf[x.Gold.CacheIndex] != fold).ToArray();

            if (steps > 0)
            {
                var glimpsePath = tower.GlimpseTower ?? tower;
                var parameters = glimpsePath.parameters().Append(tower.GlimpseZoomCodes!).ToArray();
                var optimizer = optim.Adam(parameters, lr: learningRate);
                tower.train();
                var running = new List<double>();
                for (var step = 1; step <= steps; step++)
                {
                    using var scope = NewDisposeScope();
                    var batch = Enumerable.Range(0, imagesPerStep).Select(_ => training[random.Next(training.Length)]).ToArray();
                    var (pixels, content) = Load(cache, batch.Select(x => x.Gold.CacheIndex).ToArray(), size, device);
                    // Augmentation: a horizontal flip and a brightness/contrast jitter per image.
                    var flips = batch.Select(_ => random.Next(2) == 1).ToArray();
                    pixels = Augment(pixels, flips, random, device);
                    content = content.Select((c, i) => flips[i] ? c with { X = size - c.X - c.Width } : c).ToArray();
                    var boxes = batch.Select((x, i) => x.Tags.Select(t => (t.Row, Boxes: t.Boxes.Select(b => Flip(Canvas(b, x.Gold.Content), flips[i], size)).ToArray(),
                        Ignore: t.Ignore.Select(b => Flip(Canvas(b, x.Gold.Content), flips[i], size)).ToArray())).ToArray()).ToArray();

                    var windows = batch.Select((_, i) => Sample(boxes[i].SelectMany(t => t.Boxes).ToArray(), content[i], size, windowsPerImage, random)).ToArray();
                    using var choice = Choice(windows, device);
                    var (cells, _) = Glimpses.Embed(tower, pixels, Glimpses.ContentBoxes(content, device), choice); // [B, D, K]

                    // Each image's tags against each of its windows: 1 yes, 0 no, -1 neither (ignored).
                    var most = boxes.Max(b => b.Length);
                    var rows = new long[batch.Length * most];
                    var labels = new float[batch.Length * most * windowsPerImage];
                    Array.Fill(labels, -1f);
                    for (var i = 0; i < batch.Length; i++)
                        for (var t = 0; t < most; t++)
                        {
                            rows[i * most + t] = boxes[i][Math.Min(t, boxes[i].Length - 1)].Row;
                            if (t >= boxes[i].Length)
                                continue;
                            for (var k = 0; k < windowsPerImage; k++)
                                labels[(i * most + t) * windowsPerImage + k] = Label(windows[i][k], boxes[i][t].Boxes, boxes[i][t].Ignore);
                        }
                    var rowEmbeddings = tags.index_select(0, tensor(rows, device: device)).reshape(batch.Length, most, dim);
                    var logits = scaleBias.ForwardDetached(rowEmbeddings.bmm(cells)).flatten(); // [B·T·K]
                    var target = tensor(labels, device: device);
                    var (positive, negative) = (target.eq(1f), target.eq(0f));
                    var perPair = nn.functional.binary_cross_entropy_with_logits(logits, target.clamp_min(0f), reduction: nn.Reduction.None);
                    var loss = (perPair * positive).sum() / positive.sum().clamp_min(1) + (perPair * negative).sum() / negative.sum().clamp_min(1);
                    optimizer.zero_grad();
                    loss.backward();
                    optimizer.step();
                    running.Add(loss.item<float>());
                    if (step % 50 == 0 || step == steps)
                    {
                        AnsiConsole.MarkupLineInterpolated($"fold {fold + 1}/{folds} step {step}/{steps}: balanced BCE {running.Average():F4}");
                        running.Clear();
                    }
                }
            }

            // The held-out images: the checkpoint's own windows, re-embedded.
            tower.eval();
            var tagValues = tags.cpu().data<float>().ToArray();
            using (no_grad())
                for (var s = 0; s < sets.Count; s++)
                    foreach (var chunk in sets[s].Gold.Where(g => foldOf.TryGetValue(g.CacheIndex, out var f) && f == fold && sets[s].Windows.ContainsKey(g.CacheIndex)).Chunk(16))
                    {
                        using var scope = NewDisposeScope();
                        var (pixels, content) = Load(cache, chunk.Select(g => g.CacheIndex).ToArray(), size, device);
                        var mapMask = SpatialMask.Build(content, size, size / tower.OutputStride, size / tower.OutputStride, device);
                        var (pooled, _) = tower.ForwardPooled(pixels, mapMask);
                        var windows = chunk.Select(g => sets[s].Windows[g.CacheIndex]).ToArray();
                        using var choice = Choice(windows, device);
                        var (cells, pooledValid) = Glimpses.Embed(tower, pixels, Glimpses.ContentBoxes(content, device), choice);
                        var slots = (int)cells.shape[2];
                        var cellValues = cells.cpu().data<float>().ToArray(); // [B, D, K]
                        var pooledValues = pooled.cpu().data<float>().ToArray();
                        var poolsAt = pooledValid.cpu().data<bool>().ToArray();
                        for (var i = 0; i < chunk.Length; i++)
                        {
                            var w = windows[i];
                            var windowEmbeddings = new float[w.Length * dim];
                            for (var k = 0; k < w.Length; k++)
                                for (var d = 0; d < dim; d++)
                                    windowEmbeddings[k * dim + d] = cellValues[((long)i * dim + d) * slots + k];
                            embeddingRows[s].Add(JsonSerializer.Serialize(new GlimpseEmbeddings(chunk[i].CacheIndex, dim, w.Select(x => new[] { x.Zoom, x.X, x.Y, x.Side }).ToArray(),
                                Convert.ToBase64String(System.Runtime.InteropServices.MemoryMarshal.AsBytes(windowEmbeddings.AsSpan())),
                                Convert.ToBase64String(System.Runtime.InteropServices.MemoryMarshal.AsBytes(pooledValues.AsSpan(i * dim, dim))))));
                            foreach (var t in Labelled(chunk[i], RowOf))
                            {
                                var logits = Enumerable.Range(0, w.Length)
                                    .Select(k => Enumerable.Range(0, dim).Sum(d => (double)windowEmbeddings[k * dim + d] * tagValues[(long)t.Row * dim + d])).ToArray();
                                var pooledLogits = Enumerable.Range(0, w.Length).Where(k => poolsAt[i * slots + k]).Select(k => logits[k]).ToList();
                                var probability = pooledLogits.Count > 0
                                    ? 1f / (1f + MathF.Exp(-scaleBias.ToLogitScaling().Apply((float)ClassificationPooling.Pool(pooledLogits, tower.ClassificationPoolingTemperature,
                                        tower.ClassificationPoolingTopK, tower.ClassificationPoolingBottomWeight))))
                                    : float.NaN;
                                pairs[s].Add(JsonSerializer.Serialize(new GlimpsePair(chunk[i].CacheIndex, t.Tag, chunk[i].Stratum,
                                    LocalizationEvaluator.SizeBucket(t.Tag, chunk[i].Instances.Where(x => x.Tag == t.Tag).ToList()),
                                    t.Boxes.Select(b => Canvas(b, chunk[i].Content)).ToArray(), w.Select(x => new[] { x.Zoom, x.X, x.Y, x.Side }).ToArray(),
                                    logits.Select(l => (float)l).ToArray(), probability, -1)));
                            }
                        }
                    }
            AnsiConsole.MarkupLineInterpolated($"fold {fold + 1}/{folds}: held-out images scored.");
            tower.Dispose();
            scaleBias.Dispose();
        }

        for (var s = 0; s < sets.Count; s++)
        {
            File.WriteAllLines(Path.Combine(outDir, $"glimpse_pairs{sets[s].Suffix}.jsonl"), pairs[s]);
            File.WriteAllLines(Path.Combine(outDir, $"glimpse_embeddings{sets[s].Suffix}.jsonl"), embeddingRows[s]);
        }
        AnsiConsole.MarkupLineInterpolated($"Wrote {pairs.Sum(p => p.Count)} held-out pairs to '{outDir}'.");
        return 0;
    }

    /// <summary>A gold image's labelled localizable tags: their certain/group boxes and ignore boxes, in content pixels.</summary>
    private static IEnumerable<(string Tag, int Row, int[][] Boxes, int[][] Ignore)> Labelled(GoldImage gold, Func<string, int?> rowOf) =>
        gold.Targets.Where(t => !gold.NotVisible.Contains(t)).Distinct()
            .Select(t => (Tag: t, Row: rowOf(t), Boxes: gold.Instances.Where(x => x.Tag == t && x.Kind != GoldKind.Ignore).Select(x => x.Box).ToArray(),
                Ignore: gold.Instances.Where(x => x.Tag == t && x.Kind == GoldKind.Ignore).Select(x => x.Box).ToArray()))
            .Where(t => t.Row is not null && t.Boxes.Length > 0)
            .Select(t => (t.Tag, t.Row!.Value, t.Boxes, t.Ignore));

    private static int[] Canvas(int[] box, LetterboxBox content) => [box[0] + content.X, box[1] + content.Y, box[2], box[3]];

    private static int[] Flip(int[] box, bool flip, int size) => flip ? [size - box[0] - box[2], box[1], box[2], box[3]] : box;

    private static double Intersection(GlimpseWindow w, int[] b)
    {
        var x = Math.Min(w.X + w.Side, b[0] + b[2]) - Math.Max(w.X, b[0]);
        var y = Math.Min(w.Y + w.Side, b[1] + b[3]) - Math.Max(w.Y, b[1]);
        return x <= 0 || y <= 0 ? 0 : (double)x * y;
    }

    /// <summary>1 when the window holds half of a box or a box fills half of it, 0 when it touches none (ignore boxes included), else -1.</summary>
    public static float Label(GlimpseWindow window, IReadOnlyList<int[]> boxes, IReadOnlyList<int[]> ignore)
    {
        foreach (var b in boxes)
        {
            var inside = Intersection(window, b);
            if (inside >= 0.5 * b[2] * b[3] || inside >= 0.5 * window.Side * window.Side)
                return 1f;
        }
        return boxes.Concat(ignore).Any(b => Intersection(window, b) > 0) ? -1f : 0f;
    }

    /// <summary>Windows at random zooms: half placed over a random box (its centre somewhere inside), half anywhere on the content.</summary>
    private static GlimpseWindow[] Sample(int[][] boxes, LetterboxBox content, int size, int count, Random random)
    {
        var windows = new GlimpseWindow[count];
        for (var k = 0; k < count; k++)
        {
            var zoom = random.Next(Glimpses.Zooms.Length);
            var side = Glimpses.SideOf(zoom);
            int x, y;
            if (k % 2 == 0 && boxes.Length > 0)
            {
                var b = boxes[random.Next(boxes.Length)];
                x = b[0] + b[2] / 2 - random.Next(side);
                y = b[1] + b[3] / 2 - random.Next(side);
            }
            else
            {
                x = content.X + random.Next(Math.Max(1, content.Width - side + 1));
                y = content.Y + random.Next(Math.Max(1, content.Height - side + 1));
            }
            // On the canvas, at a position the zoom's canvas has a pixel for (Glimpses.Crops rounds X·zoom).
            var step = Math.Max(1, (int)Math.Round(1 / Glimpses.Zooms[zoom]));
            x = Math.Clamp(x, 0, size - side) / step * step;
            y = Math.Clamp(y, 0, size - side) / step * step;
            windows[k] = new GlimpseWindow(zoom, x, y, side);
        }
        return windows;
    }

    private static GlimpseChoice Choice(IReadOnlyList<GlimpseWindow[]> windows, Device device)
    {
        var count = windows.Max(w => w.Length);
        var (zoom, x, y) = (new long[windows.Count * count], new long[windows.Count * count], new long[windows.Count * count]);
        var valid = new bool[windows.Count * count];
        for (var i = 0; i < windows.Count; i++)
            for (var k = 0; k < windows[i].Length; k++)
            {
                (zoom[i * count + k], x[i * count + k], y[i * count + k]) = (windows[i][k].Zoom, windows[i][k].X, windows[i][k].Y);
                valid[i * count + k] = true;
            }
        return new GlimpseChoice(tensor(zoom, [windows.Count, count], device: device), tensor(x, [windows.Count, count], device: device),
            tensor(y, [windows.Count, count], device: device), tensor(valid, [windows.Count, count], device: device));
    }

    private static (Tensor Pixels, LetterboxBox[] Content) Load(PreprocessedDatasetCacheReader cache, IReadOnlyList<int> indices, int size, Device device)
    {
        var encoded = cache.ReadEncoded(indices);
        var floats = 3 * size * size;
        var pixels = new float[indices.Count * floats];
        Parallel.For(0, indices.Count, i => ImagePreprocessing.ReconstructInto(encoded[i], size, pixels.AsSpan(i * floats, floats)));
        return (tensor(pixels, [indices.Count, 3, size, size]).to(device), encoded.Select(e => e.Content).ToArray());
    }

    private static Tensor Augment(Tensor pixels, bool[] flips, Random random, Device device)
    {
        var flipped = pixels.flip(3);
        var mask = tensor(flips, device: device).view(-1, 1, 1, 1);
        var gain = tensor(flips.Select(_ => 0.8f + 0.4f * (float)random.NextDouble()).ToArray(), device: device).view(-1, 1, 1, 1);
        var shift = tensor(flips.Select(_ => -0.1f + 0.2f * (float)random.NextDouble()).ToArray(), device: device).view(-1, 1, 1, 1);
        return where(mask, flipped, pixels) * gain + shift;
    }
}
