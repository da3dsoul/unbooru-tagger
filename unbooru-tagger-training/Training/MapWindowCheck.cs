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
/// How good would window labels read off a checkpoint's own localization map be (<c>evaluate --map-window-check</c>, 2026-11-03;
/// experiment 0 of review/open-vocab-research/PLAN.md, after Detic/OWL-ST: train windows from a localizer, not from image labels)?
/// For each labelled localizable gold pair, the tag's map is pooled (mean) over a grid of 64px and 128px windows (stride half a
/// side) on the content, and each window is labelled against the gold boxes with the ceiling test's rule
/// (<see cref="WindowProbeCommandHandler.Label"/>). Reported per size: "yes" precision of the window centred on the map's peak and
/// of the top 15% of windows (MIST, kept 0.2-IoU apart), the share of the bottom 40% of windows ("no") that touch the object
/// against the share of all windows that do (the base rate), and the map's own within-pair on/off window AUC.
/// </summary>
public static class MapWindowCheck
{
    private static readonly int[] Sides = [64, 128];
    private const double NoQuantile = 0.4, TopShare = 0.15, Diversity = 0.2;

    private sealed class Tally
    {
        public int Pairs, PeakYes, PeakWindows, TopYes, TopWindows, NoTouching, NoWindows, Touching, Windows, AucPairs;
        public double AucSum, StudentAucSum;
        public void Add(Tally o)
        {
            Pairs += o.Pairs; PeakYes += o.PeakYes; PeakWindows += o.PeakWindows; TopYes += o.TopYes; TopWindows += o.TopWindows;
            NoTouching += o.NoTouching; NoWindows += o.NoWindows; Touching += o.Touching; Windows += o.Windows; AucPairs += o.AucPairs; AucSum += o.AucSum; StudentAucSum += o.StudentAucSum;
        }
        public string Row(string name)
        {
            static string Rate(int n, int d) => d > 0 ? $"{(double)n / d,4:P0}" : "   -";
            return $"{name,-22} pairs {Pairs,4}  peak-window yes {Rate(PeakYes, PeakWindows)}  top-15% yes {Rate(TopYes, TopWindows)}  " +
                   $"'no' touching {Rate(NoTouching, NoWindows)} (base {Rate(Touching, Windows)})  map on/off AUC {(AucPairs > 0 ? AucSum / AucPairs : double.NaN):F2}, glimpses {(AucPairs > 0 ? StudentAucSum / AucPairs : double.NaN):F2} ({AucPairs})";
        }
    }

    public static void Run(ImageTower tower, float[] tagEmbeddings, int dim, IReadOnlyList<GoldImage> gold, TagVocabulary vocabulary,
        TagLocalizability localizability, PreprocessedDatasetCacheReader cache, Device device, string? dumpPath)
    {
        var size = cache.InputSize;
        var rowOf = vocabulary.Records.ToDictionary(r => r.Tag, r => r.RowIndex);
        var tallies = new Dictionary<string, Tally>();
        var dump = new List<string>();
        tower.eval();
        foreach (var batch in gold.Chunk(16))
        {
            float[] maps;
            float[]? pixelBatchValues = null;
            long mapH, mapW;
            using (no_grad())
            using (NewDisposeScope())
            {
                var encoded = cache.ReadEncoded(batch.Select(g => g.CacheIndex).ToArray());
                var floats = 3 * size * size;
                var pixelValues = new float[batch.Length * floats];
                Parallel.For(0, batch.Length, i => ImagePreprocessing.ReconstructInto(encoded[i], size, pixelValues.AsSpan(i * floats, floats)));
                pixelBatchValues = pixelValues;
                var pixels = tensor(pixelValues, [batch.Length, 3, size, size]).to(device);
                var mapMask = SpatialMask.Build(encoded.Select(e => e.Content).ToArray(), size, size / tower.OutputStride, size / tower.OutputStride, device);
                var (_, map) = tower.ForwardPooled(pixels, mapMask);
                (mapH, mapW) = (map.shape[2], map.shape[3]);
                maps = map.cpu().data<float>().ToArray(); // [B, D, h, w]
            }
            var stride = size / (int)mapW;
            for (var b = 0; b < batch.Length; b++)
            {
                var image = batch[b];
                // Every grid window of the image, embedded as a glimpse (the student's view of the same windows).
                var grids = Sides.Where(side => side <= Math.Min(image.Content.Width, image.Content.Height)).ToDictionary(side => side, side => Grid(image.Content, side));
                var allWindows = grids.Values.SelectMany(g => g).ToArray();
                float[] studentCells;
                using (no_grad())
                using (NewDisposeScope())
                {
                    var pixels = tensor(pixelBatchValues!, [batch.Length, 3, size, size]).narrow(0, b, 1).to(device);
                    var zoomOf = Sides.ToDictionary(side => side, side => (long)Enumerable.Range(0, Glimpses.Zooms.Length).First(z => Glimpses.SideOf(z) == side));
                    var choice = new GlimpseChoice(tensor(allWindows.Select(w => zoomOf[w.Side]).ToArray(), [1, allWindows.Length]).to(device),
                        tensor(allWindows.Select(w => (long)w.X).ToArray(), [1, allWindows.Length]).to(device),
                        tensor(allWindows.Select(w => (long)w.Y).ToArray(), [1, allWindows.Length]).to(device),
                        ones([1, allWindows.Length], ScalarType.Bool, device: device));
                    var contentBox = Glimpses.ContentBoxes([image.Content], device);
                    var (cells, _) = Glimpses.Embed(tower, pixels, contentBox, choice, requireZoomCodes: false);
                    studentCells = cells[0].t().contiguous().cpu().data<float>().ToArray(); // [K, D]
                }
                var windowIndex = allWindows.Select((w, k) => (w, k)).ToDictionary(x => (x.w.Side, x.w.X, x.w.Y), x => x.k);
                foreach (var tag in image.Targets.Where(t => !image.NotVisible.Contains(t)).Distinct())
                {
                    if (!rowOf.TryGetValue(tag, out var row) || !ContextRelianceEvaluator.IsGeneralTag(tag) || !localizability.IsLocalizable(tag))
                        continue;
                    var instances = image.Instances.Where(x => x.Tag == tag).ToList();
                    int[] Canvas(GoldInstance x) => [x.Box[0] + image.Content.X, x.Box[1] + image.Content.Y, x.Box[2], x.Box[3]];
                    var boxes = instances.Where(x => x.Kind != GoldKind.Ignore).Select(Canvas).ToArray();
                    var ignore = instances.Where(x => x.Kind == GoldKind.Ignore).Select(Canvas).ToArray();
                    if (boxes.Length == 0)
                        continue;

                    // The tag's map over the content, and integral sums for window means.
                    var heat = new double[mapH, mapW];
                    var (cx0, cy0) = (image.Content.X / stride, image.Content.Y / stride);
                    var (cx1, cy1) = ((image.Content.X + image.Content.Width + stride - 1) / stride, (image.Content.Y + image.Content.Height + stride - 1) / stride);
                    var (peakX, peakY, peakV) = (0, 0, double.NegativeInfinity);
                    for (var y = cy0; y < cy1; y++)
                        for (var x = cx0; x < cx1; x++)
                        {
                            double v = 0;
                            for (var d = 0; d < dim; d++)
                                v += tagEmbeddings[(long)row * dim + d] * maps[(((long)b * dim + d) * mapH + y) * mapW + x];
                            heat[y, x] = v;
                            if (v > peakV)
                                (peakX, peakY, peakV) = (x, y, v);
                        }
                    double Mean(int wx, int wy, int side)
                    {
                        int x0 = Math.Max(cx0, wx / stride), y0 = Math.Max(cy0, wy / stride);
                        int x1 = Math.Min(cx1, (wx + side) / stride), y1 = Math.Min(cy1, (wy + side) / stride);
                        double s = 0; var n = 0;
                        for (var y = y0; y < y1; y++)
                            for (var x = x0; x < x1; x++) { s += heat[y, x]; n++; }
                        return n > 0 ? s / n : double.NegativeInfinity;
                    }

                    var bucket = LocalizationEvaluator.SizeBucket(tag, instances);
                    var pair = new Tally { Pairs = 1 };
                    var record = new Dictionary<string, object> { ["cacheIndex"] = image.CacheIndex, ["tag"] = tag, ["size"] = bucket };
                    foreach (var side in Sides)
                    {
                        if (!grids.ContainsKey(side))
                            continue;
                        int ClampX(int x) => Math.Clamp(x, image.Content.X, image.Content.X + image.Content.Width - side);
                        int ClampY(int y) => Math.Clamp(y, image.Content.Y, image.Content.Y + image.Content.Height - side);
                        double Student(GlimpseWindow w)
                        {
                            var k = windowIndex[(w.Side, w.X, w.Y)];
                            double v = 0;
                            for (var d = 0; d < dim; d++)
                                v += tagEmbeddings[(long)row * dim + d] * studentCells[(long)k * dim + d];
                            return v;
                        }
                        var grid = grids[side].Select(w => (W: w, Score: Mean(w.X, w.Y, side), Label: WindowProbeCommandHandler.Label(w, boxes, ignore), Student: Student(w))).ToList();
                        // The window centred on the map's peak.
                        var peak = new GlimpseWindow(0, ClampX(peakX * stride + stride / 2 - side / 2), ClampY(peakY * stride + stride / 2 - side / 2), side);
                        var peakLabel = WindowProbeCommandHandler.Label(peak, boxes, ignore);
                        pair.PeakWindows++;
                        if (peakLabel == 1f) pair.PeakYes++;
                        // MIST: the top 15% of windows by the map, kept apart.
                        var top = new List<GlimpseWindow>();
                        foreach (var g in grid.OrderByDescending(g => g.Score).Take(Math.Max(1, (int)Math.Ceiling(grid.Count * TopShare))))
                            if (top.All(t => IoU(t, g.W) < Diversity))
                                top.Add(g.W);
                        pair.TopWindows += top.Count;
                        pair.TopYes += top.Count(t => WindowProbeCommandHandler.Label(t, boxes, ignore) == 1f);
                        // "No": the bottom 40%.
                        var no = grid.OrderBy(g => g.Score).Take((int)Math.Floor(grid.Count * NoQuantile)).ToList();
                        pair.NoWindows += no.Count;
                        pair.NoTouching += no.Count(g => g.Label != 0f);
                        pair.Windows += grid.Count;
                        pair.Touching += grid.Count(g => g.Label != 0f);
                        // The map's own within-pair window AUC (on = label 1, off = label 0).
                        var on = grid.Where(g => g.Label == 1f).ToArray();
                        var off = grid.Where(g => g.Label == 0f).ToArray();
                        if (on.Length > 0 && off.Length > 0)
                        {
                            pair.AucSum += Auc(on.Select(g => g.Score), off.Select(g => g.Score));
                            pair.StudentAucSum += Auc(on.Select(g => g.Student), off.Select(g => g.Student));
                            pair.AucPairs++;
                        }
                        record[$"peak{side}"] = peakLabel;
                    }
                    foreach (var name in new[] { "all", bucket })
                        (tallies.TryGetValue(name, out var t) ? t : tallies[name] = new()).Add(pair);
                    dump.Add(JsonSerializer.Serialize(record));
                }
            }
        }

        AnsiConsole.MarkupLineInterpolated($"Map window check: 64px and 128px windows, labelled from the gold boxes; 'no' = the bottom {NoQuantile:P0} of each pair's windows by the map.");
        foreach (var (name, tally) in tallies.OrderBy(kv => kv.Key == "all" ? 0 : 1).ThenBy(kv => kv.Key))
            AnsiConsole.WriteLine(tally.Row(name.Replace("size: ", "")));
        if (dumpPath is not null)
            File.WriteAllLines(dumpPath, dump);
    }

    private static double Auc(IEnumerable<double> on, IEnumerable<double> off)
    {
        var (a, c) = (on.ToArray(), off.ToArray());
        double wins = 0;
        foreach (var x in a)
            foreach (var y in c)
                wins += x > y ? 1 : x == y ? 0.5 : 0;
        return wins / (a.Length * c.Length);
    }

    /// <summary>The content's windows of <paramref name="side"/>, stride half a side, the last row and column moved in to the edge.</summary>
    private static List<GlimpseWindow> Grid(LetterboxBox content, int side)
    {
        int ClampX(int x) => Math.Clamp(x, content.X, content.X + content.Width - side);
        int ClampY(int y) => Math.Clamp(y, content.Y, content.Y + content.Height - side);
        var grid = new List<GlimpseWindow>();
        for (var y = content.Y; ; y += side / 2)
        {
            for (var x = content.X; ; x += side / 2)
            {
                var w = new GlimpseWindow(0, ClampX(x), ClampY(y), side);
                if (!grid.Contains(w))
                    grid.Add(w);
                if (x + side >= content.X + content.Width) break;
            }
            if (y + side >= content.Y + content.Height) break;
        }
        return grid;
    }

    private static double IoU(GlimpseWindow a, GlimpseWindow b)
    {
        var x = Math.Min(a.X + a.Side, b.X + b.Side) - Math.Max(a.X, b.X);
        var y = Math.Min(a.Y + a.Side, b.Y + b.Side) - Math.Max(a.Y, b.Y);
        if (x <= 0 || y <= 0) return 0;
        var inter = (double)x * y;
        return inter / ((double)a.Side * a.Side + (double)b.Side * b.Side - inter);
    }
}
