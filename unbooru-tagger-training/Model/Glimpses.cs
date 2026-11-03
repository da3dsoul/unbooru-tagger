using UnbooruTagger.Core.Encoding;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace UnbooruTagger.Training.Model;

/// <summary>One glimpse window: which zoom (index into <see cref="Glimpses.Zooms"/>) and its square on the input canvas.</summary>
public readonly record struct GlimpseWindow(int Zoom, int X, int Y, int Side);

/// <summary>
/// A batch's chosen windows, on the device (<see cref="Glimpses.Propose"/>): [B, K] each, in priority order; slots past an
/// image's last window are <see cref="Valid"/> false.
/// </summary>
/// <param name="Zoom">Index into <see cref="Glimpses.Zooms"/> (int64).</param>
/// <param name="X">The window's left edge on the input canvas (int64).</param>
/// <param name="Y">Its top edge (int64).</param>
public sealed record GlimpseChoice(Tensor Zoom, Tensor X, Tensor Y, Tensor Valid) : IDisposable
{
    /// <summary>Read back (tests, diagnostics; it waits for the device): each image's windows.</summary>
    public GlimpseWindow[][] ToWindows()
    {
        var (batch, count) = ((int)Zoom.shape[0], (int)Zoom.shape[1]);
        var (zoom, x, y) = (Zoom.cpu().data<long>().ToArray(), X.cpu().data<long>().ToArray(), Y.cpu().data<long>().ToArray());
        var valid = Valid.cpu().data<bool>().ToArray();
        return Enumerable.Range(0, batch)
            .Select(b => Enumerable.Range(b * count, count).Where(k => valid[k])
                .Select(k => new GlimpseWindow((int)zoom[k], (int)x[k], (int)y[k], Glimpses.SideOf((int)zoom[k]))).ToArray())
            .ToArray();
    }

    public void Dispose()
    {
        Zoom.Dispose();
        X.Dispose();
        Y.Dispose();
        Valid.Dispose();
    }
}

/// <summary>
/// Object tags scored from small zoomed views instead of the whole image (<see cref="ImageTower.ClassificationGlimpses"/>).
/// The whole-image embedding sees the character, so "this character, therefore gloves" is free to learn: the averaged and
/// map-cell scores alike, since even a fused map cell sees most of a 512px image, and the pooled readout can't separate them
/// afterwards (2026-10-30: a last-layer refit traded false positives for recall 1:1). A glimpse is a square window of the
/// canvas resized to <see cref="Size"/>px and run through the same tower, with a learned code for its zoom
/// (<see cref="ImageTower.GlimpseZoomCodes"/>) added to its pooled embedding. Zoom is magnification: 2 = a 32px window
/// enlarged (bilinear), 0.25 = a 256px window shrunk (box filter, an exact average over each 4x4 block). Small windows keep
/// the face out of view for objects away from it; earrings and glasses still share a window with the face, which is fine
/// for those. Windows overlap at the zoomed-out scales, so an object cut by one window is whole in a larger one.
/// <para>
/// A localizable tag's score is <see cref="ClassificationPooling.RawScores"/>' log-sum-exp over the image's glimpses, so the
/// best-framed glimpse decides; every other tag keeps the whole-image embedding.
/// </para>
/// <para>
/// Where to look is the model's own choice, the same in training and evaluation (<see cref="Propose"/>): the localization
/// map ranks the tags, and the top ones each get a window around their peak, at the most magnified zoom that holds their
/// response. Until 2026-10-31 training framed the labelled positives' peaks instead (<see cref="Sample"/>) and evaluation
/// scored every window of the grid (<see cref="Layout"/>, 378 per image): the labels chose where training looked, a wrong
/// tag's peak was never looked at closely, and evaluation cost ~6x a whole-image pass.
/// </para>
/// </summary>
public static class Glimpses
{
    /// <summary>Glimpses per tower pass in <see cref="Embed"/>: 512 runs a training step's 480 (10 images x 48) in one pass (67 -> 64 ms with backward on the B70).</summary>
    public const int DefaultChunk = 512;

    /// <summary>Glimpse side in pixels.</summary>
    public const int Size = 64;

    /// <summary>Magnifications: a window of <see cref="Size"/> / zoom canvas pixels each. 4x (a 16px window) holds nothing useful at 512px.</summary>
    public static readonly float[] Zooms = [2f, 1f, 0.5f, 0.25f];

    /// <summary>Fewer content pixels than this and a window isn't a glimpse (letterbox bars).</summary>
    public const double MinContent = 0.5;

    /// <summary>Every window, zoom by zoom, row-major: non-overlapping when zoomed in or at 1x, half-overlapping when zoomed out.</summary>
    public static GlimpseWindow[] Layout(int inputSize)
    {
        var windows = new List<GlimpseWindow>();
        for (var z = 0; z < Zooms.Length; z++)
        {
            var (side, step) = WindowAndStep(z);
            for (var y = 0; y + side <= inputSize; y += step)
                for (var x = 0; x + side <= inputSize; x += step)
                    windows.Add(new GlimpseWindow(z, x, y, side));
        }
        return windows.ToArray();
    }

    /// <summary>Canvas pixels per window side and between windows, for zoom index <paramref name="zoom"/>.</summary>
    private static (int Side, int Step) WindowAndStep(int zoom)
    {
        var side = (int)Math.Round(Size / Zooms[zoom]);
        return (side, Zooms[zoom] >= 1 ? side : side / 2);
    }

    /// <summary>The fraction of <paramref name="window"/> inside <paramref name="content"/>.</summary>
    public static double ContentFraction(GlimpseWindow window, LetterboxBox content)
    {
        var w = Math.Min(window.X + window.Side, content.X + content.Width) - Math.Max(window.X, content.X);
        var h = Math.Min(window.Y + window.Side, content.Y + content.Height) - Math.Max(window.Y, content.Y);
        return w <= 0 || h <= 0 ? 0 : (double)w * h / ((double)window.Side * window.Side);
    }

    /// <summary>Every window of <paramref name="layout"/> with enough content, per image.</summary>
    public static int[][] All(GlimpseWindow[] layout, IReadOnlyList<LetterboxBox> content) =>
        content.Select(box => Usable(layout, box)).ToArray();

    /// <summary>
    /// The windows holding at least <see cref="MinContent"/> of <paramref name="content"/>; for a thin strip of an image that
    /// no window is half covered by, each zoom's fullest window instead, so every image has glimpses (none would score its
    /// object tags at -infinity).
    /// </summary>
    private static int[] Usable(GlimpseWindow[] layout, LetterboxBox content)
    {
        var usable = Enumerable.Range(0, layout.Length).Where(g => ContentFraction(layout[g], content) >= MinContent).ToArray();
        return usable.Length > 0
            ? usable
            : Enumerable.Range(0, Zooms.Length)
                .Select(z => Enumerable.Range(0, layout.Length).Where(g => layout[g].Zoom == z).MaxBy(g => ContentFraction(layout[g], content)))
                .Where(g => ContentFraction(layout[g], content) > 0).ToArray();
    }

    /// <summary>Two windows of one zoom overlapping at least this much (intersection over union) are the same look.</summary>
    public const double SameLook = 0.5;

    /// <summary>
    /// Up to <paramref name="count"/> windows per image, chosen by the model without labels: what training and evaluation
    /// both score. All on the device, so a training step never waits for it.
    /// <list type="number">
    /// <item>The tags of <paramref name="tagEmbeddings"/> (the localizable ones) are ranked by <paramref name="proposals"/>, the
    /// whole-image proposal scores (<see cref="ImageTower.Proposals"/>): what the image might hold, from everything in view.
    /// The localization map can't rank them: it is trained on ~100 pairs per image, so its scores for the rest aren't comparable
    /// (on run 4's gold set, 6% of gold tags ranked in the map's top 32 against 68% in the whole-image classifier's).</item>
    /// <item>In that order (after <paramref name="firstTags"/>, if any), each of the top <paramref name="candidates"/> tags is
    /// looked at with the glimpse sensor at its map peak (<see cref="Sensor"/>: one window per zoom, centred there); then the
    /// zoomed-out grid windows holding enough of the image, for coverage.</item>
    /// <item>A window repeating an earlier one in that order (<see cref="SameLookAs"/>) is dropped, whether or not the
    /// earlier one was kept itself (the parallel form of the rule; on the CPU it compared only with kept windows), and the
    /// first <paramref name="count"/> left are the choice.</item>
    /// </list>
    /// Which zoom shows a tag best is the classifier's to learn (its log-sum-exp over the glimpses), not decided here: on the
    /// gold set the map's response extent said little about an object's size (2026-10-31, <c>glimpse-frames</c>).
    /// </summary>
    /// <param name="proposals">[B, D], the proposal head's output (<see cref="ImageTower.Proposals"/>).</param>
    /// <param name="localizationMap">[B, D, H, W], the tower's localization map (detached here).</param>
    /// <param name="tagEmbeddings">[L, D] the tags to rank: the localizable ones.</param>
    /// <param name="mapMask">[B, 1, H, W], the content cells of the map (<see cref="SpatialMask"/>).</param>
    /// <param name="contentBoxes">[B, 4] each image's content box (<see cref="ContentBoxes"/>).</param>
    /// <param name="firstTags">[B, P, D] tags to look at before the model's own (training's early label framing), or null.</param>
    /// <param name="firstValid">[B, P] which of <paramref name="firstTags"/> are real.</param>
    public static GlimpseChoice Propose(
        Tensor proposals, Tensor localizationMap, Tensor tagEmbeddings, Tensor mapMask, Tensor contentBoxes, int inputSize, int outputStride, int count,
        int candidates = 0, Tensor? firstTags = null, Tensor? firstValid = null)
    {
        using var _ = no_grad();
        using var scope = NewDisposeScope();
        var device = localizationMap.device;
        var batch = localizationMap.shape[0];
        candidates = Math.Min(candidates > 0 ? candidates : count, (int)tagEmbeddings.shape[0]);
        var tags = tagEmbeddings.detach();
        var top = proposals.detach().matmul(tags.t()).topk(candidates, 1).indexes;
        var looked = tags.index_select(0, top.flatten()).reshape(batch, candidates, -1);
        var lookedValid = ones(batch, candidates, dtype: ScalarType.Bool, device: device);
        if (firstTags is not null)
        {
            looked = cat([firstTags.detach().to_type(looked.dtype), looked], 1);
            lookedValid = cat([firstValid!, lookedValid], 1);
        }

        // Each tag's sensor at its peak, tag by tag, most magnified first; then the grid.
        var (peakX, peakY) = Peaks(localizationMap, looked, mapMask, outputStride);
        var (sensorX, sensorY) = SensorOn(peakX, peakY, contentBoxes, inputSize, Zooms.Length);
        var sensorZoom = arange(Zooms.Length, device: device).expand(sensorX.shape);
        var grid = Layout(inputSize).Where(w => w.Zoom == Zooms.Length - 1).ToArray();
        var gridX = tensor(grid.Select(g => (long)g.X).ToArray()).Upload(device).expand(batch, grid.Length);
        var gridY = tensor(grid.Select(g => (long)g.Y).ToArray()).Upload(device).expand(batch, grid.Length);
        var gridZoom = full(batch, grid.Length, Zooms.Length - 1, dtype: ScalarType.Int64, device: device);
        var zoom = cat([sensorZoom.flatten(1), gridZoom], 1);
        var x = cat([sensorX.flatten(1), gridX], 1);
        var y = cat([sensorY.flatten(1), gridY], 1);
        var valid = cat([lookedValid.repeat_interleave(Zooms.Length, 1), ContentFractionOn(gridX, gridY, Size * 4, contentBoxes).ge(MinContent)], 1);

        // Drop a window repeating an earlier one; keep the first count.
        var side = Sides(device).index_select(0, zoom.flatten()).view_as(zoom).to_type(ScalarType.Float32);
        var (xf, yf) = (x.to_type(ScalarType.Float32), y.to_type(ScalarType.Float32));
        var width = (minimum((xf + side).unsqueeze(2), (xf + side).unsqueeze(1)) - maximum(xf.unsqueeze(2), xf.unsqueeze(1))).clamp_min(0);
        var height = (minimum((yf + side).unsqueeze(2), (yf + side).unsqueeze(1)) - maximum(yf.unsqueeze(2), yf.unsqueeze(1))).clamp_min(0);
        var intersection = width * height;
        var area = side * side;
        var overlap = intersection / (area.unsqueeze(2) + area.unsqueeze(1) - intersection);
        var windows = zoom.shape[1];
        var earlier = ones(windows, windows, dtype: ScalarType.Bool, device: device).triu(1); // [i, j]: i before j
        var repeats = zoom.unsqueeze(2).eq(zoom.unsqueeze(1)).logical_and(overlap.ge(SameLook)).logical_and(valid.unsqueeze(2)).logical_and(earlier);
        var keep = valid.logical_and(repeats.any(1).logical_not());
        var rank = keep.to_type(ScalarType.Int64).cumsum(1) - 1;
        var take = keep.logical_and(rank.lt(count));
        var slot = where(take, rank, full_like(rank, count));
        Tensor Slots(Tensor values) => zeros(batch, count + 1, dtype: values.dtype, device: device).scatter(1, slot, values).narrow(1, 0, count);
        var choice = new GlimpseChoice(Slots(zoom), Slots(x), Slots(y), Slots(take));
        scope.MoveToOuter(choice.Zoom, choice.X, choice.Y, choice.Valid);
        return choice;
    }

    /// <summary>
    /// Each tag of <paramref name="tags"/> ([B, T, D], per image): where on the canvas its localization-map response peaks,
    /// within the content (<paramref name="mapMask"/>, [B, 1, H, W]), as [B, T] canvas x and y, on the device.
    /// </summary>
    public static (Tensor X, Tensor Y) Peaks(Tensor localizationMap, Tensor tags, Tensor mapMask, int outputStride)
    {
        using var _ = no_grad();
        using var scope = NewDisposeScope();
        var map = localizationMap.detach();
        var width = map.shape[3];
        var outside = mapMask.flatten(1).eq(0).unsqueeze(1);
        var peak = tags.detach().to_type(map.dtype).bmm(map.flatten(2)).masked_fill(outside, float.NegativeInfinity).argmax(2);
        var x = peak.remainder(width) * outputStride + outputStride / 2;
        var y = peak.div(width, rounding_mode: RoundingMode.floor) * outputStride + outputStride / 2;
        return (x.MoveToOuterDisposeScope(), y.MoveToOuterDisposeScope());
    }

    /// <summary>[zooms] int64: how many of <paramref name="choice"/>'s windows are at each zoom, on the device.</summary>
    public static Tensor ZoomCounts(GlimpseChoice choice)
    {
        using var scope = NewDisposeScope();
        var counts = choice.Zoom.unsqueeze(2).eq(arange(Zooms.Length, device: choice.Zoom.device).reshape(1, 1, -1))
            .logical_and(choice.Valid.unsqueeze(2)).to_type(ScalarType.Int64).sum([0, 1]);
        return counts.MoveToOuterDisposeScope();
    }

    /// <summary>[B, 4] the images' content boxes (x, y, width, height) on the device.</summary>
    public static Tensor ContentBoxes(IReadOnlyList<LetterboxBox> content, Device device) =>
        tensor(content.SelectMany(c => new long[] { c.X, c.Y, c.Width, c.Height }).ToArray(), [content.Count, 4]).Upload(device);

    /// <summary>A window's side in canvas pixels, for zoom index <paramref name="zoom"/>.</summary>
    public static int SideOf(int zoom) => (int)Math.Round(Size / Zooms[zoom]);

    private static Tensor Sides(Device device) => tensor(Enumerable.Range(0, Zooms.Length).Select(z => (long)SideOf(z)).ToArray()).Upload(device);

    /// <summary>
    /// The glimpse sensor at [B, T] points: [B, T, zooms] window corners, each window as <see cref="Centered"/> places it,
    /// on the device; the first <paramref name="zooms"/> zooms.
    /// </summary>
    public static (Tensor X, Tensor Y) SensorOn(Tensor x, Tensor y, Tensor contentBoxes, int inputSize, int zooms)
    {
        var (startX, startY) = (contentBoxes[.., 0].unsqueeze(1), contentBoxes[.., 1].unsqueeze(1));
        var (lengthX, lengthY) = (contentBoxes[.., 2].unsqueeze(1), contentBoxes[.., 3].unsqueeze(1));
        Tensor Place(Tensor centre, Tensor start, Tensor length, long side)
        {
            var inside = (centre - side / 2).maximum(start).minimum(start + length - side);
            var narrow = start + (length - side).div(2, rounding_mode: RoundingMode.trunc);
            return where(length.ge(side), inside, narrow).clamp(0, inputSize - side);
        }
        var placedX = Enumerable.Range(0, zooms).Select(z => Place(x, startX, lengthX, SideOf(z))).ToArray();
        var placedY = Enumerable.Range(0, zooms).Select(z => Place(y, startY, lengthY, SideOf(z))).ToArray();
        return (stack(placedX, 2), stack(placedY, 2));
    }

    /// <summary>[B, W] the fraction of each window (corners <paramref name="x"/>, <paramref name="y"/>, side <paramref name="side"/>) inside its image's content.</summary>
    private static Tensor ContentFractionOn(Tensor x, Tensor y, long side, Tensor contentBoxes)
    {
        var (cx, cy) = (contentBoxes[.., 0].unsqueeze(1), contentBoxes[.., 1].unsqueeze(1));
        var (cw, ch) = (contentBoxes[.., 2].unsqueeze(1), contentBoxes[.., 3].unsqueeze(1));
        var w = (minimum(x + side, cx + cw) - maximum(x, cx)).clamp_min(0);
        var h = (minimum(y + side, cy + ch) - maximum(y, cy)).clamp_min(0);
        return (w * h).to_type(ScalarType.Float32) / (side * side);
    }

    /// <summary>
    /// [B, P] whether <paramref name="choice"/> looks closely at a tag whose map peak is at (<paramref name="x"/>,
    /// <paramref name="y"/>): one of its sensor's windows other than the most zoomed-out one is a chosen look
    /// (<see cref="Frames"/>, on the device).
    /// </summary>
    public static Tensor Framed(GlimpseChoice choice, Tensor x, Tensor y, Tensor contentBoxes, int inputSize)
    {
        using var _ = no_grad();
        using var scope = NewDisposeScope();
        return SensorLooks(choice, x, y, contentBoxes, inputSize, Zooms.Length - 1).any(3).any(2).MoveToOuterDisposeScope();
    }

    /// <summary>
    /// [B, P, K] which of <paramref name="choice"/>'s windows are one of the sensor's looks (any zoom) at a tag whose map peak is
    /// at (<paramref name="x"/>, <paramref name="y"/>): the windows that frame it, on the device.
    /// </summary>
    public static Tensor FramingWindows(GlimpseChoice choice, Tensor x, Tensor y, Tensor contentBoxes, int inputSize)
    {
        using var _ = no_grad();
        using var scope = NewDisposeScope();
        return SensorLooks(choice, x, y, contentBoxes, inputSize, Zooms.Length).any(2).MoveToOuterDisposeScope();
    }

    /// <summary>[B, P, zooms, K]: window k is the same look (<see cref="SameLookAs"/>) as the sensor's window of that zoom at peak p.</summary>
    private static Tensor SensorLooks(GlimpseChoice choice, Tensor x, Tensor y, Tensor contentBoxes, int inputSize, int zooms)
    {
        var (sensorX, sensorY) = SensorOn(x, y, contentBoxes, inputSize, zooms); // [B, P, zooms]
        var side = Sides(choice.Zoom.device).index_select(0, choice.Zoom.flatten()).view_as(choice.Zoom).to_type(ScalarType.Float32).unsqueeze(1).unsqueeze(1); // [B, 1, 1, K]
        var sensorSide = Sides(choice.Zoom.device).narrow(0, 0, zooms).to_type(ScalarType.Float32).reshape(1, 1, zooms, 1);
        var (sx, sy) = (sensorX.to_type(ScalarType.Float32).unsqueeze(3), sensorY.to_type(ScalarType.Float32).unsqueeze(3));
        var (cx, cy) = (choice.X.to_type(ScalarType.Float32).unsqueeze(1).unsqueeze(1), choice.Y.to_type(ScalarType.Float32).unsqueeze(1).unsqueeze(1));
        var width = (minimum(sx + sensorSide, cx + side) - maximum(sx, cx)).clamp_min(0);
        var height = (minimum(sy + sensorSide, cy + side) - maximum(sy, cy)).clamp_min(0);
        var intersection = width * height;
        var overlap = intersection / (sensorSide * sensorSide + side * side - intersection);
        var sameZoom = choice.Zoom.unsqueeze(1).unsqueeze(1).eq(arange(zooms, device: choice.Zoom.device).reshape(1, 1, zooms, 1));
        return sameZoom.logical_and(overlap.ge(SameLook)).logical_and(choice.Valid.unsqueeze(1).unsqueeze(1));
    }

    /// <summary>
    /// The glimpse sensor at a point: one window per zoom (<see cref="Zooms"/> order, most magnified first) centred on it, each
    /// moved inside the content and the canvas (<see cref="Centered"/>).
    /// </summary>
    public static GlimpseWindow[] Sensor(int x, int y, LetterboxBox content, int inputSize) =>
        Enumerable.Range(0, Zooms.Length).Select(z => Centered(z, x, y, content, inputSize)).ToArray();

    /// <summary>
    /// Whether <paramref name="chosen"/> looks closely at a tag whose sensor is <paramref name="sensor"/>: one of its windows
    /// other than the most zoomed-out one is a chosen look (<see cref="SameLookAs"/>). The zoomed-out window alone shrinks the
    /// image 4x and holds a quarter of it, so it frames nothing in particular.
    /// </summary>
    public static bool Frames(IEnumerable<GlimpseWindow> chosen, GlimpseWindow[] sensor) =>
        sensor.Take(Zooms.Length - 1).Any(w => chosen.Any(c => SameLookAs(c, w)));

    /// <summary>Whether two windows are the same look: one zoom, overlapping at least <see cref="SameLook"/>.</summary>
    public static bool SameLookAs(GlimpseWindow a, GlimpseWindow b) => a.Zoom == b.Zoom && Overlap(a, b) >= SameLook;

    /// <summary>
    /// Evaluation's glimpses ([B, D, K] embeddings and [B, K] which are real): the windows the model chooses
    /// (<see cref="Propose"/>, <see cref="ImageTower.GlimpseCount"/> of them, as in training), or with
    /// <see cref="ImageTower.GlimpseGrid"/> every window of the grid, as evaluations scored before 2026-10-31.
    /// </summary>
    /// <param name="localizableTags">[L, D] the localizable tags' embeddings, which <see cref="Propose"/> ranks.</param>
    public static (Tensor Cells, Tensor Valid) Look(ImageTower tower, Tensor pixels, IReadOnlyList<LetterboxBox> content, Tensor pooled, Tensor localizationMap, Tensor localizableTags)
    {
        var inputSize = (int)pixels.shape[2];
        if (tower.GlimpseGrid)
        {
            var grid = Layout(inputSize);
            return Embed(tower, pixels, content, grid, All(grid, content));
        }
        using var scope = NewDisposeScope();
        Tensor proposals;
        using (no_grad())
            proposals = tower.Proposals(pooled);
        var boxes = ContentBoxes(content, pixels.device);
        var mapMask = SpatialMask.Build(content, inputSize, localizationMap.shape[2], localizationMap.shape[3], pixels.device);
        using var choice = Propose(proposals, localizationMap, localizableTags, mapMask, boxes, inputSize, tower.OutputStride, tower.GlimpseCount);
        var (cells, valid) = Embed(tower, pixels, boxes, choice);
        return (cells.MoveToOuterDisposeScope(), valid.MoveToOuterDisposeScope());
    }

    /// <summary>Whether <paramref name="window"/> holds the canvas point (<paramref name="x"/>, <paramref name="y"/>).</summary>
    public static bool Contains(GlimpseWindow window, int x, int y) =>
        x >= window.X && x < window.X + window.Side && y >= window.Y && y < window.Y + window.Side;

    /// <summary>Zoom <paramref name="zoom"/>'s window centred on (<paramref name="x"/>, <paramref name="y"/>), moved inside the
    /// content (or centred on it, where the content is narrower than the window) and the canvas.</summary>
    public static GlimpseWindow Centered(int zoom, int x, int y, LetterboxBox content, int inputSize)
    {
        var side = (int)Math.Round(Size / Zooms[zoom]);
        int Place(int centre, int start, int length) =>
            Math.Clamp(length >= side ? Math.Clamp(centre - side / 2, start, start + length - side) : start + (length - side) / 2, 0, inputSize - side);
        return new GlimpseWindow(zoom, Place(x, content.X, content.Width), Place(y, content.Y, content.Height), side);
    }

    /// <summary>Intersection over union of two windows.</summary>
    public static double Overlap(GlimpseWindow a, GlimpseWindow b)
    {
        var w = Math.Min(a.X + a.Side, b.X + b.Side) - Math.Max(a.X, b.X);
        var h = Math.Min(a.Y + a.Side, b.Y + b.Side) - Math.Max(a.Y, b.Y);
        if (w <= 0 || h <= 0)
            return 0;
        var intersection = (double)w * h;
        return intersection / ((double)a.Side * a.Side + (double)b.Side * b.Side - intersection);
    }


    /// <summary>
    /// Up to <paramref name="count"/> windows for one image: for each point in <paramref name="targets"/> (where the
    /// localization map puts one of its positive tags) a window containing it at a random zoom, then random windows with
    /// enough content. A positive then usually has a glimpse that frames it, and the random ones supply glimpses without it.
    /// </summary>
    public static int[] Sample(GlimpseWindow[] layout, LetterboxBox content, IReadOnlyList<(int X, int Y)> targets, int count, Random random)
    {
        var valid = Usable(layout, content);
        var chosen = new List<int>();
        foreach (var (px, py) in targets.Take(count / 2))
        {
            var zoom = random.Next(Zooms.Length);
            var containing = valid.Where(g => layout[g].Zoom == zoom && px >= layout[g].X && px < layout[g].X + layout[g].Side
                                              && py >= layout[g].Y && py < layout[g].Y + layout[g].Side).ToArray();
            if (containing.Length > 0)
                chosen.Add(containing[random.Next(containing.Length)]);
        }
        var rest = valid.Except(chosen).OrderBy(_ => random.Next()).Take(count - chosen.Count);
        return chosen.Distinct().Concat(rest).ToArray();
    }

    /// <summary>The tower that embeds glimpses: <see cref="ImageTower.GlimpseTower"/> (computing as the main tower does) or the main tower itself.</summary>
    private static ImageTower GlimpsePath(ImageTower tower)
    {
        if (tower.GlimpseTower is not { } glimpses)
            return tower;
        glimpses.ComputeType = tower.ComputeType;
        glimpses.ChannelsLast = tower.ChannelsLast;
        return glimpses;
    }

    /// <summary>
    /// [B·K, 3, <see cref="Size"/>, <see cref="Size"/>] the chosen windows' pixels, image-major: each window's square of
    /// <paramref name="pixels"/> resampled to <see cref="Size"/> (box filter to zoom out, bilinear to zoom in).
    /// </summary>
    public static Tensor Crops(Tensor pixels, GlimpseChoice choice)
    {
        using var scope = NewDisposeScope();
        var device = pixels.device;
        var (batch, count) = (choice.Zoom.shape[0], choice.Zoom.shape[1]);
        var glimpses = batch * count;
        var inputSize = pixels.shape[2];

        // Every zoom's canvas in one buffer: box filter (exact block averages) to zoom out, bilinear to zoom in.
        var canvases = new List<Tensor>();
        var bases = new long[Zooms.Length];
        var canvasSides = new long[Zooms.Length];
        long offset = 0;
        for (var z = 0; z < Zooms.Length; z++)
        {
            var factor = Zooms[z];
            using (no_grad())
                canvases.Add((factor == 1 ? pixels
                    : factor < 1 ? functional.avg_pool2d(pixels, (long)Math.Round(1 / factor))
                    : functional.interpolate(pixels, scale_factor: [factor, factor], mode: InterpolationMode.Bilinear, align_corners: false)).flatten());
            canvasSides[z] = (long)Math.Round(inputSize * factor);
            bases[z] = offset;
            offset += canvases[z].shape[0];
        }
        var buffer = cat(canvases, 0);

        var zoom = choice.Zoom.flatten();
        var image = arange(batch, device: device).repeat_interleave(count);
        var factors = tensor(Zooms).Upload(device).index_select(0, zoom);
        var canvasSide = tensor(canvasSides).Upload(device).index_select(0, zoom);
        var left = (choice.X.flatten().to_type(ScalarType.Float32) * factors).round().to_type(ScalarType.Int64); // as Math.Round: half to even
        var top = (choice.Y.flatten().to_type(ScalarType.Float32) * factors).round().to_type(ScalarType.Int64);
        var plane = canvasSide * canvasSide;
        var start = tensor(bases).Upload(device).index_select(0, zoom) + image * 3 * plane + top * canvasSide + left; // [N]
        var steps = arange(Size, device: device);
        var index = start.reshape(-1, 1, 1, 1)
                    + arange(3, device: device).reshape(1, 3, 1, 1) * plane.reshape(-1, 1, 1, 1)
                    + steps.reshape(1, 1, Size, 1) * canvasSide.reshape(-1, 1, 1, 1)
                    + steps.reshape(1, 1, 1, Size);
        return buffer.index_select(0, index.flatten()).view(glimpses, 3, Size, Size).MoveToOuterDisposeScope();
    }

    /// <summary>
    /// [B, D, K] glimpse embeddings (zoom code added) and [B, K] which are real, for <paramref name="choice"/>'s windows: the
    /// same glimpses as the layout-based <see cref="Embed(ImageTower, Tensor, IReadOnlyList{LetterboxBox}, GlimpseWindow[], IReadOnlyList{int[]}, int, int)"/>,
    /// computed without reading the choice back: the crops are one gather from the zoom canvases, the content masks are
    /// computed on the device. The tower runs <paramref name="chunk"/> glimpses at a time; with gradients when the caller has them.
    /// </summary>
    /// <param name="pixels">[B, 3, S, S] canvases.</param>
    /// <param name="contentBoxes">[B, 4] (<see cref="ContentBoxes"/>); glimpses pool over content only, as the whole image does.</param>
    public static (Tensor Cells, Tensor Valid) Embed(ImageTower tower, Tensor pixels, Tensor contentBoxes, GlimpseChoice choice, int chunk = DefaultChunk)
    {
        if (tower.GlimpseZoomCodes is null)
            throw new InvalidOperationException("This tower has no glimpse zoom codes (--classification-glimpses).");
        using var scope = NewDisposeScope();
        var device = pixels.device;
        var (batch, count) = (choice.Zoom.shape[0], choice.Zoom.shape[1]);
        var glimpses = batch * count;
        var inputSize = pixels.shape[2];

        var crops = Crops(pixels, choice);
        var zoom = choice.Zoom.flatten();
        var image = arange(batch, device: device).repeat_interleave(count);

        // Each glimpse's content mask at the tower's map resolution: its cells' centres inside the image's content box.
        var mapSide = Size / tower.OutputStride;
        var side = Sides(device).index_select(0, zoom).to_type(ScalarType.Float32);
        var centres = (arange(mapSide, device: device).to_type(ScalarType.Float32) + 0.5f).unsqueeze(0) * (side / mapSide).unsqueeze(1); // [N, mapSide]
        var box = contentBoxes.index_select(0, image).to_type(ScalarType.Float32);
        var centreX = choice.X.flatten().to_type(ScalarType.Float32).unsqueeze(1) + centres;
        var centreY = choice.Y.flatten().to_type(ScalarType.Float32).unsqueeze(1) + centres;
        var insideX = centreX.ge(box[.., 0].unsqueeze(1)).logical_and(centreX.lt((box[.., 0] + box[.., 2]).unsqueeze(1)));
        var insideY = centreY.ge(box[.., 1].unsqueeze(1)).logical_and(centreY.lt((box[.., 1] + box[.., 3]).unsqueeze(1)));
        var masks = insideY.unsqueeze(2).logical_and(insideX.unsqueeze(1)).unsqueeze(1).to_type(ScalarType.Float32); // [N, 1, mapSide, mapSide]

        var pass = Math.Max(1, Math.Min(chunk, glimpses));
        var embedded = new List<Tensor>();
        for (long first = 0; first < glimpses; first += pass)
        {
            var length = Math.Min(pass, glimpses - first);
            embedded.Add(GlimpsePath(tower).ForwardPooledOnly(crops.narrow(0, first, length), masks.narrow(0, first, length)));
        }
        var flat = cat(embedded, 0) + tower.GlimpseZoomCodes.index_select(0, zoom);
        var cells = flat.view(batch, count, -1).transpose(1, 2);
        if (tower.GlimpseObjectOnly && tower.GlimpseObjectDims > 0)
            cells = ObjectOnly(cells, tower.GlimpseObjectDims, tower.GlimpseContextMean);
        // Windows of the zooms left out of pooling (ImageTower.GlimpsePoolMinZoom, GlimpsePoolZooms) are embedded but not real for pooling.
        var pooledValid = tower.GlimpsePoolZooms < Zooms.Length || tower.GlimpsePoolMinZoom > 0
            ? choice.Valid.logical_and(choice.Zoom.lt(tower.GlimpsePoolZooms)).logical_and(choice.Zoom.ge(tower.GlimpsePoolMinZoom))
            : choice.Valid.clone();
        return (cells.MoveToOuterDisposeScope(), pooledValid.MoveToOuterDisposeScope());
    }

    /// <summary>
    /// [B, D, K] <paramref name="cells"/> with the context half (dimensions from <paramref name="objectDims"/> on,
    /// <see cref="ImageTower.GlimpseObjectDims"/>) replaced by <paramref name="contextMean"/> (zeros if null): the feature split's
    /// scoring of exclusive images, whose gradient reaches the object half only.
    /// </summary>
    public static Tensor ObjectOnly(Tensor cells, int objectDims, Tensor contextMean)
    {
        var (batch, dim, count) = (cells.shape[0], cells.shape[1], cells.shape[2]);
        var context = contextMean.to_type(cells.dtype).view(1, -1, 1).expand(batch, dim - objectDims, count);
        return cat([cells.narrow(1, 0, objectDims), context], 1);
    }

    /// <inheritdoc cref="ObjectOnly(Tensor, int, Tensor)"/>
    public static Tensor ObjectOnly(Tensor cells, int objectDims, float[]? contextMean)
    {
        using var mean = contextMean is not null ? tensor(contextMean, device: cells.device) : zeros(cells.shape[1] - objectDims, device: cells.device);
        return ObjectOnly(cells, objectDims, mean);
    }

    /// <summary>
    /// [B, D, K] glimpse embeddings (zoom code added) and [B, K] which are real, for each image's
    /// <paramref name="windows"/> (indices into <paramref name="layout"/>; images with fewer than K are padded and
    /// marked invalid). The tower runs <paramref name="chunk"/> glimpses at a time; with gradients when the caller has them.
    /// </summary>
    /// <param name="pixels">[B, 3, S, S] canvases, S = the layout's input size.</param>
    /// <param name="content">Each image's content box; glimpses pool over content only, as the whole image does.</param>
    /// <param name="padTo">
    /// Run at least this many glimpses, padding with blank ones that are thrown away. Each tower pass then has the same
    /// batch size from call to call (the last chunk is padded to a full one too): on the B70 every new batch size compiles
    /// new oneDNN kernels for every convolution, ~2.5s, and with training's batch size changing every step (its sampling
    /// gives some images fewer windows) the kernel cache, ~7 batch sizes' worth, kept evicting them: 178 ms of a 426 ms step.
    /// </param>
    public static (Tensor Cells, Tensor Valid) Embed(
        ImageTower tower, Tensor pixels, IReadOnlyList<LetterboxBox> content, GlimpseWindow[] layout, IReadOnlyList<int[]> windows, int chunk = DefaultChunk,
        int padTo = 0)
    {
        if (tower.GlimpseZoomCodes is null)
            throw new InvalidOperationException("This tower has no glimpse zoom codes (--classification-glimpses).");
        var batch = (int)pixels.shape[0];
        var inputSize = (int)pixels.shape[2];
        // Padded too: [B, D, K]'s K sets the scoring matmuls' shapes.
        var k = Math.Max(Math.Max(1, windows.Max(w => w.Length)), padTo / batch);
        var dim = tower.EmbeddingDim;

        using var scope = NewDisposeScope();
        var crops = Crops(pixels, layout, windows, inputSize);

        // Each glimpse's content mask at the tower's map resolution, from its window's overlap with the content box.
        var mapSide = Size / tower.OutputStride;
        var masks = new float[crops.Count * mapSide * mapSide];
        var zoomIndex = new long[crops.Count];
        var n = 0;
        for (var b = 0; b < batch; b++)
            foreach (var g in windows[b])
            {
                var window = layout[g];
                zoomIndex[n] = window.Zoom;
                var cell = (double)window.Side / mapSide;
                var box = content[b];
                for (var y = 0; y < mapSide; y++)
                    for (var x = 0; x < mapSide; x++)
                    {
                        var (cx, cy) = (window.X + (x + 0.5) * cell, window.Y + (y + 0.5) * cell);
                        masks[(n * mapSide + y) * mapSide + x] = cx >= box.X && cx < box.X + box.Width && cy >= box.Y && cy < box.Y + box.Height ? 1 : 0;
                    }
                n++;
            }

        // Padded with copies of the first crop, masked out (they pool to zero), to a whole number of equal passes.
        var count = crops.Count;
        var padded = Math.Max(count, padTo);
        var pass = Math.Max(1, Math.Min(chunk, padded));
        padded = (padded + pass - 1) / pass * pass;
        while (count > 0 && crops.Count < padded)
            crops.Add(crops[0]);
        Array.Resize(ref masks, padded * mapSide * mapSide);

        // Crops are copied out of the zoom canvases a chunk at a time: all of a gold batch's at once is ~12K glimpses.
        var embedded = new List<Tensor>();
        var maskTensor = tensor(masks, [padded, 1, mapSide, mapSide], device: pixels.device);
        for (var start = 0; start < crops.Count; start += pass)
        {
            using var part = NewDisposeScope();
            var pooled = GlimpsePath(tower).ForwardPooledOnly(cat(crops.GetRange(start, pass), 0), maskTensor.narrow(0, start, pass));
            embedded.Add(pooled.MoveToOuterDisposeScope());
        }
        var flat = embedded.Count > 0 ? cat(embedded, 0).narrow(0, 0, count) : zeros(0, dim, device: pixels.device);
        flat = flat + tower.GlimpseZoomCodes.index_select(0, tensor(zoomIndex, device: pixels.device));

        // Scatter into [B, K, D], padded.
        var slots = new long[count];
        var valid = new bool[batch * k];
        n = 0;
        for (var b = 0; b < batch; b++)
            for (var j = 0; j < windows[b].Length; j++)
            {
                slots[n++] = b * k + j;
                valid[b * k + j] = true;
            }
        var cells = zeros(batch * k, dim, device: pixels.device).index_copy(0, tensor(slots, device: pixels.device), flat)
            .reshape(batch, k, dim).transpose(1, 2);
        var validTensor = tensor(valid, device: pixels.device).reshape(batch, k);
        return (cells.MoveToOuterDisposeScope(), validTensor.MoveToOuterDisposeScope());
    }

    /// <summary>The windows' crops at <see cref="Size"/>px, image by image in <paramref name="windows"/>' order.</summary>
    private static List<Tensor> Crops(Tensor pixels, GlimpseWindow[] layout, IReadOnlyList<int[]> windows, int inputSize)
    {
        // One resampled canvas per zoom that any image uses: box filter (exact block averages) to zoom out, bilinear to
        // zoom in; then every glimpse is a Size x Size slice of its zoom's canvas.
        var canvases = new Tensor?[Zooms.Length];
        Tensor Canvas(int zoom)
        {
            if (canvases[zoom] is { } ready)
                return ready;
            var factor = Zooms[zoom];
            var canvas = factor == 1 ? pixels
                : factor < 1 ? functional.avg_pool2d(pixels, (long)Math.Round(1 / factor))
                : functional.interpolate(pixels, scale_factor: [factor, factor], mode: InterpolationMode.Bilinear, align_corners: false);
            return canvases[zoom] = canvas;
        }

        var crops = new List<Tensor>();
        for (var b = 0; b < windows.Count; b++)
            foreach (var g in windows[b])
            {
                var window = layout[g];
                var factor = Zooms[window.Zoom];
                var (x, y) = ((int)Math.Round(window.X * factor), (int)Math.Round(window.Y * factor));
                crops.Add(Canvas(window.Zoom).narrow(0, b, 1).narrow(2, y, Size).narrow(3, x, Size));
            }
        return crops;
    }
}
