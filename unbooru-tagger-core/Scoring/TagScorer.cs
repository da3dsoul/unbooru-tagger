using System.Numerics.Tensors;
using UnbooruTagger.Core.Encoding;

namespace UnbooruTagger.Core.Scoring;

/// <summary>
/// SigLIP-style sigmoid scoring between image and tag embeddings: every image-tag
/// pair is an independent match/no-match decision (see CLAUDE.md's training objective).
/// </summary>
public static class TagScorer
{
    /// <summary>
    /// <c>detect</c>'s box extraction defaults, which <c>evaluate</c>'s gold box metrics use too. Before 2026-11-01 the
    /// evaluator kept every region (no minimum size or count), so on a 128x128 map each above-cutoff speck was a box: about
    /// 150-190 boxes against a couple of gold boxes per pair, box precision 0.1-0.3% in every run.
    /// </summary>
    /// <remarks>
    /// 2026-11-04: boxes are cut <see cref="DefaultBoxBelowPeak"/> log-odds below the tag's own peak, not at a share of its
    /// probability range. Drawn as regions over the gold images, the maps peak on the objects, but the probability cut kept about a
    /// fifth of the image for every tag, an earring as much as a coat. On run 4 at 2.12M over the gold set's object tags: box
    /// precision 0.08 -> 0.19, recall 0.12 -> 0.25, small objects 0 -> 0.11 / 0.14 (with the minimum region lowered from 0.2% to
    /// 0.05% of the map, below which a small object's region fell). Cuts of 2.5-3 suit small objects, 4-5 large ones.
    /// </remarks>
    public const float DefaultBoxThreshold = 0.5f;
    public const float DefaultBoxPercentile = 0.6f;
    public const float DefaultBoxBelowPeak = 4f;
    /// <summary>
    /// With a below-peak cut, a tag's further regions are boxed only when their own peak is within this many log-odds of the tag's
    /// peak. A 4-log-odds band also takes in faint secondary blobs that a heatmap drawn by opacity shows as nearly clear, and those
    /// made most of the boxes the user saw as worse than the maps (2026-11-04, `review/regions/tune3.py`/`tune4.py`, run 4 at 2.12M,
    /// 678 object-tag gold pairs): box precision 0.19 -> 0.27, recall 0.25 -> 0.22 at IoU 0.3 (0.06 -> 0.09 / 0.08 -> 0.07 at 0.5),
    /// boxes 1,660 -> 1,028. No cut or smoothing did much better (best F1 0.26 vs 0.24): a map's extent isn't its object's.
    /// </summary>
    public const float DefaultBoxPeakMargin = 1f;
    public const float DefaultMinHeatmapRange = 0.05f;
    public const float DefaultMinBoxFraction = 0.0005f;
    public const int DefaultMaxBoxesPerTag = 3;
    /// <summary>
    /// A character is boxed by <see cref="FigureBox"/>: its figure is the cells within this many log-odds of some other called tag's
    /// own peak. A character's map peaks on the face and marks who, not where the figure ends: the 4-below-peak cut boxed a face (box
    /// precision 0.17 -> 0.01), and no cut of the character's own map beat one box over the whole image on any model from run 1 to run
    /// 5 (2026-11-04, `review/regions/tune_chars.py`). Run 4 at 2.12M, 308 called character gold pairs (`tune_figure.py`): box F1 at
    /// IoU 0.5 0.39 (whole image 0.33, own map 0.00); solo images 89% found, several characters 31%, where touching figures merge.
    /// On an image with several characters that figure takes in everyone touching: 96% of those boxes covered another character's
    /// gold box, 87% held another character's peak. So unless the model calls "solo", the figure is split between the called
    /// characters, each cell to the nearest character's peak along the figure (<see cref="DefaultSamePersonDistance"/>): 2 gold
    /// characters found 0.48 -> 0.54, another character's peak inside the box 0.85 -> 0.26; 3-5 characters 0.09 -> 0.25 and 0.85 ->
    /// 0.22; solo images 0.80 -> 0.78. Split on every image it cost solo images 13 points (a wrongly called second character peaks on
    /// the same figure), and split by straight-line distance or by the characters' own maps it did worse.
    /// </summary>
    public const float DefaultFigureBand = 3f;

    /// <summary>
    /// Characters whose peaks are closer than this share of the map's width are one person in <see cref="FigureBox"/>'s split (48px of
    /// a 512px input): a character and its costume variant (gold ship and gold ship (red strife)) are called on the same face.
    /// </summary>
    public const float DefaultSamePersonDistance = 12f / 128;

    /// <summary>Whether <see cref="FigureBox"/>, not <see cref="DetectBoxes"/>, boxes <paramref name="tag"/>.</summary>
    public static bool IsCharacter(string tag) => tag.StartsWith("character:", StringComparison.Ordinal);

    /// <summary>Confidence that <paramref name="tagEmbedding"/> applies to the whole image. <paramref name="scaling"/> defaults to <see cref="LogitScaling.Identity"/> (plain <c>sigmoid(dot)</c>) for a model bundle exported before <see cref="LogitScaling"/> existed.</summary>
    public static float Score(ReadOnlySpan<float> imageEmbedding, ReadOnlySpan<float> tagEmbedding, LogitScaling? scaling = null) =>
        Sigmoid((scaling ?? LogitScaling.Identity).Apply(Dot(imageEmbedding, tagEmbedding)));

    /// <summary>
    /// A rough localization heatmap for one tag: sigmoid(scale * dot(tag, spatial location) + bias) at
    /// every position in the pre-pool feature map. No CAM/Grad-CAM step needed — this
    /// falls directly out of the dual-encoder design (MaskCLIP-style).
    /// </summary>
    public static float[,] Heatmap(ReadOnlySpan<float> tagEmbedding, SpatialFeatureMap spatialFeatures, LogitScaling? scaling = null)
    {
        var resolvedScaling = scaling ?? LogitScaling.Identity;
        var heatmap = new float[spatialFeatures.Height, spatialFeatures.Width];
        for (var y = 0; y < spatialFeatures.Height; y++)
        for (var x = 0; x < spatialFeatures.Width; x++)
            heatmap[y, x] = Sigmoid(resolvedScaling.Apply(Dot(tagEmbedding, spatialFeatures[y, x])));

        return heatmap;
    }

    /// <summary>
    /// Rough bounding boxes for one tag: thresholds its <see cref="Heatmap"/> and groups
    /// contiguous above-threshold grid cells (4-connectivity) into boxes. <paramref name="heatmap"/>
    /// covers a <paramref name="canvasSize"/> x <paramref name="canvasSize"/> letterboxed
    /// canvas (padding bars included), so each box is first mapped from grid space to that
    /// canvas, then from <paramref name="content"/> (where the real image sits within it) to
    /// <paramref name="imageWidth"/> x <paramref name="imageHeight"/> pixels — the original
    /// image's own space. Approximate localization, not a trained detector — see CLAUDE.md.
    /// </summary>
    /// <param name="threshold">Absolute sigmoid floor — a tag whose whole heatmap sits below this never gets a box, regardless of <paramref name="relativePercentile"/>.</param>
    /// <param name="relativePercentile">
    /// Where to cut within THIS tag's own heatmap range, from 0 (its weakest cell) to 1 (only
    /// its single strongest cell). The effective cutoff is the higher of this and
    /// <paramref name="threshold"/>, so this can only tighten boxes, never loosen them — it
    /// keeps only the locations near a tag's own peak instead of everything that happens to
    /// clear the global absolute bar.
    /// </param>
    /// <param name="content">Where the original image's real content sits within the letterboxed canvas, in canvas pixel space (0..<paramref name="canvasSize"/>).</param>
    /// <param name="canvasSize">The side length of the square letterboxed canvas <paramref name="heatmap"/>'s grid covers.</param>
    /// <param name="minRange">
    /// A tag whose heatmap's own max-min spread sits below this is treated as carrying no
    /// usable spatial signal (an under-discriminated tag scores near-uniformly everywhere,
    /// not because it genuinely applies everywhere) and gets no boxes at all, regardless of
    /// <paramref name="threshold"/>/<paramref name="relativePercentile"/>. Without this, a
    /// flat heatmap either clears <paramref name="threshold"/> almost everywhere (one
    /// box spanning most of the canvas) or, after <see cref="HeatmapRefiner.Refine"/>'s
    /// upsample+bilateral-filter, fragments into small islands of interpolation/filter
    /// noise near the (meaningless) percentile cutoff — both worse than reporting nothing.
    /// Trade-off: a tag that genuinely, legitimately covers almost the whole image (e.g. a
    /// background tag on a mostly-background composition) can also have a small range and
    /// gets suppressed the same way — hence a small default rather than an aggressive one.
    /// </param>
    /// <param name="minComponentFraction">Drops connected regions covering fewer than this fraction of the grid's cells — isolated specks, not localization. 0 keeps every region.</param>
    /// <param name="maxBoxes">Keeps only this many boxes, highest peak confidence first. 0 keeps every box.</param>
    /// <param name="belowPeak">When above 0, cut this many log-odds below the heatmap's peak instead of at
    /// <paramref name="relativePercentile"/> of its probability range (<paramref name="threshold"/> then only decides whether the
    /// tag gets boxes at all: its peak must reach it).</param>
    /// <param name="peakMargin">With <paramref name="belowPeak"/>, keeps only regions whose own peak is within this many log-odds of
    /// the tag's peak (<see cref="DefaultBoxPeakMargin"/>). 0 keeps every region.</param>
    public static List<BoundingBox> DetectBoxes(
        float[,] heatmap, float threshold, float relativePercentile,
        LetterboxBox content, int canvasSize, int imageWidth, int imageHeight, float minRange = 0.05f,
        float minComponentFraction = 0f, int maxBoxes = 0, float belowPeak = 0f, float peakMargin = 0f)
    {
        var height = heatmap.GetLength(0);
        var width = heatmap.GetLength(1);

        var min = float.MaxValue;
        var max = float.MinValue;
        foreach (var value in heatmap)
        {
            min = Math.Min(min, value);
            max = Math.Max(max, value);
        }

        if (max - min < minRange)
            return [];

        float effectiveThreshold;
        var regionFloor = float.MinValue;
        if (belowPeak > 0)
        {
            if (max < threshold)
                return [];
            var peak = Math.Clamp(max, 1e-7f, 1 - 1e-7f);
            effectiveThreshold = Sigmoid(MathF.Log(peak / (1 - peak)) - belowPeak);
            if (peakMargin > 0)
                regionFloor = Sigmoid(MathF.Log(peak / (1 - peak)) - peakMargin);
        }
        else
        {
            effectiveThreshold = Math.Max(threshold, min + relativePercentile * (max - min));
        }

        var visited = new bool[height, width];

        // Grid cell -> canvas pixel, then canvas pixel -> original image pixel: canvas
        // pixels inside `content` map linearly onto the original image; anything outside
        // it (the letterbox bars) clamps to the image's own edge instead of overshooting
        // past it or going negative.
        var gridToCanvasX = canvasSize / (float)width;
        var gridToCanvasY = canvasSize / (float)height;
        var canvasToImageX = imageWidth / (float)content.Width;
        var canvasToImageY = imageHeight / (float)content.Height;

        var minCells = (int)Math.Ceiling(minComponentFraction * width * height);

        var boxes = new List<BoundingBox>();
        for (var y = 0; y < height; y++)
        for (var x = 0; x < width; x++)
        {
            if (visited[y, x] || heatmap[y, x] < effectiveThreshold)
                continue;

            var (box, cellCount) = FloodFillComponent(
                heatmap, visited, x, y, effectiveThreshold,
                gridToCanvasX, gridToCanvasY, canvasToImageX, canvasToImageY, content, imageWidth, imageHeight);
            if (cellCount >= minCells && box.Confidence >= regionFloor)
                boxes.Add(box);
        }

        if (maxBoxes > 0 && boxes.Count > maxBoxes)
            boxes = boxes.OrderByDescending(b => b.Confidence).Take(maxBoxes).ToList();

        return boxes;
    }

    /// <summary>
    /// One box for a character (<see cref="DefaultFigureBand"/>): the connected region holding the character's peak, over the cells
    /// within <paramref name="ownBand"/> log-odds of its own peak or within <paramref name="figureBand"/> of the peak of any of
    /// <paramref name="figureHeatmaps"/> (the other localizable tags the model calls on the image, same grid). With none, the whole
    /// image. With <paramref name="otherCharacters"/> (the other called characters' maps; pass none on an image the model calls solo),
    /// the figure is split first: each cell goes to whichever peak a path through the figure reaches first, this character's and those
    /// within <paramref name="samePersonDistance"/> of it against the rest. Same spaces, <paramref name="threshold"/> and
    /// <paramref name="minRange"/> as <see cref="DetectBoxes"/>; the box's confidence is the character's peak.
    /// </summary>
    public static List<BoundingBox> FigureBox(
        float[,] heatmap, IReadOnlyList<float[,]> figureHeatmaps, float threshold,
        LetterboxBox content, int canvasSize, int imageWidth, int imageHeight, float minRange = DefaultMinHeatmapRange,
        float ownBand = DefaultBoxBelowPeak, float figureBand = DefaultFigureBand,
        IReadOnlyList<float[,]>? otherCharacters = null, float samePersonDistance = DefaultSamePersonDistance)
    {
        var height = heatmap.GetLength(0);
        var width = heatmap.GetLength(1);
        var (min, max, peakX, peakY) = PeakOf(heatmap);
        if (max - min < minRange || max < threshold)
            return [];

        // Cuts in log-odds below a map's peak, compared as probabilities (the sigmoid is monotone).
        static float CutBelow(float peak, float band)
        {
            var p = Math.Clamp(peak, 1e-7f, 1 - 1e-7f);
            return Sigmoid(MathF.Log(p / (1 - p)) - band);
        }
        var mask = new float[height, width];
        var ownCut = CutBelow(max, ownBand);
        for (var y = 0; y < height; y++)
        for (var x = 0; x < width; x++)
            mask[y, x] = figureHeatmaps.Count == 0 || heatmap[y, x] >= ownCut ? 1 : 0;
        foreach (var figure in figureHeatmaps)
        {
            if (figure.GetLength(0) != height || figure.GetLength(1) != width)
                throw new ArgumentException("Figure heatmaps must share the character heatmap's grid.", nameof(figureHeatmaps));
            var figureMax = float.MinValue;
            foreach (var value in figure)
                figureMax = Math.Max(figureMax, value);
            var cut = CutBelow(figureMax, figureBand);
            for (var y = 0; y < height; y++)
            for (var x = 0; x < width; x++)
                if (figure[y, x] >= cut)
                    mask[y, x] = 1;
        }

        if (otherCharacters is { Count: > 0 })
            SplitFigure(mask, peakX, peakY, otherCharacters, samePersonDistance * width);
        mask[peakY, peakX] = 1;

        var (box, _) = FloodFillComponent(
            mask, new bool[height, width], peakX, peakY, 0.5f,
            canvasSize / (float)width, canvasSize / (float)height, imageWidth / (float)content.Width, imageHeight / (float)content.Height,
            content, imageWidth, imageHeight);
        return [box with { Confidence = max }];
    }

    private static (float Min, float Max, int X, int Y) PeakOf(float[,] map)
    {
        var (min, max, peakX, peakY) = (float.MaxValue, float.MinValue, 0, 0);
        for (var y = 0; y < map.GetLength(0); y++)
        for (var x = 0; x < map.GetLength(1); x++)
        {
            min = Math.Min(min, map[y, x]);
            if (map[y, x] > max)
                (max, peakX, peakY) = (map[y, x], x, y);
        }
        return (min, max, peakX, peakY);
    }

    /// <summary>
    /// Clears the mask's cells that a breadth-first flood through the mask reaches from another person's peak before this one's (ties
    /// to this one). The other characters' peaks within <paramref name="samePerson"/> cells of (peakX, peakY) seed this person.
    /// </summary>
    private static void SplitFigure(float[,] mask, int peakX, int peakY, IReadOnlyList<float[,]> otherCharacters, float samePerson)
    {
        var height = mask.GetLength(0);
        var width = mask.GetLength(1);
        var owner = new int[height, width]; // 0 unreached, 1 this person, 2 someone else
        var queue = new Queue<(int X, int Y)>();
        var seeds = new List<(int X, int Y, int Owner)> { (peakX, peakY, 1) };
        foreach (var other in otherCharacters)
        {
            if (other.GetLength(0) != height || other.GetLength(1) != width)
                throw new ArgumentException("Character heatmaps must share the character heatmap's grid.", nameof(otherCharacters));
            var (_, _, x, y) = PeakOf(other);
            seeds.Add((x, y, MathF.Sqrt((x - peakX) * (x - peakX) + (y - peakY) * (y - peakY)) < samePerson ? 1 : 2));
        }
        foreach (var (x, y, who) in seeds.OrderBy(s => s.Owner))
            if (owner[y, x] == 0)
            {
                owner[y, x] = who;
                queue.Enqueue((x, y));
            }
        while (queue.TryDequeue(out var cell))
        {
            var (x, y) = cell;
            foreach (var (nx, ny) in new[] { (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1) })
                if (nx >= 0 && nx < width && ny >= 0 && ny < height && owner[ny, nx] == 0 && mask[ny, nx] > 0)
                {
                    owner[ny, nx] = owner[y, x];
                    queue.Enqueue((nx, ny));
                }
        }
        for (var y = 0; y < height; y++)
        for (var x = 0; x < width; x++)
            if (owner[y, x] != 1)
                mask[y, x] = 0;
    }

    /// <summary>BFS over 4-connected above-threshold grid cells starting at (startX, startY), returning their pixel-space bounding box and how many grid cells the region covers.</summary>
    private static (BoundingBox Box, int CellCount) FloodFillComponent(
        float[,] heatmap, bool[,] visited, int startX, int startY, float threshold,
        float gridToCanvasX, float gridToCanvasY, float canvasToImageX, float canvasToImageY,
        LetterboxBox content, int imageWidth, int imageHeight)
    {
        var height = heatmap.GetLength(0);
        var width = heatmap.GetLength(1);

        var queue = new Queue<(int X, int Y)>();
        queue.Enqueue((startX, startY));
        visited[startY, startX] = true;

        int minX = startX, maxX = startX, minY = startY, maxY = startY;
        var peakConfidence = heatmap[startY, startX];
        var cellCount = 0;

        while (queue.TryDequeue(out var cell))
        {
            cellCount++;
            var (x, y) = cell;
            minX = Math.Min(minX, x);
            maxX = Math.Max(maxX, x);
            minY = Math.Min(minY, y);
            maxY = Math.Max(maxY, y);
            peakConfidence = Math.Max(peakConfidence, heatmap[y, x]);

            foreach (var (nx, ny) in new[] { (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1) })
            {
                if (nx < 0 || nx >= width || ny < 0 || ny >= height)
                    continue;
                if (visited[ny, nx] || heatmap[ny, nx] < threshold)
                    continue;

                visited[ny, nx] = true;
                queue.Enqueue((nx, ny));
            }
        }

        var canvasLeft = minX * gridToCanvasX;
        var canvasTop = minY * gridToCanvasY;
        var canvasRight = MathF.Ceiling((maxX + 1) * gridToCanvasX);
        var canvasBottom = MathF.Ceiling((maxY + 1) * gridToCanvasY);

        var pixelX = (int)Math.Clamp((canvasLeft - content.X) * canvasToImageX, 0, imageWidth);
        var pixelY = (int)Math.Clamp((canvasTop - content.Y) * canvasToImageY, 0, imageHeight);
        var pixelRight = (int)Math.Clamp(MathF.Ceiling((canvasRight - content.X) * canvasToImageX), 0, imageWidth);
        var pixelBottom = (int)Math.Clamp(MathF.Ceiling((canvasBottom - content.Y) * canvasToImageY), 0, imageHeight);

        return (new BoundingBox(pixelX, pixelY, pixelRight - pixelX, pixelBottom - pixelY, peakConfidence), cellCount);
    }

    private static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vectors must be the same length to dot-product.");

        // SIMD: a 128x128 heatmap is 16K of these per tag at 512 dims.
        return TensorPrimitives.Dot(a, b);
    }

    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));
}
