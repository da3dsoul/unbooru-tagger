using System.Globalization;
using SkiaSharp;
using UnbooruTagger.Training.Checkpoints;
using UnbooruTagger.Training.Training;

namespace UnbooruTagger.Training.Commands;

/// <summary>Everything the charts show for one run, read from its checkpoint directory.</summary>
/// <param name="CurrentStep">Global step of the latest checkpoint save.</param>
/// <param name="References">Other runs' evaluations to compare against (e.g. "v3 e1"), drawn as dashed lines.</param>
/// <param name="Comparisons">Other runs drawn as curves over the same steps (e.g. run 4 beside a fresh run testing a change): their evaluations up to this run's latest one, their loss up to the current step.</param>
public sealed record TrainingSeries(
    string Name, LossHistory Loss, IReadOnlyList<EvaluationPoint> Points, long CurrentStep, int StepsPerEpoch, int Epochs,
    IReadOnlyList<(string Label, EvaluationPoint Point)> References,
    IReadOnlyList<(string Label, TrainingSeries Run)>? Comparisons = null)
{
    /// <summary>
    /// Reads <paramref name="checkpointDir"/>: the saved loss history, every epoch evaluation
    /// (history/epoch-N) and finished snapshot evaluation (snapshots/step-N).
    /// </summary>
    /// <param name="comparisons">Other runs' checkpoint directories, read the same way, drawn as curves up to this run's step.</param>
    public static TrainingSeries Load(string checkpointDir, int epochs, IReadOnlyList<(string Label, string Directory)>? references = null,
        IReadOnlyList<(string Label, string Directory)>? comparisons = null)
    {
        var progress = TrainingState.HasProgress(checkpointDir) ? TrainingState.LoadProgress(checkpointDir) : TrainingProgress.Initial;
        var stepsPerEpoch = Math.Max(1, progress.StepsPerEpoch);
        var currentStep = (long)progress.CompletedEpochs * stepsPerEpoch + progress.CompletedStepsInEpoch;

        var points = new List<EvaluationPoint>();
        for (var e = 1; e <= progress.CompletedEpochs; e++)
            if (EvaluationPoint.Read(Path.Combine(checkpointDir, "history", $"epoch-{e}"), (long)e * stepsPerEpoch, endOfEpoch: true) is { } point)
                points.Add(point);
        var snapshots = Path.Combine(checkpointDir, SnapshotEvaluations.DirectoryName);
        if (Directory.Exists(snapshots))
            foreach (var dir in Directory.GetDirectories(snapshots, "step-*"))
                if (long.TryParse(Path.GetFileName(dir)["step-".Length..], out var step) && step <= currentStep
                    && File.Exists(Path.Combine(dir, OffloadedEvaluations.DoneMarker))
                    && EvaluationPoint.Read(dir, step, endOfEpoch: false) is { } point)
                    points.Add(point);

        var referencePoints = (references ?? [])
            .Select(r => (r.Label, Point: EvaluationPoint.Read(r.Directory, 0, endOfEpoch: true)))
            .Where(r => r.Point is not null)
            .Select(r => (r.Label, r.Point!))
            .ToList();
        var name = Path.GetFileName(Path.TrimEndingDirectorySeparator(Path.GetFullPath(checkpointDir)));
        return new TrainingSeries(name, LossHistory.Load(checkpointDir, currentStep), points.OrderBy(p => p.Step).ToList(),
            currentStep, stepsPerEpoch, epochs, referencePoints,
            (comparisons ?? []).Select(c => (c.Label, Load(c.Directory, epochs))).ToList());
    }
}

/// <summary>
/// Draws a run's <see cref="TrainingSeries"/> as one image: a grid of panels, the loss over the whole run
/// first, then one per headline evaluation metric, with epoch boundaries and other runs' values for
/// comparison. Shown in the terminal as Sixel by the <c>charts</c> command, or saved as a PNG.
/// </summary>
public static class TrainingCharts
{
    private static readonly SKColor Background = new(0x17, 0x19, 0x1e);
    private static readonly SKColor Panel = new(0x20, 0x23, 0x2a);
    private static readonly SKColor Grid = new(0x33, 0x37, 0x40);
    private static readonly SKColor Axis = new(0x8a, 0x90, 0x9c);
    private static readonly SKColor Text = new(0xe6, 0xe8, 0xec);
    private static readonly SKColor Better = new(0x4c, 0xc3, 0x7a);
    private static readonly SKColor Worse = new(0xf0, 0x5d, 0x5d);
    private static readonly SKColor LossColor = new(0x4f, 0xb3, 0xe8);
    private static readonly SKColor MetricColor = new(0xf2, 0xb1, 0x4c);
    private static readonly SKColor ReferenceColor = new(0xb4, 0x8e, 0xe0);

    /// <param name="Split">Draw these lines instead of <paramref name="Value"/>'s (which still gives the title's latest value and change),
    /// each with an optional faint dashed companion (<c>Tolerant</c>); other runs and references aren't drawn in such a panel.</param>
    private sealed record Metric(string Name, Func<EvaluationPoint, double> Value, bool LowerIsBetter, bool Percent, SplitLine[]? Split = null);

    private sealed record SplitLine(string Label, Func<EvaluationPoint, double> Value, Func<EvaluationPoint, double>? Tolerant, SKColor Color);

    // Gold pointing by object size: the overall rate is a mix of sizes (characters ~90%, small objects ~38%), so it hides where it moves.
    private static readonly SplitLine[] PointingBySize =
    [
        .. EvaluationPoint.SizeRows.Zip(new SKColor[] { new(0x4f, 0xb3, 0xe8), new(0x4c, 0xc3, 0x7a), new(0xf2, 0xb1, 0x4c), new(0xf0, 0x5d, 0x5d) },
            (size, color) => new SplitLine(size.Short, p => EvaluationPoint.Get(p.GoldPointingBySize, size.Short), p => EvaluationPoint.Get(p.GoldTolerantBySize, size.Short), color)),
    ];

    private static readonly SKColor[] SizeColors = [new(0x4f, 0xb3, 0xe8), new(0x4c, 0xc3, 0x7a), new(0xf2, 0xb1, 0x4c), new(0xf0, 0x5d, 0x5d)];

    // The same at 1024px (HigherScalePoint), for mixed-scale training: the gold sets from their originals, and the SAM 3 set.
    private static readonly SplitLine[] PointingBySize1024 =
    [
        .. EvaluationPoint.SizeRows.Zip(SizeColors,
            (size, color) => new SplitLine(size.Short, p => EvaluationPoint.Get(p.At1024?.GoldBySize, size.Short), p => EvaluationPoint.Get(p.At1024?.GoldTolerantBySize, size.Short), color)),
    ];

    private static readonly SplitLine[] AutoPointingBySize1024 =
    [
        .. EvaluationPoint.SizeRows.Skip(1).Zip(SizeColors.Skip(1),
            (size, color) => new SplitLine(size.Short, p => EvaluationPoint.Get(p.At1024?.AutoBySize, size.Short), p => EvaluationPoint.Get(p.At1024?.AutoTolerantBySize, size.Short), color)),
    ];

    // Region precision by size, recall dashed (GoldResult.RegionPrecision: the map within 4 log-odds of its peak, what detect's boxes
    // are cut from): whether the map's hot region is the object, without the box extraction in between.
    private static readonly SplitLine[] RegionBySize =
    [
        .. EvaluationPoint.SizeRows.Skip(1).Zip(SizeColors.Skip(1),
            (size, color) => new SplitLine(size.Short, p => EvaluationPoint.Get(p.RegionPrecisionBySize, size.Short), p => EvaluationPoint.Get(p.RegionRecallBySize, size.Short), color)),
    ];

    private static readonly SplitLine[] AutoRegionBySize1024 =
    [
        .. EvaluationPoint.SizeRows.Skip(1).Zip(SizeColors.Skip(1),
            (size, color) => new SplitLine(size.Short, p => EvaluationPoint.Get(p.At1024?.AutoRegionPrecisionBySize, size.Short), p => EvaluationPoint.Get(p.At1024?.AutoRegionRecallBySize, size.Short), color)),
    ];

    private static readonly Metric[] Metrics =
    [
        new("held-out mAP", p => p.MeanAveragePrecision, false, false),
        new("gold pointing by size", p => p.GoldPointing, false, true, PointingBySize),
        new("gold-small pointing", p => p.SmallGoldPointing, false, true),
        new("exclusive-tag violations", p => p.ExclusiveViolations, true, true),
        // Object effect, not the still-called rate: that one counts only pairs found at p >= 0.5 and flips on small
        // score changes near 0.5 (a quarter of its pairs change between checkpoints 4,500 steps apart).
        new("object effect (log-odds)", p => p.ObjectEffect, false, false),
        new("gold-small object effect", p => p.SmallObjectEffect, false, false),
        new("context recall gap", p => p.RecallGap, true, true),
        new("context false-positive gap", p => p.FalsePositiveGap, true, true),
        // The gap counts untagged objects the model finds as false positives; these are images checked by eye. Log-odds
        // past the threshold, not the called rate, for the same reason as the object effect: ~200 pairs at thresholds
        // recalibrated each evaluation, so a few pairs flipping moved the rate by points.
        new("absent objects (labelled)", p => p.LabelledAbsentLogit, true, false),
        new("untagged objects (labelled)", p => p.LabelledPresentLogit, false, false),
        new("gold pointing by size @1024", p => p.At1024?.GoldPointing ?? double.NaN, false, true, PointingBySize1024),
        new("gold-auto pointing by size @1024", p => p.At1024?.AutoPointing ?? double.NaN, false, true, AutoPointingBySize1024),
        new("gold-small pointing @1024", p => p.At1024?.SmallGoldPointing ?? double.NaN, false, true),
        new("object effect @1024 (log-odds)", p => p.At1024?.ObjectEffect ?? double.NaN, false, false),
        new("gold region precision by size (dashed: recall)", p => EvaluationPoint.Get(p.RegionPrecisionBySize, "medium"), false, true, RegionBySize),
        new("gold-auto region precision @1024 (dashed: recall)", p => EvaluationPoint.Get(p.At1024?.AutoRegionPrecisionBySize, "medium"), false, true, AutoRegionBySize1024),
    ];

    /// <summary>Panels per row; the loss takes the first two.</summary>
    private const int Columns = 4;

    private const int LossColumns = 2;

    /// <summary>A snapshot evaluation's dot radius (before scaling): just wider than the line, so a point shows without hiding it.</summary>
    private const float SnapshotDot = 1.5f;

    /// <param name="header">A title line with the run and step; the training dashboard leaves it out, its panel already says so.</param>
    /// <param name="textScale">Text and line size (1 = 13 px labels); in a terminal, matched to its own text. Default: from the height.</param>
    public static SKBitmap Render(TrainingSeries series, int width, int height, DateTime now, bool header = true, float? textScale = null)
    {
        var bitmap = new SKBitmap(width, height, SKColorType.Rgba8888, SKAlphaType.Premul);
        using var canvas = new SKCanvas(bitmap);
        canvas.Clear(Background);
        var scale = textScale ?? Math.Clamp(height / 900f, 0.6f, 2f);
        using var typeface = SKTypeface.FromFamilyName("DejaVu Sans") ?? SKTypeface.Default;
        using var font = new SKFont(typeface, 13 * scale);
        using var bold = new SKFont(SKTypeface.FromFamilyName("DejaVu Sans", SKFontStyle.Bold) ?? typeface, 15 * scale);
        using var big = new SKFont(SKTypeface.FromFamilyName("DejaVu Sans", SKFontStyle.Bold) ?? typeface, 17 * scale);

        var margin = 12 * scale;
        var top = margin;
        if (header)
        {
            var epoch = series.CurrentStep / series.StepsPerEpoch + 1;
            Draw(canvas, $"{series.Name} · epoch {Math.Min(epoch, series.Epochs)}/{series.Epochs} · step {series.CurrentStep:N0} of {(long)series.Epochs * series.StepsPerEpoch:N0}",
                margin, margin + 16 * scale, big, Text);
            Draw(canvas, $"{series.Points.Count} evaluations · updated {now:HH:mm}", width - margin, margin + 16 * scale, font, Axis, SKTextAlign.Right);
            top += 26 * scale;
        }
        var gap = 10 * scale;
        var rows = (Metrics.Length + LossColumns + Columns - 1) / Columns;
        var cellWidth = (width - 2 * margin - (Columns - 1) * gap) / Columns;
        var cellHeight = (height - margin - top - (rows - 1) * gap) / rows;
        SKRect Cell(int k) => SKRect.Create(margin + k % Columns * (cellWidth + gap), top + k / Columns * (cellHeight + gap), cellWidth, cellHeight);
        DrawLoss(canvas, series, SKRect.Union(Cell(0), Cell(LossColumns - 1)), scale, font, bold);

        // Evaluation panels share one x range, so the same step lines up across them.
        var steps = series.Points.Select(p => (double)p.Step).ToList();
        // From the run's first loss bucket (its start, or where it resumed from), so the first evaluation isn't on the edge.
        var start = series.Loss.Buckets.Count > 0 ? series.Loss.Buckets[0].End - series.Loss.BucketSteps : series.CurrentStep;
        var xMin = Math.Max(0, Math.Min(start, steps.Count > 0 ? steps.Min() : series.CurrentStep));
        var xMax = Math.Max(series.CurrentStep, steps.Count > 0 ? steps.Max() : 1);
        for (var k = 0; k < Metrics.Length; k++)
            DrawMetric(canvas, series, Metrics[k], Cell(k + LossColumns), xMin, xMax, scale, font, bold);
        return bitmap;
    }

    public static void SavePng(SKBitmap bitmap, string path)
    {
        using var image = SKImage.FromBitmap(bitmap);
        using var data = image.Encode(SKEncodedImageFormat.Png, 100);
        using var file = File.Create(path);
        data.SaveTo(file);
    }

    private static void DrawLoss(SKCanvas canvas, TrainingSeries series, SKRect box, float scale, SKFont font, SKFont bold)
    {
        var buckets = series.Loss.Buckets;
        var title = $"loss · per {TrainingDashboard.Steps(series.Loss.BucketSteps)} steps";
        var plot = PanelFrame(canvas, box, scale, title, buckets.Count > 0 ? buckets[^1].Mean.ToString("G4", CultureInfo.InvariantCulture) : null, null, false, font, bold);
        if (buckets.Count < 2)
        {
            Draw(canvas, "the loss history fills in as checkpoints are saved", plot.MidX, plot.MidY, font, Axis, SKTextAlign.Center);
            return;
        }

        // Raw bucket means, and a smoothed line over them: the trend the noise hides.
        var raw = buckets.Select(b => ((double)b.End, (double)b.Mean)).ToList();
        var smoothed = Smooth(raw);
        var (xMin, xMax) = (raw[0].Item1, Math.Max(raw[^1].Item1, series.CurrentStep));
        // Comparison runs over this run's steps, smoothed only (their raw buckets would bury this run's).
        var others = (series.Comparisons ?? [])
            .Select(c => (c.Label, Points: Smooth(c.Run.Loss.Buckets.Select(b => ((double)b.End, (double)b.Mean)).ToList()).Where(p => p.X >= xMin && p.X <= xMax).ToList()))
            .Where(c => c.Points.Count > 1).ToList();
        var (yMin, yMax) = Range(raw.Select(p => p.Item2).Concat(others.SelectMany(c => c.Points.Select(p => p.Y))), false);
        var area = Axes(canvas, plot, xMin, xMax, yMin, yMax, false, series, scale, font);
        foreach (var (label, points) in others)
            Comparison(canvas, area, label, points, xMin, xMax, yMin, yMax, scale, font);
        Line(canvas, area, raw, xMin, xMax, yMin, yMax, LossColor.WithAlpha(90), 1.2f * scale);
        Line(canvas, area, smoothed, xMin, xMax, yMin, yMax, LossColor, 2.4f * scale);
    }

    private static void DrawMetric(SKCanvas canvas, TrainingSeries series, Metric metric, SKRect box, double xMin, double xMax, float scale, SKFont font, SKFont bold)
    {
        var values = series.Points.Select(p => ((double)p.Step, metric.Value(p), p.EndOfEpoch)).Where(p => !double.IsNaN(p.Item2)).ToList();
        string? latest = values.Count > 0 ? Format(values[^1].Item2, metric.Percent) : null;
        (string, string, SKColor)? change = null;
        if (values.Count > 1)
        {
            var delta = values[^1].Item2 - values[^2].Item2;
            var better = metric.LowerIsBetter ? delta < 0 : delta > 0;
            var magnitude = metric.Percent ? (100 * Math.Abs(delta)).ToString("0.0", CultureInfo.InvariantCulture) + " pts" : Format(Math.Abs(delta), false);
            change = ($"{(delta >= 0 ? "+" : "−")}{magnitude}", $" since {TrainingDashboard.Steps(values[^2].Item1)}", delta == 0 ? Axis : better ? Better : Worse);
        }
        var plot = PanelFrame(canvas, box, scale, metric.Name, latest, change, metric.LowerIsBetter, font, bold);
        if (values.Count == 0)
        {
            Draw(canvas, "no evaluation yet", plot.MidX, plot.MidY, font, Axis, SKTextAlign.Center);
            return;
        }

        if (metric.Split is { } split)
        {
            DrawSplit(canvas, series, metric, split, plot, xMin, xMax, scale, font);
            return;
        }
        var references = series.References.Select(r => (r.Label, Value: metric.Value(r.Point))).Where(r => !double.IsNaN(r.Value)).ToList();
        // Other runs only up to this run's latest evaluation, not the current step: this run's evaluations land a while
        // after their step (they run in another process), so otherwise the other run's newer points come in alone.
        var compareTo = Math.Min(xMax, values[^1].Item1);
        var others = (series.Comparisons ?? [])
            .Select(c => (c.Label, Points: c.Run.Points.Where(p => p.Step >= xMin && p.Step <= compareTo).Select(p => ((double)p.Step, metric.Value(p)))
                .Where(p => !double.IsNaN(p.Item2)).ToList()))
            .Where(c => c.Points.Count > 0).ToList();
        var (yMin, yMax) = Range(values.Select(v => v.Item2).Concat(references.Select(r => r.Value)).Concat(others.SelectMany(c => c.Points.Select(p => p.Item2))),
            metric.Percent);
        var area = Axes(canvas, plot, xMin, xMax, yMin, yMax, metric.Percent, series, scale, font);

        using var dashed = new SKPaint { Color = ReferenceColor.WithAlpha(170), StrokeWidth = 1.2f * scale, IsAntialias = true, Style = SKPaintStyle.Stroke,
            PathEffect = SKPathEffect.CreateDash([6 * scale, 4 * scale], 0) };
        using var small = new SKFont(font.Typeface, font.Size * 0.85f);
        // Labels of lines close together go side by side instead of on top of each other.
        var placed = new List<(float Y, float Right)>();
        foreach (var (label, value) in references.OrderBy(r => r.Value))
        {
            var y = Y(area, value, yMin, yMax);
            canvas.DrawLine(area.Left, y, area.Right, y, dashed);
            var x = area.Left + 4 * scale;
            foreach (var p in placed.Where(p => Math.Abs(p.Y - y) < small.Size))
                x = Math.Max(x, p.Right + 8 * scale);
            placed.Add((y, Draw(canvas, label, x, y - 3 * scale, small, ReferenceColor)));
        }

        foreach (var (label, points) in others)
            Comparison(canvas, area, label, points, xMin, xMax, yMin, yMax, scale, font);
        Line(canvas, area, values.Select(v => (v.Item1, v.Item2)).ToList(), xMin, xMax, yMin, yMax, MetricColor, 2.2f * scale);
        using var dot = new SKPaint { Color = MetricColor, IsAntialias = true };
        using var ring = new SKPaint { Color = Text, IsAntialias = true, Style = SKPaintStyle.Stroke, StrokeWidth = 1.5f * scale };
        // Snapshot dots small enough that the line stays readable where hundreds of them sit side by side; epochs keep a ring.
        foreach (var (x, y, endOfEpoch) in values)
        {
            var point = new SKPoint(X(area, x, xMin, xMax), Y(area, y, yMin, yMax));
            canvas.DrawCircle(point, (endOfEpoch ? 4.5f : SnapshotDot) * scale, dot);
            if (endOfEpoch)
                canvas.DrawCircle(point, 6.5f * scale, ring);
        }
    }

    /// <summary>One line per <paramref name="split"/> entry, labelled at its end, with its tolerant companion dashed and faint.</summary>
    private static void DrawSplit(SKCanvas canvas, TrainingSeries series, Metric metric, SplitLine[] split, SKRect plot, double xMin, double xMax, float scale, SKFont font)
    {
        List<(double X, double Y, bool EndOfEpoch)> Points(Func<EvaluationPoint, double> value) =>
            series.Points.Select(p => ((double)p.Step, value(p), p.EndOfEpoch)).Where(p => !double.IsNaN(p.Item2)).ToList();
        var lines = split.Select(s => (s.Label, s.Color, Strict: Points(s.Value), Tolerant: s.Tolerant is null ? [] : Points(s.Tolerant))).ToList();
        if (lines.All(l => l.Strict.Count == 0 && l.Tolerant.Count == 0))
        {
            // Evaluations without the split (no gold set, or reports from before it): nothing to draw yet.
            Draw(canvas, "no evaluation yet", plot.MidX, plot.MidY, font, Axis, SKTextAlign.Center);
            return;
        }
        var (yMin, yMax) = Range(lines.SelectMany(l => l.Strict.Concat(l.Tolerant)).Select(p => p.Y), metric.Percent);
        var area = Axes(canvas, plot, xMin, xMax, yMin, yMax, metric.Percent, series, scale, font);
        using var small = new SKFont(font.Typeface, font.Size * 0.85f);
        var placed = new List<float>();
        foreach (var (label, color, strict, tolerant) in lines)
        {
            if (tolerant.Count > 1)
            {
                using var dashed = new SKPaint { Color = color.WithAlpha(120), StrokeWidth = 1.4f * scale, IsAntialias = true, Style = SKPaintStyle.Stroke,
                    PathEffect = SKPathEffect.CreateDash([5 * scale, 4 * scale], 0) };
                using var path = new SKPath();
                for (var k = 0; k < tolerant.Count; k++)
                {
                    var p = new SKPoint(X(area, tolerant[k].X, xMin, xMax), Y(area, tolerant[k].Y, yMin, yMax));
                    if (k == 0) path.MoveTo(p); else path.LineTo(p);
                }
                canvas.DrawPath(path, dashed);
            }
            if (strict.Count == 0)
                continue;
            Line(canvas, area, strict.Select(v => (v.X, v.Y)).ToList(), xMin, xMax, yMin, yMax, color, 2f * scale);
            using var dot = new SKPaint { Color = color, IsAntialias = true };
            foreach (var (x, y, endOfEpoch) in strict)
                canvas.DrawCircle(X(area, x, xMin, xMax), Y(area, y, yMin, yMax), (endOfEpoch ? 3.8f : SnapshotDot) * scale, dot);
            // The label and latest value above the line's end, where the lines are furthest apart; moved off any label already there.
            var ly = Y(area, strict[^1].Y, yMin, yMax) - 6 * scale;
            while (placed.Any(p => Math.Abs(p - ly) < small.Size))
                ly -= small.Size;
            placed.Add(ly);
            Draw(canvas, $"{label} {Format(strict[^1].Y, metric.Percent)}", X(area, strict[^1].X, xMin, xMax) - 4 * scale, ly, small, color, SKTextAlign.Right);
        }
    }

    /// <summary>
    /// The panel background and its title line; returns the area left for the plot. In a narrow panel the title's
    /// extras give way, least needed first: "lower is better" becomes an arrow, then the arrow goes (the change's color
    /// still says better or worse), then the change loses its "since", then the change goes. The name and the latest value stay.
    /// </summary>
    /// <param name="change">The change since the previous evaluation, the step it was at (" since 1.2M"), and its color.</param>
    private static SKRect PanelFrame(SKCanvas canvas, SKRect box, float scale, string name, string? latest, (string Text, string Since, SKColor Color)? change, bool lowerIsBetter,
        SKFont font, SKFont bold)
    {
        using var panel = new SKPaint { Color = Panel, IsAntialias = true };
        canvas.DrawRoundRect(box, 6 * scale, 6 * scale, panel);
        var x = box.Left + 10 * scale;
        var right = box.Right - 10 * scale;
        var baseline = box.Top + 20 * scale;
        x = Draw(canvas, name, x, baseline, font, Axis) + 8 * scale;
        if (latest is not null)
            x = Draw(canvas, latest, x, baseline, bold, Text) + 8 * scale;

        string[] hints = lowerIsBetter ? ["lower is better", "↓", ""] : [""];
        string[] changes = change is { } c ? [c.Text + c.Since, c.Text, ""] : [""];
        var fits = (from text in changes from hint in hints
                    let width = (text.Length > 0 ? font.MeasureText(text) + 8 * scale : 0) + (hint.Length > 0 ? font.MeasureText(hint) : 0)
                    where x + width <= right
                    select (Change: text, Hint: hint)).FirstOrDefault();
        if (fits.Change is { Length: > 0 })
            Draw(canvas, fits.Change, x, baseline, font, change!.Value.Color);
        if (fits.Hint is { Length: > 0 })
            Draw(canvas, fits.Hint, right, baseline, font, Axis.WithAlpha(150), SKTextAlign.Right);
        return new SKRect(box.Left + 8 * scale, baseline + 8 * scale, box.Right - 12 * scale, box.Bottom - 6 * scale);
    }

    /// <summary>Y gridlines with labels, x labels in steps, epoch boundaries; returns the data area.</summary>
    private static SKRect Axes(SKCanvas canvas, SKRect plot, double xMin, double xMax, double yMin, double yMax, bool percent, TrainingSeries series, float scale, SKFont font)
    {
        var ticks = Ticks(yMin, yMax, 4);
        var labelWidth = ticks.Max(t => font.MeasureText(Format(t, percent))) + 8 * scale;
        var area = new SKRect(plot.Left + labelWidth, plot.Top + 4 * scale, plot.Right, plot.Bottom - font.Size - 8 * scale);

        using var grid = new SKPaint { Color = Grid, StrokeWidth = 1, IsAntialias = true };
        foreach (var t in ticks)
        {
            var y = Y(area, t, yMin, yMax);
            canvas.DrawLine(area.Left, y, area.Right, y, grid);
            Draw(canvas, Format(t, percent), area.Left - 6 * scale, y + font.Size * 0.35f, font, Axis, SKTextAlign.Right);
        }

        // Epoch boundaries inside the range, labelled on their left with the epoch that ends there, whose evaluation sits on the
        // line. Labelled with the epoch starting there, "epoch 3" sat beside epoch 2's evaluation and read as epoch 3's.
        using var epochLine = new SKPaint { Color = Axis.WithAlpha(110), StrokeWidth = 1.2f * scale, IsAntialias = true,
            PathEffect = SKPathEffect.CreateDash([3 * scale, 4 * scale], 0) };
        foreach (var (step, label) in EpochEnds(series, xMin, xMax))
        {
            var x = X(area, step, xMin, xMax);
            canvas.DrawLine(x, area.Top, x, area.Bottom, epochLine);
            if (x - 4 * scale - font.MeasureText(label) >= area.Left)
                Draw(canvas, label, x - 4 * scale, area.Top + font.Size, font, Axis, SKTextAlign.Right);
        }

        var labelY = area.Bottom + font.Size + 4 * scale;
        var xTicks = Ticks(xMin, xMax, Math.Max(2, (int)(area.Width / (90 * scale))));
        foreach (var t in xTicks.Where(t => t >= xMin && t <= xMax))
            Draw(canvas, TrainingDashboard.Steps(t), X(area, t, xMin, xMax), labelY, font, Axis, SKTextAlign.Center);
        return area;
    }

    /// <summary>
    /// A centred running mean over a run's loss buckets (7 wide, narrower at the ends): the trend the noise hides. Not an
    /// exponential average, which lags: early in a run, with few buckets, it sat well above the falling loss.
    /// </summary>
    private static List<(double X, double Y)> Smooth(IReadOnlyList<(double X, double Y)> raw)
    {
        var smoothed = new List<(double, double)>(raw.Count);
        for (var i = 0; i < raw.Count; i++)
        {
            var half = Math.Min(3, Math.Min(i, raw.Count - 1 - i));
            var sum = 0.0;
            for (var j = i - half; j <= i + half; j++)
                sum += raw[j].Y;
            smoothed.Add((raw[i].X, sum / (2 * half + 1)));
        }
        return smoothed;
    }

    /// <summary>Another run's curve (<see cref="TrainingSeries.Comparisons"/>), labelled at its last point.</summary>
    private static void Comparison(SKCanvas canvas, SKRect area, string label, IReadOnlyList<(double X, double Y)> points,
        double xMin, double xMax, double yMin, double yMax, float scale, SKFont font)
    {
        Line(canvas, area, points, xMin, xMax, yMin, yMax, ReferenceColor, 1.6f * scale);
        using var dot = new SKPaint { Color = ReferenceColor, IsAntialias = true };
        foreach (var (x, y) in points)
            canvas.DrawCircle(X(area, x, xMin, xMax), Y(area, y, yMin, yMax), SnapshotDot * scale, dot);
        using var small = new SKFont(font.Typeface, font.Size * 0.85f);
        var (lx, ly) = (X(area, points[^1].X, xMin, xMax), Y(area, points[^1].Y, yMin, yMax));
        var right = lx + 5 * scale + small.MeasureText(label) > area.Right;
        Draw(canvas, label, right ? lx - 5 * scale : lx + 5 * scale, ly - 4 * scale, small, ReferenceColor, right ? SKTextAlign.Right : SKTextAlign.Left);
    }

    private static void Line(SKCanvas canvas, SKRect area, IReadOnlyList<(double X, double Y)> points, double xMin, double xMax, double yMin, double yMax, SKColor color, float width)
    {
        using var paint = new SKPaint { Color = color, StrokeWidth = width, IsAntialias = true, Style = SKPaintStyle.Stroke, StrokeJoin = SKStrokeJoin.Round, StrokeCap = SKStrokeCap.Round };
        using var path = new SKPath();
        for (var k = 0; k < points.Count; k++)
        {
            var p = new SKPoint(X(area, points[k].X, xMin, xMax), Y(area, points[k].Y, yMin, yMax));
            if (k == 0)
                path.MoveTo(p);
            else
                path.LineTo(p);
        }
        canvas.DrawPath(path, paint);
    }

    private static float X(SKRect area, double x, double min, double max) => area.Left + (float)((x - min) / Math.Max(max - min, 1e-12) * area.Width);

    private static float Y(SKRect area, double y, double min, double max) => area.Bottom - (float)((y - min) / Math.Max(max - min, 1e-12) * area.Height);

    /// <summary>A padded range around the values, never flat.</summary>
    private static (double Min, double Max) Range(IEnumerable<double> values, bool percent)
    {
        var list = values.ToList();
        var (min, max) = (list.Min(), list.Max());
        var pad = Math.Max((max - min) * 0.12, percent ? 0.005 : Math.Max(Math.Abs(max) * 0.02, 1e-3));
        return (min - pad, max + pad);
    }

    /// <summary>The epoch ends strictly inside (<paramref name="xMin"/>, <paramref name="xMax"/>): global step, and the label of the epoch ending there.</summary>
    public static IReadOnlyList<(double Step, string Label)> EpochEnds(TrainingSeries series, double xMin, double xMax) =>
        Enumerable.Range(1, Math.Max(0, series.Epochs - 1))
            .Select(e => ((double)e * series.StepsPerEpoch, $"epoch {e}"))
            .Where(end => end.Item1 > xMin && end.Item1 < xMax)
            .ToList();

    /// <summary>About <paramref name="count"/> round tick values (1, 2, 2.5, 5 x 10^n apart) inside [min, max].</summary>
    public static IReadOnlyList<double> Ticks(double min, double max, int count)
    {
        var raw = (max - min) / Math.Max(1, count);
        if (raw <= 0)
            return [min];
        var magnitude = Math.Pow(10, Math.Floor(Math.Log10(raw)));
        var step = new[] { 1, 2, 2.5, 5, 10 }.Select(m => m * magnitude).First(s => s >= raw);
        var ticks = new List<double>();
        // k * step, rounded to the step's own precision, so 3 x 0.2 is 0.6 and not 0.6000000000000001.
        var digits = Math.Clamp(2 - (int)Math.Floor(Math.Log10(step)), 0, 15);
        for (var k = Math.Ceiling(min / step); k * step <= max + step * 1e-9; k++)
            ticks.Add(Math.Round(k * step, digits));
        return ticks;
    }

    private static string Format(double value, bool percent) =>
        Math.Abs(value) < 1e-9 ? (percent ? "0%" : "0")
        : percent ? (100 * value).ToString(Math.Abs(value) < 0.1 ? "0.0" : "0.#", CultureInfo.InvariantCulture) + "%"
        : value.ToString("0.000", CultureInfo.InvariantCulture);

    /// <summary>Draws text; returns where it ends (left-aligned).</summary>
    private static float Draw(SKCanvas canvas, string text, float x, float y, SKFont font, SKColor color, SKTextAlign align = SKTextAlign.Left)
    {
        using var paint = new SKPaint { Color = color, IsAntialias = true };
        canvas.DrawText(text, x, y, align, font, paint);
        return x + font.MeasureText(text);
    }
}
