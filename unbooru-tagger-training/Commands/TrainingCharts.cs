using System.Globalization;
using SkiaSharp;
using UnbooruTagger.Training.Checkpoints;
using UnbooruTagger.Training.Training;

namespace UnbooruTagger.Training.Commands;

/// <summary>Everything the charts show for one run, read from its checkpoint directory.</summary>
/// <param name="CurrentStep">Global step of the latest checkpoint save.</param>
/// <param name="References">Other runs' evaluations to compare against (e.g. "v3 e1"), drawn as dashed lines.</param>
public sealed record TrainingSeries(
    string Name, LossHistory Loss, IReadOnlyList<EvaluationPoint> Points, long CurrentStep, int StepsPerEpoch, int Epochs,
    IReadOnlyList<(string Label, EvaluationPoint Point)> References)
{
    /// <summary>
    /// Reads <paramref name="checkpointDir"/>: the saved loss history, every epoch evaluation
    /// (history/epoch-N) and finished snapshot evaluation (snapshots/step-N).
    /// </summary>
    public static TrainingSeries Load(string checkpointDir, int epochs, IReadOnlyList<(string Label, string Directory)>? references = null)
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
            currentStep, stepsPerEpoch, epochs, referencePoints);
    }
}

/// <summary>
/// Draws a run's <see cref="TrainingSeries"/> as one image: the loss over the whole run on top, then
/// one panel per headline evaluation metric, with epoch boundaries and other runs' values for
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

    private sealed record Metric(string Name, Func<EvaluationPoint, double> Value, bool LowerIsBetter, bool Percent);

    private static readonly Metric[] Metrics =
    [
        new("held-out mAP", p => p.MeanAveragePrecision, false, false),
        new("gold pointing", p => p.GoldPointing, false, true),
        new("blanked object still called", p => p.OccludedStillCalled, true, true),
        new("context recall gap", p => p.RecallGap, true, true),
        new("context false-positive gap", p => p.FalsePositiveGap, true, true),
        new("exclusive-tag violations", p => p.ExclusiveViolations, true, true),
    ];

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
        var lossHeight = (height - top - margin) * 0.34f;
        var gap = 10 * scale;
        var lossBox = new SKRect(margin, top, width - margin, top + lossHeight);
        DrawLoss(canvas, series, lossBox, scale, font, bold);

        // Evaluation panels share one x range, so the same step lines up across them.
        var metricsTop = lossBox.Bottom + gap;
        var cellWidth = (width - 2 * margin - 2 * gap) / 3;
        var cellHeight = (height - margin - metricsTop - gap) / 2;
        var steps = series.Points.Select(p => (double)p.Step).ToList();
        var xMin = steps.Count > 0 ? Math.Min(steps.Min(), series.CurrentStep) : 0;
        var xMax = Math.Max(series.CurrentStep, steps.Count > 0 ? steps.Max() : 1);
        for (var k = 0; k < Metrics.Length; k++)
        {
            var (column, row) = (k % 3, k / 3);
            var box = SKRect.Create(margin + column * (cellWidth + gap), metricsTop + row * (cellHeight + gap), cellWidth, cellHeight);
            DrawMetric(canvas, series, Metrics[k], box, xMin, xMax, scale, font, bold);
        }
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
        var title = $"loss · mean per {series.Loss.BucketSteps:N0} steps";
        var plot = PanelFrame(canvas, box, scale, title, buckets.Count > 0 ? buckets[^1].Mean.ToString("G4", CultureInfo.InvariantCulture) : null, null, false, font, bold);
        if (buckets.Count < 2)
        {
            Draw(canvas, "the loss history fills in as checkpoints are saved", plot.MidX, plot.MidY, font, Axis, SKTextAlign.Center);
            return;
        }

        // Raw bucket means, and a smoothed line over them: the trend the noise hides.
        var raw = buckets.Select(b => ((double)b.End, (double)b.Mean)).ToList();
        var smoothed = new List<(double, double)>();
        var ema = raw[0].Item2;
        foreach (var (x, y) in raw)
            smoothed.Add((x, ema = 0.85 * ema + 0.15 * y));
        var (yMin, yMax) = Range(raw.Select(p => p.Item2), false);
        var (xMin, xMax) = (raw[0].Item1, Math.Max(raw[^1].Item1, series.CurrentStep));
        var area = Axes(canvas, plot, xMin, xMax, yMin, yMax, false, series, scale, font);
        Line(canvas, area, raw, xMin, xMax, yMin, yMax, LossColor.WithAlpha(90), 1.2f * scale);
        Line(canvas, area, smoothed, xMin, xMax, yMin, yMax, LossColor, 2.4f * scale);
    }

    private static void DrawMetric(SKCanvas canvas, TrainingSeries series, Metric metric, SKRect box, double xMin, double xMax, float scale, SKFont font, SKFont bold)
    {
        var values = series.Points.Select(p => ((double)p.Step, metric.Value(p), p.EndOfEpoch)).Where(p => !double.IsNaN(p.Item2)).ToList();
        string? latest = values.Count > 0 ? Format(values[^1].Item2, metric.Percent) : null;
        (string, SKColor)? change = null;
        if (values.Count > 1)
        {
            var delta = values[^1].Item2 - values[^2].Item2;
            var better = metric.LowerIsBetter ? delta < 0 : delta > 0;
            var magnitude = metric.Percent ? (100 * Math.Abs(delta)).ToString("0.0", CultureInfo.InvariantCulture) + " pts" : Format(Math.Abs(delta), false);
            change = ($"{(delta >= 0 ? "+" : "−")}{magnitude} since {TrainingDashboard.Steps(values[^2].Item1)}",
                delta == 0 ? Axis : better ? Better : Worse);
        }
        var plot = PanelFrame(canvas, box, scale, metric.Name, latest, change, metric.LowerIsBetter, font, bold);
        if (values.Count == 0)
        {
            Draw(canvas, "no evaluation yet", plot.MidX, plot.MidY, font, Axis, SKTextAlign.Center);
            return;
        }

        var references = series.References.Select(r => (r.Label, Value: metric.Value(r.Point))).Where(r => !double.IsNaN(r.Value)).ToList();
        var (yMin, yMax) = Range(values.Select(v => v.Item2).Concat(references.Select(r => r.Value)), metric.Percent);
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

        Line(canvas, area, values.Select(v => (v.Item1, v.Item2)).ToList(), xMin, xMax, yMin, yMax, MetricColor, 2.2f * scale);
        using var dot = new SKPaint { Color = MetricColor, IsAntialias = true };
        using var ring = new SKPaint { Color = Text, IsAntialias = true, Style = SKPaintStyle.Stroke, StrokeWidth = 1.5f * scale };
        foreach (var (x, y, endOfEpoch) in values)
        {
            var point = new SKPoint(X(area, x, xMin, xMax), Y(area, y, yMin, yMax));
            canvas.DrawCircle(point, (endOfEpoch ? 4.5f : 2.8f) * scale, dot);
            if (endOfEpoch)
                canvas.DrawCircle(point, 6.5f * scale, ring);
        }
    }

    /// <summary>The panel background and its title line; returns the area left for the plot.</summary>
    private static SKRect PanelFrame(SKCanvas canvas, SKRect box, float scale, string name, string? latest, (string Text, SKColor Color)? change, bool lowerIsBetter,
        SKFont font, SKFont bold)
    {
        using var panel = new SKPaint { Color = Panel, IsAntialias = true };
        canvas.DrawRoundRect(box, 6 * scale, 6 * scale, panel);
        var x = box.Left + 10 * scale;
        var baseline = box.Top + 20 * scale;
        x = Draw(canvas, name, x, baseline, font, Axis) + 8 * scale;
        if (latest is not null)
            x = Draw(canvas, latest, x, baseline, bold, Text) + 8 * scale;
        if (change is { } c)
            x = Draw(canvas, c.Text, x, baseline, font, c.Color) + 8 * scale;
        if (lowerIsBetter)
            Draw(canvas, "lower is better", box.Right - 10 * scale, baseline, font, Axis.WithAlpha(150), SKTextAlign.Right);
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

        // Epoch boundaries inside the range.
        using var epochLine = new SKPaint { Color = Axis.WithAlpha(110), StrokeWidth = 1.2f * scale, IsAntialias = true,
            PathEffect = SKPathEffect.CreateDash([3 * scale, 4 * scale], 0) };
        for (var e = 1; e < series.Epochs; e++)
        {
            var step = (double)e * series.StepsPerEpoch;
            if (step <= xMin || step >= xMax)
                continue;
            var x = X(area, step, xMin, xMax);
            canvas.DrawLine(x, area.Top, x, area.Bottom, epochLine);
            Draw(canvas, $"epoch {e + 1}", x + 4 * scale, area.Top + font.Size, font, Axis);
        }

        var labelY = area.Bottom + font.Size + 4 * scale;
        var xTicks = Ticks(xMin, xMax, Math.Max(2, (int)(area.Width / (90 * scale))));
        foreach (var t in xTicks.Where(t => t >= xMin && t <= xMax))
            Draw(canvas, TrainingDashboard.Steps(t), X(area, t, xMin, xMax), labelY, font, Axis, SKTextAlign.Center);
        return area;
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
