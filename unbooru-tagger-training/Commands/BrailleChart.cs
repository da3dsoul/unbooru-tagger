using Spectre.Console;
using Spectre.Console.Rendering;

namespace UnbooruTagger.Training.Commands;

/// <summary>
/// A line chart in braille characters (each cell a 2x4 grid of dots), for the training dashboard.
/// It takes whatever width it's given, with the y range labeled on the left and the x range
/// underneath. Points that land in the same dot column are averaged; consecutive points are joined.
/// </summary>
public sealed class BrailleChart(
    IReadOnlyList<(double X, double Y)> points, int rows, Func<double, string> formatY, Func<double, string> formatX, Color color) : Renderable
{
    // Braille dot bits by [dot row, dot column] within a cell.
    private static readonly int[,] Bits = { { 0x01, 0x08 }, { 0x02, 0x10 }, { 0x04, 0x20 }, { 0x40, 0x80 } };

    protected override Measurement Measure(RenderOptions options, int maxWidth) => new(Math.Min(maxWidth, 20), maxWidth);

    protected override IEnumerable<Segment> Render(RenderOptions options, int maxWidth)
    {
        var axis = new Style(Color.Grey);
        var valid = points.Where(p => !double.IsNaN(p.Y) && !double.IsInfinity(p.Y)).OrderBy(p => p.X).ToList();
        if (valid.Count == 0)
            return [new Segment("no data yet", axis), Segment.LineBreak];

        var (yMin, yMax) = (valid.Min(p => p.Y), valid.Max(p => p.Y));
        if (yMax - yMin < 1e-12)
        {
            var pad = Math.Max(Math.Abs(yMax) * 0.05, 1e-3);
            (yMin, yMax) = (yMin - pad, yMax + pad);
        }
        var (top, bottom) = (formatY(yMax), formatY(yMin));
        var labelWidth = Math.Max(top.Length, bottom.Length) + 1;
        var cells = Math.Max(4, maxWidth - labelWidth);
        var (dotColumns, dotRows) = (cells * 2, rows * 4);

        // One averaged value per dot column the points fall in.
        var (xMin, xMax) = (valid[0].X, valid[^1].X);
        var columns = valid
            .GroupBy(p => xMax > xMin ? (int)Math.Round((p.X - xMin) / (xMax - xMin) * (dotColumns - 1)) : 0)
            .Select(g => (Column: g.Key, Row: (int)Math.Round((yMax - g.Average(p => p.Y)) / (yMax - yMin) * (dotRows - 1))))
            .OrderBy(c => c.Column)
            .ToList();

        var grid = new int[rows, cells];
        void Dot(int column, int row) => grid[row / 4, column / 2] |= Bits[row % 4, column % 2];
        Dot(columns[0].Column, columns[0].Row);
        for (var k = 1; k < columns.Count; k++)
        {
            var (c0, r0) = columns[k - 1];
            var (c1, r1) = columns[k];
            var steps = Math.Max(Math.Abs(c1 - c0), Math.Abs(r1 - r0));
            for (var i = 1; i <= steps; i++)
                Dot(c0 + (int)Math.Round((c1 - c0) * (double)i / steps), r0 + (int)Math.Round((r1 - r0) * (double)i / steps));
        }

        var line = new Style(color);
        var segments = new List<Segment>();
        for (var row = 0; row < rows; row++)
        {
            var label = row == 0 ? top : row == rows - 1 ? bottom : "";
            segments.Add(new Segment(label.PadLeft(labelWidth - 1) + " ", axis));
            segments.Add(new Segment(new string(Enumerable.Range(0, cells).Select(c => (char)(0x2800 + grid[row, c])).ToArray()), line));
            segments.Add(Segment.LineBreak);
        }

        var (first, last) = (formatX(xMin), formatX(xMax));
        var xLabels = xMax > xMin && first.Length + last.Length + 1 <= cells ? first + last.PadLeft(cells - first.Length) : first;
        segments.Add(new Segment(new string(' ', labelWidth) + xLabels, axis));
        segments.Add(Segment.LineBreak);
        return segments;
    }
}
