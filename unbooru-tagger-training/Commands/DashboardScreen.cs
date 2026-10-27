using System.Text;
using Spectre.Console;

namespace UnbooruTagger.Training.Commands;

/// <summary>
/// The training dashboard with graphical charts, for terminals that show Sixel images
/// (<see cref="TerminalGraphics.SupportsSixel"/>). Spectre's live display can't host an image: it
/// rewrites every line each refresh, and overwriting an image's lines erases it (tmux drops it). So
/// this draws the screen itself, on the alternate screen: the panel (<see cref="TrainingDashboard.Render"/>,
/// rendered to text by Spectre) in the top rows every second, and the charts
/// (<see cref="TrainingCharts"/>) as one image in the rows below, redrawn only when their data, the
/// panel's height or the terminal's size changes. The panel never writes into the image's rows.
/// </summary>
public sealed class DashboardScreen : IDisposable
{
    private const int MinimumChartRows = 10;

    private readonly TrainingDashboard _dashboard;
    private readonly TextWriter _output;
    private readonly Func<(int Columns, int Rows)> _size;
    private readonly (int Width, int Height) _cell;
    private readonly object _drawing = new();
    private readonly Timer _timer;
    private string? _chartKey;
    private int _panelRows;
    private bool _disposed;

    /// <param name="output">Where to draw (default: the console).</param>
    /// <param name="size">The terminal's size in characters (default: the console window's).</param>
    /// <param name="cell">A character cell in pixels (default: <see cref="TerminalGraphics.CellSize"/>).</param>
    /// <param name="refresh">Redraw interval; null draws only on <see cref="Draw"/> calls (tests).</param>
    public DashboardScreen(TrainingDashboard dashboard, TextWriter? output = null, Func<(int, int)>? size = null, (int, int)? cell = null,
        TimeSpan? refresh = null)
    {
        _dashboard = dashboard;
        _output = output ?? Console.Out;
        _size = size ?? (() => (Console.WindowWidth, Console.WindowHeight));
        _cell = cell ?? TerminalGraphics.CellSize();
        _dashboard.GraphicalCharts = true;
        // Alternate screen (the shell's scrollback is left as it was), cursor hidden, cleared.
        _output.Write("\u001b[?1049h\u001b[?25l\u001b[2J");
        _output.Flush();
        _timer = new Timer(_ => Draw(), null, refresh is null ? Timeout.InfiniteTimeSpan : TimeSpan.Zero, refresh ?? Timeout.InfiniteTimeSpan);
        // Ctrl+C or a kill: don't leave the terminal on the alternate screen with the cursor hidden.
        AppDomain.CurrentDomain.ProcessExit += (_, _) => Dispose();
    }

    public void Draw()
    {
        lock (_drawing)
        {
            if (_disposed)
                return;
            try
            {
                var (columns, rows) = _size();
                (columns, rows) = (Math.Max(40, columns), Math.Max(10, rows));
                var panel = PanelLines(columns, rows);
                var frame = new StringBuilder();
                frame.Append("\u001b[?2026h"); // synchronized update: the terminal shows the frame at once

                // The image: one blank row below the panel, then the rest of the screen but its last two rows.
                // After an image the terminal moves the cursor below it, and an image reaching the bottom
                // row scrolls the whole screen up, pushing the image's top under the panel.
                var chartTop = panel.Count + 1;
                var chartRows = rows - chartTop - 2;
                var (series, version) = _dashboard.ChartData();
                var key = $"{version}|{columns}x{rows}|{panel.Count}";
                if (key != _chartKey)
                {
                    // Erasing a row clears tmux's copy but leaves an old image's pixels in Konsole; only text
                    // drawn over them removes them. So blank rows are written with the blank braille character.
                    var blank = new string('⠀', columns);
                    for (var row = Math.Min(_panelRows, panel.Count) + 1; row <= rows; row++)
                        frame.Append($"\u001b[{row};1H\u001b[2K").Append(blank);
                    if (chartRows >= MinimumChartRows)
                    {
                        var (cellWidth, cellHeight) = _cell;
                        // Sixel draws in bands of six pixel rows: a height that isn't a multiple of six is sent
                        // rounded up, one row taller than planned.
                        var height = chartRows * cellHeight / 6 * 6;
                        using var bitmap = TrainingCharts.Render(series, columns * cellWidth, height, DateTime.Now, header: false, textScale: cellHeight / 15f);
                        frame.Append($"\u001b[{chartTop + 1};1H").Append(Sixel.Encode(bitmap));
                    }
                    _chartKey = key;
                }

                for (var row = 0; row < panel.Count; row++)
                    frame.Append($"\u001b[{row + 1};1H").Append(panel[row]).Append("\u001b[0m\u001b[K");
                _panelRows = panel.Count;
                frame.Append("\u001b[?2026l");
                _output.Write(frame.ToString());
                _output.Flush();
            }
            catch (Exception e) when (e is IOException or ArgumentException or InvalidOperationException)
            {
                // A resize mid-frame or a closed terminal: the next tick draws again.
            }
        }
    }

    /// <summary>The panel rendered by Spectre to ANSI text lines, cut to leave the charts room.</summary>
    private List<string> PanelLines(int columns, int rows)
    {
        var writer = new StringWriter();
        var console = AnsiConsole.Create(new AnsiConsoleSettings
        {
            Ansi = AnsiSupport.Yes,
            ColorSystem = (ColorSystemSupport)AnsiConsole.Profile.Capabilities.ColorSystem,
            Out = new AnsiConsoleOutput(writer),
            Interactive = InteractionSupport.No,
        });
        console.Profile.Width = columns;
        console.Profile.Height = rows;
        console.Write(_dashboard.Render());
        var lines = writer.ToString().Replace("\r\n", "\n").TrimEnd('\n').Split('\n').ToList();
        // On a short terminal the panel wins; the charts need at least MinimumChartRows to be drawn at all.
        return lines.Count <= rows ? lines : lines.Take(rows).ToList();
    }

    /// <summary>Back to the normal screen, where the caller prints the final panel.</summary>
    public void Dispose()
    {
        lock (_drawing)
        {
            if (_disposed)
                return;
            _disposed = true;
        }
        _timer.Dispose();
        _dashboard.GraphicalCharts = false;
        _output.Write("\u001b[?25h\u001b[?1049l");
        _output.Flush();
    }
}
