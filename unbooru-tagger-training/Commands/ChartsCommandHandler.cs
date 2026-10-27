using Spectre.Console;
using UnbooruTagger.Training.Training;

namespace UnbooruTagger.Training.Commands;

/// <summary>
/// <c>charts</c>: draws a run's loss and evaluation trends (<see cref="TrainingCharts"/>) as real
/// graphics, in the terminal as a Sixel image and/or to a PNG. Reads only the checkpoint directory,
/// so it runs alongside training (typically in its own tmux pane) without touching it; with
/// <c>--watch</c> it redraws whenever a checkpoint save or a finished evaluation changes the data.
/// </summary>
public static class ChartsCommandHandler
{
    public static int Run(string checkpointDir, int epochs, IReadOnlyList<string> references, string? pngPath, bool watch, int intervalSeconds, int width, int height)
    {
        var parsedReferences = new List<(string, string)>();
        foreach (var reference in references)
        {
            var split = reference.IndexOf('=');
            if (split <= 0)
            {
                AnsiConsole.MarkupLineInterpolated($"[red]--reference takes label=report directory, got '{reference}'.[/]");
                return 1;
            }
            parsedReferences.Add((reference[..split], reference[(split + 1)..]));
        }

        var toTerminal = !Console.IsOutputRedirected && (watch || pngPath is null);
        if (!watch)
        {
            Draw(checkpointDir, epochs, parsedReferences, pngPath, toTerminal, width, height);
            return 0;
        }

        Console.Write("\u001b[?25l"); // hide the cursor while the chart is up
        Console.CancelKeyPress += (_, _) => Console.Write("\u001b[?25h\n");
        string? shown = null;
        while (true)
        {
            var signature = Signature(checkpointDir) + $"|{Console.WindowWidth}x{Console.WindowHeight}";
            if (signature != shown)
            {
                Draw(checkpointDir, epochs, parsedReferences, pngPath, toTerminal, width, height);
                shown = signature;
            }
            Thread.Sleep(TimeSpan.FromSeconds(Math.Max(1, intervalSeconds)));
        }
    }

    private static void Draw(string checkpointDir, int epochs, IReadOnlyList<(string, string)> references, string? pngPath, bool toTerminal, int width, int height)
    {
        var series = TrainingSeries.Load(checkpointDir, epochs, references);
        var (cellWidth, cellHeight) = TerminalGraphics.CellSize();
        // Leave the last terminal row for the status line.
        var (w, h) = (width > 0 ? width : Console.WindowWidth * cellWidth, height > 0 ? height : (Console.WindowHeight - 1) * cellHeight);
        if (!toTerminal)
            (w, h) = (width > 0 ? width : 1800, height > 0 ? height : 950);
        using var bitmap = TrainingCharts.Render(series, Math.Max(400, w), Math.Max(300, h), DateTime.Now, textScale: toTerminal && width == 0 ? cellHeight / 15f : null);
        if (pngPath is not null)
            TrainingCharts.SavePng(bitmap, pngPath);
        if (!toTerminal)
            return;
        var sixel = Sixel.Encode(bitmap);
        Console.Write("\u001b[H\u001b[2J" + sixel);
        Console.Write($"\u001b[{Console.WindowHeight};1H\u001b[2K{series.Name}: {series.Points.Count} evaluations, step {series.CurrentStep:N0} · redrawn {DateTime.Now:HH:mm} when the data changes · Ctrl+C to exit");
    }

    /// <summary>Changes whenever there's something new to draw: a checkpoint save or a finished evaluation.</summary>
    private static string Signature(string checkpointDir)
    {
        string Stamp(string path) => File.Exists(path) ? File.GetLastWriteTimeUtc(path).Ticks.ToString() : "-";
        var snapshots = Path.Combine(checkpointDir, SnapshotEvaluations.DirectoryName);
        var done = Directory.Exists(snapshots) ? Directory.GetFiles(snapshots, OffloadedEvaluations.DoneMarker, SearchOption.AllDirectories).Length : 0;
        var history = Path.Combine(checkpointDir, "history");
        var epochsDone = Directory.Exists(history) ? Directory.GetFiles(history, "evaluation_report.json", SearchOption.AllDirectories).Length : 0;
        return $"{Stamp(Path.Combine(checkpointDir, LossHistory.FileName))}|{Stamp(Path.Combine(checkpointDir, "training_progress.json"))}|{done}|{epochsDone}";
    }
}
