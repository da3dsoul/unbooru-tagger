using Spectre.Console;
using UnbooruTagger.Training.Commands;

namespace UnbooruTagger.Tests.Training;

public class BrailleChartTests
{
    private static string[] Render(BrailleChart chart, int width)
    {
        var writer = new StringWriter();
        var console = AnsiConsole.Create(new AnsiConsoleSettings { Ansi = AnsiSupport.No, Out = new AnsiConsoleOutput(writer) });
        console.Profile.Width = width;
        console.Write(chart);
        return writer.ToString().TrimEnd('\n').Split('\n');
    }

    [Fact]
    public void Line_RunsFromBottomLeftToTopRight_WithBothAxesLabeled()
    {
        var chart = new BrailleChart([(0, 0), (100, 1)], rows: 2, v => v.ToString("F1"), x => $"s{x}", Color.Green);
        var lines = Render(chart, 14);

        Assert.Equal(3, lines.Length); // two rows of dots, then the x labels
        Assert.StartsWith("1.0 ", lines[0]);
        Assert.StartsWith("0.0 ", lines[1]);
        Assert.Equal("    s0    s100", lines[2]);
        // Ten cells, 20x8 dots: the bottom-left dot and the top-right dot are both drawn.
        var (top, bottom) = (lines[0][4..], lines[1][4..]);
        Assert.Equal(10, top.Length);
        Assert.True((bottom[0] - 0x2800 & 0x40) != 0);
        Assert.True((top[^1] - 0x2800 & 0x08) != 0);
        Assert.Equal('⠀', top[0]);
        Assert.Equal('⠀', bottom[^1]);
    }

    [Fact]
    public void NoValues_SaysSo()
    {
        var chart = new BrailleChart([(0, double.NaN)], rows: 2, v => v.ToString("F1"), x => $"{x}", Color.Green);
        Assert.Equal(["no data yet"], Render(chart, 30));
    }

    [Fact]
    public void FlatSeries_StillDrawsALine()
    {
        var chart = new BrailleChart([(0, 0.5), (1, 0.5), (2, 0.5)], rows: 2, v => v.ToString("F3"), x => $"{x}", Color.Green);
        var lines = Render(chart, 30);
        Assert.Contains(lines.Take(2), l => l.Any(c => c > '⠀' && c <= '⣿'));
    }
}
