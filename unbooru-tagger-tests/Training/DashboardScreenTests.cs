using UnbooruTagger.Training.Commands;
using UnbooruTagger.Training.Training;

namespace UnbooruTagger.Tests.Training;

public class DashboardScreenTests
{
    private static readonly TrainingRunInfo Info = new("training · test", Epochs: 6, StepsPerEpoch: 484_921, StartEpoch: 0, StartStepInEpoch: 199_000,
        ImagesPerStep: 10, CheckpointInterval: TimeSpan.FromMinutes(15), EvaluationsOffloaded: true, SnapshotEvaluationSteps: 10_000);

    [Fact]
    public void ChartsAreDrawnOnceBelowThePanel_AndAgainOnlyWhenTheirDataChanges()
    {
        var dashboard = new TrainingDashboard(Info);
        dashboard.AddEvaluationPoint(new EvaluationPoint(190_000, false, 0.40, 0.29, 0.056, 0.107, 0.86, 0.27));
        var output = new StringWriter();
        using (var screen = new DashboardScreen(dashboard, output, size: () => (120, 50), cell: (8, 15)))
        {
            screen.Draw();
            var first = output.ToString();
            Assert.StartsWith("\u001b[?1049h", first); // alternate screen
            Assert.Single(Sixels(first));
            Assert.Contains("every 10,000 steps", first); // the panel
            Assert.DoesNotMatch("[\u2801-\u28FF]", first); // without its text charts (blank rows are U+2800)

            // The panel's rows are rewritten every tick; the image isn't, until there's something new.
            output.GetStringBuilder().Clear();
            screen.Draw();
            Assert.Empty(Sixels(output.ToString()));
            Assert.Contains("every 10,000 steps", output.ToString());

            dashboard.AddEvaluationPoint(new EvaluationPoint(200_000, false, 0.42, 0.30, 0.060, 0.100, 0.84, 0.25));
            output.GetStringBuilder().Clear();
            screen.Draw();
            Assert.Single(Sixels(output.ToString()));
        }
        Assert.EndsWith("\u001b[?1049l", output.ToString()); // back to the normal screen
    }

    [Fact]
    public void ImageStartsBelowThePanelRows_AndStopsAboveTheLastRow()
    {
        var dashboard = new TrainingDashboard(Info);
        dashboard.AddEvaluationPoint(new EvaluationPoint(190_000, false, 0.40, 0.29, 0.056, 0.107, 0.86, 0.27));
        var output = new StringWriter();
        using var screen = new DashboardScreen(dashboard, output, size: () => (120, 50), cell: (8, 15));
        screen.Draw();
        var text = output.ToString();

        // The panel's rows (not the rows cleared below it), then the image positioned below them.
        var panelRows = System.Text.RegularExpressions.Regex.Matches(text, @"\u001b\[(\d+);1H(?!\u001bP|\u001b\[2K)").Select(m => int.Parse(m.Groups[1].Value)).ToList();
        var image = System.Text.RegularExpressions.Regex.Match(text, @"\u001b\[(\d+);1H\u001bP");
        Assert.True(image.Success);
        var imageRow = int.Parse(image.Groups[1].Value);
        var header = System.Text.RegularExpressions.Regex.Match(text, "\"1;1;(\\d+);(\\d+)");
        Assert.Equal(120 * 8, int.Parse(header.Groups[1].Value));
        var imageHeight = int.Parse(header.Groups[2].Value);
        Assert.Equal(0, imageHeight % 6); // sent exactly as tall as planned, not rounded up a band
        var lastImageRow = imageRow + (imageHeight + 14) / 15 - 1;
        // Two rows stay free under the image: the cursor moving below it must not scroll the screen.
        Assert.True(lastImageRow <= 48, $"image rows {imageRow}..{lastImageRow} come within two rows of the bottom");
        Assert.True(imageRow > panelRows.Max(), "the image starts below the panel");
    }

    private static IEnumerable<int> Sixels(string text)
    {
        for (var at = text.IndexOf("\u001bP", StringComparison.Ordinal); at >= 0; at = text.IndexOf("\u001bP", at + 2, StringComparison.Ordinal))
            yield return at;
    }
}
