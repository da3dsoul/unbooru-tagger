using UnbooruTagger.Core.Scoring;

namespace UnbooruTagger.Tests.Core;

public class HeatmapRefinerTests
{
    [Fact]
    public void Refine_ReturnsAGridSizedToTheGuide()
    {
        var heatmap = new float[,] { { 1f, 0f }, { 0f, 0f } };
        var guide = new float[8, 8, 3];

        var refined = HeatmapRefiner.Refine(heatmap, guide);

        Assert.Equal(8, refined.GetLength(0));
        Assert.Equal(8, refined.GetLength(1));
    }

    [Fact]
    public void Refine_KeepsHeatOnItsOwnSideOfASharpColorEdge()
    {
        // Hot in the left half of the grid, cold in the right half.
        var heatmap = new float[,]
        {
            { 1f, 1f, 0f, 0f },
            { 1f, 1f, 0f, 0f },
            { 1f, 1f, 0f, 0f },
            { 1f, 1f, 0f, 0f }
        };

        const int size = 16;
        var splitGuide = new float[size, size, 3];
        var uniformGuide = new float[size, size, 3];
        for (var y = 0; y < size; y++)
        for (var x = 0; x < size; x++)
        {
            // Left half red, right half blue — matches the heatmap's hot/cold split.
            splitGuide[y, x, 0] = x < size / 2 ? 1f : 0f;
            splitGuide[y, x, 2] = x < size / 2 ? 0f : 1f;

            uniformGuide[y, x, 0] = 0.5f;
            uniformGuide[y, x, 1] = 0.5f;
            uniformGuide[y, x, 2] = 0.5f;
        }

        var refinedWithSplitGuide = HeatmapRefiner.Refine(heatmap, splitGuide);
        var refinedWithUniformGuide = HeatmapRefiner.Refine(heatmap, uniformGuide);

        // Just left of the edge: with a color-aware guide, this cell should stay closer to
        // its own (hot) side's value than a plain spatial blur that mixes evenly across the
        // edge with the (cold) far side.
        const int probeY = 8;
        var probeX = size / 2 - 1;

        Assert.True(refinedWithSplitGuide[probeY, probeX] > refinedWithUniformGuide[probeY, probeX]);
    }
}
