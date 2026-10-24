using UnbooruTagger.Core.Encoding;
using UnbooruTagger.Core.Scoring;

namespace UnbooruTagger.Tests.Core;

public class TagScorerTests
{
    [Fact]
    public void Score_IsHigherForAlignedThanOrthogonalVectors()
    {
        float[] image = [1, 0, 0];
        float[] tagAligned = [1, 0, 0];
        float[] tagOrthogonal = [0, 1, 0];

        var scoreAligned = TagScorer.Score(image, tagAligned);
        var scoreOrthogonal = TagScorer.Score(image, tagOrthogonal);

        Assert.True(scoreAligned > scoreOrthogonal);
        Assert.True(scoreAligned > 0.5f);
        Assert.Equal(0.5, scoreOrthogonal, 3);
    }

    [Fact]
    public void Heatmap_IsHighestAtTheLocationMatchingTheTag()
    {
        float[] tagEmbedding = [1, 0];
        var data = new float[]
        {
            1, 0, 0, 0,
            0, 0, 0, 0
        };
        var spatial = new SpatialFeatureMap(data, height: 2, width: 2, channels: 2);

        var heatmap = TagScorer.Heatmap(tagEmbedding, spatial);

        Assert.True(heatmap[0, 0] > heatmap[0, 1]);
        Assert.True(heatmap[0, 0] > heatmap[1, 0]);
        Assert.True(heatmap[0, 0] > heatmap[1, 1]);
    }

    [Fact]
    public void DetectBoxes_HigherPercentileNeverProducesALargerBoxThanALowerOne()
    {
        var heatmap = new float[,]
        {
            { 1.0f, 0.8f, 0.2f, 0.1f },
            { 0.9f, 0.7f, 0.2f, 0.1f },
            { 0.2f, 0.2f, 0.2f, 0.1f },
            { 0.1f, 0.1f, 0.1f, 0.1f }
        };

        var fullCanvas = new LetterboxBox(0, 0, 4, 4);
        var loosePercentile = TagScorer.DetectBoxes(heatmap, threshold: -10f, relativePercentile: 0f, fullCanvas, canvasSize: 4, imageWidth: 4, imageHeight: 4);
        var tightPercentile = TagScorer.DetectBoxes(heatmap, threshold: -10f, relativePercentile: 0.9f, fullCanvas, canvasSize: 4, imageWidth: 4, imageHeight: 4);

        var looseArea = loosePercentile.Sum(b => b.Width * b.Height);
        var tightArea = tightPercentile.Sum(b => b.Width * b.Height);

        Assert.True(tightArea < looseArea);
    }

    [Fact]
    public void DetectBoxes_AbsoluteThresholdActsAsAFloorRegardlessOfPercentile()
    {
        var heatmap = new float[,] { { 0.3f, 0.2f }, { 0.1f, 0.05f } };

        // Even at percentile 0 (its own weakest cell), a tag whose whole heatmap sits
        // below the absolute floor should get no boxes.
        var boxes = TagScorer.DetectBoxes(heatmap, threshold: 0.5f, relativePercentile: 0f, new LetterboxBox(0, 0, 2, 2), canvasSize: 2, imageWidth: 2, imageHeight: 2);

        Assert.Empty(boxes);
    }

    [Fact]
    public void DetectBoxes_NearFlatHeatmapProducesNoBoxesRegardlessOfThresholdOrPercentile()
    {
        // Every cell clears both the absolute floor and the percentile cut, but the
        // heatmap's own max-min spread (0.02) is too small to carry real spatial signal —
        // an under-discriminated tag scores near-uniformly, not because it genuinely
        // applies everywhere.
        var heatmap = new float[,] { { 0.51f, 0.50f }, { 0.52f, 0.49f } };

        var boxes = TagScorer.DetectBoxes(
            heatmap, threshold: -10f, relativePercentile: 0f, new LetterboxBox(0, 0, 2, 2),
            canvasSize: 2, imageWidth: 2, imageHeight: 2, minRange: 0.05f);

        Assert.Empty(boxes);
    }

    [Fact]
    public void DetectBoxes_HeatmapRangeAboveMinRangeIsUnaffectedByTheGate()
    {
        var heatmap = new float[,] { { 1.0f, 0.8f }, { 0.2f, 0.1f } };

        var boxes = TagScorer.DetectBoxes(
            heatmap, threshold: -10f, relativePercentile: 0f, new LetterboxBox(0, 0, 2, 2),
            canvasSize: 2, imageWidth: 2, imageHeight: 2, minRange: 0.05f);

        Assert.NotEmpty(boxes);
    }
}
