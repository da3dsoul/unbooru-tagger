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
    public void DetectBoxes_BelowPeak_CutsInLogOddsUnderTheTagsOwnPeak()
    {
        // Log-odds 6, 3 and 1 around a peak of 8, the rest -6: a cut 4 below the peak keeps the 8 and the 6 only, where a cut at
        // 60% of the probability range (0.6) would also keep the 3 and the 1.
        static float P(float logit) => 1 / (1 + MathF.Exp(-logit));
        var heatmap = new float[,]
        {
            { P(8), P(6), P(3), P(1) },
            { P(-6), P(-6), P(-6), P(-6) },
            { P(-6), P(-6), P(-6), P(-6) },
            { P(-6), P(-6), P(-6), P(-6) }
        };
        var canvas = new LetterboxBox(0, 0, 4, 4);

        var belowPeak = TagScorer.DetectBoxes(heatmap, threshold: 0.5f, relativePercentile: 0.6f, canvas, canvasSize: 4, imageWidth: 4, imageHeight: 4, belowPeak: 4f);
        var percentile = TagScorer.DetectBoxes(heatmap, threshold: 0.5f, relativePercentile: 0.6f, canvas, canvasSize: 4, imageWidth: 4, imageHeight: 4);

        Assert.Equal(new BoundingBox(0, 0, 2, 1, belowPeak[0].Confidence), Assert.Single(belowPeak));
        Assert.Equal(4, Assert.Single(percentile).Width);
        // Below the absolute threshold at its peak, a tag still gets no boxes.
        var weak = new float[,] { { P(-1), P(-3) }, { P(-6), P(-6) } };
        Assert.Empty(TagScorer.DetectBoxes(weak, threshold: 0.5f, relativePercentile: 0f, new LetterboxBox(0, 0, 2, 2), canvasSize: 2, imageWidth: 2, imageHeight: 2, belowPeak: 4f));
    }

    [Fact]
    public void DetectBoxes_PeakMarginDropsFaintSecondaryRegions()
    {
        static float P(float logit) => 1f / (1f + MathF.Exp(-logit));
        // Three separate regions in the 4-log-odds band: the peak (8), one 0.5 below it, one 3 below it.
        var heatmap = new float[,]
        {
            { P(8), P(-6), P(7.5f), P(-6) },
            { P(-6), P(-6), P(-6), P(-6) },
            { P(5), P(-6), P(-6), P(-6) },
            { P(-6), P(-6), P(-6), P(-6) }
        };
        var canvas = new LetterboxBox(0, 0, 4, 4);

        var all = TagScorer.DetectBoxes(heatmap, threshold: 0.5f, relativePercentile: 0.6f, canvas, canvasSize: 4, imageWidth: 4, imageHeight: 4, belowPeak: 4f);
        var near = TagScorer.DetectBoxes(heatmap, threshold: 0.5f, relativePercentile: 0.6f, canvas, canvasSize: 4, imageWidth: 4, imageHeight: 4, belowPeak: 4f, peakMargin: 1f);

        Assert.Equal(3, all.Count);
        Assert.Equal([0, 2], near.Select(b => b.X).Order());
    }

    [Fact]
    public void FigureBox_SpansTheFigureConnectedToTheCharactersPeak()
    {
        static float P(float logit) => 1f / (1f + MathF.Exp(-logit));
        // The character peaks on the face (top left); a clothing tag covers the body below it, a background tag the far column.
        var character = new float[,]
        {
            { P(6), P(-8), P(-8), P(-8) },
            { P(-8), P(-8), P(-8), P(-8) },
            { P(-8), P(-8), P(-8), P(-8) },
            { P(-8), P(-8), P(-8), P(-8) }
        };
        var dress = new float[,]
        {
            { P(-9), P(-9), P(-9), P(-9) },
            { P(4), P(3), P(-9), P(-9) },
            { P(3.5f), P(2), P(-9), P(-9) },
            { P(-9), P(-9), P(-9), P(-9) }
        };
        var sky = new float[,]
        {
            { P(-9), P(-9), P(-9), P(5) },
            { P(-9), P(-9), P(-9), P(5) },
            { P(-9), P(-9), P(-9), P(5) },
            { P(-9), P(-9), P(-9), P(5) }
        };
        var canvas = new LetterboxBox(0, 0, 4, 4);

        var box = Assert.Single(TagScorer.FigureBox(character, [dress, sky], threshold: 0.5f, canvas, canvasSize: 4, imageWidth: 4, imageHeight: 4));
        var own = Assert.Single(TagScorer.DetectBoxes(character, 0.5f, 0.6f, canvas, 4, 4, 4, belowPeak: 4f));

        Assert.Equal((0, 0, 2, 3), (box.X, box.Y, box.Width, box.Height));
        Assert.Equal(P(6), box.Confidence);
        Assert.Equal((1, 1), (own.Width, own.Height));
    }

    [Fact]
    public void FigureBox_SplitsATouchingFigureBetweenCharactersButNotFromTheSamePerson()
    {
        static float P(float logit) => 1f / (1f + MathF.Exp(-logit));
        static float[,] PeakAt(int x, int width) { var m = new float[1, width]; for (var i = 0; i < width; i++) m[0, i] = P(i == x ? 6 : -8); return m; }
        // One row of 16 cells, all covered by clothing tags: two characters side by side, touching.
        var clothes = new float[1, 16];
        for (var i = 0; i < 16; i++)
            clothes[0, i] = P(3);
        var canvas = new LetterboxBox(0, 0, 16, 1);
        var a = PeakAt(2, 16);

        var merged = Assert.Single(TagScorer.FigureBox(a, [clothes], 0.5f, canvas, 16, 16, 1));
        var split = Assert.Single(TagScorer.FigureBox(a, [clothes], 0.5f, canvas, 16, 16, 1, otherCharacters: [PeakAt(12, 16)]));
        // A costume variant peaking on the same face (within the same-person distance) doesn't split it.
        var variant = Assert.Single(TagScorer.FigureBox(a, [clothes], 0.5f, canvas, 16, 16, 1, otherCharacters: [PeakAt(3, 16)], samePersonDistance: 0.2f));

        Assert.Equal((0, 16), (merged.X, merged.Width));
        Assert.Equal((0, 8), (split.X, split.Width));
        Assert.Equal((0, 16), (variant.X, variant.Width));
    }

    [Fact]
    public void FigureBox_WithoutOtherTagsBoxesTheWholeImageAndBelowThresholdNothing()
    {
        var character = new float[,] { { 0.9f, 0.01f }, { 0.01f, 0.01f } };
        var canvas = new LetterboxBox(0, 0, 2, 2);

        var whole = Assert.Single(TagScorer.FigureBox(character, [], threshold: 0.5f, canvas, canvasSize: 2, imageWidth: 2, imageHeight: 2));

        Assert.Equal((0, 0, 2, 2), (whole.X, whole.Y, whole.Width, whole.Height));
        Assert.Empty(TagScorer.FigureBox(character, [], threshold: 0.95f, canvas, canvasSize: 2, imageWidth: 2, imageHeight: 2));
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
