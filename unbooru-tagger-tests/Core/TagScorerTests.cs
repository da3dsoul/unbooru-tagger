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
}
