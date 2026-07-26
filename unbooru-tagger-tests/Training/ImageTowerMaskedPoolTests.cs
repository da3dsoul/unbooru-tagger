using UnbooruTagger.Training.Model;
using static TorchSharp.torch;

namespace UnbooruTagger.Tests.Training;

public class ImageTowerMaskedPoolTests
{
    [Fact]
    public void MaskedPool_IgnoresMaskedOutLocations()
    {
        // [1, 1, 2, 2]: three real locations near 0, one extreme outlier that's masked
        // out. A plain average would be dragged far off; MaskedPool should land close
        // to the three real locations instead.
        var spatial = tensor(new float[] { 1f, 1f, 1f, 1000f }, [1, 1, 2, 2]);
        var mask = tensor(new float[] { 1f, 1f, 1f, 0f }, [1, 1, 2, 2]);

        var pooled = ImageTower.MaskedPool(spatial, mask);

        Assert.Equal(1f, pooled.item<float>(), 3);
    }

    [Fact]
    public void MaskedPool_MatchesPlainMean_WhenEveryLocationIsValid()
    {
        var spatial = tensor(new float[] { 2f, 4f, 6f, 8f }, [1, 1, 2, 2]);
        var mask = ones([1, 1, 2, 2]);

        var pooled = ImageTower.MaskedPool(spatial, mask);

        Assert.Equal(5f, pooled.item<float>(), 3);
    }
}
