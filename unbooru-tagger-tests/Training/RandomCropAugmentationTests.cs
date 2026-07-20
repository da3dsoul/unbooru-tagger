using UnbooruTagger.Training.Training;
using static TorchSharp.torch;

namespace UnbooruTagger.Tests.Training;

public class RandomCropAugmentationTests
{
    [Fact]
    public void Apply_PreservesBatchShape()
    {
        var batch = rand([4, 3, 16, 16]);

        var augmented = RandomCropAugmentation.Apply(batch, random: new Random(1));

        Assert.Equal(batch.shape, augmented.shape);
    }

    [Fact]
    public void Apply_OnlyReusesPixelsFromTheSourceImage()
    {
        // Every non-zero cell in the output must have come from somewhere in the source
        // image (the augmentation crops+pastes, it never invents new pixel values) --
        // guards against e.g. an off-by-one in the crop/paste bounds pulling in garbage.
        var batch = rand([2, 3, 12, 12]) + 1f; // shift away from 0 so "untouched" border cells are distinguishable
        var sourceMin = batch.min().item<float>();
        var sourceMax = batch.max().item<float>();

        var augmented = RandomCropAugmentation.Apply(batch, minScale: 0.5, random: new Random(42));

        var augmentedMin = augmented.min().item<float>();
        var augmentedMax = augmented.max().item<float>();

        Assert.True(augmentedMin >= Math.Min(0f, sourceMin));
        Assert.True(augmentedMax <= sourceMax);
    }
}
