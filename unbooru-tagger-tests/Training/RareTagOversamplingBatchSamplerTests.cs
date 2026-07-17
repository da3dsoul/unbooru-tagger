using UnbooruTagger.Training.Training;

namespace UnbooruTagger.Tests.Training;

public class RareTagOversamplingBatchSamplerTests
{
    [Fact]
    public void SampleBatch_OversamplesTheImageWithTheRarerTag()
    {
        // Image 0 carries only a very common tag; image 1 carries only a very rare one.
        // Natural-frequency sampling would pick image 0 far more often — the sampler
        // should invert that.
        IReadOnlyList<IReadOnlyList<int>> imageTagRows = [[0], [1]];
        var tagFrequencies = new Dictionary<int, int> { [0] = 1000, [1] = 1 };
        var sampler = new RareTagOversamplingBatchSampler(imageTagRows, tagFrequencies, new Random(42));

        var batch = sampler.SampleBatch(10_000);

        var rareImageCount = batch.Count(i => i == 1);
        var commonImageCount = batch.Count(i => i == 0);
        Assert.True(rareImageCount > commonImageCount);
    }
}
