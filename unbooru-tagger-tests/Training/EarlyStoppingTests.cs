using UnbooruTagger.Training.Training;

namespace UnbooruTagger.Tests.Training;

public class EarlyStoppingTests
{
    [Fact]
    public void ShouldStop_StaysFalseWhileLossKeepsImproving()
    {
        var earlyStopping = new EarlyStopping(patience: 2);

        Assert.False(earlyStopping.ShouldStop(1.0));
        Assert.False(earlyStopping.ShouldStop(0.8));
        Assert.False(earlyStopping.ShouldStop(0.6));
    }

    [Fact]
    public void ShouldStop_FiresAfterPatienceExhaustedWithoutImprovement()
    {
        var earlyStopping = new EarlyStopping(patience: 2, minDelta: 1e-4);

        Assert.False(earlyStopping.ShouldStop(1.0)); // best so far
        Assert.False(earlyStopping.ShouldStop(1.0)); // 1st evaluation without improvement
        Assert.True(earlyStopping.ShouldStop(1.0));  // 2nd — patience exhausted
    }

    [Fact]
    public void ShouldStop_ResetsPatienceOnImprovement()
    {
        var earlyStopping = new EarlyStopping(patience: 2, minDelta: 1e-4);

        Assert.False(earlyStopping.ShouldStop(1.0));
        Assert.False(earlyStopping.ShouldStop(1.0)); // no improvement, patience used once
        Assert.False(earlyStopping.ShouldStop(0.5)); // improved — patience resets
        Assert.False(earlyStopping.ShouldStop(0.5));
        Assert.True(earlyStopping.ShouldStop(0.5));
    }
}
