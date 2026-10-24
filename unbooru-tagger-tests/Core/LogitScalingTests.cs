using UnbooruTagger.Core.Scoring;

namespace UnbooruTagger.Tests.Core;

public class LogitScalingTests
{
    [Fact]
    public void Identity_LeavesALogitUnchanged()
    {
        Assert.Equal(0.42f, LogitScaling.Identity.Apply(0.42f), 5);
    }

    [Fact]
    public void Apply_ScalesThenAddsBias()
    {
        var scaling = new LogitScaling(Scale: 2f, Bias: -1f);

        Assert.Equal(3f, scaling.Apply(2f), 5);
    }

    [Fact]
    public void Load_FallsBackToIdentityWhenTheFileDoesNotExist()
    {
        var path = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid()}.json");

        var loaded = LogitScaling.Load(path);

        Assert.Equal(LogitScaling.Identity, loaded);
    }

    [Fact]
    public void SaveThenLoad_RoundTrips()
    {
        var path = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid()}.json");
        try
        {
            var original = new LogitScaling(Scale: 9.5f, Bias: -8.25f);
            original.Save(path);

            var loaded = LogitScaling.Load(path);

            Assert.Equal(original, loaded);
        }
        finally
        {
            File.Delete(path);
        }
    }
}
