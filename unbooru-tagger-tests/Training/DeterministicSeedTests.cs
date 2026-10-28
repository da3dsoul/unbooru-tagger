using UnbooruTagger.Training.Training;

namespace UnbooruTagger.Tests.Training;

public class DeterministicSeedTests
{
    [Fact]
    public void TheSameSeedAndIndex_GiveTheSameSeedInEveryProcess()
    {
        // A fixed value, not just a repeat within this process: HashCode.Combine passed that and still changed per run.
        Assert.Equal(DeterministicSeed.For(7, 3), DeterministicSeed.For(7, 3));
        Assert.Equal(2065550767, DeterministicSeed.For(0, 0));
    }

    [Fact]
    public void NeighbouringIndicesAndSeeds_GiveDifferentSeeds()
    {
        var seeds = new HashSet<int>();
        for (var seed = 0; seed < 4; seed++)
            for (var index = 0; index < 1000; index++)
                seeds.Add(DeterministicSeed.For(seed, index));
        Assert.Equal(4000, seeds.Count);
    }
}
