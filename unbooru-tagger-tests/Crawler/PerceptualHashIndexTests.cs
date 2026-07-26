using UnbooruTagger.Crawler;

namespace UnbooruTagger.Tests.Crawler;

public class PerceptualHashIndexTests
{
    [Fact]
    public void FindNear_ReturnsNull_WhenIndexIsEmpty()
    {
        var index = new PerceptualHashIndex(maxHammingDistance: 2, existing: []);
        Assert.Null(index.FindNear(0x1234_5678_9ABC_DEF0));
    }

    [Fact]
    public void FindNear_FindsExactMatch()
    {
        var index = new PerceptualHashIndex(maxHammingDistance: 2, existing: []);
        index.Add(0x1234_5678_9ABC_DEF0, "abc");

        Assert.Equal("abc", index.FindNear(0x1234_5678_9ABC_DEF0));
    }

    [Fact]
    public void FindNear_FindsHashWithinThreshold()
    {
        var index = new PerceptualHashIndex(maxHammingDistance: 2, existing: []);
        var original = 0x0000_0000_0000_0000UL;
        index.Add(original, "abc");

        // Flip 2 low bits — within the distance-2 threshold.
        var nearby = original ^ 0b11UL;
        Assert.Equal("abc", index.FindNear(nearby));
    }

    [Fact]
    public void FindNear_ReturnsNull_WhenBeyondThreshold()
    {
        var index = new PerceptualHashIndex(maxHammingDistance: 2, existing: []);
        var original = 0x0000_0000_0000_0000UL;
        index.Add(original, "abc");

        // Flip 3 bits, spread across different bands so no band matches exactly.
        var farther = original ^ ((1UL << 5) | (1UL << 30) | (1UL << 50));
        Assert.Null(index.FindNear(farther));
    }

    [Fact]
    public void Constructor_SeedsFromExistingEntries()
    {
        var index = new PerceptualHashIndex(maxHammingDistance: 2, existing: [(0x1111_1111_1111_1111UL, "seed-a"), (0xFFFF_FFFF_FFFF_FFFFUL, "seed-b")]);

        Assert.Equal("seed-a", index.FindNear(0x1111_1111_1111_1111UL));
        Assert.Equal("seed-b", index.FindNear(0xFFFF_FFFF_FFFF_FFFFUL));
    }

    /// <summary>
    /// The banding scheme is a correctness-critical optimization (any pair within the
    /// threshold MUST still be found — the whole point is speed without dropping real
    /// duplicates), so this checks it against a brute-force reference over many random
    /// hashes and thresholds rather than only a few hand-picked bit patterns.
    /// </summary>
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(4)]
    public void FindNear_AgreesWithBruteForceReference(int maxHammingDistance)
    {
        var random = new Random(42);
        var entries = Enumerable.Range(0, 300)
            .Select(i => (Hash: NextHash(random), Md5: $"img-{i}"))
            .ToList();

        var index = new PerceptualHashIndex(maxHammingDistance, entries);

        for (var q = 0; q < 200; q++)
        {
            var query = NextHash(random);
            var bruteForceMatch = entries.Any(e => PerceptualHash.HammingDistance(e.Hash, query) <= maxHammingDistance);
            var indexMatch = index.FindNear(query) is not null;
            Assert.Equal(bruteForceMatch, indexMatch);
        }
    }

    private static ulong NextHash(Random random)
    {
        Span<byte> bytes = stackalloc byte[8];
        random.NextBytes(bytes);
        return BitConverter.ToUInt64(bytes);
    }
}
