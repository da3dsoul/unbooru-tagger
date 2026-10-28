namespace UnbooruTagger.Training.Training;

/// <summary>
/// Per-item seeds derived from a run's seed. <see cref="HashCode.Combine{T1,T2}"/> is randomized per process, so the
/// samples it seeded (tag statistics, evaluation negatives, context priors) changed on every run with the same seed:
/// two evaluations of one checkpoint differed by ~0.001 mAP and by dozens of images per context-prior band.
/// </summary>
public static class DeterministicSeed
{
    /// <summary>A seed for item <paramref name="index"/> of a run seeded <paramref name="seed"/> (SplitMix64's finalizer).</summary>
    public static int For(int seed, int index)
    {
        var z = ((ulong)(uint)seed << 32 | (uint)index) + 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return (int)(z ^ (z >> 31));
    }
}
