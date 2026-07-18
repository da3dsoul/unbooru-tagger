namespace UnbooruTagger.Training.Training;

/// <summary>
/// Random weight initialization can occasionally land in a numerically unstable
/// configuration for this architecture — investigated at length, not fully eliminable
/// via architecture changes alone within reasonable effort. Real training doesn't fix a
/// seed, so a fresh process gets fresh random weights and very likely won't repeat it:
/// fail fast with an actionable message instead of grinding through the rest of a run
/// producing garbage.
/// </summary>
public static class NaNGuard
{
    public const string Message =
        "Training diverged to NaN loss. This can happen when random weight " +
        "initialization lands in a numerically unstable configuration for this " +
        "architecture. Simply re-running (a fresh process gets fresh random weights) " +
        "will very likely avoid it; if it keeps happening, try a lower --lr.";
}
