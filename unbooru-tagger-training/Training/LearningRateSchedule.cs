namespace UnbooruTagger.Training.Training;

/// <summary>
/// Linear warmup to the base rate over <c>warmupSteps</c>, then cosine decay to
/// <c>minFactor</c>·base by the last step. Warmup matters for a larger tower trained from
/// scratch with Adam: early per-parameter variance estimates are noisy, and full-size steps
/// in the first few hundred updates can push LayerScale/normalization into a bad region.
/// Computed from the global step so a resumed run lands on exactly the same rate.
/// </summary>
public static class LearningRateSchedule
{
    public static double At(long globalStep, long totalSteps, double baseRate, long warmupSteps, double minFactor = 0.05)
    {
        if (warmupSteps > 0 && globalStep < warmupSteps)
            return baseRate * (globalStep + 1) / warmupSteps;

        var decaySteps = Math.Max(1, totalSteps - warmupSteps);
        var progress = Math.Clamp((globalStep - warmupSteps) / (double)decaySteps, 0, 1);
        return baseRate * (minFactor + (1 - minFactor) * 0.5 * (1 + Math.Cos(Math.PI * progress)));
    }
}
