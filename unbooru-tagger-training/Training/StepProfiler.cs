using System.Diagnostics;
using static TorchSharp.torch;

namespace UnbooruTagger.Training.Training;

/// <summary>
/// Where a training step's time goes, for tuning (<c>UNBOORU_PROFILE_STEPS=N</c>: N steps after
/// <see cref="WarmupSteps"/> are timed, then the totals are written to <c>step_profile.txt</c> in the checkpoint
/// directory). The device runs asynchronously, so each mark first waits for the phase's output tensor; that
/// serializes CPU and device work, so the phases add up to more than an unprofiled step, but each one's share is
/// what it costs on its own.
/// </summary>
public sealed class StepProfiler
{
    public const int WarmupSteps = 50;
    private readonly int _steps;
    private readonly string _path;
    private readonly Dictionary<string, double> _totals = new();
    private readonly List<string> _order = [];
    private readonly Stopwatch _clock = new();
    private int _step = -1;

    private StepProfiler(int steps, string path) => (_steps, _path) = (steps, path);

    /// <summary>A profiler when <c>UNBOORU_PROFILE_STEPS</c> is set, else null (no cost).</summary>
    public static StepProfiler? FromEnvironment(string checkpointDir) =>
        int.TryParse(Environment.GetEnvironmentVariable("UNBOORU_PROFILE_STEPS"), out var steps) && steps > 0
            ? new StepProfiler(steps, Path.Combine(checkpointDir, "step_profile.txt"))
            : null;

    private bool Active => _step >= WarmupSteps && _step < WarmupSteps + _steps;

    public void StepStarted()
    {
        _step++;
        _clock.Restart();
    }

    /// <summary>Ends phase <paramref name="name"/> once <paramref name="output"/> (if any) is computed.</summary>
    public void Mark(string name, Tensor? output = null)
    {
        if (!Active)
            return;
        if (output is not null)
            using (no_grad())
                _ = output.flatten()[0].to_type(ScalarType.Float32).item<float>();
        if (!_totals.ContainsKey(name))
            _order.Add(name);
        _totals[name] = _totals.GetValueOrDefault(name) + _clock.Elapsed.TotalMilliseconds;
        _clock.Restart();
    }

    /// <summary>Writes the totals after the last profiled step; true once written.</summary>
    public bool StepFinished()
    {
        if (_step != WarmupSteps + _steps - 1)
            return false;
        var total = _totals.Values.Sum();
        var lines = _order.Select(n => $"{n,-28} {_totals[n] / _steps,8:0.0} ms  {100 * _totals[n] / total,5:0.0}%").ToList();
        lines.Add($"{"total (synchronized)",-28} {total / _steps,8:0.0} ms over {_steps} steps");
        File.WriteAllLines(_path, lines);
        return true;
    }
}
