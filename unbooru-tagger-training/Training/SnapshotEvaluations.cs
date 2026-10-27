using System.Diagnostics;
using System.Text.Json;
using UnbooruTagger.Training.Commands;

namespace UnbooruTagger.Training.Training;

/// <summary>
/// The headline numbers of one evaluation, as a point on the dashboard's trend charts. Read from
/// the report files rather than kept in the trainer, so a restarted run gets its whole history back
/// from disk.
/// </summary>
/// <param name="Step">Global training step (over the whole run) the evaluated weights are from.</param>
/// <param name="EndOfEpoch">An epoch evaluation rather than a mid-epoch snapshot.</param>
/// <param name="GoldPointing">Gold-set pointing accuracy over all pairs; NaN without a gold set.</param>
/// <param name="RecallGap">Context reliance's macro recall gap (see <see cref="ContextRelianceReport"/>).</param>
/// <param name="FalsePositiveGap">Its macro false-positive gap.</param>
/// <param name="OccludedStillCalled">Occlusion check's still-called rate over all pairs; NaN without a gold set.</param>
public sealed record EvaluationPoint(
    long Step, bool EndOfEpoch, double MeanAveragePrecision, double GoldPointing, double RecallGap, double FalsePositiveGap,
    double OccludedStillCalled, double ExclusiveViolations)
{
    /// <summary>Reads <paramref name="directory"/>'s evaluation_report.json (and localization_report.json, if any); null if there's no report.</summary>
    public static EvaluationPoint? Read(string directory, long step, bool endOfEpoch)
    {
        var reportPath = Path.Combine(directory, TrainCommandHandler.EvaluationReportFileName);
        if (!File.Exists(reportPath))
            return null;
        try
        {
            // Only a few fields of reports holding every tag's results: JsonDocument, not the records.
            using var report = JsonDocument.Parse(File.ReadAllBytes(reportPath));
            var root = report.RootElement;
            var context = root.TryGetProperty(nameof(EvaluationReport.ContextReliance), out var c) && c.ValueKind == JsonValueKind.Object ? c : default;

            double goldPointing = double.NaN, occluded = double.NaN;
            var localizationPath = Path.Combine(directory, EvaluateCommandHandler.LocalizationReportFileName);
            if (File.Exists(localizationPath))
            {
                using var localization = JsonDocument.Parse(File.ReadAllBytes(localizationPath));
                goldPointing = All(localization.RootElement, nameof(LocalizationReport.Gold), nameof(GoldResult.PointingAccuracy));
                occluded = All(localization.RootElement, nameof(LocalizationReport.Occlusion), nameof(OcclusionResult.StillYesRate));
            }

            return new EvaluationPoint(step, endOfEpoch,
                Number(root, nameof(EvaluationReport.MeanAveragePrecision)), goldPointing,
                Number(context, nameof(ContextRelianceReport.MacroRecallGap)), Number(context, nameof(ContextRelianceReport.MacroFalsePositiveGap)),
                occluded, Number(root, nameof(EvaluationReport.ExclusiveViolationRate)));
        }
        catch (Exception e) when (e is JsonException or IOException)
        {
            return null; // still being written, or damaged: not a point
        }
    }

    /// <summary>The "all" row's <paramref name="field"/> of a list of per-group results.</summary>
    private static double All(JsonElement root, string list, string field)
    {
        if (!root.TryGetProperty(list, out var rows) || rows.ValueKind != JsonValueKind.Array)
            return double.NaN;
        foreach (var row in rows.EnumerateArray())
            if (row.TryGetProperty("Name", out var name) && name.GetString() == "all")
                return Number(row, field);
        return double.NaN;
    }

    /// <summary>A number, or NaN when missing (reports write NaN as the string "NaN").</summary>
    private static double Number(JsonElement element, string name) =>
        element.ValueKind == JsonValueKind.Object && element.TryGetProperty(name, out var value)
            ? value.ValueKind == JsonValueKind.Number ? value.GetDouble()
            : value.ValueKind == JsonValueKind.String && double.TryParse(value.GetString(), System.Globalization.CultureInfo.InvariantCulture, out var parsed) ? parsed
            : double.NaN
            : double.NaN;
}

/// <summary>
/// Mid-epoch evaluations for the dashboard's trend charts (<c>train --snapshot-evaluation-steps</c>):
/// every N global steps the trainer writes a model-only snapshot to <c>snapshots/step-N/checkpoint/</c>
/// and runs the same <c>--evaluation-command</c> as the epoch evaluations
/// (<see cref="OffloadedEvaluations"/>) on it, reports into <c>snapshots/step-N/</c>. Unlike epoch
/// evaluations these never feed early stopping, so results are taken in whatever order they finish.
/// Once one succeeds its snapshot weights are deleted (only the reports are kept). Markers
/// (<c>evaluation.done</c> / <c>evaluation.failed</c>) make it survive a restart; see <see cref="Resume"/>.
/// </summary>
public sealed class SnapshotEvaluations(string checkpointDir, string command) : IDisposable
{
    public const string DirectoryName = "snapshots";

    private readonly Dictionary<long, (Process Process, StreamWriter Log)> _running = new();

    public sealed record Result(long Step, EvaluationPoint? Point, string? Failure, string Directory);

    public string Root => Path.Combine(checkpointDir, DirectoryName);

    public string StepDirectory(long step) => Path.Combine(Root, $"step-{step}");

    public string SnapshotDirectory(long step) => Path.Combine(StepDirectory(step), OffloadedEvaluations.SnapshotDirectoryName);

    /// <summary>Steps whose evaluation is running in this process.</summary>
    public IReadOnlyList<long> Running => _running.Keys.Order().ToList();

    /// <summary>Writes step <paramref name="step"/>'s snapshot via <paramref name="writeSnapshot"/>, then starts its evaluation.</summary>
    public void Launch(long step, Action<string> writeSnapshot)
    {
        var stepDir = StepDirectory(step);
        if (Directory.Exists(stepDir))
            Directory.Delete(stepDir, recursive: true);
        Directory.CreateDirectory(SnapshotDirectory(step));
        writeSnapshot(SnapshotDirectory(step));
        _running[step] = OffloadedEvaluations.StartProcess(command, SnapshotDirectory(step), stepDir);
    }

    /// <summary>
    /// On startup, resuming at global step <paramref name="resumedStep"/>: deletes snapshots past it
    /// (taken after the checkpoint being resumed was saved, so from weights this run no longer has;
    /// they're taken again when training gets there), relaunches the ones whose evaluation never
    /// finished, and returns the finished ones in step order: the charts' history.
    /// </summary>
    public (IReadOnlyList<Result> Finished, IReadOnlyList<long> Relaunched) Resume(long resumedStep)
    {
        var finished = new List<Result>();
        var relaunched = new List<long>();
        foreach (var step in Steps())
        {
            var stepDir = StepDirectory(step);
            if (step > resumedStep)
            {
                Directory.Delete(stepDir, recursive: true);
                continue;
            }
            if (File.Exists(Path.Combine(stepDir, OffloadedEvaluations.DoneMarker)) || File.Exists(Path.Combine(stepDir, OffloadedEvaluations.FailedMarker)))
                finished.Add(Finish(step));
            else if (Directory.Exists(SnapshotDirectory(step)))
            {
                _running[step] = OffloadedEvaluations.StartProcess(command, SnapshotDirectory(step), stepDir);
                relaunched.Add(step);
            }
        }
        return (finished, relaunched);
    }

    /// <summary>Evaluations that finished since the last call, marked and (on success) with their snapshot weights deleted.</summary>
    public IReadOnlyList<Result> TakeFinished()
    {
        List<Result>? results = null;
        foreach (var (step, (process, log)) in _running)
        {
            if (!process.HasExited)
                continue;
            process.WaitForExit(); // drains the redirected output handlers
            lock (log)
                log.Dispose();
            var stepDir = StepDirectory(step);
            var ok = process.ExitCode == 0 && File.Exists(Path.Combine(stepDir, TrainCommandHandler.EvaluationReportFileName));
            File.WriteAllText(Path.Combine(stepDir, ok ? OffloadedEvaluations.DoneMarker : OffloadedEvaluations.FailedMarker), $"exit code {process.ExitCode}\n");
            process.Dispose();
            (results ??= []).Add(Finish(step));
        }
        if (results is null)
            return [];
        foreach (var result in results)
            _running.Remove(result.Step);
        return results;
    }

    /// <summary>Blocks until every running evaluation has exited.</summary>
    public void WaitAll()
    {
        foreach (var (process, _) in _running.Values)
            process.WaitForExit();
    }

    private Result Finish(long step)
    {
        var stepDir = StepDirectory(step);
        var failed = Path.Combine(stepDir, OffloadedEvaluations.FailedMarker);
        if (File.Exists(failed))
            return new Result(step, null, File.ReadAllText(failed).Trim(), stepDir); // weights kept, to look into
        if (Directory.Exists(SnapshotDirectory(step)))
            Directory.Delete(SnapshotDirectory(step), recursive: true);
        var point = EvaluationPoint.Read(stepDir, step, endOfEpoch: false);
        return new Result(step, point, point is null ? "no readable evaluation_report.json" : null, stepDir);
    }

    private IEnumerable<long> Steps() =>
        Directory.Exists(Root)
            ? Directory.GetDirectories(Root, "step-*")
                .Select(d => long.TryParse(Path.GetFileName(d)["step-".Length..], out var s) ? s : -1)
                .Where(s => s >= 0)
                .Order()
            : [];

    /// <summary>Running evaluations are left to finish on their own; a restarted trainer picks up their results.</summary>
    public void Dispose()
    {
        foreach (var (process, log) in _running.Values)
        {
            process.Dispose();
            lock (log)
                log.Dispose();
        }
        _running.Clear();
    }
}
