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
/// <param name="SmallGoldPointing">The same as <paramref name="GoldPointing"/> for the <c>evaluate --gold-small</c> set; NaN without one.</param>
/// <param name="SmallOccludedStillCalled">The same as <paramref name="OccludedStillCalled"/> for that set.</param>
/// <param name="ObjectEffect">Occlusion check's object effect over every gold pair (<see cref="OcclusionResult.ObjectEffect"/>); NaN before it existed.</param>
/// <param name="SmallObjectEffect">The same for the gold-small set.</param>
/// <param name="LabelledAbsentLogit">Hand-labelled context negatives where the object is absent: mean log-odds past the tag's threshold
/// (<see cref="LabelledContextReport.AbsentMeanLogit"/>); NaN without <c>evaluate --context-check</c>. Not the called rate: about 200 pairs
/// called at thresholds recalibrated each evaluation, so a few pairs flipping moves it by points.</param>
/// <param name="LabelledPresentLogit">The same for those where it's visible but untagged.</param>
/// <param name="GoldPointingBySize">Gold pointing per size bucket ("characters", "large", "medium", "small"); empty without a gold set.</param>
/// <param name="GoldTolerantBySize">The same with <see cref="GoldResult.TolerantPointing"/> (and "all"); empty in reports from before it existed.</param>
/// <param name="At1024">The gold checks at 1024px (<see cref="HigherScalePoint"/>); null where the evaluation didn't run them.</param>
public sealed record EvaluationPoint(
    long Step, bool EndOfEpoch, double MeanAveragePrecision, double GoldPointing, double RecallGap, double FalsePositiveGap,
    double OccludedStillCalled, double ExclusiveViolations, double SmallGoldPointing = double.NaN, double SmallOccludedStillCalled = double.NaN,
    double ObjectEffect = double.NaN, double SmallObjectEffect = double.NaN,
    double LabelledAbsentLogit = double.NaN, double LabelledPresentLogit = double.NaN,
    IReadOnlyDictionary<string, double>? GoldPointingBySize = null, IReadOnlyDictionary<string, double>? GoldTolerantBySize = null,
    HigherScalePoint? At1024 = null)
{
    /// <summary>The gold set's size buckets, as <see cref="LocalizationEvaluator.SizeBucket"/> names them, by short name.</summary>
    public static readonly (string Short, string Row)[] SizeRows =
        [("characters", "size: characters"), ("large", "size: large >=96px"), ("medium", "size: medium 32-96px"), ("small", "size: small <32px")];

    /// <summary>A per-size value, NaN when missing.</summary>
    public static double Get(IReadOnlyDictionary<string, double>? values, string key) => values is not null && values.TryGetValue(key, out var v) ? v : double.NaN;

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
            var labelled = root.TryGetProperty(nameof(EvaluationReport.LabelledContext), out var l) && l.ValueKind == JsonValueKind.Object ? l : default;

            var goldPath = Path.Combine(directory, EvaluateCommandHandler.LocalizationReportFileName);
            var (goldPointing, occluded, effect) = Localization(goldPath);
            var (bySize, tolerantBySize) = Sizes(goldPath);
            var (smallPointing, smallOccluded, smallEffect) = Localization(Path.Combine(directory, EvaluateCommandHandler.SmallGoldReportFileName));

            return new EvaluationPoint(step, endOfEpoch,
                Number(root, nameof(EvaluationReport.MeanAveragePrecision)), goldPointing,
                Number(context, nameof(ContextRelianceReport.MacroRecallGap)), Number(context, nameof(ContextRelianceReport.MacroFalsePositiveGap)),
                occluded, Number(root, nameof(EvaluationReport.ExclusiveViolationRate)), smallPointing, smallOccluded, effect, smallEffect,
                Number(labelled, nameof(LabelledContextReport.AbsentMeanLogit)), Number(labelled, nameof(LabelledContextReport.PresentMeanLogit)),
                bySize, tolerantBySize, HigherScalePoint.Read(directory, 1024));
        }
        catch (Exception e) when (e is JsonException or IOException)
        {
            return null; // still being written, or damaged: not a point
        }
    }

    /// <summary>A localization report's overall gold pointing, occlusion still-called rate and object effect; NaN if it doesn't exist.</summary>
    internal static (double Pointing, double StillCalled, double ObjectEffect) Localization(string path)
    {
        if (!File.Exists(path))
            return (double.NaN, double.NaN, double.NaN);
        using var localization = JsonDocument.Parse(File.ReadAllBytes(path));
        return (All(localization.RootElement, nameof(LocalizationReport.Gold), nameof(GoldResult.PointingAccuracy)),
            All(localization.RootElement, nameof(LocalizationReport.Occlusion), nameof(OcclusionResult.StillYesRate)),
            All(localization.RootElement, nameof(LocalizationReport.Occlusion), nameof(OcclusionResult.ObjectEffect)));
    }

    /// <summary>A localization report's gold pointing per size bucket, strict and tolerant (the tolerant one with "all" too).</summary>
    internal static (Dictionary<string, double> Strict, Dictionary<string, double> Tolerant) Sizes(string path)
    {
        var (strict, tolerant) = (new Dictionary<string, double>(), new Dictionary<string, double>());
        if (!File.Exists(path))
            return (strict, tolerant);
        using var localization = JsonDocument.Parse(File.ReadAllBytes(path));
        if (!localization.RootElement.TryGetProperty(nameof(LocalizationReport.Gold), out var rows) || rows.ValueKind != JsonValueKind.Array)
            return (strict, tolerant);
        foreach (var row in rows.EnumerateArray())
        {
            var name = row.TryGetProperty("Name", out var n) ? n.GetString() : null;
            var key = name == "all" ? "all" : SizeRows.FirstOrDefault(s => s.Row == name).Short;
            if (key is null)
                continue;
            if (key != "all" && Number(row, nameof(GoldResult.PointingAccuracy)) is var p && !double.IsNaN(p))
                strict[key] = p;
            if (Number(row, nameof(GoldResult.TolerantPointing)) is var t && !double.IsNaN(t))
                tolerant[key] = t;
        }
        return (strict, tolerant);
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
/// The gold checks at a larger input size, beside an evaluation's own (2026-11-04: training mixes 512-2048px batches,
/// <c>--train-scales</c>, while the trend evaluations score at the checkpoint's 512px, the size it gains least at). The evaluation
/// command writes them into <c>at{size}/</c> (gold and gold-small, from the originals shrunk to that size: review/aliasing/area1024)
/// and <c>at{size}-auto/</c> (gold-auto) beside its own reports.
/// </summary>
/// <param name="GoldPointing">Gold pointing over all pairs.</param>
/// <param name="GoldBySize">Gold pointing per size bucket (as <see cref="EvaluationPoint.GoldPointingBySize"/>; sizes are the 512px ones).</param>
/// <param name="GoldTolerantBySize">The same within 16px.</param>
/// <param name="SmallGoldPointing">Gold-small pointing over all pairs.</param>
/// <param name="ObjectEffect">The gold occlusion object effect.</param>
/// <param name="AutoPointing">Gold-auto (SAM 3 boxes, 1,020 images) pointing over all pairs.</param>
/// <param name="AutoBySize">Its pointing per size bucket.</param>
/// <param name="AutoTolerantBySize">The same within 16px.</param>
public sealed record HigherScalePoint(
    double GoldPointing, IReadOnlyDictionary<string, double> GoldBySize, IReadOnlyDictionary<string, double> GoldTolerantBySize,
    double SmallGoldPointing, double ObjectEffect,
    double AutoPointing, IReadOnlyDictionary<string, double> AutoBySize, IReadOnlyDictionary<string, double> AutoTolerantBySize)
{
    public static string DirectoryName(int size) => $"at{size}";

    public static string AutoDirectoryName(int size) => $"at{size}-auto";

    /// <summary>The checks at <paramref name="size"/> beside an evaluation in <paramref name="directory"/>; null if neither was run.</summary>
    public static HigherScalePoint? Read(string directory, int size)
    {
        var gold = Path.Combine(directory, DirectoryName(size), EvaluateCommandHandler.LocalizationReportFileName);
        var auto = Path.Combine(directory, AutoDirectoryName(size), EvaluateCommandHandler.LocalizationReportFileName);
        if (!File.Exists(gold) && !File.Exists(auto))
            return null;
        var (goldPointing, _, effect) = EvaluationPoint.Localization(gold);
        var (goldBySize, goldTolerant) = EvaluationPoint.Sizes(gold);
        var (smallPointing, _, _) = EvaluationPoint.Localization(Path.Combine(directory, DirectoryName(size), EvaluateCommandHandler.SmallGoldReportFileName));
        var (autoPointing, _, _) = EvaluationPoint.Localization(auto);
        var (autoBySize, autoTolerant) = EvaluationPoint.Sizes(auto);
        return new HigherScalePoint(goldPointing, goldBySize, goldTolerant, smallPointing, effect, autoPointing, autoBySize, autoTolerant);
    }
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
