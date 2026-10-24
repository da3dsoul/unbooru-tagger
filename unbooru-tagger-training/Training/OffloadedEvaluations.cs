using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Serialization;
using UnbooruTagger.Training.Commands;

namespace UnbooruTagger.Training.Training;

/// <summary>
/// Runs each epoch's evaluation in a separate process (<c>train --evaluation-command</c>) instead
/// of inside the training loop, so training carries on while it runs — typically <c>evaluate</c>
/// on a second GPU. It has to be a separate process: CUDA and XPU libtorch builds can't be
/// loaded into one.
/// <para>
/// At an epoch boundary the trainer writes a model-only snapshot to
/// <c>history/epoch-N/checkpoint/</c> and starts <c>&lt;command&gt; &lt;snapshot dir&gt; &lt;history/epoch-N&gt;</c>;
/// the command is expected to write evaluation_report.json (and rating_report.json,
/// tag_thresholds.json, localization_report.json) into the second directory and exit 0. Results
/// come back in epoch order, so early stopping sees the same sequence it would inline, just
/// later. Marker files make it survive a restart:
/// <c>evaluation.done</c> / <c>evaluation.failed</c> once the process exits, and
/// <c>evaluation.applied</c> once the trainer has acted on it and saved its state. A snapshot with
/// neither done nor failed on startup is evaluated again (a still-running process from before the
/// restart can't be tracked, so it may run twice).
/// </para>
/// </summary>
public sealed class OffloadedEvaluations(string checkpointDir, string command) : IDisposable
{
    public const string SnapshotDirectoryName = "checkpoint";
    private const string DoneMarker = "evaluation.done";
    private const string FailedMarker = "evaluation.failed";
    private const string AppliedMarker = "evaluation.applied";
    private const string LogFileName = "evaluation.log";

    private static readonly JsonSerializerOptions JsonOptions = new() { NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals };

    private readonly SortedDictionary<int, (Process Process, StreamWriter Log)> _running = new();

    public sealed record Result(int Epoch, EvaluationReport? Report, RatingReport? Ratings, string? Failure)
    {
        public string Directory { get; init; } = "";
    }

    public string EpochDirectory(int epoch) => Path.Combine(checkpointDir, "history", $"epoch-{epoch}");

    public string SnapshotDirectory(int epoch) => Path.Combine(EpochDirectory(epoch), SnapshotDirectoryName);

    public int Running => _running.Count;

    /// <summary>Writes epoch <paramref name="epoch"/>'s snapshot via <paramref name="writeSnapshot"/>, then starts its evaluation.</summary>
    public void Launch(int epoch, Action<string> writeSnapshot)
    {
        var epochDir = Directory.CreateDirectory(EpochDirectory(epoch)).FullName;
        foreach (var marker in new[] { DoneMarker, FailedMarker, AppliedMarker })
            File.Delete(Path.Combine(epochDir, marker));
        var snapshot = SnapshotDirectory(epoch);
        Directory.CreateDirectory(snapshot);
        writeSnapshot(snapshot);
        Start(epoch);
    }

    /// <summary>Relaunches or re-queues the evaluations of completed epochs that a previous process left unapplied.</summary>
    public IReadOnlyList<int> ResumePending(int completedEpochs)
    {
        var relaunched = new List<int>();
        for (var epoch = 1; epoch <= completedEpochs; epoch++)
        {
            var epochDir = EpochDirectory(epoch);
            if (!Directory.Exists(SnapshotDirectory(epoch)) || File.Exists(Path.Combine(epochDir, AppliedMarker)))
                continue;
            if (File.Exists(Path.Combine(epochDir, DoneMarker)) || File.Exists(Path.Combine(epochDir, FailedMarker)))
                continue; // finished while no trainer was running: TakeFinished picks it up
            Start(epoch);
            relaunched.Add(epoch);
        }
        return relaunched;
    }

    /// <summary>
    /// Finished, not yet applied evaluations in epoch order, stopping at the first one still running
    /// (so a later epoch never reaches early stopping before an earlier one).
    /// </summary>
    public IReadOnlyList<Result> TakeFinished()
    {
        foreach (var (epoch, (process, log)) in _running.ToList())
        {
            if (!process.HasExited)
                continue;
            process.WaitForExit(); // drains the redirected output handlers
            lock (log)
                log.Dispose();
            var ok = process.ExitCode == 0 && File.Exists(Path.Combine(EpochDirectory(epoch), TrainCommandHandler.EvaluationReportFileName));
            File.WriteAllText(Path.Combine(EpochDirectory(epoch), ok ? DoneMarker : FailedMarker), $"exit code {process.ExitCode}\n");
            process.Dispose();
            _running.Remove(epoch);
        }

        var results = new List<Result>();
        var historyDir = Path.Combine(checkpointDir, "history");
        if (!Directory.Exists(historyDir))
            return results;
        var epochs = Directory.GetDirectories(historyDir, "epoch-*")
            .Select(d => int.TryParse(Path.GetFileName(d)["epoch-".Length..], out var e) ? e : -1)
            .Where(e => e > 0 && Directory.Exists(SnapshotDirectory(e)) && !File.Exists(Path.Combine(EpochDirectory(e), AppliedMarker)))
            .Order();
        foreach (var epoch in epochs)
        {
            var epochDir = EpochDirectory(epoch);
            if (File.Exists(Path.Combine(epochDir, FailedMarker)))
                results.Add(new Result(epoch, null, null, File.ReadAllText(Path.Combine(epochDir, FailedMarker)).Trim()) { Directory = epochDir });
            else if (File.Exists(Path.Combine(epochDir, DoneMarker)))
                results.Add(new Result(epoch, Read<EvaluationReport>(epochDir, TrainCommandHandler.EvaluationReportFileName),
                    Read<RatingReport>(epochDir, TrainCommandHandler.RatingReportFileName), null) { Directory = epochDir });
            else
                break;
        }
        return results;
    }

    /// <summary>Call once the trainer's state reflecting <paramref name="epoch"/>'s result has been saved.</summary>
    public void MarkApplied(int epoch) =>
        File.WriteAllText(Path.Combine(EpochDirectory(epoch), AppliedMarker), DateTime.Now.ToString("O") + "\n");

    /// <summary>Blocks until every running evaluation has exited.</summary>
    public void WaitAll()
    {
        foreach (var (process, _) in _running.Values)
            process.WaitForExit();
    }

    private void Start(int epoch)
    {
        var epochDir = EpochDirectory(epoch);
        var start = new ProcessStartInfo(command)
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
        };
        start.ArgumentList.Add(SnapshotDirectory(epoch));
        start.ArgumentList.Add(epochDir);
        var log = new StreamWriter(Path.Combine(epochDir, LogFileName), append: false) { AutoFlush = true };
        var process = new Process { StartInfo = start };
        DataReceivedEventHandler write = (_, e) =>
        {
            if (e.Data is null)
                return;
            lock (log)
            {
                // After Dispose (trainer exiting) the evaluation keeps running, unlogged.
                if (log.BaseStream is { CanWrite: true })
                    log.WriteLine(e.Data);
            }
        };
        process.OutputDataReceived += write;
        process.ErrorDataReceived += write;
        process.Start();
        process.BeginOutputReadLine();
        process.BeginErrorReadLine();
        _running[epoch] = (process, log);
    }

    private static T? Read<T>(string directory, string fileName) where T : class
    {
        var path = Path.Combine(directory, fileName);
        return File.Exists(path) ? JsonSerializer.Deserialize<T>(File.ReadAllText(path), JsonOptions) : null;
    }

    /// <summary>Running evaluations are left to finish on their own: they write their results to disk, and a restarted trainer picks them up.</summary>
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
