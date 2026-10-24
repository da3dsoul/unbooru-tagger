using System.Text.Json;
using System.Text.Json.Serialization;
using UnbooruTagger.Training.Training;

namespace UnbooruTagger.Tests.Training;

public class OffloadedEvaluationsTests
{
    // A stand-in for `evaluate`: writes a report whose mAP is the epoch number / 10, unless the
    // snapshot holds a "fail" file (exit 3) or a "wait" file (blocks until it's deleted).
    private const string Script = """
        #!/usr/bin/env bash
        set -eu
        snapshot="$1"; out="$2"
        while [ -e "$snapshot/wait" ]; do sleep 0.05; done
        [ -e "$snapshot/fail" ] && exit 3
        epoch="${out##*epoch-}"
        sed "s/__MAP__/0.$epoch/" "$(dirname "$0")/report-template.json" > "$out/evaluation_report.json"
        """;

    [Fact]
    public void FinishedEvaluation_ComesBackOnceWithItsReport()
    {
        if (!OperatingSystem.IsLinux())
            return;
        using var fixture = new Fixture();
        using var offloaded = fixture.Create();

        offloaded.Launch(1, snapshot => File.WriteAllText(Path.Combine(snapshot, "marker"), "x"));
        var results = WaitForResults(offloaded, 1);

        var result = Assert.Single(results);
        Assert.Equal(1, result.Epoch);
        Assert.Null(result.Failure);
        Assert.Equal(0.1, result.Report!.MeanAveragePrecision, 6);
        Assert.True(File.Exists(Path.Combine(offloaded.SnapshotDirectory(1), "marker")));

        // Until it's marked applied it keeps being returned (the trainer applies it at its next save).
        Assert.Single(offloaded.TakeFinished());
        offloaded.MarkApplied(1);
        Assert.Empty(offloaded.TakeFinished());
    }

    [Fact]
    public void FailedEvaluation_IsReportedAsAFailure()
    {
        if (!OperatingSystem.IsLinux())
            return;
        using var fixture = new Fixture();
        using var offloaded = fixture.Create();

        offloaded.Launch(1, snapshot => File.WriteAllText(Path.Combine(snapshot, "fail"), ""));
        var result = Assert.Single(WaitForResults(offloaded, 1));

        Assert.Null(result.Report);
        Assert.Contains("exit code 3", result.Failure);
    }

    [Fact]
    public void LaterEpoch_WaitsForAnEarlierOneStillRunning()
    {
        if (!OperatingSystem.IsLinux())
            return;
        using var fixture = new Fixture();
        using var offloaded = fixture.Create();

        offloaded.Launch(1, snapshot => File.WriteAllText(Path.Combine(snapshot, "wait"), ""));
        offloaded.Launch(2, _ => { });
        // Epoch 2 finishes first, but early stopping must see epoch 1's result before it.
        WaitUntil(() => File.Exists(Path.Combine(offloaded.EpochDirectory(2), "evaluation_report.json")));
        Thread.Sleep(200);
        Assert.Empty(offloaded.TakeFinished());

        File.Delete(Path.Combine(offloaded.SnapshotDirectory(1), "wait"));
        var results = WaitForResults(offloaded, 2);
        Assert.Equal([1, 2], results.Select(r => r.Epoch));
    }

    [Fact]
    public void ResumePending_RerunsUnfinishedAndKeepsFinishedUnapplied()
    {
        if (!OperatingSystem.IsLinux())
            return;
        using var fixture = new Fixture();
        using (var first = fixture.Create())
        {
            first.Launch(1, _ => { });
            WaitForResults(first, 1); // finished, never applied
            // Epoch 2's snapshot was written but its process died with the trainer: no markers.
            Directory.CreateDirectory(first.SnapshotDirectory(2));
        }

        using var resumed = fixture.Create();
        Assert.Equal([2], resumed.ResumePending(completedEpochs: 2));
        var results = WaitForResults(resumed, 2);
        Assert.Equal([1, 2], results.Select(r => r.Epoch));
        Assert.All(results, r => Assert.NotNull(r.Report));
    }

    private static IReadOnlyList<OffloadedEvaluations.Result> WaitForResults(OffloadedEvaluations offloaded, int count)
    {
        IReadOnlyList<OffloadedEvaluations.Result> results = [];
        WaitUntil(() => (results = offloaded.TakeFinished()).Count >= count);
        return results;
    }

    private static void WaitUntil(Func<bool> condition)
    {
        var deadline = DateTime.UtcNow + TimeSpan.FromSeconds(30);
        while (!condition())
        {
            if (DateTime.UtcNow > deadline)
                throw new TimeoutException("condition not reached in 30s");
            Thread.Sleep(20);
        }
    }

    private sealed class Fixture : IDisposable
    {
        private readonly string _root = Directory.CreateTempSubdirectory().FullName;
        private readonly string _script;

        public Fixture()
        {
            var tools = Directory.CreateDirectory(Path.Combine(_root, "tools")).FullName;
            _script = Path.Combine(tools, "fake-evaluate.sh");
            File.WriteAllText(_script, Script.Replace("\r\n", "\n") + "\n");
            File.SetUnixFileMode(_script, UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute);
            var template = new EvaluationReport(10, 5, double.NaN, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, []);
            var json = JsonSerializer.Serialize(template, new JsonSerializerOptions { NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals })
                .Replace("\"MeanAveragePrecision\":\"NaN\"", "\"MeanAveragePrecision\":__MAP__");
            File.WriteAllText(Path.Combine(tools, "report-template.json"), json);
            Directory.CreateDirectory(Path.Combine(_root, "checkpoint"));
        }

        public OffloadedEvaluations Create() => new(Path.Combine(_root, "checkpoint"), _script);

        public void Dispose() => Directory.Delete(_root, recursive: true);
    }
}
