using UnbooruTagger.Training.Training;

namespace UnbooruTagger.Tests.Training;

public class SnapshotEvaluationsTests
{
    // A stand-in for `evaluate`: reports mAP = step / 1,000,000 and gold pointing 0.3, unless the
    // snapshot holds a "fail" file (exit 3).
    private const string Script = """
        #!/usr/bin/env bash
        set -eu
        snapshot="$1"; out="$2"
        [ -e "$snapshot/fail" ] && exit 3
        step="${out##*step-}"
        map=$(printf '0.%06d' "$step")
        echo "{\"MeanAveragePrecision\":$map,\"ExclusiveViolationRate\":0.05,\"ContextReliance\":{\"MacroRecallGap\":0.1,\"MacroFalsePositiveGap\":\"NaN\"}}" > "$out/evaluation_report.json"
        echo '{"Gold":[{"Name":"clean","PointingAccuracy":0.9},{"Name":"all","PointingAccuracy":0.3}],"Occlusion":[{"Name":"all","StillYesRate":0.8}]}' > "$out/localization_report.json"
        """;

    [Fact]
    public void FinishedEvaluation_BecomesAPoint_AndItsWeightsAreDeleted()
    {
        if (!OperatingSystem.IsLinux())
            return;
        using var fixture = new Fixture();
        using var snapshots = fixture.Create();

        snapshots.Launch(20_000, snapshot => File.WriteAllText(Path.Combine(snapshot, "weights"), "x"));
        Assert.Equal([20_000L], snapshots.Running);
        var result = Assert.Single(WaitForResults(snapshots, 1));

        Assert.Null(result.Failure);
        var point = result.Point!;
        Assert.Equal(20_000, point.Step);
        Assert.False(point.EndOfEpoch);
        Assert.Equal(0.02, point.MeanAveragePrecision, 6);
        Assert.Equal(0.3, point.GoldPointing, 6);
        Assert.Equal(0.8, point.OccludedStillCalled, 6);
        Assert.Equal(0.1, point.RecallGap, 6);
        Assert.True(double.IsNaN(point.FalsePositiveGap));
        Assert.Equal(0.05, point.ExclusiveViolations, 6);
        Assert.False(Directory.Exists(snapshots.SnapshotDirectory(20_000)));
        Assert.True(File.Exists(Path.Combine(snapshots.StepDirectory(20_000), "evaluation_report.json")));
        Assert.Empty(snapshots.TakeFinished());
        Assert.Empty(snapshots.Running);
    }

    [Fact]
    public void FailedEvaluation_KeepsItsWeights()
    {
        if (!OperatingSystem.IsLinux())
            return;
        using var fixture = new Fixture();
        using var snapshots = fixture.Create();

        snapshots.Launch(10_000, snapshot => File.WriteAllText(Path.Combine(snapshot, "fail"), ""));
        var result = Assert.Single(WaitForResults(snapshots, 1));

        Assert.Null(result.Point);
        Assert.Contains("exit code 3", result.Failure);
        Assert.True(Directory.Exists(snapshots.SnapshotDirectory(10_000)));
    }

    [Fact]
    public void Resume_ReturnsFinishedRerunsUnfinishedAndDropsStepsPastTheResumePoint()
    {
        if (!OperatingSystem.IsLinux())
            return;
        using var fixture = new Fixture();
        using (var first = fixture.Create())
        {
            first.Launch(10_000, _ => { });
            WaitForResults(first, 1);
            // Taken but never evaluated (the trainer died), one before and one after the checkpoint being resumed.
            Directory.CreateDirectory(first.SnapshotDirectory(20_000));
            Directory.CreateDirectory(first.SnapshotDirectory(30_000));
        }

        using var resumed = fixture.Create();
        var (finished, relaunched) = resumed.Resume(resumedStep: 25_000);

        Assert.Equal([10_000L], finished.Select(f => f.Step));
        Assert.Equal(0.01, finished[0].Point!.MeanAveragePrecision, 6);
        Assert.Equal([20_000L], relaunched);
        Assert.False(Directory.Exists(resumed.StepDirectory(30_000)));
        var rerun = Assert.Single(WaitForResults(resumed, 1));
        Assert.Equal(0.02, rerun.Point!.MeanAveragePrecision, 6);
    }

    [Fact]
    public void Point_IsNullWithoutAReport()
    {
        using var fixture = new Fixture();
        Assert.Null(EvaluationPoint.Read(fixture.Root, 1, endOfEpoch: true));
    }

    private static IReadOnlyList<SnapshotEvaluations.Result> WaitForResults(SnapshotEvaluations snapshots, int count)
    {
        var results = new List<SnapshotEvaluations.Result>();
        var deadline = DateTime.UtcNow + TimeSpan.FromSeconds(30);
        while (results.Count < count)
        {
            if (DateTime.UtcNow > deadline)
                throw new TimeoutException("evaluations not finished in 30s");
            results.AddRange(snapshots.TakeFinished());
            Thread.Sleep(20);
        }
        return results;
    }

    private sealed class Fixture : IDisposable
    {
        public string Root { get; } = Directory.CreateTempSubdirectory().FullName;
        private readonly string _script;

        public Fixture()
        {
            _script = Path.Combine(Directory.CreateDirectory(Path.Combine(Root, "tools")).FullName, "fake-evaluate.sh");
            File.WriteAllText(_script, Script.Replace("\r\n", "\n") + "\n");
            if (OperatingSystem.IsLinux())
                File.SetUnixFileMode(_script, UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute);
            Directory.CreateDirectory(Path.Combine(Root, "checkpoint"));
        }

        public SnapshotEvaluations Create() => new(Path.Combine(Root, "checkpoint"), _script);

        public void Dispose() => Directory.Delete(Root, recursive: true);
    }
}
