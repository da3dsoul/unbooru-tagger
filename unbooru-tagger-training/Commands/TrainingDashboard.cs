using System.Globalization;
using System.Text;
using Spectre.Console;
using Spectre.Console.Rendering;
using UnbooruTagger.Training.Training;

namespace UnbooruTagger.Training.Commands;

/// <summary>Fixed facts about a training run, shown in the dashboard header and used for its estimates.</summary>
public sealed record TrainingRunInfo(
    string Title,
    int Epochs,
    int StepsPerEpoch,
    int StartEpoch,
    int StartStepInEpoch,
    int ImagesPerStep,
    TimeSpan CheckpointInterval,
    bool EvaluationsOffloaded = false,
    int SnapshotEvaluationSteps = 0);

/// <summary>
/// Live full-width status panel for <see cref="TrainCommandHandler"/>, replacing two
/// Spectre progress rows whose fixed 70-column description cut off the per-epoch ETA and
/// whose "Overall" ETA (the whole multi-epoch run) read as a wrong epoch estimate.
/// <para>
/// Estimates use the wall-clock step rate over the last <see cref="RateWindow"/>, so
/// checkpoint saves count at their real cost and a changed step time shows up within
/// minutes instead of being averaged against the whole run. The run estimate adds the
/// measured per-epoch evaluation time once one has run. The per-step loss is too noisy to
/// read directly, so the panel also shows a trailing mean and a chart of
/// <see cref="LossHistory"/>'s bucket means over the whole run (saved with each checkpoint, so a
/// restart keeps it). Below that, one chart per headline metric across every evaluation so far,
/// epoch and mid-epoch snapshot alike (<see cref="SnapshotEvaluations"/>).
/// </para>
/// State is written from the training thread and rendered from a timer thread, so every
/// access goes through <see cref="_lock"/>.
/// </summary>
public sealed class TrainingDashboard
{
    public static readonly TimeSpan RateWindow = TimeSpan.FromMinutes(30);
    private static readonly TimeSpan RateSampleInterval = TimeSpan.FromSeconds(5);
    private const int RecentLossSteps = 1000;
    private const int BarWidth = 40;
    private const int ChartRows = 4;

    private readonly TrainingRunInfo _info;
    private readonly Func<DateTime> _now;
    private readonly object _lock = new();
    private readonly DateTime _startedAt;

    private readonly Queue<(DateTime Time, long Step)> _rateSamples = new();
    private readonly Queue<float> _recentLosses = new();
    private double _recentLossSum;
    private readonly LossHistory _lossHistory;
    private readonly SortedDictionary<long, EvaluationPoint> _points = new();
    private readonly SortedDictionary<long, DateTime> _runningSnapshots = new();

    private int _epoch;
    private int _stepInEpoch;
    private long _globalStep;
    private float _loss = float.NaN, _tagLoss = float.NaN, _localizationLoss = float.NaN;
    private double _learningRate = double.NaN;
    private double _dataWaitMs, _stepMs;

    private string _phase = "starting";
    private (int Done, int Total, DateTime StartedAt)? _phaseProgress;
    private DateTime? _lastCheckpoint;
    private DateTime? _evaluationStartedAt;
    private TimeSpan? _evaluationDuration;
    private string? _lastEvaluation;
    // --evaluation-command: completed epochs whose evaluation runs in another process, and since when.
    private readonly SortedDictionary<int, DateTime> _offloadedEvaluations = new();
    private bool _finished;
    private bool _printEvents;
    private (int TrustedTags, double CroppedFraction)? _cropTeacher;
    private readonly List<string> _events = [];
    private const int ShownEvents = 8;

    public TrainingDashboard(TrainingRunInfo info, Func<DateTime>? now = null, LossHistory? lossHistory = null)
    {
        _info = info;
        _lossHistory = lossHistory ?? new LossHistory();
        _now = now ?? (() => DateTime.Now);
        _startedAt = _now();
        _epoch = info.StartEpoch;
        _stepInEpoch = info.StartStepInEpoch;
        _globalStep = (long)info.StartEpoch * info.StepsPerEpoch + info.StartStepInEpoch;
        _rateSamples.Enqueue((_startedAt, _globalStep));
    }

    private long TotalSteps => (long)_info.Epochs * _info.StepsPerEpoch;

    /// <summary>Re-renders into <paramref name="ctx"/> every second until disposed.</summary>
    public IDisposable Attach(LiveDisplayContext ctx) =>
        new Timer(_ =>
        {
            ctx.UpdateTarget(Render());
            ctx.Refresh();
        }, null, TimeSpan.Zero, TimeSpan.FromSeconds(1));

    /// <summary>For redirected output: prints <see cref="Summary"/> every <paramref name="interval"/> until disposed.</summary>
    public IDisposable AttachLog(TimeSpan interval)
    {
        lock (_lock)
            _printEvents = true;
        return new Timer(_ => AnsiConsole.MarkupLine(Summary()), null, interval, interval);
    }

    /// <summary>
    /// A message (Spectre markup) from inside the training loop. Printing a line while the
    /// live panel is up leaves fragments of the old panel around it, so in panel mode it is
    /// shown in the panel's event list instead (and printed to scrollback once the panel
    /// stops, from <see cref="Events"/>); in log mode it is printed directly.
    /// </summary>
    public void Log(string markup)
    {
        bool print;
        lock (_lock)
        {
            _events.Add($"[grey]{_now():ddd HH:mm}[/] {markup}");
            print = _printEvents;
        }
        if (print)
            AnsiConsole.MarkupLine(markup);
    }

    public IReadOnlyList<string> Events
    {
        get
        {
            lock (_lock)
                return _events.ToList();
        }
    }

    /// <summary>One line with the essentials, for logs.</summary>
    public string Summary()
    {
        lock (_lock)
        {
            var rate = StepsPerSecond();
            var text = $"{Markup.Escape(_phase)} · epoch {_epoch + 1}/{_info.Epochs} step {_stepInEpoch:N0}/{_info.StepsPerEpoch:N0} " +
                       $"({Percent(_stepInEpoch, _info.StepsPerEpoch).Trim()})";
            if (rate > 0)
                text += $" · {1000 / rate:F0} ms/step · epoch ends in {Duration(TimeSpan.FromSeconds((_info.StepsPerEpoch - _stepInEpoch) / rate))}" +
                        $" · run ends in {Duration(TimeSpan.FromSeconds((TotalSteps - _globalStep) / rate))}";
            if (_recentLosses.Count > 0)
                text += $" · loss (mean of last {_recentLosses.Count:N0}) {_recentLossSum / _recentLosses.Count:G4}";
            if (_points.Count > 0)
            {
                var p = _points.Values.Last();
                text += Markup.Escape($" · evaluation at step {p.Step:N0}: mAP {p.MeanAveragePrecision:F3}, gold pointing {p.GoldPointing:P1}, " +
                                      $"context gaps {p.RecallGap:P1} / {p.FalsePositiveGap:P1}, blanked still called {p.OccludedStillCalled:P1}");
            }
            return text;
        }
    }

    public void EpochStarted(int epoch, int stepInEpoch)
    {
        lock (_lock)
        {
            _epoch = epoch;
            _stepInEpoch = stepInEpoch;
            _globalStep = (long)epoch * _info.StepsPerEpoch + stepInEpoch;
            _phase = "training";
            _phaseProgress = null;
        }
    }

    /// <param name="stepInEpoch">Steps completed in the current epoch, including this one.</param>
    public void StepCompleted(int stepInEpoch, float loss, float tagLoss, float localizationLoss, double learningRate, double stepMs, double dataWaitMs)
    {
        lock (_lock)
        {
            var now = _now();
            _stepInEpoch = stepInEpoch;
            _globalStep = (long)_epoch * _info.StepsPerEpoch + stepInEpoch;
            (_loss, _tagLoss, _localizationLoss) = (loss, tagLoss, localizationLoss);
            _learningRate = learningRate;
            (_stepMs, _dataWaitMs) = (stepMs, dataWaitMs);
            _phase = "training";
            _phaseProgress = null;

            if (now - _rateSamples.Last().Time >= RateSampleInterval)
                _rateSamples.Enqueue((now, _globalStep));
            // Keep one sample older than the window so the rate always spans a full window.
            while (_rateSamples.Count > 2 && now - _rateSamples.ElementAt(1).Time >= RateWindow)
                _rateSamples.Dequeue();

            _recentLosses.Enqueue(loss);
            _recentLossSum += loss;
            if (_recentLosses.Count > RecentLossSteps)
                _recentLossSum -= _recentLosses.Dequeue();

            _lossHistory.Add(_globalStep, loss);
        }
    }

    /// <summary>Writes the loss history next to a checkpoint being saved (<see cref="LossHistory.Load"/> reads it back on resume).</summary>
    public void SaveLossHistory(string checkpointDir)
    {
        lock (_lock)
            _lossHistory.Save(checkpointDir);
    }

    /// <summary>An evaluation's headline numbers, epoch or snapshot, for the trend charts.</summary>
    public void AddEvaluationPoint(EvaluationPoint point)
    {
        lock (_lock)
        {
            _points[point.Step] = point;
            _runningSnapshots.Remove(point.Step);
        }
    }

    /// <summary>A snapshot evaluation started (or restarted) in another process.</summary>
    public void SnapshotEvaluationStarted(long step)
    {
        lock (_lock)
            _runningSnapshots[step] = _now();
    }

    /// <summary>A snapshot evaluation ended without a result; it's logged separately.</summary>
    public void SnapshotEvaluationFailed(long step)
    {
        lock (_lock)
            _runningSnapshots.Remove(step);
    }

    /// <summary>A step-free activity (checkpoint save, ranking evaluation); cleared by the next step.</summary>
    public void Phase(string phase)
    {
        lock (_lock)
        {
            _phase = phase;
            _phaseProgress = null;
        }
    }

    public void PhaseProgress(string phase, int done, int total)
    {
        lock (_lock)
        {
            var startedAt = _phase == phase && _phaseProgress is { } p ? p.StartedAt : _now();
            _phase = phase;
            _phaseProgress = (done, total, startedAt);
        }
    }

    /// <summary>Crop consistency's state: tags trusted as teachers, and the share of images cropped so far (its extra cost).</summary>
    public void CropTeacher(int trustedTags, double croppedFraction)
    {
        lock (_lock)
            _cropTeacher = (trustedTags, croppedFraction);
    }

    public void CheckpointSaved()
    {
        lock (_lock)
        {
            _lastCheckpoint = _now();
            _phase = "training";
        }
    }

    public void EvaluationStarted()
    {
        lock (_lock)
            _evaluationStartedAt = _now();
    }

    /// <summary>Epoch <paramref name="completedEpoch"/> (1-based) is being evaluated by another process.</summary>
    public void EvaluationOffloaded(int completedEpoch)
    {
        lock (_lock)
            _offloadedEvaluations[completedEpoch] = _now();
    }

    /// <summary>An offloaded evaluation came back without a report; it's logged separately.</summary>
    public void OffloadedEvaluationFailed(int completedEpoch)
    {
        lock (_lock)
            _offloadedEvaluations.Remove(completedEpoch);
    }

    public void EvaluationFinished(int epoch, EvaluationReport report, RatingReport? ratings)
    {
        lock (_lock)
        {
            _offloadedEvaluations.Remove(epoch + 1);
            if (_evaluationStartedAt is { } started)
                _evaluationDuration = _now() - started;
            _evaluationStartedAt = null;

            var text = $"epoch {epoch + 1}: mAP {report.MeanAveragePrecision:F4} · hard-negative mAP {report.MeanHardAveragePrecision:F4} · " +
                       $"image precision {report.MicroPrecision:F3} / recall {report.MicroRecall:F3} · exclusive violations {report.ExclusiveViolationRate:P2}";
            if (ratings is not null)
                text += $" · ratings {ratings.Accuracy:P1}, questionable/explicit called general {ratings.ExplicitOrQuestionableCalledGeneral:P2}";
            _lastEvaluation = text;
        }
    }

    public void Finished(string phase)
    {
        lock (_lock)
        {
            _finished = true;
            _phase = phase;
            _phaseProgress = null;
        }
    }

    /// <summary>Steps per second over the trailing <see cref="RateWindow"/> (wall clock, so including checkpoint saves).</summary>
    public double StepsPerSecond()
    {
        lock (_lock)
        {
            var (oldTime, oldStep) = _rateSamples.Peek();
            var seconds = (_now() - oldTime).TotalSeconds;
            return seconds > 0 ? (_globalStep - oldStep) / seconds : 0;
        }
    }

    public IRenderable Render()
    {
        lock (_lock)
        {
            var now = _now();
            var rate = StepsPerSecond();
            var grid = new Grid().AddColumn(new GridColumn().NoWrap().PadRight(2)).AddColumn();

            void Row(string label, string markup) => grid.AddRow(new Markup($"[grey]{label}[/]"), new Markup(markup));

            Row("status", PhaseText(now));

            var epochLeft = _finished ? TimeSpan.Zero
                : rate > 0 ? TimeSpan.FromSeconds((_info.StepsPerEpoch - _stepInEpoch) / rate) : (TimeSpan?)null;
            Row($"epoch {_epoch + 1}/{_info.Epochs}",
                $"{Bar(_stepInEpoch, _info.StepsPerEpoch)} {Percent(_stepInEpoch, _info.StepsPerEpoch)}  " +
                $"step {_stepInEpoch:N0} / {_info.StepsPerEpoch:N0}{Remaining(epochLeft, now, "epoch ends")}");

            // Whole run: remaining steps plus one evaluation per remaining epoch boundary
            // (including this epoch's) once an evaluation has been timed.
            TimeSpan? runLeft = _finished ? TimeSpan.Zero : null;
            if (rate > 0 && !_finished)
            {
                runLeft = TimeSpan.FromSeconds((TotalSteps - _globalStep) / rate);
                if (_evaluationDuration is { } evaluation)
                    runLeft += evaluation * (_info.Epochs - _epoch);
            }
            var evaluationNote = _info.EvaluationsOffloaded
                ? " [grey](epoch evaluations run in another process alongside training; early stopping can end it sooner)[/]"
                : _evaluationDuration is { } d
                    ? $" [grey](incl. {Duration(d)} evaluation per epoch; early stopping can end it sooner)[/]"
                    : " [grey](plus an evaluation per epoch; early stopping can end it sooner)[/]";
            Row($"run",
                $"{Bar(_globalStep, TotalSteps)} {Percent(_globalStep, TotalSteps)}  " +
                $"step {_globalStep:N0} / {TotalSteps:N0}{Remaining(runLeft, now, "finishes")}{evaluationNote}");

            var imagesPerSecond = rate * _info.ImagesPerStep;
            Row("speed",
                $"{(rate > 0 ? 1000 / rate : 0):F0} ms/step over the last {Duration(Min(now - _startedAt, RateWindow))} · {imagesPerSecond:F1} img/s · " +
                $"{_stepMs:F0} ms compute + {_dataWaitMs:F0} ms waiting on data per step (epoch average) · running {Duration(now - _startedAt)}");

            Row("loss", LossText());
            if (_cropTeacher is { } crop)
                Row("crop teacher", crop.TrustedTags == 0
                    ? "[grey]off: no tag trusted yet (re-checked after each epoch evaluation)[/]"
                    : $"{crop.TrustedTags:N0} tags trusted · {crop.CroppedFraction:P1} of images cropped this run (extra forward pass for those only)");
            Row("learning rate", double.IsNaN(_learningRate) ? "[grey]-[/]" : _learningRate.ToString("0.00e+0", CultureInfo.InvariantCulture));

            var nextCheckpoint = (_lastCheckpoint ?? _startedAt) + _info.CheckpointInterval;
            Row("checkpoint",
                (_lastCheckpoint is { } saved ? $"saved {saved:HH:mm} ({Duration(now - saved)} ago)" : "none saved by this process yet") +
                (_info.CheckpointInterval < TimeSpan.FromDays(365) && !_finished ? $" · next ≈ {nextCheckpoint:HH:mm}" : ""));

            if (_offloadedEvaluations.Count > 0)
                Row("evaluating", string.Join(" · ", _offloadedEvaluations.Select(p =>
                        $"epoch {p.Key} in another process since {p.Value:HH:mm}, running {Duration(now - p.Value)}")) +
                    " [grey](applied at the first checkpoint save after it finishes)[/]");
            if (_info.SnapshotEvaluationSteps > 0)
                Row("snapshots", SnapshotText(now, rate));
            Row("last evaluation", _lastEvaluation is not null
                ? Markup.Escape(_lastEvaluation)
                : _offloadedEvaluations.Count > 0
                    ? "[grey]none applied yet[/]"
                    : $"[grey]none yet: runs at the end of epoch {_epoch + 1}{(epochLeft is { } e ? $" (≈ {Clock(now + e, now)})" : "")}[/]");

            var body = new List<IRenderable> { grid };
            body.AddRange(Charts());

            if (_events.Count > 0)
            {
                var events = new Grid().AddColumn(new GridColumn().NoWrap().PadRight(2)).AddColumn();
                var shown = _events.Skip(Math.Max(0, _events.Count - ShownEvents)).ToList();
                for (var k = 0; k < shown.Count; k++)
                    events.AddRow(new Markup(k == 0 ? $"[grey]events{(_events.Count > ShownEvents ? $" (last {ShownEvents})" : "")}[/]" : ""), new Markup(shown[k]));
                body.Add(new Text(""));
                body.Add(events);
            }

            return new Panel(new Rows(body))
                .Header($" {Markup.Escape(_info.Title)} ")
                .Border(BoxBorder.Rounded)
                .Expand();
        }
    }

    private string PhaseText(DateTime now)
    {
        var phase = Markup.Escape(_phase);
        if (_phaseProgress is not { } p)
            return _phase == "training" ? $"[green]{phase}[/]" : $"[yellow]{phase}[/]";

        var elapsed = (now - p.StartedAt).TotalSeconds;
        var left = p.Done > 0 && elapsed > 0 ? TimeSpan.FromSeconds((p.Total - p.Done) * elapsed / p.Done) : (TimeSpan?)null;
        return $"[yellow]{phase}[/] {Bar(p.Done, p.Total)} {p.Done:N0} / {p.Total:N0}{Remaining(left, now, "done")}";
    }

    private string LossText()
    {
        if (float.IsNaN(_loss))
            return "[grey]-[/]";

        var text = new StringBuilder();
        text.Append(CultureInfo.InvariantCulture, $"{_loss:G4} this step");
        if (!float.IsNaN(_localizationLoss) && _localizationLoss != 0)
            text.Append(CultureInfo.InvariantCulture, $" [grey](tags {_tagLoss:G4} + weighted localization {_localizationLoss:G4})[/]");
        if (_recentLosses.Count > 0)
            text.Append(CultureInfo.InvariantCulture, $" · mean of last {_recentLosses.Count:N0}: [bold]{_recentLossSum / _recentLosses.Count:G4}[/]");
        return text.ToString();
    }

    private string SnapshotText(DateTime now, double rate)
    {
        var every = _info.SnapshotEvaluationSteps;
        var snapshots = _points.Values.Count(p => !p.EndOfEpoch);
        var next = (_globalStep / every + 1) * every;
        var text = $"every {every:N0} steps · {snapshots:N0} evaluated so far · next at step {next:N0}" +
                   (rate > 0 ? $" (≈ {Clock(now + TimeSpan.FromSeconds((next - _globalStep) / rate), now)})" : "");
        if (_runningSnapshots.Count > 0)
            text += " · evaluating " + string.Join(", ", _runningSnapshots.Select(r => $"step {r.Key:N0} (since {r.Value:HH:mm})"));
        return text + " [grey](in another process; trend charts below)[/]";
    }

    /// <summary>The loss chart, then one chart per headline evaluation metric.</summary>
    private IEnumerable<IRenderable> Charts()
    {
        if (_lossHistory.Buckets.Count >= 2)
        {
            yield return new Text("");
            var loss = _lossHistory.Buckets.Select(b => ((double)b.End, (double)b.Mean)).ToList();
            yield return Chart($"loss, mean per {_lossHistory.BucketSteps:N0} steps", loss, false, v => v.ToString("G3", CultureInfo.InvariantCulture), Color.Aqua);
        }
        if (_points.Count == 0)
            yield break;

        static string Fraction(double v) => v.ToString("F3", CultureInfo.InvariantCulture);
        static string Percent(double v) => (100 * v).ToString("F1", CultureInfo.InvariantCulture) + "%";
        var metrics = new (string Name, Func<EvaluationPoint, double> Value, bool LowerIsBetter, Func<double, string> Format)[]
        {
            ("held-out mAP", p => p.MeanAveragePrecision, false, Fraction),
            ("gold pointing", p => p.GoldPointing, false, Percent),
            ("blanked object still called", p => p.OccludedStillCalled, true, Percent),
            ("context recall gap", p => p.RecallGap, true, Percent),
            ("context false-positive gap", p => p.FalsePositiveGap, true, Percent),
            ("exclusive-tag violations", p => p.ExclusiveViolations, true, Percent),
        };
        var grid = new Grid();
        for (var c = 0; c < 3; c++)
            grid.AddColumn(new GridColumn().PadRight(c < 2 ? 4 : 0));
        var charts = metrics.Select(m => Chart(m.Name, _points.Values.Select(p => ((double)p.Step, m.Value(p))).ToList(), m.LowerIsBetter, m.Format, Color.Green)).ToList();
        for (var k = 0; k < charts.Count; k += 3)
            grid.AddRow(charts.Skip(k).Take(3).ToArray());
        yield return new Text("");
        yield return new Markup($"[grey]evaluations: {_points.Count:N0} ({_points.Values.Count(p => p.EndOfEpoch)} at epoch ends), by training step · ↓ = lower is better[/]");
        yield return grid;
    }

    /// <summary>A chart under a title line with the latest value and its change since the point before.</summary>
    private static IRenderable Chart(string name, List<(double X, double Y)> points, bool lowerIsBetter, Func<double, string> format, Color color)
    {
        var valid = points.Where(p => !double.IsNaN(p.Y)).ToList();
        var title = $"[grey]{Markup.Escape(name)}[/]";
        if (valid.Count > 0)
            title += $" [bold]{Markup.Escape(format(valid[^1].Y))}[/]";
        if (valid.Count > 1)
        {
            var change = valid[^1].Y - valid[^2].Y;
            var better = lowerIsBetter ? change < 0 : change > 0;
            var sign = change >= 0 ? "+" : "−";
            var magnitude = format(Math.Abs(change));
            title += $" [{(change == 0 ? "grey" : better ? "green" : "red")}]{sign}{Markup.Escape(magnitude)}[/] [grey]since step {Steps(valid[^2].X)}[/]";
        }
        if (lowerIsBetter)
            title += " [grey]↓[/]";
        return new Rows(new Markup(title), new BrailleChart(points, ChartRows, format, Steps, color));
    }

    /// <summary>A step count in thousands or millions: "24.9K", "1.45M".</summary>
    public static string Steps(double step) =>
        step >= 1_000_000 ? (step / 1_000_000).ToString("0.##", CultureInfo.InvariantCulture) + "M"
        : step >= 1_000 ? (step / 1_000).ToString("0.#", CultureInfo.InvariantCulture) + "K"
        : step.ToString("0", CultureInfo.InvariantCulture);

    public static string Sparkline(IReadOnlyList<float> values)
    {
        const string levels = "▁▂▃▄▅▆▇█";
        var min = values.Min();
        var max = values.Max();
        var range = max - min;
        return new string(values.Select(v => levels[range > 0 ? (int)Math.Round((v - min) / range * (levels.Length - 1)) : 0]).ToArray());
    }

    private static string Bar(long done, long total)
    {
        var filled = total > 0 ? (int)Math.Clamp(done * BarWidth / total, 0, BarWidth) : 0;
        return $"[green]{new string('━', filled)}[/][grey23]{new string('━', BarWidth - filled)}[/]";
    }

    private static string Percent(long done, long total) =>
        (total > 0 ? 100.0 * done / total : 0).ToString("F1", CultureInfo.InvariantCulture).PadLeft(5) + "%";

    private static string Remaining(TimeSpan? left, DateTime now, string verb) =>
        left is { } done && done <= TimeSpan.Zero ? " · done"
        : left is { } l ? $" · [bold]{Duration(l)}[/] left · {verb} {Clock(now + l, now)}" : " · [grey]estimating…[/]";

    private static TimeSpan Min(TimeSpan a, TimeSpan b) => a < b ? a : b;

    /// <summary>Compact duration: "8d 13h", "21h 40m", "4m 05s".</summary>
    public static string Duration(TimeSpan span)
    {
        if (span < TimeSpan.Zero)
            span = TimeSpan.Zero;
        if (span.TotalDays >= 1)
            return $"{(int)span.TotalDays}d {span.Hours}h";
        if (span.TotalHours >= 1)
            return $"{(int)span.TotalHours}h {span.Minutes:D2}m";
        return $"{span.Minutes}m {span.Seconds:D2}s";
    }

    /// <summary>A wall-clock time, with the day when it isn't today: "23:12", "Fri 23:12", "Oct 31 02:05".</summary>
    public static string Clock(DateTime at, DateTime now)
    {
        if (at.Date == now.Date)
            return at.ToString("HH:mm", CultureInfo.InvariantCulture);
        if (at.Date < now.Date.AddDays(7))
            return at.ToString("ddd HH:mm", CultureInfo.InvariantCulture);
        return at.ToString("MMM d HH:mm", CultureInfo.InvariantCulture);
    }
}
