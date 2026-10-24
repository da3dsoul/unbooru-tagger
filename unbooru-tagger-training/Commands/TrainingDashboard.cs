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
    TimeSpan CheckpointInterval);

/// <summary>
/// Live full-width status panel for <see cref="TrainCommandHandler"/>, replacing two
/// Spectre progress rows whose fixed 70-column description cut off the per-epoch ETA and
/// whose "Overall" ETA (the whole multi-epoch run) read as a wrong epoch estimate.
/// <para>
/// Estimates use the wall-clock step rate over the last <see cref="RateWindow"/>, so
/// checkpoint saves count at their real cost and a changed step time shows up within
/// minutes instead of being averaged against the whole run. The run estimate adds the
/// measured per-epoch evaluation time once one has run. The per-step loss is too noisy to
/// read directly, so the panel also shows a trailing mean and a sparkline of
/// <see cref="LossBucketSteps"/>-step means.
/// </para>
/// State is written from the training thread and rendered from a timer thread, so every
/// access goes through <see cref="_lock"/>.
/// </summary>
public sealed class TrainingDashboard
{
    public static readonly TimeSpan RateWindow = TimeSpan.FromMinutes(30);
    private static readonly TimeSpan RateSampleInterval = TimeSpan.FromSeconds(5);
    public const int LossBucketSteps = 2000;
    private const int RecentLossSteps = 1000;
    private const int SparklineWidth = 48;
    private const int BarWidth = 40;

    private readonly TrainingRunInfo _info;
    private readonly Func<DateTime> _now;
    private readonly object _lock = new();
    private readonly DateTime _startedAt;

    private readonly Queue<(DateTime Time, long Step)> _rateSamples = new();
    private readonly Queue<float> _recentLosses = new();
    private double _recentLossSum;
    private readonly List<float> _lossBuckets = [];
    private double _bucketSum;
    private int _bucketCount;

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
    private bool _finished;
    private bool _printEvents;
    private (int TrustedTags, double CroppedFraction)? _cropTeacher;
    private readonly List<string> _events = [];
    private const int ShownEvents = 8;

    public TrainingDashboard(TrainingRunInfo info, Func<DateTime>? now = null)
    {
        _info = info;
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

            _bucketSum += loss;
            if (++_bucketCount == LossBucketSteps)
            {
                _lossBuckets.Add((float)(_bucketSum / _bucketCount));
                if (_lossBuckets.Count > SparklineWidth)
                    _lossBuckets.RemoveAt(0);
                (_bucketSum, _bucketCount) = (0, 0);
            }
        }
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

    public void EvaluationFinished(int epoch, EvaluationReport report, RatingReport? ratings)
    {
        lock (_lock)
        {
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
            var evaluationNote = _evaluationDuration is { } d
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

            Row("last evaluation", _lastEvaluation is not null
                ? Markup.Escape(_lastEvaluation)
                : $"[grey]none yet: runs at the end of epoch {_epoch + 1}{(epochLeft is { } e ? $" (≈ {Clock(now + e, now)})" : "")}[/]");

            if (_events.Count > 0)
            {
                grid.AddEmptyRow();
                var shown = _events.Skip(Math.Max(0, _events.Count - ShownEvents)).ToList();
                for (var k = 0; k < shown.Count; k++)
                    grid.AddRow(new Markup(k == 0 ? $"[grey]events{(_events.Count > ShownEvents ? $" (last {ShownEvents})" : "")}[/]" : ""), new Markup(shown[k]));
            }

            return new Panel(grid)
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
        if (_lossBuckets.Count >= 2)
            text.Append(CultureInfo.InvariantCulture,
                $" · trend {_lossBuckets[0]:G3} [aqua]{Sparkline(_lossBuckets)}[/] {_lossBuckets[^1]:G3} [grey](mean per {LossBucketSteps:N0} steps)[/]");
        return text.ToString();
    }

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
