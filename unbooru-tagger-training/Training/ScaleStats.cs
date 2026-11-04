namespace UnbooruTagger.Training.Training;

/// <summary>
/// What mixed-scale training (<see cref="TrainingScales"/>) actually did, for the dashboard: per size, the share of micro-batches
/// drawn (against the plan: a size trained only on stored rows is skipped while no store holds any), the time a micro-batch takes,
/// its mean loss, and how many of its images came from a hi-res store (drawn from stored rows, or sampled as usual and held by one).
/// <para>A step's own time can't be measured: the device runs behind the CPU, and only the read-backs every few steps wait for it. So
/// the time per size is fitted by least squares over the intervals between read-backs (each interval's length against how many
/// micro-batches of each size it held), which converges once the window has intervals with different mixes.</para>
/// </summary>
public sealed class ScaleStats(IReadOnlyList<TrainingScale> scales)
{
    private readonly int[] _sizes = scales.Select(s => s.Size).ToArray();
    private readonly int[] _batchOf = scales.Select(s => s.Batch).ToArray();
    private readonly double[] _plan = scales.Select(s => s.Share / scales.Sum(x => x.Share)).ToArray();
    private Window _window = new(scales.Count);
    private int[] _interval = new int[scales.Count];

    private sealed class Window(int n)
    {
        public readonly long[] Steps = new long[n], Images = new long[n], FromStore = new long[n], DrawnStored = new long[n], LossCount = new long[n];
        public readonly double[] LossSum = new double[n];
        // Normal equations of the time fit: A t = b, A = sum of c c^T, b = sum of c T.
        public readonly double[,] A = new double[n, n];
        public readonly double[] B = new double[n];
        public double Milliseconds;
        public long TimedImages;
        public long PeakDeviceMb;
    }

    private int IndexOf(int size) => Array.IndexOf(_sizes, size);

    /// <summary>A micro-batch at <paramref name="size"/> was trained.</summary>
    public void Step(int size, int images, int fromStore, int drawnStored)
    {
        if (IndexOf(size) is var i and >= 0)
        {
            _window.Steps[i]++;
            _window.Images[i] += images;
            _window.FromStore[i] += fromStore;
            _window.DrawnStored[i] += drawnStored;
            _interval[i]++;
        }
    }

    /// <summary>A step's loss, read back later than the step itself.</summary>
    public void Loss(int size, double loss)
    {
        if (IndexOf(size) is var i and >= 0 && double.IsFinite(loss))
        {
            _window.LossSum[i] += loss;
            _window.LossCount[i]++;
        }
    }

    /// <summary>The device caught up (a read-back) after <paramref name="milliseconds"/> since the last one; the steps since then made it.</summary>
    public void Interval(double milliseconds, long deviceMb)
    {
        var c = _interval;
        _interval = new int[c.Length];
        _window.PeakDeviceMb = Math.Max(_window.PeakDeviceMb, deviceMb);
        // An interval with a checkpoint save or a snapshot in it says nothing about the steps.
        if (milliseconds > c.Sum() * MaxStepMilliseconds)
            return;
        for (var j = 0; j < c.Length; j++)
        {
            for (var k = 0; k < c.Length; k++)
                _window.A[j, k] += c[j] * c[k];
            _window.B[j] += c[j] * milliseconds;
        }
        _window.Milliseconds += milliseconds;
        _window.TimedImages += Enumerable.Range(0, c.Length).Sum(i => (long)c[i] * _batchOf[i]);
    }

    /// <summary>Longer than any micro-batch takes (a 2048px one is ~0.4 s on the B70): intervals averaging more had something else in them.</summary>
    private const double MaxStepMilliseconds = 3000;

    /// <summary>Milliseconds per micro-batch at each size, fitted over the window; NaN where a size wasn't drawn.</summary>
    public double[] FittedMilliseconds()
    {
        var n = _sizes.Length;
        var used = Enumerable.Range(0, n).Where(i => _window.Steps[i] > 0).ToArray();
        var result = Enumerable.Repeat(double.NaN, n).ToArray();
        if (used.Length == 0)
            return result;
        var m = used.Length;
        var a = new double[m, m + 1];
        for (var j = 0; j < m; j++)
        {
            for (var k = 0; k < m; k++)
                a[j, k] = _window.A[used[j], used[k]];
            a[j, m] = _window.B[used[j]];
        }
        // Gaussian elimination with partial pivoting; a singular fit (every interval the same mix) leaves NaN.
        for (var col = 0; col < m; col++)
        {
            var pivot = Enumerable.Range(col, m - col).MaxBy(r => Math.Abs(a[r, col]));
            if (Math.Abs(a[pivot, col]) < 1e-9)
                return result;
            for (var k = 0; k <= m; k++)
                (a[col, k], a[pivot, k]) = (a[pivot, k], a[col, k]);
            for (var r = 0; r < m; r++)
            {
                if (r == col)
                    continue;
                var f = a[r, col] / a[col, col];
                for (var k = col; k <= m; k++)
                    a[r, k] -= f * a[col, k];
            }
        }
        for (var j = 0; j < m; j++)
            result[used[j]] = a[j, m] / a[j, j];
        return result;
    }

    /// <summary>The window's summary line (then a new window starts), or null if no step was trained in it.</summary>
    public string? TakeSummary(string stores)
    {
        var w = _window;
        var steps = w.Steps.Sum();
        if (steps == 0)
            return null;
        var ms = FittedMilliseconds();
        var parts = _sizes.Select((size, i) => w.Steps[i] == 0
            ? $"{size}px skipped (plan {_plan[i]:P0})"
            : $"{size}px {w.Steps[i] / (double)steps:P0} (plan {_plan[i]:P0}) {(double.IsNaN(ms[i]) ? "?" : $"{ms[i]:F0}")} ms loss {(w.LossCount[i] > 0 ? $"{w.LossSum[i] / w.LossCount[i]:F3}" : "?")}"
              + $" store {w.FromStore[i] / (double)w.Images[i]:P0} ({w.DrawnStored[i] / (double)w.Images[i]:P0} drawn)");
        var imagesPerSecond = w.Milliseconds > 0 ? w.TimedImages / (w.Milliseconds / 1000) : double.NaN;
        _window = new Window(_sizes.Length);
        return $"scales, last {steps:N0} steps: {string.Join(" | ", parts)} · {imagesPerSecond:F1} img/s · {stores}"
               + (w.PeakDeviceMb > 0 ? $" · VRAM peak {w.PeakDeviceMb / 1024.0:F1} GB" : "");
    }
}
