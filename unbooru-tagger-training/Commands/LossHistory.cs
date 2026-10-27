using System.Text.Json;

namespace UnbooruTagger.Training.Commands;

/// <summary>
/// The training loss as means over fixed <see cref="BucketSteps"/>-step buckets of global steps,
/// for the dashboard's loss chart. Saved with every checkpoint (<see cref="FileName"/>) and read
/// back on resume, so the chart covers the whole run, not just the current process.
/// </summary>
public sealed class LossHistory
{
    public const string FileName = "loss_history.json";
    public const int DefaultBucketSteps = 2000;

    private readonly List<(long End, float Mean)> _buckets = [];
    private double _partialSum;
    private int _partialCount;
    private long _lastStep = -1;

    public LossHistory(int bucketSteps = DefaultBucketSteps) => BucketSteps = bucketSteps;

    public int BucketSteps { get; }

    /// <summary>Completed buckets: the global step each ends at, and its mean loss.</summary>
    public IReadOnlyList<(long End, float Mean)> Buckets => _buckets;

    /// <param name="globalStep">Steps completed over the whole run, including this one.</param>
    public void Add(long globalStep, float loss)
    {
        _partialSum += loss;
        _partialCount++;
        _lastStep = globalStep;
        if (globalStep % BucketSteps != 0)
            return;
        _buckets.Add((globalStep, (float)(_partialSum / _partialCount)));
        (_partialSum, _partialCount) = (0, 0);
    }

    /// <summary>An independent copy, for drawing from another thread.</summary>
    public LossHistory Copy()
    {
        var copy = new LossHistory(BucketSteps) { _partialSum = _partialSum, _partialCount = _partialCount, _lastStep = _lastStep };
        copy._buckets.AddRange(_buckets);
        return copy;
    }

    private sealed record Saved(int BucketSteps, long[] Ends, float[] Means, double PartialSum, int PartialCount, long LastStep);

    public void Save(string directory)
    {
        var path = Path.Combine(directory, FileName);
        var saved = new Saved(BucketSteps, _buckets.Select(b => b.End).ToArray(), _buckets.Select(b => b.Mean).ToArray(), _partialSum, _partialCount, _lastStep);
        File.WriteAllText(path + ".tmp", JsonSerializer.Serialize(saved));
        File.Move(path + ".tmp", path, overwrite: true);
    }

    /// <summary>
    /// The history saved in <paramref name="directory"/>, cut back to <paramref name="resumedStep"/>
    /// (anything later was trained on weights the resumed run doesn't have); an empty history if
    /// there's none or it can't be read.
    /// </summary>
    public static LossHistory Load(string directory, long resumedStep)
    {
        var path = Path.Combine(directory, FileName);
        if (!File.Exists(path))
            return new LossHistory();
        Saved? saved;
        try
        {
            saved = JsonSerializer.Deserialize<Saved>(File.ReadAllText(path));
        }
        catch (JsonException)
        {
            return new LossHistory();
        }
        if (saved is null || saved.BucketSteps <= 0 || saved.Ends.Length != saved.Means.Length)
            return new LossHistory();

        var history = new LossHistory(saved.BucketSteps);
        for (var k = 0; k < saved.Ends.Length; k++)
            if (saved.Ends[k] <= resumedStep)
                history._buckets.Add((saved.Ends[k], saved.Means[k]));
        // The unfinished bucket only lines up if it was saved at exactly the step being resumed.
        if (saved.LastStep == resumedStep)
            (history._partialSum, history._partialCount, history._lastStep) = (saved.PartialSum, saved.PartialCount, saved.LastStep);
        return history;
    }
}
