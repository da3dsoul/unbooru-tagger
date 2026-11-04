using System.Globalization;

namespace UnbooruTagger.Training.Training;

/// <summary>
/// One input size in mixed-scale training (<c>--train-scales</c>).
/// </summary>
/// <param name="Size">The canvas, in pixels.</param>
/// <param name="Batch">Images per micro-batch at this size (what fits in memory: on the B70, 10 at 512, 5 at 768, 3-4 at 1024, 1-2 at 2048).</param>
/// <param name="Share">This size's share of the micro-batches (normalized over the scales).</param>
/// <param name="StoredShare">The share of each batch drawn from rows a hi-res store holds at this size or larger, uniformly
/// (the rest are sampled as usual, and still come from a store when one holds them). 1 for a size worth training only on real
/// pixels (2048: enlarging the 512px cache 4x shows nothing new).</param>
public sealed record TrainingScale(int Size, int Batch, double Share, double StoredShare = 0);

/// <summary>
/// Mixed-scale training (2026-11-04): fine-tuning at 1024 found small objects 10-20 points better but cost large objects up to
/// 7, and real hi-res pixels gained more where the store covers a tag. So each micro-batch is drawn at one of several sizes, each
/// with its own batch size, and a share of its images can come from the hi-res stores.
/// <para>Spec: comma-separated <c>size:batch:share[:stored]</c>, e.g. <c>512:10:0.3,1024:4:0.5:0.25,2048:1:0.2:1</c>.</para>
/// </summary>
public sealed class TrainingScales
{
    private readonly double[] _cumulative;
    private readonly Random _random;

    public IReadOnlyList<TrainingScale> Scales { get; }

    public TrainingScales(IReadOnlyList<TrainingScale> scales, int seed)
    {
        if (scales.Count == 0)
            throw new ArgumentException("At least one training scale is needed.", nameof(scales));
        foreach (var scale in scales)
            if (scale.Size < 32 || scale.Batch < 1 || scale.Share <= 0 || scale.StoredShare is < 0 or > 1)
                throw new ArgumentException($"Bad training scale {scale}: size >= 32, batch >= 1, share > 0, stored share in [0, 1].");
        if (scales.Select(s => s.Size).Distinct().Count() != scales.Count)
            throw new ArgumentException("Each training scale's size must differ.");
        Scales = scales;
        var total = scales.Sum(s => s.Share);
        _cumulative = new double[scales.Count];
        var sum = 0.0;
        for (var i = 0; i < scales.Count; i++)
            _cumulative[i] = sum += scales[i].Share / total;
        _random = new Random(seed);
    }

    public static IReadOnlyList<TrainingScale> Parse(string spec) =>
        spec.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(part =>
        {
            var fields = part.Split(':');
            if (fields.Length is < 3 or > 4)
                throw new ArgumentException($"--train-scales: '{part}' isn't size:batch:share[:stored].");
            return new TrainingScale(
                int.Parse(fields[0], CultureInfo.InvariantCulture),
                int.Parse(fields[1], CultureInfo.InvariantCulture),
                double.Parse(fields[2], CultureInfo.InvariantCulture),
                fields.Length == 4 ? double.Parse(fields[3], CultureInfo.InvariantCulture) : 0);
        }).ToArray();

    /// <summary>The next micro-batch's scale, drawn by share.</summary>
    public TrainingScale Next()
    {
        var draw = _random.NextDouble();
        for (var i = 0; i < _cumulative.Length - 1; i++)
            if (draw < _cumulative[i])
                return Scales[i];
        return Scales[^1];
    }

    /// <summary>How many of a batch at <paramref name="scale"/> come from stored rows: the share, rounded at random so its mean is exact.</summary>
    public int StoredCount(TrainingScale scale)
    {
        var exact = scale.StoredShare * scale.Batch;
        var whole = (int)Math.Floor(exact);
        return whole + (_random.NextDouble() < exact - whole ? 1 : 0);
    }

    public override string ToString() => string.Join(", ", Scales.Select(s =>
        $"{s.Size}px x{s.Batch} {s.Share / Scales.Sum(x => x.Share):P0}{(s.StoredShare > 0 ? $" ({s.StoredShare:P0} stored)" : "")}"));
}
