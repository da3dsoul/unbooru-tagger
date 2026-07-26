namespace UnbooruTagger.Crawler;

/// <summary>
/// Near-duplicate lookup for 64-bit perceptual hashes under a small Hamming-distance
/// threshold, without a full scan per query.
///
/// Splits each hash into <c>maxHammingDistance + 1</c> disjoint bands and indexes
/// entries by each band's exact value. For any two hashes within
/// <c>maxHammingDistance</c> bits of each other, the differing bits can affect at most
/// <c>maxHammingDistance</c> of the bands (each differing bit lands in exactly one
/// band) — so with one more band than that, at least one band is guaranteed to match
/// exactly between them (pigeonhole). A query only needs to check entries sharing a
/// band with it, instead of every entry ever added.
///
/// A plain list with a linear scan-and-compare per insert made a crawl run
/// progressively slower as the corpus grew — O(n) per new image, O(n^2) over a run —
/// which is exactly why downloads visibly slow down as the dataset gets bigger. This
/// keeps both insert and lookup close to O(1) regardless of corpus size.
/// </summary>
public sealed class PerceptualHashIndex
{
    private const int TotalBits = 64;

    private readonly int _maxHammingDistance;
    private readonly (int Shift, ulong Mask)[] _bands;
    private readonly Dictionary<ulong, List<int>>[] _buckets;
    private readonly List<(ulong Hash, string Md5)> _entries = [];

    public PerceptualHashIndex(int maxHammingDistance, IEnumerable<(ulong Hash, string Md5)> existing)
    {
        _maxHammingDistance = maxHammingDistance;
        _bands = BuildBands(maxHammingDistance + 1);
        _buckets = [.. _bands.Select(_ => new Dictionary<ulong, List<int>>())];

        foreach (var (hash, md5) in existing)
            Add(hash, md5);
    }

    private static (int Shift, ulong Mask)[] BuildBands(int bandCount)
    {
        var baseWidth = TotalBits / bandCount;
        var remainder = TotalBits % bandCount;
        var bands = new (int Shift, ulong Mask)[bandCount];
        var shift = 0;
        for (var i = 0; i < bandCount; i++)
        {
            var width = baseWidth + (i < remainder ? 1 : 0);
            bands[i] = (shift, width >= TotalBits ? ulong.MaxValue : (1UL << width) - 1);
            shift += width;
        }

        return bands;
    }

    public void Add(ulong hash, string md5)
    {
        var index = _entries.Count;
        _entries.Add((hash, md5));
        for (var band = 0; band < _bands.Length; band++)
        {
            var key = BandKey(hash, band);
            if (!_buckets[band].TryGetValue(key, out var bucket))
                _buckets[band][key] = bucket = [];
            bucket.Add(index);
        }
    }

    /// <summary>The Md5 of an existing entry within the configured Hamming-distance threshold of <paramref name="hash"/>, or null if none.</summary>
    public string? FindNear(ulong hash)
    {
        HashSet<int>? checkedIndexes = null;
        for (var band = 0; band < _bands.Length; band++)
        {
            if (!_buckets[band].TryGetValue(BandKey(hash, band), out var bucket))
                continue;

            foreach (var index in bucket)
            {
                checkedIndexes ??= [];
                if (!checkedIndexes.Add(index))
                    continue; // already ruled out (or matched) via an earlier band

                var candidate = _entries[index];
                if (PerceptualHash.HammingDistance(candidate.Hash, hash) <= _maxHammingDistance)
                    return candidate.Md5;
            }
        }

        return null;
    }

    private ulong BandKey(ulong hash, int band)
    {
        var (shift, mask) = _bands[band];
        return (hash >> shift) & mask;
    }
}
