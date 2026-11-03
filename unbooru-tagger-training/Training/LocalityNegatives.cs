using UnbooruTagger.Training.Model;
using static TorchSharp.torch;

namespace UnbooruTagger.Training.Training;

/// <summary>
/// Window-level negatives from where tags can't be together (<c>--locality-negative-weight</c>, 2026-11-01). The glimpse score
/// is close to a max over an image's windows, so a labelled "yes" can come from any window, and nothing says which window a
/// tag is not in: on run 5's gold set the highest-scoring glimpse for a medium object was on the object 28% of the time (13%
/// by chance). But some tags can't share a close look: a window on the glasses doesn't hold the shoes. So where a labelled
/// tag A is confidently found in a close window (32-128px), each other labelled tag B is trained "no" in that window, weighted
/// by how rarely A and B are together and by the confidence in A.
/// <para>
/// How rarely is learned as training runs: for every image with both labelled, whether A's and B's best close windows are
/// within <see cref="Distance"/> of each other, weighted by both calls' confidence, in a table on the device (keyed by the
/// unordered pair, open addressing without probing: a pair whose slot another pair holds is skipped). Optionally it starts
/// from a prior measured on a trained map (<see cref="LoadPrior"/>, <c>evaluate --locality-prior-out</c>): learned from the
/// model's own calls alone, it can inherit their bias (if the best "shoes" window is the face, shoes and glasses look
/// together). A pair gives negatives once it has <see cref="MinObservations"/> of weighted observations.
/// </para>
/// </summary>
public sealed class LocalityNegatives : IDisposable
{
    /// <summary>Table slots by default: about 4M, 64 MB on the device (run 4's map saw 385,691 pairs over 20K images).</summary>
    public const int DefaultSlotBits = 22;

    /// <summary>
    /// Labelled tags per image the caller passes (<see cref="Cap"/>), padded to exactly this many: the term's memory then
    /// doesn't depend on the batch. Sized by the busiest image, it spiked on rare tag-heavy batches at the step's peak, and
    /// run 5 (~27 GB of the B70's 32) died with xe "VM worker error: -12" after 14-47 minutes (2026-11-01).
    /// </summary>
    public const int PositivesPerImage = 64;

    /// <summary>Each image's rows, a random subset of <see cref="PositivesPerImage"/> where it has more.</summary>
    public static List<int>[] Cap(IReadOnlyList<List<int>> rows, Random random) =>
        rows.Select(r => r.Count <= PositivesPerImage ? r : r.OrderBy(_ => random.Next()).Take(PositivesPerImage).ToList()).ToArray();

    private readonly long _slotMask;

    private readonly Tensor _keys;
    private readonly Tensor _weight;
    private readonly Tensor _together;
    private readonly Tensor _stats; // [3]: pair observations, pairs with negatives, their summed weight (for the dashboard)
    private long _steps;

    public LocalityNegatives(Device device, int slotBits = DefaultSlotBits)
    {
        var slots = 1L << slotBits;
        _slotMask = slots - 1;
        _keys = full(slots, -1L, dtype: ScalarType.Int64, device: device);
        _weight = zeros(slots, device: device);
        _together = zeros(slots, device: device);
        _stats = zeros(3, dtype: ScalarType.Float64, device: device);
    }

    /// <summary>Canvas pixels: two tags whose best windows' centres are closer than this on both axes count as together.</summary>
    public float Distance { get; init; } = DefaultDistance;

    public const float DefaultDistance = 64;

    /// <summary>Weighted observations a pair needs before it gives negatives.</summary>
    public float MinObservations { get; init; } = 20;

    /// <summary>The table decays by this factor every <see cref="DecayEvery"/> steps, so early, poorly-located calls fade.</summary>
    public float Decay { get; init; } = 0.99f;

    public int DecayEvery { get; init; } = 100;

    /// <summary>
    /// Only windows where A's call is at least this sure give negatives (0 = any; the table still learns from every call). At
    /// weight 1 ungated, the first A/B halved precision: a negative only says where a tag isn't, and fires it elsewhere.
    /// </summary>
    public float ConfidenceGate { get; init; }

    /// <summary>
    /// Each image's term is the weighted mean over its pairs given a negative, not their sum: an image with 30 labelled tags has
    /// ~900 pairs, so summed, busy images dominated the term.
    /// </summary>
    public bool PerPair { get; init; }

    /// <summary>
    /// No gradient into the shared logit scale and bias: the negatives move only the tower and the tag embeddings, so they can't
    /// shift every tag's operating point (the global bias is the cheapest way to lower many windows' logits at once).
    /// </summary>
    public bool DetachScaling { get; init; }

    /// <summary>A pair's prior observations are capped at this weight, so training's own can outweigh them.</summary>
    public float PriorCap { get; init; } = 50;

    /// <summary>
    /// One step's loss term (to be scaled by the caller) and its table update.
    /// </summary>
    /// <param name="cells">[B, D, K] glimpse embeddings (<see cref="Glimpses.Embed"/>).</param>
    /// <param name="choice">The windows (<see cref="Glimpses.Propose"/>).</param>
    /// <param name="rows">[B, P] each image's labelled localizable positives (vocabulary rows), padded.</param>
    /// <param name="rowsValid">[B, P] which are real.</param>
    /// <param name="tagEmbeddings">[V, D].</param>
    public Tensor Step(Tensor cells, GlimpseChoice choice, Tensor rows, Tensor rowsValid, Tensor tagEmbeddings, LogitScaleBias scaling)
    {
        using var scope = NewDisposeScope();
        var device = cells.device;
        var (batch, positives) = (rows.shape[0], rows.shape[1]);
        var embeddings = tagEmbeddings.index_select(0, rows.flatten()).reshape(batch, positives, -1);
        var dots = embeddings.bmm(cells.to_type(embeddings.dtype));
        var logits = DetachScaling ? scaling.ForwardDetached(dots) : scaling.forward(dots); // [B, P, K]

        Tensor negativeWeight, best;
        using (no_grad())
        {
            // Each positive's best close window and how sure the model is of it there.
            var close = choice.Valid.logical_and(choice.Zoom.lt(Glimpses.Zooms.Length - 1)); // [B, K]
            var masked = logits.detach().masked_fill(close.unsqueeze(1).logical_not(), float.NegativeInfinity);
            var (top, index) = masked.max(2); // [B, P]
            best = index.MoveToOuterDisposeScope();
            var present = rowsValid.logical_and(close.any(1, keepdim: true)).logical_and(top.isfinite());
            var confident = sigmoid(top.masked_fill(present.logical_not(), float.NegativeInfinity)); // 0 where absent
            var side = tensor(Enumerable.Range(0, Glimpses.Zooms.Length).Select(z => (float)Glimpses.SideOf(z)).ToArray(), device: device)
                .index_select(0, choice.Zoom.flatten()).view_as(choice.Zoom);
            var centreX = (choice.X.to_type(ScalarType.Float32) + side / 2).gather(1, index); // [B, P]
            var centreY = (choice.Y.to_type(ScalarType.Float32) + side / 2).gather(1, index);
            var colocated = (centreX.unsqueeze(2) - centreX.unsqueeze(1)).abs().lt(Distance)
                .logical_and((centreY.unsqueeze(2) - centreY.unsqueeze(1)).abs().lt(Distance)).to_type(ScalarType.Float32); // [B, A, B']

            // Ordered pairs (A, B) of distinct real positives; the table keys the unordered pair.
            var distinct = present.unsqueeze(2).logical_and(present.unsqueeze(1))
                .logical_and(rows.unsqueeze(2).ne(rows.unsqueeze(1)));
            var (low, high) = (minimum(rows.unsqueeze(2), rows.unsqueeze(1)), maximum(rows.unsqueeze(2), rows.unsqueeze(1)));
            var key = (low * 1_000_003L + high).masked_fill(distinct.logical_not(), -1);
            var slot = Hash(key);
            var weight = (confident.unsqueeze(2) * confident.unsqueeze(1)).masked_fill(distinct.logical_not(), 0);

            // Claim empty slots, then read and update the pairs that own theirs.
            var flatKey = key.flatten();
            var flatSlot = slot.flatten();
            var real = flatKey.ge(0);
            var empty = _keys.index_select(0, flatSlot).eq(-1).logical_and(real);
            _keys.index_put_(flatKey.masked_select(empty), flatSlot.masked_select(empty));
            var owned = _keys.index_select(0, flatSlot).eq(flatKey).logical_and(real).view_as(key);
            var seen = _weight.index_select(0, flatSlot).view_as(key);
            var together = _together.index_select(0, flatSlot).view_as(key);
            var known = owned.logical_and(seen.ge(MinObservations)).to_type(ScalarType.Float32);
            var rate = together / seen.clamp_min(1e-6f);

            // Negative weight for B in A's best window: how rarely they are together, times the confidence in A.
            var gated = ConfidenceGate > 0 ? confident * confident.ge(ConfidenceGate).to_type(ScalarType.Float32) : confident;
            var negative = (1 - rate).clamp(0, 1) * gated.unsqueeze(2) * known; // [B, A, B']
            weight = weight * owned.to_type(ScalarType.Float32);
            _weight.index_add_(0, flatSlot, weight.flatten(), 1);
            _together.index_add_(0, flatSlot, (weight * colocated).flatten(), 1);
            _stats.index_add_(0, arange(3, device: device),
                stack([weight.gt(0).sum().to_type(ScalarType.Float64), negative.gt(0).sum().to_type(ScalarType.Float64), negative.sum().to_type(ScalarType.Float64)]), 1);
            if (++_steps % DecayEvery == 0)
            {
                _weight.mul_(Decay);
                _together.mul_(Decay);
            }
            negativeWeight = negative.MoveToOuterDisposeScope();
        }

        // B's logit in A's best window: [B, A, B'] = logits[b, B', best[b, A]].
        var inWindow = logits.unsqueeze(1).expand(batch, positives, positives, -1)
            .gather(3, best.unsqueeze(2).unsqueeze(3).expand(batch, positives, positives, 1)).squeeze(3);
        var terms = negativeWeight * nn.functional.softplus(inWindow);
        var loss = PerPair
            ? (terms.sum([1, 2]) / negativeWeight.gt(0).sum([1, 2]).clamp_min(1).to_type(terms.dtype)).sum() / batch
            : terms.sum() / batch;
        negativeWeight.Dispose();
        best.Dispose();
        return loss.MoveToOuterDisposeScope();
    }

    /// <summary>Pair observations, pairs given a negative and their summed weight since the last call; resets them (reads the device).</summary>
    public (double Observed, double Negatives, double Weight) TakeStats()
    {
        var values = _stats.cpu().data<double>().ToArray();
        _stats.zero_();
        return (values[0], values[1], values[2]);
    }

    /// <summary>
    /// Seeds the table from a prior (lines "tagA \t tagB \t together \t observations", as <c>evaluate --locality-prior-out</c>
    /// writes), matched to <paramref name="rowOf"/> by tag name, each pair's observations capped at <see cref="PriorCap"/>.
    /// Returns how many pairs were loaded.
    /// </summary>
    public int LoadPrior(string path, IReadOnlyDictionary<string, int> rowOf)
    {
        var keys = new List<long>();
        var weights = new List<float>();
        var together = new List<float>();
        foreach (var line in File.ReadLines(path))
        {
            var parts = line.Split('\t');
            if (parts.Length != 4 || !rowOf.TryGetValue(parts[0], out var a) || !rowOf.TryGetValue(parts[1], out var b) || a == b)
                continue;
            var seen = float.Parse(parts[3], System.Globalization.CultureInfo.InvariantCulture);
            var share = float.Parse(parts[2], System.Globalization.CultureInfo.InvariantCulture) / Math.Max(seen, 1e-6f);
            var capped = Math.Min(seen, PriorCap);
            keys.Add(Math.Min(a, b) * 1_000_003L + Math.Max(a, b));
            weights.Add(capped);
            together.Add(capped * share);
        }
        if (keys.Count == 0)
            return 0;
        using var scope = NewDisposeScope();
        var device = _keys.device;
        var key = tensor(keys.ToArray(), device: device);
        var slot = Hash(key);
        // First come, first served, as in training: a slot keeps the first pair written to it.
        var empty = _keys.index_select(0, slot).eq(-1);
        _keys.index_put_(key.masked_select(empty), slot.masked_select(empty));
        var owned = _keys.index_select(0, slot).eq(key).to_type(ScalarType.Float32);
        _weight.index_add_(0, slot, tensor(weights.ToArray(), device: device) * owned, 1);
        _together.index_add_(0, slot, tensor(together.ToArray(), device: device) * owned, 1);
        return (int)owned.sum().item<float>();
    }

    /// <summary>The weighted share of observations together for a pair of rows, or null below <see cref="MinObservations"/> or when its slot is another pair's (tests, diagnostics).</summary>
    public double? Together(int a, int b)
    {
        var key = Math.Min(a, b) * 1_000_003L + Math.Max(a, b);
        using var keyTensor = tensor(new[] { key }, device: _keys.device);
        using var slot = Hash(keyTensor);
        var index = slot.cpu().data<long>()[0];
        if (_keys[index].item<long>() != key)
            return null;
        var seen = _weight[index].item<float>();
        return seen >= MinObservations ? _together[index].item<float>() / seen : null;
    }

    // Multiplicative hashing of the 64-bit key into the table.
    private Tensor Hash(Tensor key) =>
        key.mul(-7046029254386353131L).bitwise_right_shift(20).bitwise_and(_slotMask).masked_fill(key.lt(0), 0);

    public void Dispose()
    {
        _keys.Dispose();
        _weight.Dispose();
        _together.Dispose();
        _stats.Dispose();
    }
}
