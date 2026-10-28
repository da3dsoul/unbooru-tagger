using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace UnbooruTagger.Training.Model;

/// <summary>
/// The image tower: a compact ConvNeXt-inspired stack that exposes both the pooled
/// global embedding and the pre-pool spatial feature map, as CLAUDE.md's image-tower
/// spec requires for localization. <see cref="Layers"/> interleaves stride-2 downsample
/// convs between stages of <see cref="ConvNeXtBlock"/>s.
///
/// <see cref="Projection"/> is a 1x1 conv (not a plain <c>Linear</c> on the pooled
/// vector alone) applied to the full spatial map BEFORE pooling, so the spatial output
/// lives in the same embeddingDim-dimensional space as tag embeddings and the pooled
/// output — required for <c>TagScorer.Heatmap</c>'s dot product against a tag embedding
/// to even be dimensionally valid, let alone meaningful. A 1x1 conv commutes with
/// average pooling (it's a linear map applied per-location), so projecting-then-pooling
/// gives the exact same pooled result as the old pooling-then-projecting order.
///
/// <see cref="FusedStages"/> (0 = off, the original layout) adds a light FPN-style top-down
/// path for sharper localization: the last stage's projected map is upsampled 2x (nearest)
/// and added to a 1x1 projection (<see cref="Laterals"/>) of the previous stage's output,
/// repeated once per fused stage. Fusing 2 stages on a 4-stage stride-32 tower yields a
/// stride-8 map (64x64 at 512px) in the same embedding space, instead of a coarse 16x16
/// one — small objects (earrings, hairclips, chokers) span only a cell or two at stride 16+.
/// </summary>
public sealed class ImageTower : Module<Tensor, (Tensor Pooled, Tensor Spatial)>
{
    public const long StemKernel = 4;
    public const long StemStride = 4;

    public readonly Conv2d Stem;
    public readonly ModuleList<Module<Tensor, Tensor>> Layers;
    public readonly Conv2d Projection;
    public readonly ModuleList<Conv2d> Laterals;

    public int EmbeddingDim { get; }
    public int StemChannels { get; }
    public int[] StageChannels { get; }
    public int[] BlocksPerStage { get; }
    public int FusedStages { get; }

    /// <summary>
    /// When set (requires <see cref="FusedStages"/> &gt; 0), <see cref="ForwardMaps"/>'s
    /// localization map is the fused laterals alone, without the upsampled last-stage
    /// projection. Classification still pools the full map. A last-stage location sees most
    /// of a 512px image, so on the full map every location can learn "this image has a hat"
    /// and the heatmap saturates (run 2, epoch 1: gold pointing 27%). The earlier stages'
    /// laterals have much smaller effective receptive fields, so training localization
    /// against them alone rewards evidence that is actually local. Not a weight change: it
    /// can be switched on when resuming.
    /// </summary>
    public bool LocalizationFromFusedOnly
    {
        get => _localizationFromFusedOnly;
        set
        {
            if (value && FusedStages == 0)
                throw new InvalidOperationException("A fused-only localization map needs --fused-stages > 0.");
            _localizationFromFusedOnly = value;
        }
    }

    private bool _localizationFromFusedOnly;

    /// <summary>
    /// With <see cref="LocalizationFromFusedOnly"/>, how many of the coarsest fused laterals the
    /// localization map leaves out (0 = all of them). Even the fused-only map carries the stride-16
    /// lateral, whose locations see most of the image: run 3 at epoch 2 put the shoes and gloves
    /// peaks on faces, a spot that can see a whole outfit, with 0% gold pointing for shoes and
    /// animal ears. Leaving it out (<c>--localization-map local</c>) keeps only laterals whose
    /// locations see roughly 100px, so a location can only claim a tag it can see. Classification
    /// still pools the full map. Not a weight change: it can be switched when resuming.
    /// </summary>
    public int LocalizationSkippedLaterals
    {
        get => _localizationSkippedLaterals;
        set
        {
            if (value < 0 || (value > 0 && (!LocalizationFromFusedOnly || value >= FusedStages)))
                throw new InvalidOperationException($"Skipping {value} laterals needs a fused-only localization map that keeps at least one (--fused-stages {FusedStages}).");
            _localizationSkippedLaterals = value;
        }
    }

    private int _localizationSkippedLaterals;

    /// <summary>
    /// Above 0, localizable tags' whole-image scores are a log-sum-exp over a grid of the full map's cells at this
    /// temperature (raw dot-product units) instead of the dot product with the averaged embedding
    /// (<see cref="ClassificationPooling"/>); 0 = the averaged embedding for every tag. Not a weight change: it
    /// can be switched when resuming, and evaluation must score the same way.
    /// </summary>
    public float ClassificationPoolingTemperature { get; set; }

    /// <summary>
    /// The dtype the trunk, projection and laterals compute in (<see cref="MixedPrecision"/>); null = the weights'
    /// own (fp32). The weights stay fp32 either way and both maps come out fp32, so everything downstream (tag scores,
    /// losses, pooling) is unchanged. Not a weight change: it can be switched when resuming.
    /// </summary>
    public ScalarType? ComputeType { get; set; }

    /// <summary>How training anneals <see cref="ClassificationPoolingTemperature"/>; null = held fixed.</summary>
    public ClassificationPooling.Schedule? ClassificationPoolingSchedule { get; set; }

    /// <summary>Input pixels per spatial-map cell: the stem's stride, x2 per downsampling stage, /2 per fused stage.</summary>
    public int OutputStride { get; }

    /// <summary>Index into <see cref="Layers"/> of each stage's last layer — where fusion taps stage outputs.</summary>
    private readonly int[] _stageEnds;

    public ImageTower(int embeddingDim, int stemChannels = 64, int[]? stageChannels = null, int[]? blocksPerStage = null, Device? device = null, int fusedStages = 0)
        : base(nameof(ImageTower))
    {
        stageChannels ??= [64, 128, 256];
        blocksPerStage ??= [2, 2, 2];
        if (fusedStages < 0 || fusedStages >= stageChannels.Length)
            throw new ArgumentException($"fusedStages must be between 0 and {stageChannels.Length - 1} (stages - 1).", nameof(fusedStages));
        EmbeddingDim = embeddingDim;
        StemChannels = stemChannels;
        StageChannels = stageChannels;
        BlocksPerStage = blocksPerStage;
        FusedStages = fusedStages;
        var stageEnds = new int[stageChannels.Length];

        Stem = Conv2d(3, stemChannels, kernel_size: StemKernel, stride: StemStride, device: device);

        // Real ConvNeXt normalizes right after the stem and before every downsample
        // (its downsample "layer" is literally LayerNorm -> Conv). Without that, plain
        // unnormalized downsample convs let activation magnitude drift across stages
        // with nothing to rein it in — traced to a real bug: by the last block of the
        // last stage, that drift occasionally pushed a ConvNeXtBlock's internal
        // GroupNorm into computing NaN, LayerScale notwithstanding (LayerScale only
        // dampens a block's finite output; it can't recover an already-NaN/Infinity
        // value produced inside the branch).
        var layers = new List<Module<Tensor, Tensor>> { ConvNeXtBlock.NewGroupNorm(stemChannels, device) };
        long inChannels = stemChannels;
        for (var stage = 0; stage < stageChannels.Length; stage++)
        {
            long outChannels = stageChannels[stage];
            if (outChannels != inChannels)
            {
                layers.Add(ConvNeXtBlock.NewGroupNorm(inChannels, device));
                layers.Add(Conv2d(inChannels, outChannels, kernel_size: 2, stride: 2, device: device));
            }
            for (var b = 0; b < blocksPerStage[stage]; b++)
                layers.Add(new ConvNeXtBlock(outChannels, device));
            stageEnds[stage] = layers.Count - 1;
            inChannels = outChannels;
        }

        _stageEnds = stageEnds;
        var stride = (int)StemStride;
        long previousChannels = stemChannels;
        foreach (var channels in stageChannels)
        {
            if (channels != previousChannels)
                stride *= 2;
            previousChannels = channels;
        }
        OutputStride = stride >> fusedStages;
        Layers = ModuleList(layers.ToArray());
        Projection = Conv2d(inChannels, embeddingDim, kernel_size: 1, device: device);
        // Laterals[k] projects stage (last - 1 - k) into the embedding space.
        Laterals = ModuleList(Enumerable.Range(1, fusedStages)
            .Select(k => Conv2d(stageChannels[stageChannels.Length - 1 - k], embeddingDim, kernel_size: 1, device: device))
            .ToArray());

        RegisterComponents();
    }

    /// <summary>Index into <see cref="Layers"/> of the last layer of <paramref name="stage"/>.</summary>
    public int StageEnd(int stage) => _stageEnds[stage];

    public override (Tensor Pooled, Tensor Spatial) forward(Tensor input)
    {
        var (spatial, _) = ForwardMaps(input);
        var pooled = functional.adaptive_avg_pool2d(spatial, [1L, 1L]).flatten(1);
        return (pooled, spatial);
    }

    /// <summary>
    /// The full spatial map (what classification pools) and the localization map (what
    /// heatmaps and the localization loss use). They are the same tensor unless
    /// <see cref="LocalizationFromFusedOnly"/> is set.
    /// </summary>
    public (Tensor Spatial, Tensor Localization) ForwardMaps(Tensor input)
    {
        // As in ConvNeXtBlock.forward: block and stage outputs and upsampled maps are freed here, not by the caller.
        using var scope = NewDisposeScope();
        var stageOutputs = Trunk(input, out var x);
        var (spatial, localization) = FusedChain(stageOutputs, x);
        var same = ReferenceEquals(localization, spatial);
        spatial = ToFloat32(spatial);
        localization = same ? spatial : ToFloat32(localization);
        scope.MoveToOuter(spatial);
        if (!ReferenceEquals(localization, spatial))
            scope.MoveToOuter(localization);
        return (spatial, localization);
    }

    /// <summary>The full map and the localization map from the trunk's outputs (<see cref="ForwardMaps"/>).</summary>
    private (Tensor Spatial, Tensor Localization) FusedChain(Tensor[] stageOutputs, Tensor last)
    {
        var spatial = MixedPrecision.Conv(Projection, last);
        Tensor? localization = null;
        for (var k = 1; k <= FusedStages; k++)
        {
            var lateral = MixedPrecision.Conv(Laterals[k - 1], stageOutputs[StageChannels.Length - 1 - k]);
            spatial = Upsample(spatial, lateral) + lateral;
            if (LocalizationFromFusedOnly && k > LocalizationSkippedLaterals)
                localization = localization is null ? lateral : Upsample(localization, lateral) + lateral;
        }
        return (spatial, localization ?? spatial);
    }

    private static Tensor ToFloat32(Tensor map) => map.dtype == ScalarType.Float32 ? map : map.to_type(ScalarType.Float32);

    private static Tensor Upsample(Tensor map, Tensor like) =>
        functional.interpolate(map, size: [like.shape[2], like.shape[3]], mode: InterpolationMode.Nearest);

    /// <summary>Stem and stages: each stage's output, and the last one in <paramref name="last"/>.</summary>
    private Tensor[] Trunk(Tensor input, out Tensor last)
    {
        var x = MixedPrecision.Conv(Stem, ComputeType is { } computeType && computeType != input.dtype ? input.to_type(computeType) : input);
        var stageOutputs = new Tensor[StageChannels.Length];
        var nextStage = 0;
        for (var i = 0; i < Layers.Count; i++)
        {
            x = MixedPrecision.Layer(Layers[i], x);
            if (nextStage < _stageEnds.Length && i == _stageEnds[nextStage])
                stageOutputs[nextStage++] = x;
        }
        last = x;
        return stageOutputs;
    }

    /// <summary>
    /// <c>MaskedPool(ForwardMaps(input).Spatial, mask)</c> and the localization map, without building the full map.
    /// The full map is the coarsest projection plus each fused lateral, nearest-upsampled 2x per level to the finest
    /// grid; training only ever pools it, and at 512px it is [batch, 512, 128, 128] for every one of those adds, in
    /// both passes. Nearest upsampling by an exact factor repeats each value over a block, so the masked sum over the
    /// fine grid is each level's map times the mask summed over its blocks, added up over the levels. Falls back to
    /// <see cref="ForwardMaps"/> when the localization map is the full map or a level isn't an exact 2x.
    /// </summary>
    /// <param name="mask">[batch, 1, height, width] at the full map's resolution, as for <see cref="MaskedPool"/>.</param>
    public (Tensor Pooled, Tensor Localization) ForwardPooled(Tensor input, Tensor mask)
    {
        using var scope = NewDisposeScope();
        var stageOutputs = Trunk(input, out var x);
        // Level k (0 = the projection of the last stage, k > 0 = lateral k) is upsampled 2^(FusedStages - k) times.
        Tensor LevelInput(int k) => k == 0 ? x : stageOutputs[StageChannels.Length - 1 - k];
        var exact = LocalizationFromFusedOnly && Enumerable.Range(0, FusedStages + 1).All(k =>
            LevelInput(k).shape[2] << (FusedStages - k) == mask.shape[2] && LevelInput(k).shape[3] << (FusedStages - k) == mask.shape[3]);

        Tensor pooled, localization;
        if (exact)
        {
            Tensor? weightedSum = null;
            Tensor? fused = null;
            for (var k = 0; k <= FusedStages; k++)
            {
                // Back to fp32 before the sums over up to 128x128 cells.
                var map = ToFloat32(k == 0 ? MixedPrecision.Conv(Projection, x) : MixedPrecision.Conv(Laterals[k - 1], LevelInput(k)));
                if (k > LocalizationSkippedLaterals)
                    fused = fused is null ? map : Upsample(fused, map) + map;
                var factor = 1L << (FusedStages - k);
                var blockMask = factor == 1 ? mask : functional.avg_pool2d(mask, factor) * (factor * factor);
                var term = (map * blockMask).sum([2, 3]);
                weightedSum = weightedSum is null ? term : weightedSum + term;
            }
            pooled = weightedSum! / mask.sum([2, 3]).clamp_min(1e-6);
            localization = fused!;
        }
        else
        {
            var (spatial, fullLocalization) = FusedChain(stageOutputs, x);
            pooled = MaskedPool(ToFloat32(spatial), mask);
            localization = ToFloat32(fullLocalization);
        }

        scope.MoveToOuter(pooled);
        scope.MoveToOuter(localization);
        return (pooled, localization);
    }

    /// <summary>
    /// Global average pool restricted to <paramref name="mask"/>'s valid locations
    /// (from <see cref="SpatialMask"/>) instead of every location uniformly —
    /// <see cref="forward"/>'s plain <c>adaptive_avg_pool2d</c> would otherwise let the
    /// letterbox padding bars (see <c>UnbooruTagger.Core.Encoding.ImagePreprocessing</c>)
    /// dilute the pooled embedding for any non-square training image.
    /// </summary>
    /// <param name="spatial">[batch, embeddingDim, height, width] — this tower's pre-pool spatial map.</param>
    /// <param name="mask">[batch, 1, height, width] — 1 for valid (real content) locations, 0 for padding.</param>
    public static Tensor MaskedPool(Tensor spatial, Tensor mask)
    {
        var maskedSum = (spatial * mask).sum([2, 3]);
        var validCount = mask.sum([2, 3]).clamp_min(1e-6);
        return maskedSum / validCount;
    }
}
