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

    /// <summary>
    /// With <see cref="ClassificationGlimpses"/>, one learned code per <see cref="Glimpses.Zooms"/> entry, added to a
    /// glimpse's pooled embedding so a tag can tell a 32px window enlarged from a 256px one shrunk; null otherwise.
    /// </summary>
    public readonly Parameter? GlimpseZoomCodes;

    /// <summary>
    /// With <see cref="ClassificationGlimpses"/>, [D, D]: the whole-image proposal head (<see cref="Proposals"/>). Starts as
    /// the identity. Null otherwise.
    /// </summary>
    public readonly Parameter? ProposalWeights;

    /// <summary>
    /// The whole-image embedding through the proposal head ([B, D]): its dot product with a localizable tag says whether the
    /// image might hold that tag, from everything in view, context included. That only decides where the glimpses look
    /// (<see cref="Glimpses.Propose"/>); the tag's score comes from the glimpses. A head of its own, so suspecting gloves
    /// from the character doesn't pull the gloves embedding the glimpses are scored with towards the character.
    /// </summary>
    public Tensor Proposals(Tensor pooled) => pooled.matmul(ProposalWeights!.t());

    /// <summary>
    /// Localizable tags are scored from zoomed glimpses of the image (<see cref="Glimpses"/>) instead of the whole map, at
    /// <see cref="ClassificationPoolingTemperature"/>. A weight change (it adds <see cref="GlimpseZoomCodes"/>), so it is
    /// set when the tower is built: a fresh run, saved in the model config.
    /// </summary>
    public bool ClassificationGlimpses => GlimpseZoomCodes is not null;

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
    /// Above 0, the pooled localizable tags' score is the mean of their <c>k</c> highest per-window (or per-cell) dots instead
    /// of the log-sum-exp (<see cref="ClassificationPooling.Pool(Tensor, Tensor, float, int)"/>): only those windows get a
    /// gradient, so a "yes" trains the windows that show the tag and a "no" the ones that fire for it. Not a weight change;
    /// saved in the model config, and evaluation must score the same way.
    /// </summary>
    public int ClassificationPoolingTopK { get; set; }

    /// <summary>
    /// WILDCAT's negative evidence (Durand et al. 2017) with <see cref="ClassificationPoolingTopK"/>: this weight times the mean
    /// of the k lowest windows' dots is added to the top-k mean, so a window where the tag clearly isn't counts too. 0 = off.
    /// </summary>
    public float ClassificationPoolingBottomWeight { get; set; }

    /// <summary>
    /// The glimpse zooms (<see cref="Glimpses.Zooms"/> indices below this) whose windows score localizable tags; the rest are
    /// still chosen and embedded but don't pool. 3 leaves out the 256px window, which holds a quarter of the image and with it
    /// the character (Saccader and GFNet keep their classifier to about a third of the image side). Saved in the model config.
    /// </summary>
    public int GlimpsePoolZooms { get; set; } = Glimpses.Zooms.Length;

    /// <summary>
    /// The smallest glimpse zoom (<see cref="Glimpses.Zooms"/> index) whose windows score localizable tags; 1 leaves out the 32px
    /// window, whose embeddings held the least of any zoom on the gold set (2026-11-02 probes). Saved in the model config.
    /// </summary>
    public int GlimpsePoolMinZoom { get; set; }

    /// <summary>
    /// Singh et al. 2020's feature split on the glimpse embedding: its first this many dimensions are the object half, the rest
    /// the context half. Where a localizable tag is a positive and the image carries none of its context sources (the
    /// character/costume/artist usually seen with it: an exclusive image), training scores it with the context half held at its
    /// running mean (<see cref="GlimpseContextMean"/>) and without gradient there, so the object half has to carry those calls.
    /// 0 = off. Saved in the model config.
    /// </summary>
    public int GlimpseObjectDims { get; set; }

    /// <summary>The context half's running mean over training glimpses (<see cref="GlimpseObjectDims"/>); saved in the model config.</summary>
    public float[]? GlimpseContextMean { get; set; }

    /// <summary>
    /// Evaluation only, not saved: glimpses score with the context half replaced by <see cref="GlimpseContextMean"/>, as training
    /// scores exclusive images, so a tag's call comes from the object half alone (a per-tag constant shift, which pooling keeps).
    /// </summary>
    public bool GlimpseObjectOnly { get; set; }

    /// <summary>
    /// A separate tower that embeds the glimpses (<see cref="Glimpses.Embed(ImageTower, Tensor, Tensor, GlimpseChoice, int)"/>)
    /// in place of this one: same shape, its own weights (copied from this tower when first made, <see cref="CopyForGlimpses"/>),
    /// its own optimizer and file (<c>glimpse_tower.dat</c>). The 2026-11-02 gold probes found every window's embedding near
    /// its whole image's (cosine ~0.85) and objects such as gloves unreadable at any zoom, so this tests whether sharing the
    /// 512px path's weights is what keeps a 64px crop from encoding its object (RA-CNN gives each scale its own parameters).
    /// Not a registered submodule: <see cref="Module.parameters"/>, saving and the main optimizer leave it out. Null = glimpses
    /// use this tower. The zoom codes and the proposal head stay here.
    /// </summary>
    public ImageTower? GlimpseTower { get; set; }

    /// <summary>A new tower of this one's shape (no glimpse codes or proposal head) with a copy of its weights, for <see cref="GlimpseTower"/>.</summary>
    public ImageTower CopyForGlimpses()
    {
        var device = Projection.weight!.device;
        var copy = new ImageTower(EmbeddingDim, StemChannels, StageChannels, BlocksPerStage, device, FusedStages)
        {
            LocalizationFromFusedOnly = LocalizationFromFusedOnly,
            LocalizationSkippedLaterals = LocalizationSkippedLaterals,
        };
        var source = named_parameters().ToDictionary(p => p.name, p => p.parameter);
        using (no_grad())
        {
            foreach (var (name, parameter) in copy.named_parameters())
                parameter.copy_(source[name]);
            var buffers = named_buffers().ToDictionary(b => b.name, b => b.buffer);
            foreach (var (name, buffer) in copy.named_buffers())
                buffer.copy_(buffers[name]);
        }
        return copy;
    }

    /// <summary>
    /// The dtype the trunk, projection and laterals compute in (<see cref="MixedPrecision"/>); null = the weights'
    /// own (fp32). The weights stay fp32 either way and both maps come out fp32, so everything downstream (tag scores,
    /// losses, pooling) is unchanged. Not a weight change: it can be switched when resuming.
    /// </summary>
    public ScalarType? ComputeType { get; set; }

    /// <summary>
    /// Run the trunk channels-last (NHWC memory order, the same NCHW shapes and values): each ConvNeXt block's MLP then
    /// reads its rows without a copy and its residual add is contiguous (<see cref="MixedPrecision.Norm"/> keeps the order).
    /// The stage outputs the laterals and projection read are put back in NCHW order once, so the maps come out as before.
    /// Not a weight change.
    /// </summary>
    public bool ChannelsLast { get; set; }

    /// <summary>How training anneals <see cref="ClassificationPoolingTemperature"/>; null = held fixed.</summary>
    public ClassificationPooling.Schedule? ClassificationPoolingSchedule { get; set; }

    /// <summary>
    /// With <see cref="ClassificationGlimpses"/>, the windows per image the model chooses to look at (<see cref="Glimpses.Propose"/>),
    /// in training and evaluation alike. Not a weight change; saved with the checkpoint so evaluation looks as much as training did.
    /// </summary>
    public int GlimpseCount { get; set; } = 16;

    /// <summary>
    /// Evaluation only: score every window of the grid (<see cref="Glimpses.Layout"/>) instead of the model's chosen ones, as
    /// evaluations did before 2026-10-31, to compare with them. Not saved.
    /// </summary>
    public bool GlimpseGrid { get; set; }

    /// <summary>
    /// Input pixels per spatial-map cell: the stem's stride, x2 per downsampling stage, /2 per fused stage, /2 again with
    /// <see cref="Fine"/> (whose maps, the full one included, then come at its resolution).
    /// </summary>
    public int OutputStride => Fine is null ? _coarseStride : _coarseStride / FineLocalization.Stride;

    private readonly int _coarseStride;

    /// <summary>
    /// A stride-2 level under the localization map (<see cref="FineLocalization"/>, <c>--fine-localization</c>); null without it.
    /// Not a registered module: saved in its own file, so checkpoints without it load unchanged. Needs
    /// <see cref="LocalizationFromFusedOnly"/> with every stage fused, so the localization map's finest level is stage 0's.
    /// </summary>
    public FineLocalization? Fine { get; set; }

    /// <summary>A new <see cref="FineLocalization"/> for this tower (zero output until trained).</summary>
    public FineLocalization NewFineLocalization(Device? device = null)
    {
        if (!LocalizationFromFusedOnly || FusedStages != StageChannels.Length - 1 || LocalizationSkippedLaterals != 0)
            throw new InvalidOperationException("--fine-localization adds a level under the localization map's stage-0 lateral: it needs --localization-map fused and every stage fused.");
        return new FineLocalization(StageChannels[0], EmbeddingDim, device: device);
    }

    /// <summary>Index into <see cref="Layers"/> of each stage's last layer — where fusion taps stage outputs.</summary>
    private readonly int[] _stageEnds;

    public ImageTower(int embeddingDim, int stemChannels = 64, int[]? stageChannels = null, int[]? blocksPerStage = null, Device? device = null, int fusedStages = 0,
        bool classificationGlimpses = false)
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
        _coarseStride = stride >> fusedStages;
        Layers = ModuleList(layers.ToArray());
        Projection = Conv2d(inChannels, embeddingDim, kernel_size: 1, device: device);
        // Laterals[k] projects stage (last - 1 - k) into the embedding space.
        Laterals = ModuleList(Enumerable.Range(1, fusedStages)
            .Select(k => Conv2d(stageChannels[stageChannels.Length - 1 - k], embeddingDim, kernel_size: 1, device: device))
            .ToArray());
        if (classificationGlimpses)
        {
            GlimpseZoomCodes = new Parameter(zeros(Glimpses.Zooms.Length, embeddingDim, device: device));
            ProposalWeights = new Parameter(eye(embeddingDim, device: device));
        }

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
        var stageOutputs = Trunk(input, out var x, out var typed);
        var (spatial, localization) = FusedChain(stageOutputs, x, typed);
        var same = ReferenceEquals(localization, spatial);
        spatial = ToFloat32(spatial);
        localization = same ? spatial : ToFloat32(localization);
        scope.MoveToOuter(spatial);
        if (!ReferenceEquals(localization, spatial))
            scope.MoveToOuter(localization);
        return (spatial, localization);
    }

    /// <summary>The full map and the localization map from the trunk's outputs (<see cref="ForwardMaps"/>).</summary>
    private (Tensor Spatial, Tensor Localization) FusedChain(Tensor[] stageOutputs, Tensor last, Tensor typed)
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
        if (Fine is not null && localization is not null)
        {
            var fine = Fine.forward(typed, stageOutputs[0]);
            localization = Upsample(localization, fine) + fine;
            // The full map at the same resolution, so callers' masks (built at OutputStride) fit both; nearest-upsampled, so
            // its masked pool is unchanged when the mask is too.
            spatial = Upsample(spatial, fine);
        }
        return (spatial, localization ?? spatial);
    }

    private static Tensor ToFloat32(Tensor map) => map.dtype == ScalarType.Float32 ? map : map.to_type(ScalarType.Float32);

    private static Tensor Upsample(Tensor map, Tensor like) =>
        functional.interpolate(map, size: [like.shape[2], like.shape[3]], mode: InterpolationMode.Nearest);

    /// <summary>Stem and stages: each stage's output, and the last one in <paramref name="last"/>.</summary>
    private Tensor[] Trunk(Tensor input, out Tensor last, out Tensor typed)
    {
        typed = ComputeType is { } computeType && computeType != input.dtype ? input.to_type(computeType) : input;
        if (ChannelsLast)
            typed = MixedPrecision.ToChannelsLast(typed);
        var x = MixedPrecision.Conv(Stem, typed);
        var stageOutputs = new Tensor[StageChannels.Length];
        var nextStage = 0;
        for (var i = 0; i < Layers.Count; i++)
        {
            x = MixedPrecision.Layer(Layers[i], x);
            if (nextStage < _stageEnds.Length && i == _stageEnds[nextStage])
                stageOutputs[nextStage++] = ChannelsLast ? x.contiguous() : x;
        }
        last = ChannelsLast ? x.contiguous() : x;
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
        var (pooled, localization) = ForwardPooled(input, mask, factored: false);
        var map = localization as Tensor ?? ((FineLocalizationMap)localization).Dense();
        scope.MoveToOuter(pooled);
        scope.MoveToOuter(map);
        return (pooled, map);
    }

    /// <summary>
    /// <see cref="ForwardPooled(Tensor, Tensor)"/> with <see cref="Fine"/> set, the localization map left in its two parts
    /// (<see cref="FineLocalizationMap"/>) instead of built at 256x256 x the embedding width.
    /// </summary>
    public (Tensor Pooled, FineLocalizationMap Localization) ForwardPooledFine(Tensor input, Tensor mask)
    {
        if (Fine is null)
            throw new InvalidOperationException("ForwardPooledFine needs a fine localization level.");
        using var scope = NewDisposeScope();
        var (pooled, localization) = ForwardPooled(input, mask, factored: true);
        var map = (FineLocalizationMap)localization;
        scope.MoveToOuter(pooled);
        scope.MoveToOuter(map.Coarse);
        scope.MoveToOuter(map.Features);
        scope.MoveToOuter(map.Weight); // a view of the projection's weight; the bias is the parameter itself
        return (pooled, map);
    }

    /// <returns>The pooled embedding and the localization map: a tensor, or with <paramref name="factored"/> a <see cref="FineLocalizationMap"/>.</returns>
    private (Tensor Pooled, object Localization) ForwardPooled(Tensor input, Tensor mask, bool factored)
    {
        var stageOutputs = Trunk(input, out var x, out var typed);
        // Level k (0 = the projection of the last stage, k > 0 = lateral k) is upsampled 2^(FusedStages - k) times, once more
        // with the fine level (the mask is then at its resolution).
        Tensor LevelInput(int k) => k == 0 ? x : stageOutputs[StageChannels.Length - 1 - k];
        var fineLevels = Fine is null ? 0 : 1;
        var exact = LocalizationFromFusedOnly && Enumerable.Range(0, FusedStages + 1).All(k =>
            LevelInput(k).shape[2] << (FusedStages - k + fineLevels) == mask.shape[2] && LevelInput(k).shape[3] << (FusedStages - k + fineLevels) == mask.shape[3]);

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
                var factor = 1L << (FusedStages - k + fineLevels);
                var blockMask = factor == 1 ? mask : functional.avg_pool2d(mask, factor) * (factor * factor);
                var term = (map * blockMask).sum([2, 3]);
                weightedSum = weightedSum is null ? term : weightedSum + term;
            }
            pooled = weightedSum! / mask.sum([2, 3]).clamp_min(1e-6);
            if (Fine is not null)
                return (pooled, FineLocalizationMap.From(fused!, ToFloat32(Fine.Features(typed, stageOutputs[0])), Fine));
            localization = fused!;
        }
        else
        {
            if (factored)
                throw new InvalidOperationException("The factored fine map needs every level an exact power-of-two multiple of the mask's grid.");
            var (spatial, fullLocalization) = FusedChain(stageOutputs, x, typed);
            pooled = MaskedPool(ToFloat32(spatial), mask);
            localization = ToFloat32(fullLocalization);
        }

        return (pooled, localization);
    }

    /// <summary>
    /// <see cref="ForwardPooled"/>'s pooled embedding alone, as glimpses need it. The projection and laterals are 1x1
    /// convolutions, so each level's masked sum is taken over its input features (at most the last stage's channels, over a
    /// few cells) and projected afterwards: <c>Σ mask · (W f + b) = W (Σ mask · f) + b Σ mask</c>. That skips building every
    /// level's embedding-wide map and the localization chain, which the glimpses never use. Falls back to
    /// <see cref="ForwardPooled"/> when that one isn't exact either (a map size that isn't a power-of-two multiple).
    /// </summary>
    public Tensor ForwardPooledOnly(Tensor input, Tensor mask)
    {
        using var scope = NewDisposeScope();
        var stageOutputs = Trunk(input, out var x, out _);
        Tensor LevelInput(int k) => k == 0 ? x : stageOutputs[StageChannels.Length - 1 - k];
        var fineLevels = Fine is null ? 0 : 1;
        var exact = LocalizationFromFusedOnly && Enumerable.Range(0, FusedStages + 1).All(k =>
            LevelInput(k).shape[2] << (FusedStages - k + fineLevels) == mask.shape[2] && LevelInput(k).shape[3] << (FusedStages - k + fineLevels) == mask.shape[3]);
        if (!exact)
        {
            var pooledFallback = ForwardPooled(input, mask).Pooled;
            return pooledFallback.MoveToOuterDisposeScope();
        }

        Tensor? weightedSum = null;
        for (var k = 0; k <= FusedStages; k++)
        {
            var conv = k == 0 ? Projection : Laterals[k - 1];
            var factor = 1L << (FusedStages - k + fineLevels);
            var blockMask = factor == 1 ? mask : functional.avg_pool2d(mask, factor) * (factor * factor); // fine cells per cell
            var summed = (ToFloat32(LevelInput(k)) * blockMask).sum([2, 3]); // [B, C_k]
            var term = summed.matmul(conv.weight!.flatten(1).t());
            if (conv.bias is { } bias)
                term = term + blockMask.sum([1, 2, 3]).unsqueeze(1) * bias;
            weightedSum = weightedSum is null ? term : weightedSum + term;
        }
        var pooled = weightedSum! / mask.sum([2, 3]).clamp_min(1e-6);
        return pooled.MoveToOuterDisposeScope();
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
