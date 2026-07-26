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
/// </summary>
public sealed class ImageTower : Module<Tensor, (Tensor Pooled, Tensor Spatial)>
{
    public const long StemKernel = 4;
    public const long StemStride = 4;

    public readonly Conv2d Stem;
    public readonly ModuleList<Module<Tensor, Tensor>> Layers;
    public readonly Conv2d Projection;

    public int EmbeddingDim { get; }

    public ImageTower(int embeddingDim, int stemChannels = 64, int[]? stageChannels = null, int[]? blocksPerStage = null, Device? device = null)
        : base(nameof(ImageTower))
    {
        stageChannels ??= [64, 128, 256];
        blocksPerStage ??= [2, 2, 2];
        EmbeddingDim = embeddingDim;

        Stem = Conv2d(3, stemChannels, kernel_size: StemKernel, stride: StemStride, device: device);

        // Real ConvNeXt normalizes right after the stem and before every downsample
        // (its downsample "layer" is literally LayerNorm -> Conv). Without that, plain
        // unnormalized downsample convs let activation magnitude drift across stages
        // with nothing to rein it in — traced to a real bug: by the last block of the
        // last stage, that drift occasionally pushed a ConvNeXtBlock's internal
        // GroupNorm into computing NaN, LayerScale notwithstanding (LayerScale only
        // dampens a block's finite output; it can't recover an already-NaN/Infinity
        // value produced inside the branch).
        var layers = new List<Module<Tensor, Tensor>> { GroupNorm(1, stemChannels, eps: 1e-3, device: device) };
        long inChannels = stemChannels;
        for (var stage = 0; stage < stageChannels.Length; stage++)
        {
            long outChannels = stageChannels[stage];
            if (outChannels != inChannels)
            {
                layers.Add(GroupNorm(1, inChannels, eps: 1e-3, device: device));
                layers.Add(Conv2d(inChannels, outChannels, kernel_size: 2, stride: 2, device: device));
            }
            for (var b = 0; b < blocksPerStage[stage]; b++)
                layers.Add(new ConvNeXtBlock(outChannels, device));
            inChannels = outChannels;
        }

        Layers = ModuleList(layers.ToArray());
        Projection = Conv2d(inChannels, embeddingDim, kernel_size: 1, device: device);

        RegisterComponents();
    }

    public override (Tensor Pooled, Tensor Spatial) forward(Tensor input)
    {
        var x = Stem.forward(input);
        foreach (var layer in Layers)
            x = layer.forward(x);

        var spatial = Projection.forward(x);
        var pooled = functional.adaptive_avg_pool2d(spatial, [1L, 1L]).flatten(1);
        return (pooled, spatial);
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
