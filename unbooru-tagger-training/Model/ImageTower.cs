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
/// </summary>
public sealed class ImageTower : Module<Tensor, (Tensor Pooled, Tensor Spatial)>
{
    public const long StemKernel = 4;
    public const long StemStride = 4;

    public readonly Conv2d Stem;
    public readonly ModuleList<Module<Tensor, Tensor>> Layers;
    public readonly Linear Projection;

    public int EmbeddingDim { get; }

    public ImageTower(int embeddingDim, int stemChannels = 64, int[]? stageChannels = null, int[]? blocksPerStage = null)
        : base(nameof(ImageTower))
    {
        stageChannels ??= [64, 128, 256];
        blocksPerStage ??= [2, 2, 2];
        EmbeddingDim = embeddingDim;

        Stem = Conv2d(3, stemChannels, kernel_size: StemKernel, stride: StemStride);

        var layers = new List<Module<Tensor, Tensor>>();
        long inChannels = stemChannels;
        for (var stage = 0; stage < stageChannels.Length; stage++)
        {
            long outChannels = stageChannels[stage];
            if (outChannels != inChannels)
                layers.Add(Conv2d(inChannels, outChannels, kernel_size: 2, stride: 2));
            for (var b = 0; b < blocksPerStage[stage]; b++)
                layers.Add(new ConvNeXtBlock(outChannels));
            inChannels = outChannels;
        }

        Layers = ModuleList(layers.ToArray());
        Projection = Linear(inChannels, embeddingDim);

        RegisterComponents();
    }

    public override (Tensor Pooled, Tensor Spatial) forward(Tensor input)
    {
        var x = Stem.forward(input);
        foreach (var layer in Layers)
            x = layer.forward(x);

        var spatial = x;
        var pooled = functional.adaptive_avg_pool2d(x, [1L, 1L]).flatten(1);
        pooled = Projection.forward(pooled);
        return (pooled, spatial);
    }
}
