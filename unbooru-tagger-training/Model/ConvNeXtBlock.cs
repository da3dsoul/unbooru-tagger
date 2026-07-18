using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace UnbooruTagger.Training.Model;

/// <summary>
/// One ConvNeXt-style residual block: a depthwise 7x7 conv, a norm, and a 4x-expanding
/// pointwise MLP — a compact stand-in for the "ViT or ConvNeXt backbone" CLAUDE.md calls
/// for. Exported to ONNX node-for-node by <see cref="Export.ImageTowerOnnxExporter"/>,
/// so keep this and that exporter in lockstep if the architecture changes.
/// </summary>
public sealed class ConvNeXtBlock : Module<Tensor, Tensor>
{
    public readonly Conv2d Depthwise;
    public readonly GroupNorm Norm;
    public readonly Conv2d PointwiseExpand;
    public readonly Conv2d PointwiseProject;
    public readonly Parameter LayerScale;

    public ConvNeXtBlock(long channels) : base(nameof(ConvNeXtBlock))
    {
        Depthwise = Conv2d(channels, channels, kernel_size: 7, padding: 3, groups: channels);
        Norm = GroupNorm(1, channels);
        PointwiseExpand = Conv2d(channels, channels * 4, kernel_size: 1);
        PointwiseProject = Conv2d(channels * 4, channels, kernel_size: 1);

        // ConvNeXt's "LayerScale": start each block as (near) an identity mapping so a
        // stack of several blocks can't compound an unlucky random init into an
        // activation blowup (observed in practice: real training runs hit NaN loss from
        // step 1 without this). The branch's contribution grows from ~0 as training
        // makes it useful, rather than being full-strength from the very first forward pass.
        LayerScale = new Parameter(ones([1, channels, 1, 1]) * 1e-6);

        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var x = Depthwise.forward(input);
        x = Norm.forward(x);
        x = PointwiseExpand.forward(x);
        x = functional.gelu(x);
        x = PointwiseProject.forward(x);
        return input + (LayerScale * x);
    }
}
