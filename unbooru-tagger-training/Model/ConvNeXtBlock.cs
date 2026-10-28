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

    public ConvNeXtBlock(long channels, Device? device = null) : base(nameof(ConvNeXtBlock))
    {
        Depthwise = Conv2d(channels, channels, kernel_size: 7, padding: 3, groups: channels, device: device);
        // A larger-than-default eps (PyTorch/TorchSharp default is 1e-5): LayerScale
        // dampens a block's *output* once it's computed, but if variance genuinely lands
        // near zero, dividing by sqrt(variance + eps) can still overflow to Infinity/NaN
        // internally before LayerScale ever gets a chance to scale it down — Infinity/NaN
        // survive multiplication by any finite scale. A bigger eps keeps the denominator
        // bounded away from zero in the first place.
        Norm = NewGroupNorm(channels, device);
        PointwiseExpand = Conv2d(channels, channels * 4, kernel_size: 1, device: device);
        PointwiseProject = Conv2d(channels * 4, channels, kernel_size: 1, device: device);

        // ConvNeXt's "LayerScale": start each block as (near) an identity mapping so a
        // stack of several blocks can't compound an unlucky random init into an
        // activation blowup (observed in practice: real training runs hit NaN loss from
        // step 1 without this). The branch's contribution grows from ~0 as training
        // makes it useful, rather than being full-strength from the very first forward pass.
        LayerScale = new Parameter(ones([1, channels, 1, 1], device: device) * 1e-6);

        RegisterComponents();
    }

    /// <summary>
    /// <c>GroupNorm(1, channels, eps: 1e-3)</c> with its affine parameters explicitly set to
    /// weight 1 / bias 0. TorchSharp 0.107's GroupNorm leaves them as uninitialized memory:
    /// on CPU that is sometimes NaN (every forward pass NaN from step 1), on CUDA usually
    /// zeros (a norm that outputs nothing until gradients grow its weight).
    /// </summary>
    internal static GroupNorm NewGroupNorm(long channels, Device? device)
    {
        var norm = GroupNorm(1, channels, eps: 1e-3, device: device);
        using (no_grad())
        {
            norm.weight!.fill_(1);
            norm.bias!.fill_(0);
        }
        return norm;
    }

    public override Tensor forward(Tensor input)
    {
        // TorchSharp frees an intermediate only when its dispose scope ends, not when it goes out of use as Python's
        // refcounting would, so without this scope every block's six intermediates lived until the caller's scope
        // ended: a 10-image forward at 512px held ~7.4GB, and the evaluator ran the A4000 out of memory on gold
        // batches of 16. Autograd keeps its own references to what backward needs, so training is unaffected.
        using var scope = NewDisposeScope();
        // In the input's dtype (MixedPrecision): bf16 when ImageTower.ComputeType says so.
        var x = MixedPrecision.Conv(Depthwise, input);
        x = MixedPrecision.Norm(Norm, x);
        x = MixedPrecision.Conv(PointwiseExpand, x);
        x = functional.gelu(x);
        x = MixedPrecision.Conv(PointwiseProject, x);
        var layerScale = LayerScale.dtype == input.dtype ? (Tensor)LayerScale : LayerScale.to_type(input.dtype);
        return scope.MoveToOuter(input + (layerScale * x));
    }
}
