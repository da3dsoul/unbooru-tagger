using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace UnbooruTagger.Training.Model;

/// <summary>
/// The image tower's layers in their input's dtype, by hand: TorchSharp 0.107 has no autocast. Weights stay fp32 (so
/// do checkpoints, gradients and the optimizer) and are cast in the graph each call, so their gradients come back fp32
/// through the cast. With an fp32 input these are just the modules' own forwards. The tower is memory-bandwidth-bound
/// on the B70, so halving the bytes its activations move is most of the gain: forward + backward of run 4's tower,
/// 12 images at 512px, 204 ms fp32 vs 161 ms bf16 (benchmark-tower, 2026-10-28).
/// </summary>
internal static class MixedPrecision
{
    public static Tensor Conv(Conv2d conv, Tensor x) => x.dtype == conv.weight!.dtype
        ? conv.forward(x)
        : functional.conv2d(x, conv.weight.to_type(x.dtype), conv.bias?.to_type(x.dtype), conv.stride, conv.padding, conv.dilation, conv.groups);

    /// <summary>libtorch's GPU group norm accumulates bf16 statistics in fp32 itself.</summary>
    public static Tensor Norm(GroupNorm norm, Tensor x) => x.dtype == norm.weight!.dtype
        ? norm.forward(x)
        : functional.group_norm(x, norm.num_groups, norm.weight.to_type(x.dtype), norm.bias!.to_type(x.dtype), norm.eps);

    /// <summary>One of <see cref="ImageTower.Layers"/>: a group norm, a downsample conv or a <see cref="ConvNeXtBlock"/>.</summary>
    public static Tensor Layer(Module<Tensor, Tensor> layer, Tensor x) => layer switch
    {
        GroupNorm norm => Norm(norm, x),
        Conv2d conv => Conv(conv, x),
        _ => layer.forward(x),
    };
}
