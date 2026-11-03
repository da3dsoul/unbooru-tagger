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
public static class MixedPrecision
{
    public static Tensor Conv(Conv2d conv, Tensor x) => x.dtype == conv.weight!.dtype
        ? conv.forward(x)
        : functional.conv2d(x, conv.weight.to_type(x.dtype), conv.bias?.to_type(x.dtype), conv.stride, conv.padding, conv.dilation, conv.groups);

    /// <summary>
    /// A 1x1 convolution on rows of channels ([rows, in] x [in, out], every pixel of every image a row): one matrix multiply
    /// however many images there are, same weights. On the B70 it replaced a 1x1 convolution, whose oneDNN weight gradient
    /// ran ~7x slower than the layer's forward, and then a batched multiply ([out, in] x [N, in, H*W]), which ran one small
    /// multiply per image: for 480 glimpses with 2x2 to 16x16 maps that wasted most of the device (glimpse pass forward and
    /// backward 79 -> 64 ms; the 10-image 512px pass 127 -> 109 ms).
    /// </summary>
    public static Tensor Rows(Conv2d conv, Tensor rows)
    {
        var weight = conv.weight!.flatten(1);
        if (weight.dtype != rows.dtype)
            weight = weight.to_type(rows.dtype);
        var y = rows.matmul(weight.t());
        if (conv.bias is { } bias)
            y = y + (bias.dtype == rows.dtype ? bias : bias.to_type(rows.dtype));
        return y;
    }

    /// <summary>libtorch's GPU group norm accumulates bf16 statistics in fp32 itself.</summary>
    public static Tensor Norm(GroupNorm norm, Tensor x)
    {
        if (norm.num_groups == 1 && IsChannelsLast(x))
        {
            // One group normalizes over (C, H, W): a layer norm over the contiguous NHWC view, then the per-channel affine,
            // which keeps the channels-last order (group norm returns NCHW whatever it is given).
            var (n, c, h, w) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]);
            var normalized = functional.layer_norm(x.permute(0, 2, 3, 1).reshape(n, h * w * c), [h * w * c], eps: norm.eps).view(n, h, w, c);
            var weight = norm.weight!.dtype == x.dtype ? norm.weight : norm.weight.to_type(x.dtype);
            var bias = norm.bias!.dtype == x.dtype ? norm.bias : norm.bias.to_type(x.dtype);
            return addcmul(bias, normalized, weight, 1).permute(0, 3, 1, 2);
        }
        return x.dtype == norm.weight!.dtype
            ? norm.forward(x)
            : functional.group_norm(x, norm.num_groups, norm.weight.to_type(x.dtype), norm.bias!.to_type(x.dtype), norm.eps);
    }

    /// <summary>Whether <paramref name="x"/> ([N, C, H, W]) is laid out channels-last (NHWC strides).</summary>
    public static bool IsChannelsLast(Tensor x) =>
        x.dim() == 4 && x.shape[1] > 1 && x.stride(1) == 1 && x.stride(3) == x.shape[1] && x.stride(2) == x.shape[3] * x.shape[1];

    /// <summary><paramref name="x"/> ([N, C, H, W]) with the same shape and values, laid out channels-last.</summary>
    public static Tensor ToChannelsLast(Tensor x) => x.permute(0, 2, 3, 1).contiguous().permute(0, 3, 1, 2);

    /// <summary>One of <see cref="ImageTower.Layers"/>: a group norm, a downsample conv or a <see cref="ConvNeXtBlock"/>.</summary>
    public static Tensor Layer(Module<Tensor, Tensor> layer, Tensor x) => layer switch
    {
        GroupNorm norm => Norm(norm, x),
        Conv2d conv => Conv(conv, x),
        _ => layer.forward(x),
    };
}
