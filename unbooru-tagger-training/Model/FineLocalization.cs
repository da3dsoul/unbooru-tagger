using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace UnbooruTagger.Training.Model;

/// <summary>
/// A stride-2 level under the localization map (<c>--fine-localization</c>, 2026-11-03). The tower's finest features are the
/// stem's, at stride 4, so this level makes its own from the pixels: a 2x2 stride-2 patchify, plus the stage-0 features
/// (1x1, nearest-upsampled 2x) so it knows what the coarse map was looking at, one ConvNeXt block, and a 1x1 projection into
/// the embedding space that is added to the coarse localization map, nearest-upsampled 2x. The projection starts at zero, so
/// a tower that gains this level scores exactly as before until training moves it. Only the localization map uses it;
/// classification still pools the coarse map.
/// Kept out of <see cref="ImageTower"/>'s registered modules (like <see cref="ImageTower.GlimpseTower"/>) so checkpoints
/// without it load unchanged: saved in its own file with its own optimizer.
/// </summary>
public sealed class FineLocalization : Module<Tensor, Tensor, Tensor>
{
    public const int Stride = 2;
    public const long DefaultChannels = 64;

    public readonly Conv2d Patchify;
    public readonly Conv2d Coarse;
    public readonly GroupNorm Norm;
    public readonly ConvNeXtBlock Block;
    public readonly Conv2d Projection;

    public long Channels { get; }

    public FineLocalization(long stageZeroChannels, long embeddingDim, long channels = DefaultChannels, Device? device = null)
        : base(nameof(FineLocalization))
    {
        Channels = channels;
        Patchify = Conv2d(3, channels, kernel_size: Stride, stride: Stride, device: device);
        Coarse = Conv2d(stageZeroChannels, channels, kernel_size: 1, device: device);
        Norm = ConvNeXtBlock.NewGroupNorm(channels, device);
        Block = new ConvNeXtBlock(channels, device);
        Projection = Conv2d(channels, embeddingDim, kernel_size: 1, device: device);
        using (no_grad())
        {
            Projection.weight!.zero_();
            Projection.bias!.zero_();
        }
        RegisterComponents();
    }

    /// <summary>
    /// [batch, embeddingDim, 2h, 2w] from <paramref name="pixels"/> (the tower's input, already in its compute type and layout)
    /// and <paramref name="stageZero"/> ([batch, C0, h, w], stride 4).
    /// </summary>
    public override Tensor forward(Tensor pixels, Tensor stageZero)
    {
        using var scope = NewDisposeScope();
        return MixedPrecision.Conv(Projection, Features(pixels, stageZero)).MoveToOuterDisposeScope();
    }

    /// <summary><see cref="forward"/> before the projection: [batch, <see cref="Channels"/>, 2h, 2w], in the compute type.</summary>
    public Tensor Features(Tensor pixels, Tensor stageZero)
    {
        using var scope = NewDisposeScope();
        var fine = MixedPrecision.Conv(Patchify, pixels);
        var coarse = MixedPrecision.Conv(Coarse, stageZero);
        fine = fine + functional.interpolate(coarse, size: [fine.shape[2], fine.shape[3]], mode: InterpolationMode.Nearest);
        return MixedPrecision.Layer(Block, MixedPrecision.Norm(Norm, fine)).MoveToOuterDisposeScope();
    }
}

/// <summary>
/// A localization map with a <see cref="FineLocalization"/> level, kept in its two parts: the coarse map
/// (<see cref="Coarse"/>, [batch, D, h, w]) and the fine level's features before its 1x1 projection (<see cref="Features"/>,
/// [batch, C, 2h, 2w]), both fp32. The map itself is <c>Upsample2x(Coarse) + W · Features + b</c>. At 512px that is
/// [batch, 512, 256, 256], 134 MB per image in fp32 for each of the upsample, the add and their gradients, while a tag's
/// logits only need <c>e · Coarse</c> and <c>(eW) · Features + e · b</c>: the projection moves onto the tag side, 64
/// channels instead of 512 at 256x256 (<see cref="Training.SigmoidContrastiveLoss.ComputeLocalizedPairs(FineLocalizationMap, Tensor, Training.LocalizationPairs, float, Tensor, LogitScaleBias?, float, float)"/>).
/// </summary>
public sealed record FineLocalizationMap(Tensor Coarse, Tensor Features, Tensor Weight, Tensor Bias)
{
    /// <summary>The parts of <paramref name="fine"/>'s output: its projection as [D, C] and [D].</summary>
    public static FineLocalizationMap From(Tensor coarse, Tensor features, FineLocalization fine) =>
        new(coarse, features, fine.Projection.weight!.flatten(1), fine.Projection.bias!);

    /// <summary>The map itself, [batch, D, 2h, 2w] (what <see cref="ImageTower.ForwardPooled"/> returns).</summary>
    public Tensor Dense()
    {
        var upsampled = functional.interpolate(Coarse, size: [Features.shape[2], Features.shape[3]], mode: InterpolationMode.Nearest);
        return upsampled + functional.conv2d(Features, Weight.unsqueeze(2).unsqueeze(3), Bias);
    }

    /// <summary>The maps of the given images (a [n] index tensor).</summary>
    public FineLocalizationMap Select(Tensor images) => this with { Coarse = Coarse.index_select(0, images), Features = Features.index_select(0, images) };

    /// <summary>
    /// The square of <paramref name="size"/> fine cells (even) at fine cell (<paramref name="top"/>, <paramref name="left"/>)
    /// (both even, so it covers whole coarse cells) of image <paramref name="image"/>.
    /// </summary>
    public FineLocalizationMap Crop(long image, long top, long left, long size)
    {
        if (top % 2 != 0 || left % 2 != 0 || size % 2 != 0)
            throw new ArgumentException("A crop of the fine map must cover whole coarse cells.");
        return this with
        {
            Coarse = Coarse.narrow(0, image, 1).narrow(2, top / 2, size / 2).narrow(3, left / 2, size / 2),
            Features = Features.narrow(0, image, 1).narrow(2, top, size).narrow(3, left, size),
        };
    }
}
