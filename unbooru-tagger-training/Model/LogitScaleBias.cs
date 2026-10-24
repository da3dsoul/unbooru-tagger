using TorchSharp;
using TorchSharp.Modules;
using UnbooruTagger.Core.Scoring;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace UnbooruTagger.Training.Model;

/// <summary>
/// Learnable logit scale and bias applied before <see cref="Training.SigmoidContrastiveLoss"/>'s
/// sigmoid: <c>sigmoid(exp(LogScale) * dot + Bias)</c> — see <see cref="LogitScaling"/> for why
/// this exists. Scale is stored in log-space (<c>Scale = exp(LogScale)</c>) so gradient
/// descent can't push it through zero, mirroring the SigLIP paper's own parametrization.
/// Bias starts strongly negative so an untrained tag defaults to "no match" instead of
/// ~50/50, rather than making every rare tag's own embedding shrink to represent that.
/// </summary>
public sealed class LogitScaleBias : Module<Tensor, Tensor>
{
    private readonly Parameter _logScale;
    private readonly Parameter _bias;

    public LogitScaleBias(float initialScale = 10f, float initialBias = -10f, Device? device = null)
        : base(nameof(LogitScaleBias))
    {
        _logScale = new Parameter(tensor(MathF.Log(initialScale), device: device));
        _bias = new Parameter(tensor(initialBias, device: device));
        RegisterComponents();
    }

    public float Scale => _logScale.exp().item<float>();
    public float Bias => _bias.item<float>();

    public LogitScaling ToLogitScaling() => new(Scale, Bias);

    public override Tensor forward(Tensor logits) => logits * _logScale.exp() + _bias;
}
