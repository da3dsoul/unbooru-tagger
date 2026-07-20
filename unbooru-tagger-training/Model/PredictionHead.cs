using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace UnbooruTagger.Training.Model;

/// <summary>
/// SimSiam-style predictor MLP (Chen &amp; He, 2021): maps one view's pooled embedding
/// toward the other view's, used only by <see cref="Training.SelfSupervisedConsistencyLoss"/>.
/// Applying this to only one side of a pair — paired with stop-gradient on the other side —
/// is what keeps that consistency loss from collapsing to a trivial constant solution,
/// without needing a separate momentum teacher network.
/// </summary>
public sealed class PredictionHead : Module<Tensor, Tensor>
{
    public readonly Linear Hidden;
    public readonly Linear Output;

    public PredictionHead(int embeddingDim, int hiddenDim, Device? device = null) : base(nameof(PredictionHead))
    {
        Hidden = Linear(embeddingDim, hiddenDim, device: device);
        Output = Linear(hiddenDim, embeddingDim, device: device);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var x = Hidden.forward(input);
        x = functional.relu(x);
        return Output.forward(x);
    }
}
