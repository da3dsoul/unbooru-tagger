using UnbooruTagger.Training.Model;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace UnbooruTagger.Tests.Training;

/// <summary>
/// <see cref="ConvNeXtBlock"/> runs its 1x1 convolutions as matrix multiplies (MixedPrecision.Rows, faster to train on
/// the B70); the checkpoint and the ONNX export still treat them as the convolutions they are, so the two must agree.
/// </summary>
public class ConvNeXtBlockTests
{
    [Fact]
    public void Forward_MatchesTheBlockBuiltFromItsConvolutions_ValuesAndGradients()
    {
        manual_seed(7);
        var block = new ConvNeXtBlock(6);
        using (no_grad())
        {
            block.LayerScale.fill_(0.5f);
            block.PointwiseExpand.bias!.normal_();
            block.PointwiseProject.bias!.normal_();
        }
        using var input = randn(3, 6, 5, 7);

        var output = block.forward(input);
        output.square().sum().backward();
        var gradients = new[] { block.PointwiseExpand.weight!.grad!.clone(), block.PointwiseProject.weight!.grad!.clone(), block.PointwiseExpand.bias!.grad!.clone() };
        block.zero_grad();

        var x = block.Norm.forward(block.Depthwise.forward(input));
        var reference = input + block.LayerScale * block.PointwiseProject.forward(functional.gelu(block.PointwiseExpand.forward(x)));
        reference.square().sum().backward();

        Assert.True(output.allclose(reference, rtol: 1e-5, atol: 1e-6));
        Assert.True(gradients[0].allclose(block.PointwiseExpand.weight!.grad!, rtol: 1e-4, atol: 1e-5));
        Assert.True(gradients[1].allclose(block.PointwiseProject.weight!.grad!, rtol: 1e-4, atol: 1e-5));
        Assert.True(gradients[2].allclose(block.PointwiseExpand.bias!.grad!, rtol: 1e-4, atol: 1e-5));
    }
}
