using UnbooruTagger.Training.Model;
using static TorchSharp.torch;

namespace UnbooruTagger.Tests.Training;

public class TagTowerFreezeTests
{
    /// <summary>
    /// The "add a tag" pipeline fine-tunes exactly one row (CLAUDE.md's tag-growth
    /// section). This exercises TagTower's frozen/trainable split against a real
    /// TorchSharp optimizer step, since a wrong Parameter-registration would either
    /// fail to train the target row or silently leak gradients into the frozen ones.
    /// </summary>
    [Fact]
    public void OptimizerStep_OnlyChangesTheTrainableRow()
    {
        float[][] rows = [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]];
        var tagTower = TagTower.CreateWithSingleTrainableRow(rows, trainableRowIndex: 1, embeddingDim: 4);

        var optimizer = optim.SGD(tagTower.parameters().ToArray(), learningRate: 1.0);
        var indices = tensor(new long[] { 0, 1, 2 });

        var output = tagTower.forward(indices);
        var loss = output.sum();
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        var updated = tagTower.ExtractRows();
        Assert.Equal(1f, updated[0][0]);
        Assert.Equal(3f, updated[2][0]);
        Assert.NotEqual(2f, updated[1][0]);
    }
}
