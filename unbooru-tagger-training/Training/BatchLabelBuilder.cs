using static TorchSharp.torch;

namespace UnbooruTagger.Training.Training;

/// <summary>Builds the [batchImages, vocabularySize] +1/-1 label matrix <see cref="SigmoidContrastiveLoss"/> expects.</summary>
public static class BatchLabelBuilder
{
    public static Tensor Build(IReadOnlyList<IReadOnlyList<int>> imageTagRows, int vocabularySize)
    {
        var labels = new float[imageTagRows.Count * vocabularySize];
        Array.Fill(labels, -1f);
        for (var i = 0; i < imageTagRows.Count; i++)
            foreach (var tagRow in imageTagRows[i])
                labels[i * vocabularySize + tagRow] = 1f;

        return tensor(labels, [imageTagRows.Count, vocabularySize]);
    }
}
