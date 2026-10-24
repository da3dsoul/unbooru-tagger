using UnbooruTagger.Training.Commands;

namespace UnbooruTagger.Tests.Training;

public class TagWarmStartEmbedderFactoryTests
{
    [Fact]
    public void Create_ReturnsNullWhenNeitherFlagIsGiven()
    {
        var result = TagWarmStartEmbedderFactory.Create(modelPath: null, vocabPath: null, embeddingDim: 512);

        Assert.Null(result);
    }

    [Fact]
    public void Create_ThrowsWhenOnlyTheModelPathIsGiven()
    {
        Assert.Throws<ArgumentException>(() =>
            TagWarmStartEmbedderFactory.Create(modelPath: "model.onnx", vocabPath: null, embeddingDim: 512));
    }

    [Fact]
    public void Create_ThrowsWhenOnlyTheVocabPathIsGiven()
    {
        Assert.Throws<ArgumentException>(() =>
            TagWarmStartEmbedderFactory.Create(modelPath: null, vocabPath: "vocab.txt", embeddingDim: 512));
    }
}
