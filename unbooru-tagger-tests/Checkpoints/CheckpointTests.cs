using UnbooruTagger.Core.Embedding;
using UnbooruTagger.Core.Vocabulary;
using UnbooruTagger.Training.Checkpoints;
using UnbooruTagger.Training.Model;

namespace UnbooruTagger.Tests.Checkpoints;

public class CheckpointTests
{
    private const int EmbeddingDim = 4;

    [Fact]
    public void SaveThenLoad_RoundTripsLogitScaleBias()
    {
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            var config = new ModelConfig(EmbeddingDim, InputSize: 32, StemChannels: 4, StageChannels: [4], BlocksPerStage: [1]);
            var imageTower = new ImageTower(config.EmbeddingDim, config.StemChannels, config.StageChannels, config.BlocksPerStage);
            var vocabulary = TagVocabulary.CreateEmpty();
            vocabulary.AddTag("solo");
            var embeddings = TagEmbeddingStore.CreateEmpty(EmbeddingDim);
            embeddings.AppendRow(new float[EmbeddingDim]);
            var logitScaleBias = new LogitScaleBias(initialScale: 7f, initialBias: -6f);

            Checkpoint.Save(directory, imageTower, config, vocabulary, embeddings, logitScaleBias);

            Assert.True(Checkpoint.HasLogitScaleBias(directory));

            var (_, _, _, _, loadedLogitScaleBias) = Checkpoint.Load(directory);
            Assert.Equal(7f, loadedLogitScaleBias.Scale, 3);
            Assert.Equal(-6f, loadedLogitScaleBias.Bias, 3);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    /// <summary>
    /// A checkpoint written before <see cref="LogitScaleBias"/> existed has every other
    /// file but not <c>logit_scale_bias.dat</c> — resuming it must still succeed, just
    /// with that module starting fresh instead of failing to load altogether.
    /// </summary>
    [Fact]
    public void Load_FallsBackToADefaultLogitScaleBiasForAnOlderCheckpoint()
    {
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            var config = new ModelConfig(EmbeddingDim, InputSize: 32, StemChannels: 4, StageChannels: [4], BlocksPerStage: [1]);
            var imageTower = new ImageTower(config.EmbeddingDim, config.StemChannels, config.StageChannels, config.BlocksPerStage);
            var vocabulary = TagVocabulary.CreateEmpty();
            vocabulary.AddTag("solo");
            var embeddings = TagEmbeddingStore.CreateEmpty(EmbeddingDim);
            embeddings.AppendRow(new float[EmbeddingDim]);

            // Simulates a pre-LogitScaleBias checkpoint: every file Save would have
            // written back then, deliberately not calling the new Save overload's
            // logit_scale_bias.dat write.
            imageTower.save(Path.Combine(directory, "image_tower.dat"));
            File.WriteAllText(Path.Combine(directory, "model_config.json"), System.Text.Json.JsonSerializer.Serialize(config));
            vocabulary.Save(Path.Combine(directory, "tag_vocabulary.json"));
            embeddings.Save(Path.Combine(directory, "tag_embeddings.bin"));

            Assert.True(Checkpoint.Exists(directory));
            Assert.False(Checkpoint.HasLogitScaleBias(directory));

            var (_, _, _, _, logitScaleBias) = Checkpoint.Load(directory);

            Assert.Equal(10f, logitScaleBias.Scale, 3);
            Assert.Equal(-10f, logitScaleBias.Bias, 3);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }
}
