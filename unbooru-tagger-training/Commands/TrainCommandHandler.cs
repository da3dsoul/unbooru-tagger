using UnbooruTagger.Core.Dataset;
using UnbooruTagger.Core.Embedding;
using UnbooruTagger.Core.Vocabulary;
using UnbooruTagger.Training.Checkpoints;
using UnbooruTagger.Training.Data;
using UnbooruTagger.Training.Model;
using UnbooruTagger.Training.Training;
using static TorchSharp.torch;

namespace UnbooruTagger.Training.Commands;

/// <summary>
/// The full/periodic fine-tune pass — CLAUDE.md build-order step 1: validate the
/// pipeline on the top few thousand most common tags before dealing with long-tail pain.
/// </summary>
public static class TrainCommandHandler
{
    public static int Run(
        string manifestPath,
        string checkpointDir,
        int embeddingDim,
        int inputSize,
        int epochs,
        int batchSize,
        double learningRate)
    {
        var manifest = DatasetManifest.Load(manifestPath);
        var (vocabulary, imageTagRows) = BuildVocabulary(manifest);
        var tagFrequencies = ComputeTagFrequencies(imageTagRows);

        var initialRows = Enumerable.Range(0, vocabulary.Records.Count)
            .Select(_ => EmbeddingInit.RandomRow(embeddingDim))
            .ToArray();

        var imageTower = new ImageTower(embeddingDim);
        var tagTower = TagTower.CreateFullyTrainable(initialRows, embeddingDim);

        var optimizer = optim.Adam(imageTower.parameters().Concat(tagTower.parameters()).ToArray(), lr: learningRate);
        var sampler = new RareTagOversamplingBatchSampler(imageTagRows, tagFrequencies);
        var allTagIndices = tensor(Enumerable.Range(0, vocabulary.Records.Count).Select(i => (long)i).ToArray());

        var stepsPerEpoch = Math.Max(1, manifest.Entries.Count / batchSize);
        for (var epoch = 0; epoch < epochs; epoch++)
        {
            for (var step = 0; step < stepsPerEpoch; step++)
            {
                var batchImageIndices = sampler.SampleBatch(batchSize);
                var imagePaths = batchImageIndices.Select(i => manifest.Entries[i].ImagePath).ToList();

                using var pixelBatch = ImageBatchLoader.Load(imagePaths, inputSize);
                var (pooled, _) = imageTower.forward(pixelBatch);
                var tagEmbeddings = tagTower.forward(allTagIndices);

                var batchTagRows = batchImageIndices.Select(i => imageTagRows[i]).ToList();
                using var labels = BatchLabelBuilder.Build(batchTagRows, vocabulary.Records.Count);
                var loss = SigmoidContrastiveLoss.Compute(pooled, tagEmbeddings, labels);

                optimizer.zero_grad();
                loss.backward();
                optimizer.step();

                Console.WriteLine($"epoch {epoch + 1}/{epochs} step {step + 1}/{stepsPerEpoch} loss {loss.item<float>():F4}");
            }
        }

        var embeddings = TagEmbeddingStore.CreateEmpty(embeddingDim);
        foreach (var row in tagTower.ExtractRows())
            embeddings.AppendRow(row);

        var config = ModelConfig.Default(embeddingDim, inputSize);
        Checkpoint.Save(checkpointDir, imageTower, config, vocabulary, embeddings);

        return 0;
    }

    private static (TagVocabulary Vocabulary, List<List<int>> ImageTagRows) BuildVocabulary(DatasetManifest manifest)
    {
        var vocabulary = TagVocabulary.CreateEmpty();
        var imageTagRows = new List<List<int>>();

        foreach (var entry in manifest.Entries)
        {
            var rows = new List<int>();
            foreach (var tag in entry.Tags)
            {
                if (!vocabulary.TryGet(tag, out var record))
                    record = vocabulary.AddTag(tag);
                record.ImageCount++;
                rows.Add(record.RowIndex);
            }
            imageTagRows.Add(rows);
        }

        return (vocabulary, imageTagRows);
    }

    private static Dictionary<int, int> ComputeTagFrequencies(IReadOnlyList<IReadOnlyList<int>> imageTagRows)
    {
        var frequencies = new Dictionary<int, int>();
        foreach (var rows in imageTagRows)
            foreach (var row in rows)
                frequencies[row] = frequencies.GetValueOrDefault(row) + 1;
        return frequencies;
    }
}
