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
/// pipeline on the top few thousand most common tags before dealing with long-tail
/// pain. Accepts either a raw <c>DatasetManifest</c> (decoded/normalized on the fly,
/// every epoch) or a <c>PreprocessedDatasetCache</c> built by unbooru-tagger-data's
/// large mode (decoded/normalized once, ahead of time — the "maximum speed" path).
/// Holds out a validation split and stops early once validation loss stops improving,
/// instead of always running the full <paramref name="epochs"/> count.
/// </summary>
public static class TrainCommandHandler
{
    public static int Run(
        string? manifestPath,
        string? cacheDir,
        string checkpointDir,
        int embeddingDim,
        int inputSize,
        int epochs,
        int batchSize,
        double learningRate,
        double validationFraction,
        int earlyStoppingPatience)
    {
        if ((manifestPath is null) == (cacheDir is null))
            throw new ArgumentException("Provide exactly one of --manifest or --cache-dir.");

        using var cacheReader = cacheDir is not null ? new PreprocessedDatasetCacheReader(cacheDir) : null;

        TagVocabulary vocabulary;
        List<List<int>> imageTagRows;
        int datasetCount;
        int resolvedInputSize;
        Func<IReadOnlyList<int>, Tensor> loadBatch;

        if (cacheReader is not null)
        {
            vocabulary = TagVocabulary.Load(Path.Combine(cacheDir!, "tag_vocabulary.json"));
            imageTagRows = cacheReader.ImageTagRows.Select(rows => rows.ToList()).ToList();
            datasetCount = cacheReader.ImageCount;
            resolvedInputSize = cacheReader.InputSize;
            loadBatch = indices => LoadCacheBatch(cacheReader, indices, resolvedInputSize);
        }
        else
        {
            var manifest = DatasetManifest.Load(manifestPath!);
            (vocabulary, imageTagRows) = BuildVocabulary(manifest);
            datasetCount = manifest.Entries.Count;
            resolvedInputSize = inputSize;
            loadBatch = indices => ImageBatchLoader.Load(indices.Select(i => manifest.Entries[i].ImagePath).ToList(), resolvedInputSize);
        }

        var tagFrequencies = ComputeTagFrequencies(imageTagRows);

        var initialRows = Enumerable.Range(0, vocabulary.Records.Count)
            .Select(_ => EmbeddingInit.RandomRow(embeddingDim))
            .ToArray();

        var imageTower = new ImageTower(embeddingDim);
        var tagTower = TagTower.CreateFullyTrainable(initialRows, embeddingDim);
        var optimizer = optim.Adam(imageTower.parameters().Concat(tagTower.parameters()).ToArray(), lr: learningRate);
        var allTagIndices = tensor(Enumerable.Range(0, vocabulary.Records.Count).Select(i => (long)i).ToArray());

        var (trainingIndices, validationIndices) = SplitForValidation(datasetCount, validationFraction);
        var trainingTagRows = trainingIndices.Select(i => imageTagRows[i]).ToList();
        var sampler = new RareTagOversamplingBatchSampler(trainingTagRows, tagFrequencies);
        var earlyStopping = new EarlyStopping(earlyStoppingPatience);

        var stepsPerEpoch = Math.Max(1, trainingIndices.Count / batchSize);
        for (var epoch = 0; epoch < epochs; epoch++)
        {
            for (var step = 0; step < stepsPerEpoch; step++)
            {
                var localBatchIndices = sampler.SampleBatch(batchSize);
                var globalBatchIndices = localBatchIndices.Select(li => trainingIndices[li]).ToList();

                using var pixelBatch = loadBatch(globalBatchIndices);
                var (pooled, _) = imageTower.forward(pixelBatch);
                var tagEmbeddings = tagTower.forward(allTagIndices);

                var batchTagRows = localBatchIndices.Select(li => trainingTagRows[li]).ToList();
                using var labels = BatchLabelBuilder.Build(batchTagRows, vocabulary.Records.Count);
                var loss = SigmoidContrastiveLoss.Compute(pooled, tagEmbeddings, labels);

                optimizer.zero_grad();
                loss.backward();
                optimizer.step();

                Console.WriteLine($"epoch {epoch + 1}/{epochs} step {step + 1}/{stepsPerEpoch} loss {loss.item<float>():F4}");
            }

            if (validationIndices.Count == 0)
            {
                Console.WriteLine("Not enough images for a validation split — running the full epoch count without early stopping.");
                continue;
            }

            var validationLoss = Evaluate(imageTower, tagTower, loadBatch, imageTagRows, validationIndices, allTagIndices, vocabulary.Records.Count);
            Console.WriteLine($"epoch {epoch + 1}/{epochs} validation loss {validationLoss:F4}");

            if (earlyStopping.ShouldStop(validationLoss))
            {
                Console.WriteLine($"Early stopping: validation loss stopped improving after epoch {epoch + 1}.");
                break;
            }
        }

        var embeddings = TagEmbeddingStore.CreateEmpty(embeddingDim);
        foreach (var row in tagTower.ExtractRows())
            embeddings.AppendRow(row);

        var config = ModelConfig.Default(embeddingDim, resolvedInputSize);
        Checkpoint.Save(checkpointDir, imageTower, config, vocabulary, embeddings);

        return 0;
    }

    private static double Evaluate(
        ImageTower imageTower,
        TagTower tagTower,
        Func<IReadOnlyList<int>, Tensor> loadBatch,
        IReadOnlyList<IReadOnlyList<int>> imageTagRows,
        IReadOnlyList<int> validationIndices,
        Tensor allTagIndices,
        int vocabularySize)
    {
        using var _ = no_grad();

        using var pixelBatch = loadBatch(validationIndices);
        var (pooled, _) = imageTower.forward(pixelBatch);
        var tagEmbeddings = tagTower.forward(allTagIndices);

        var batchTagRows = validationIndices.Select(i => imageTagRows[i]).ToList();
        using var labels = BatchLabelBuilder.Build(batchTagRows, vocabularySize);
        using var loss = SigmoidContrastiveLoss.Compute(pooled, tagEmbeddings, labels);

        return loss.item<float>();
    }

    private static Tensor LoadCacheBatch(PreprocessedDatasetCacheReader cache, IReadOnlyList<int> indices, int inputSize)
    {
        var imageSize = 3 * inputSize * inputSize;
        var flat = new float[indices.Count * imageSize];
        for (var i = 0; i < indices.Count; i++)
            cache.ReadImage(indices[i]).CopyTo(flat, i * imageSize);

        return tensor(flat, [indices.Count, 3, inputSize, inputSize]);
    }

    /// <summary>Shuffles all dataset indices once and carves off a validation slice, so the same held-out images are used for every epoch's evaluation.</summary>
    private static (List<int> Training, List<int> Validation) SplitForValidation(int datasetCount, double validationFraction)
    {
        var indices = Enumerable.Range(0, datasetCount).ToList();
        var random = Random.Shared;
        for (var i = indices.Count - 1; i > 0; i--)
        {
            var j = random.Next(i + 1);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }

        var validationCount = datasetCount <= 1 ? 0 : Math.Clamp((int)(datasetCount * validationFraction), 1, datasetCount - 1);
        return (indices.Skip(validationCount).ToList(), indices.Take(validationCount).ToList());
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
