using UnbooruTagger.Core.Dataset;
using UnbooruTagger.Core.Encoding;
using UnbooruTagger.Training.Checkpoints;
using UnbooruTagger.Training.Data;
using UnbooruTagger.Training.Model;
using UnbooruTagger.Training.Training;
using static TorchSharp.torch;

namespace UnbooruTagger.Training.Commands;

/// <summary>
/// The "add a tag" pipeline: warm-start a brand-new tag's row and fine-tune ONLY that
/// row against its newly tagged images, image encoder frozen — CLAUDE.md's tag-growth
/// section, and its own explicit deliverable (add one held-out tag, confirm no full
/// retrain is needed). Uses the same "train a bit, check its work" early stopping as
/// `train`, but only when there are enough images for a meaningful validation split —
/// add-tag datasets can be as small as CLAUDE.md's ~10-20 image minimum threshold.
/// </summary>
public static class AddTagCommandHandler
{
    private const int MinImagesForValidationSplit = 6;

    public static int Run(
        string checkpointDir,
        string tag,
        string imagesManifestPath,
        int steps,
        double learningRate,
        int minImageThreshold,
        int earlyStoppingPatience = 5,
        ITagTextEmbedder? warmStartEmbedder = null)
    {
        var (imageTower, config, vocabulary, embeddings) = Checkpoint.Load(checkpointDir);

        if (!vocabulary.TryGet(tag, out var record))
        {
            var warmStart = warmStartEmbedder?.Embed(tag) ?? EmbeddingInit.RandomRow(embeddings.EmbeddingDim);
            record = vocabulary.AddTag(tag);
            embeddings.AppendRow(warmStart);
        }

        var manifest = DatasetManifest.Load(imagesManifestPath);
        record.ImageCount += manifest.Entries.Count;

        // Image encoder frozen: no gradients are computed for its parameters, so the
        // optimizer below (built only from tagTower's parameters) never touches them.
        foreach (var parameter in imageTower.parameters())
            parameter.requires_grad = false;

        var allRows = Enumerable.Range(0, embeddings.RowCount).Select(i => embeddings.GetRow(i).ToArray()).ToArray();
        var tagTower = TagTower.CreateWithSingleTrainableRow(allRows, record.RowIndex, embeddings.EmbeddingDim);
        var optimizer = optim.Adam(tagTower.parameters().ToArray(), lr: learningRate);

        var imagePaths = manifest.Entries.Select(e => e.ImagePath).ToList();
        var imageTagRows = manifest.Entries
            .Select(e => e.Tags
                .Select(t => vocabulary.TryGet(t, out var r) ? r.RowIndex : -1)
                .Where(rowIndex => rowIndex >= 0)
                .ToList())
            .ToList();
        var allTagIndices = tensor(Enumerable.Range(0, vocabulary.Records.Count).Select(i => (long)i).ToArray());

        var (trainingIndices, validationIndices) = SplitForValidation(manifest.Entries.Count);
        var useEarlyStopping = validationIndices.Count > 0;

        using var trainingPixelBatch = ImageBatchLoader.Load(trainingIndices.Select(i => imagePaths[i]).ToList(), config.InputSize);
        using var trainingLabels = BatchLabelBuilder.Build(trainingIndices.Select(i => imageTagRows[i]).ToList(), vocabulary.Records.Count);

        Tensor? validationPixelBatch = null;
        Tensor? validationLabels = null;
        if (useEarlyStopping)
        {
            validationPixelBatch = ImageBatchLoader.Load(validationIndices.Select(i => imagePaths[i]).ToList(), config.InputSize);
            validationLabels = BatchLabelBuilder.Build(validationIndices.Select(i => imageTagRows[i]).ToList(), vocabulary.Records.Count);
        }
        else
        {
            Console.WriteLine($"Only {manifest.Entries.Count} images — too few for a validation split, running the full {steps}-step count without early stopping.");
        }

        var earlyStopping = new EarlyStopping(earlyStoppingPatience);

        for (var step = 0; step < steps; step++)
        {
            var (pooled, _) = imageTower.forward(trainingPixelBatch);
            var tagEmbeddings = tagTower.forward(allTagIndices);
            var loss = SigmoidContrastiveLoss.Compute(pooled, tagEmbeddings, trainingLabels);

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            Console.WriteLine($"step {step + 1}/{steps} loss {loss.item<float>():F4}");

            if (!useEarlyStopping)
                continue;

            using var _ = no_grad();
            var (valPooled, _) = imageTower.forward(validationPixelBatch!);
            var valTagEmbeddings = tagTower.forward(allTagIndices);
            var validationLoss = SigmoidContrastiveLoss.Compute(valPooled, valTagEmbeddings, validationLabels!).item<float>();

            if (earlyStopping.ShouldStop(validationLoss))
            {
                Console.WriteLine($"Early stopping: validation loss stopped improving after step {step + 1}.");
                break;
            }
        }

        validationPixelBatch?.Dispose();
        validationLabels?.Dispose();

        var updatedRows = tagTower.ExtractRows();
        for (var i = 0; i < updatedRows.Length; i++)
            embeddings.SetRow(i, updatedRows[i]);

        vocabulary.PromoteIfThresholdMet(tag, minImageThreshold);
        Checkpoint.Save(checkpointDir, imageTower, config, vocabulary, embeddings);

        return 0;
    }

    /// <summary>Below <see cref="MinImagesForValidationSplit"/>, every image goes to training and no split is made — too little data for a meaningful held-out signal.</summary>
    private static (List<int> Training, List<int> Validation) SplitForValidation(int count)
    {
        if (count < MinImagesForValidationSplit)
            return (Enumerable.Range(0, count).ToList(), []);

        var indices = Enumerable.Range(0, count).ToList();
        var random = Random.Shared;
        for (var i = indices.Count - 1; i > 0; i--)
        {
            var j = random.Next(i + 1);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }

        var validationCount = Math.Max(1, count / 5);
        return (indices.Skip(validationCount).ToList(), indices.Take(validationCount).ToList());
    }
}
