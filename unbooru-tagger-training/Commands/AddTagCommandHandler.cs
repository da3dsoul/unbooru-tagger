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
/// retrain is needed).
/// </summary>
public static class AddTagCommandHandler
{
    public static int Run(
        string checkpointDir,
        string tag,
        string imagesManifestPath,
        int steps,
        double learningRate,
        int minImageThreshold,
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

        using var pixelBatch = ImageBatchLoader.Load(imagePaths, config.InputSize);
        var allTagIndices = tensor(Enumerable.Range(0, vocabulary.Records.Count).Select(i => (long)i).ToArray());
        using var labels = BatchLabelBuilder.Build(imageTagRows, vocabulary.Records.Count);

        for (var step = 0; step < steps; step++)
        {
            var (pooled, _) = imageTower.forward(pixelBatch);
            var tagEmbeddings = tagTower.forward(allTagIndices);
            var loss = SigmoidContrastiveLoss.Compute(pooled, tagEmbeddings, labels);

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            Console.WriteLine($"step {step + 1}/{steps} loss {loss.item<float>():F4}");
        }

        var updatedRows = tagTower.ExtractRows();
        for (var i = 0; i < updatedRows.Length; i++)
            embeddings.SetRow(i, updatedRows[i]);

        vocabulary.PromoteIfThresholdMet(tag, minImageThreshold);
        Checkpoint.Save(checkpointDir, imageTower, config, vocabulary, embeddings);

        return 0;
    }
}
