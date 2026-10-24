using Spectre.Console;
using UnbooruTagger.Core.Dataset;
using UnbooruTagger.Core.Encoding;
using UnbooruTagger.Core.Vocabulary;
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
///
/// With <c>--cache-dir</c>, the new row also trains against hard negatives drawn from the
/// existing corpus: images carrying the tags that commonly accompany the new one in its
/// manifest (the base character for a new costume, the series for a new character) but
/// not the new tag itself — plus some random corpus images. Without negatives, a manifest
/// of only positives teaches the row nothing but "always yes", and it fires on everything
/// that looks remotely similar: adding a tag would cost precision instead of being free.
///
/// Since the image encoder is frozen, every image is embedded exactly once up front and
/// the row trains on those cached embeddings — no per-step forward passes, and no need to
/// fit the whole image set on the GPU at once.
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
        string? warmStartModelPath = null,
        string? warmStartVocabPath = null,
        string? negativesCacheDir = null,
        int negativesPerPositive = 4)
    {
        var device = DeviceSelector.Best();
        var (imageTower, config, vocabulary, embeddings, logitScaleBias) = Checkpoint.Load(checkpointDir, device);

        // Frozen, like the image encoder above: the corpus-wide operating point a single
        // new tag's handful of images shouldn't be allowed to shift.
        foreach (var parameter in logitScaleBias.parameters())
            parameter.requires_grad = false;

        if (!vocabulary.TryGet(tag, out var record))
        {
            // Built here, not passed in, since it needs embeddings.EmbeddingDim -- only
            // known once the checkpoint above is loaded.
            using var warmStartEmbedder = TagWarmStartEmbedderFactory.Create(warmStartModelPath, warmStartVocabPath, embeddings.EmbeddingDim);
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
        var tagTower = TagTower.CreateWithSingleTrainableRow(allRows, record.RowIndex, embeddings.EmbeddingDim, device);
        var optimizer = optim.Adam(tagTower.parameters().ToArray(), lr: learningRate);

        var imagePaths = manifest.Entries.Select(e => e.ImagePath).ToList();
        var imageTagRows = manifest.Entries
            .Select(e => e.Tags
                .Select(t => vocabulary.TryGet(t, out var r) ? r.RowIndex : -1)
                .Where(rowIndex => rowIndex >= 0)
                .ToList())
            .ToList();
        var allTagIndices = tensor(Enumerable.Range(0, vocabulary.Records.Count).Select(i => (long)i).ToArray(), device: device);

        var (trainingIndices, validationIndices) = SplitForValidation(manifest.Entries.Count);
        var useEarlyStopping = validationIndices.Count > 0;

        var trainingPooled = EmbedPooled(imageTower, ids => ImageBatchLoader.Load(ids.Select(i => imagePaths[i]).ToList(), config.InputSize), trainingIndices, config.InputSize, device);
        var trainingTagRows = trainingIndices.Select(i => (IReadOnlyList<int>)imageTagRows[i]).ToList();
        Tensor? validationPooled = useEarlyStopping
            ? EmbedPooled(imageTower, ids => ImageBatchLoader.Load(ids.Select(i => imagePaths[i]).ToList(), config.InputSize), validationIndices, config.InputSize, device)
            : null;
        var validationTagRows = validationIndices.Select(i => (IReadOnlyList<int>)imageTagRows[i]).ToList();

        if (!useEarlyStopping)
            AnsiConsole.MarkupLineInterpolated($"Only {manifest.Entries.Count} images — too few for a validation split, running the full {steps}-step count without early stopping.");

        if (negativesCacheDir is not null)
        {
            using var cacheReader = new PreprocessedDatasetCacheReader(negativesCacheDir);
            var cacheVocabularyCount = TagVocabulary.Load(Path.Combine(negativesCacheDir, "tag_vocabulary.json")).Records.Count;
            if (cacheVocabularyCount > vocabulary.Records.Count)
                throw new InvalidOperationException(
                    $"--cache-dir has {cacheVocabularyCount} tags but the checkpoint only {vocabulary.Records.Count} -- it isn't the corpus this checkpoint was trained on.");

            var negatives = SelectNegatives(
                cacheReader.ImageTagRows, imageTagRows, record.RowIndex, negativesPerPositive * manifest.Entries.Count, vocabulary);
            var validationNegativeCount = useEarlyStopping ? negatives.Count * validationIndices.Count / manifest.Entries.Count : 0;
            var validationNegatives = negatives.Take(validationNegativeCount).ToList();
            var trainingNegatives = negatives.Skip(validationNegativeCount).ToList();

            AnsiConsole.MarkupLineInterpolated($"Embedding {negatives.Count} corpus negatives for '{tag}'...");
            Func<IReadOnlyList<int>, (Tensor, IReadOnlyList<LetterboxBox>)> loadNegatives =
                ids => TrainCommandHandler.LoadCacheBatch(cacheReader, ids, cacheReader.InputSize);

            trainingPooled = cat([trainingPooled, EmbedPooled(imageTower, loadNegatives, trainingNegatives, config.InputSize, device)], 0);
            trainingTagRows.AddRange(trainingNegatives.Select(i => cacheReader.ImageTagRows[i]));
            if (validationPooled is not null && validationNegatives.Count > 0)
            {
                validationPooled = cat([validationPooled, EmbedPooled(imageTower, loadNegatives, validationNegatives, config.InputSize, device)], 0);
                validationTagRows.AddRange(validationNegatives.Select(i => cacheReader.ImageTagRows[i]));
            }
        }
        else
        {
            AnsiConsole.MarkupLine("[yellow]No --cache-dir: the new tag only sees the manifest's own images as negatives. If they all carry the tag, it learns no negatives at all and will over-fire.[/]");
        }

        using var trainingLabels = BatchLabelBuilder.Build(trainingTagRows, vocabulary.Records.Count).to(device);
        using var validationLabels = useEarlyStopping ? BatchLabelBuilder.Build(validationTagRows, vocabulary.Records.Count).to(device) : null;

        var earlyStopping = new EarlyStopping(earlyStoppingPatience);

        // Periodic saves so a crash/interrupt partway through doesn't lose completed
        // work — add-tag runs are usually quick, but not always fast enough that this
        // doesn't matter.
        void SaveProgress()
        {
            var updatedRows = tagTower.ExtractRows();
            for (var i = 0; i < updatedRows.Length; i++)
                embeddings.SetRow(i, updatedRows[i]);
            Checkpoint.Save(checkpointDir, imageTower, config, vocabulary, embeddings, logitScaleBias);
        }

        var result = 0;

        AnsiConsole.Progress()
            .Columns(new TaskDescriptionColumn(), new ProgressBarColumn(), new PercentageColumn(), new RemainingTimeColumn())
            .Start(ctx =>
            {
                var task = ctx.AddTask($"Fine-tuning '{tag}'", maxValue: steps);

                for (var step = 0; step < steps; step++)
                {
                    // Same reasoning as TrainCommandHandler's stepScope: every tensor a
                    // step allocates (forward pass, pooled embedding, tag embeddings,
                    // loss, the validation forward pass below) is native/CUDA memory the
                    // .NET GC does not reclaim promptly. Without this, `steps` iterations
                    // leak without bound until the process (or the whole machine, on CPU
                    // training) runs out of memory.
                    using var stepScope = NewDisposeScope();

                    var tagEmbeddings = tagTower.forward(allTagIndices);
                    var loss = SigmoidContrastiveLoss.Compute(trainingPooled, tagEmbeddings, trainingLabels, logitScaleBias);

                    optimizer.zero_grad();
                    loss.backward();
                    optimizer.step();

                    var lossValue = loss.item<float>();
                    if (float.IsNaN(lossValue))
                    {
                        AnsiConsole.MarkupLineInterpolated($"[red]step {step + 1}/{steps}: {NaNGuard.Message}[/]");
                        result = 1;
                        return;
                    }

                    task.Increment(1);
                    // G4 (not F4): a diverging-but-not-yet-NaN loss can be enormous, and
                    // F4's full decimal expansion of a huge float is long enough to wrap
                    // the terminal line and corrupt Spectre's live-region redraw.
                    task.Description = $"Fine-tuning '{tag}' step {step + 1}/{steps} loss {lossValue:G4}";

                    if (step % 20 == 19)
                        SaveProgress();

                    if (!useEarlyStopping)
                        continue;

                    using var _ = no_grad();
                    var valTagEmbeddings = tagTower.forward(allTagIndices);
                    var validationLoss = SigmoidContrastiveLoss.Compute(validationPooled!, valTagEmbeddings, validationLabels!, logitScaleBias).item<float>();

                    if (earlyStopping.ShouldStop(validationLoss))
                    {
                        AnsiConsole.MarkupLineInterpolated($"[yellow]Early stopping: validation loss stopped improving after step {step + 1}.[/]");
                        break;
                    }
                }
            });

        trainingPooled.Dispose();
        validationPooled?.Dispose();

        if (result != 0)
            return result;

        vocabulary.PromoteIfThresholdMet(tag, minImageThreshold);
        SaveProgress();

        return 0;
    }

    /// <summary>Pooled embeddings of <paramref name="ids"/> through the frozen tower, a few images at a time, with no autograd graph.</summary>
    private static Tensor EmbedPooled(
        ImageTower imageTower, Func<IReadOnlyList<int>, (Tensor Pixels, IReadOnlyList<LetterboxBox> Boxes)> load,
        IReadOnlyList<int> ids, int inputSize, Device device, int chunkSize = 16)
    {
        using var _ = no_grad();
        var parts = new List<Tensor>();
        for (var start = 0; start < ids.Count; start += chunkSize)
        {
            using var scope = NewDisposeScope();
            var (pixels, boxes) = load(ids.Skip(start).Take(chunkSize).ToList());
            var (_, spatial) = imageTower.forward(pixels.to(device));
            var mask = SpatialMask.Build(boxes, inputSize, spatial.shape[2], spatial.shape[3], device);
            parts.Add(ImageTower.MaskedPool(spatial, mask).MoveToOuterDisposeScope());
        }

        if (parts.Count == 0)
            return zeros([0, imageTower.EmbeddingDim], device: device);
        var result = cat(parts, 0);
        foreach (var part in parts)
            part.Dispose();
        return result;
    }

    /// <summary>
    /// Corpus images to train the new tag against: half hard negatives (images carrying a
    /// tag that appears on at least 30% of the manifest's images — the base character,
    /// the series — up to the 8 strongest such tags), half random corpus images. The new
    /// tag is brand new to the corpus, so no corpus image carries it; the risk is a corpus
    /// image that SHOULD have carried it, which is why hard negatives are capped at half.
    /// </summary>
    private static List<int> SelectNegatives(
        IReadOnlyList<IReadOnlyList<int>> corpusTagRows, IReadOnlyList<List<int>> manifestTagRows, int newTagRow, int count,
        TagVocabulary vocabulary)
    {
        var random = new Random(newTagRow);
        var sources = manifestTagRows
            .SelectMany(rows => rows.Where(r => r != newTagRow).Distinct())
            .GroupBy(r => r)
            .Where(g => g.Count() >= 0.3 * manifestTagRows.Count)
            .OrderByDescending(g => g.Count())
            .Take(8)
            .Select(g => g.Key)
            .ToHashSet();

        if (sources.Count > 0)
            AnsiConsole.MarkupLineInterpolated($"Hard-negative sources: {string.Join(", ", sources.Select(r => vocabulary.GetByRowIndex(r).Tag))}");

        var hardCandidates = new List<int>();
        if (sources.Count > 0)
            for (var i = 0; i < corpusTagRows.Count; i++)
                if (corpusTagRows[i].Any(sources.Contains))
                    hardCandidates.Add(i);

        var chosen = new HashSet<int>();
        var hardTarget = Math.Min(count / 2, hardCandidates.Count);
        while (chosen.Count < hardTarget)
            chosen.Add(hardCandidates[random.Next(hardCandidates.Count)]);
        for (var attempt = 0; chosen.Count < count && attempt < count * 4; attempt++)
            chosen.Add(random.Next(corpusTagRows.Count));

        return chosen.OrderBy(_ => random.Next()).ToList();
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
