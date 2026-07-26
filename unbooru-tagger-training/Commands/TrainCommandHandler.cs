using Spectre.Console;
using UnbooruTagger.Core.Dataset;
using UnbooruTagger.Core.Embedding;
using UnbooruTagger.Core.Encoding;
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
        int earlyStoppingPatience,
        double localizationWeight = 0.1,
        double localizationTemperature = 0.35,
        double selfSupervisedWeight = 0.1)
    {
        if ((manifestPath is null) == (cacheDir is null))
            throw new ArgumentException("Provide exactly one of --manifest or --cache-dir.");

        using var cacheReader = cacheDir is not null ? new PreprocessedDatasetCacheReader(cacheDir) : null;

        TagVocabulary vocabulary;
        List<List<int>> imageTagRows;
        int datasetCount;
        int resolvedInputSize;
        Func<IReadOnlyList<int>, (Tensor Pixels, IReadOnlyList<LetterboxBox> Boxes)> loadBatch;

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

        var device = DeviceSelector.Best();

        ImageTower imageTower;
        TagTower tagTower;
        var startEpoch = 0;
        var resumedProgress = TrainingProgress.Initial;
        var resumingOptimizerState = false;
        if (Checkpoint.Exists(checkpointDir))
        {
            // Resume instead of starting over: re-running `train` against the same
            // --checkpoint-dir after a crash/kill previously silently discarded every
            // completed epoch's work (SaveCheckpoint wrote it, but nothing ever read it
            // back), which is the whole point of saving a checkpoint every epoch in the
            // first place.
            var (loadedImageTower, config, checkpointVocabulary, embeddings) = Checkpoint.Load(checkpointDir, device);

            // The embedding table's row count is baked into its saved tensor shape, so
            // the vocabulary used from here on must be the exact one that produced it —
            // not a fresh load of the dataset's current vocabulary, which could have
            // grown (e.g. a still-running build-large-cache) since this checkpoint was
            // written.
            if (checkpointVocabulary.Records.Count != vocabulary.Records.Count)
                throw new InvalidOperationException(
                    $"Checkpoint at '{checkpointDir}' has {checkpointVocabulary.Records.Count} tags but the dataset vocabulary has {vocabulary.Records.Count} -- resuming would misalign tag-row indices. Use a fresh --checkpoint-dir, or a dataset that matches the one this checkpoint was trained on.");

            if (config.EmbeddingDim != embeddingDim)
                AnsiConsole.MarkupLineInterpolated($"[yellow]Resuming: checkpoint embedding dim {config.EmbeddingDim} overrides --embedding-dim {embeddingDim}.[/]");

            embeddingDim = config.EmbeddingDim;
            vocabulary = checkpointVocabulary;
            imageTower = loadedImageTower;

            var resumedRows = Enumerable.Range(0, embeddings.RowCount)
                .Select(i => embeddings.GetRow(i).ToArray())
                .ToArray();
            tagTower = TagTower.CreateFullyTrainable(resumedRows, embeddingDim, device);

            // Model weights resume unconditionally above (Checkpoint.Exists), but epoch
            // count / EarlyStopping history / optimizer momentum are only there if this
            // checkpoint was written by a build that already had this feature -- an
            // older checkpoint directory still resumes, just with those three reset,
            // rather than failing to resume at all.
            if (TrainingState.Exists(checkpointDir))
            {
                resumedProgress = TrainingState.LoadProgress(checkpointDir);
                startEpoch = resumedProgress.CompletedEpochs;
                resumingOptimizerState = true;
                AnsiConsole.MarkupLineInterpolated($"Resuming from checkpoint in '{checkpointDir}' (epoch {startEpoch}, optimizer state, early-stopping history).");
            }
            else
            {
                AnsiConsole.MarkupLineInterpolated($"[yellow]Resuming from checkpoint in '{checkpointDir}', but no training_progress.json/optimizer.dat found -- epoch count, optimizer momentum, and early-stopping history all restart from scratch.[/]");
            }
        }
        else
        {
            var initialRows = Enumerable.Range(0, vocabulary.Records.Count)
                .Select(_ => EmbeddingInit.RandomRow(embeddingDim))
                .ToArray();
            imageTower = new ImageTower(embeddingDim, device: device);
            tagTower = TagTower.CreateFullyTrainable(initialRows, embeddingDim, device);
        }

        // Not checkpointed/resumed: it only shapes how the image tower trains (see the
        // self-supervised loss below), never reaches the exported model, and is cheap
        // enough to relearn from scratch within a handful of steps after a resume.
        var predictionHead = new PredictionHead(embeddingDim, Math.Max(embeddingDim / 4, 32), device);

        var optimizer = optim.Adam(
            imageTower.parameters().Concat(tagTower.parameters()).Concat(predictionHead.parameters()).ToArray(), lr: learningRate);
        if (resumingOptimizerState)
            TrainingState.LoadOptimizerState(checkpointDir, optimizer);

        var allTagIndices = tensor(Enumerable.Range(0, vocabulary.Records.Count).Select(i => (long)i).ToArray(), device: device);

        var (trainingIndices, validationIndices) = SplitForValidation(datasetCount, validationFraction);
        var trainingTagRows = trainingIndices.Select(i => imageTagRows[i]).ToList();
        var sampler = new RareTagOversamplingBatchSampler(trainingTagRows, tagFrequencies);
        var earlyStopping = new EarlyStopping(earlyStoppingPatience);
        if (resumingOptimizerState)
            earlyStopping.Restore(resumedProgress.EarlyStoppingBestLoss, resumedProgress.EarlyStoppingEvaluationsSinceImprovement);

        var stepsPerEpoch = Math.Max(1, trainingIndices.Count / batchSize);
        var result = 0;

        if (startEpoch >= epochs)
        {
            AnsiConsole.MarkupLineInterpolated($"[yellow]Checkpoint already completed {startEpoch}/{epochs} epochs -- nothing to do. Pass a larger --epochs to continue training.[/]");
            return result;
        }

        AnsiConsole.Progress()
            .Columns(TrainingProgressColumns.Columns)
            .Start(ctx =>
            {
                var reporter = TrainingProgressColumns.AddTasks(ctx, epochs * stepsPerEpoch, startEpoch * stepsPerEpoch);
                try
                {
                    for (var epoch = startEpoch; epoch < epochs; epoch++)
                    {
                        for (var step = 0; step < stepsPerEpoch; step++)
                        {
                            // Every tensor allocated in a step (pooled/spatial embeddings, the full
                            // tagTower forward pass over the whole vocabulary, each `loss +=`
                            // reassignment's intermediate) is native/CUDA memory that .NET's GC does
                            // not reclaim promptly. Without this scope those accumulate every step
                            // until VRAM is exhausted; the scope frees them all once `lossValue` has
                            // been extracted as a plain float below.
                            using var stepScope = NewDisposeScope();

                            var localBatchIndices = sampler.SampleBatch(batchSize);
                            var globalBatchIndices = localBatchIndices.Select(li => trainingIndices[li]).ToList();

                            var (rawPixelBatch, boxes) = loadBatch(globalBatchIndices);
                            using var pixelBatch = rawPixelBatch.to(device);
                            var (_, spatial) = imageTower.forward(pixelBatch);
                            var spatialMask = SpatialMask.Build(boxes, resolvedInputSize, spatial.shape[2], spatial.shape[3], device);
                            var pooled = ImageTower.MaskedPool(spatial, spatialMask);
                            var tagEmbeddings = tagTower.forward(allTagIndices);

                            var batchTagRows = localBatchIndices.Select(li => trainingTagRows[li]).ToList();
                            using var labels = BatchLabelBuilder.Build(batchTagRows, vocabulary.Records.Count).to(device);
                            var loss = SigmoidContrastiveLoss.Compute(pooled, tagEmbeddings, labels);

                            if (localizationWeight > 0)
                                loss += localizationWeight * SigmoidContrastiveLoss.ComputeLocalized(spatial, tagEmbeddings, labels, (float)localizationTemperature, spatialMask);

                            if (selfSupervisedWeight > 0)
                            {
                                // Two independent crop+flip views of the same already-loaded batch --
                                // no extra image I/O, just extra forward passes through imageTower.
                                using var viewA = RandomCropAugmentation.Apply(pixelBatch);
                                using var viewB = RandomCropAugmentation.Apply(pixelBatch);
                                var (pooledA, _) = imageTower.forward(viewA);
                                var (pooledB, _) = imageTower.forward(viewB);
                                var predictionA = predictionHead.forward(pooledA);
                                var predictionB = predictionHead.forward(pooledB);
                                loss += selfSupervisedWeight * SelfSupervisedConsistencyLoss.Compute(pooledA, predictionA, pooledB, predictionB);
                            }

                            optimizer.zero_grad();
                            loss.backward();
                            optimizer.step();

                            var lossValue = loss.item<float>();
                            if (float.IsNaN(lossValue))
                            {
                                AnsiConsole.MarkupLineInterpolated($"[red]epoch {epoch + 1}/{epochs} step {step + 1}/{stepsPerEpoch}: {NaNGuard.Message}[/]");
                                result = 1;
                                return;
                            }

                            reporter.ReportStepComplete();
                            // G4 (not F4): a diverging-but-not-yet-NaN loss can be enormous, and
                            // F4's full decimal expansion of a huge float is long enough to wrap
                            // the terminal line and corrupt Spectre's live-region redraw. Tracks
                            // a real fraction of *this epoch* (not the whole run), so a run with
                            // a huge total step count doesn't sit looking stuck near 0% for the
                            // entire first epoch -- the Overall row above already covers that.
                            reporter.ReportPhaseProgress($"epoch {epoch + 1}/{epochs} step {step + 1}/{stepsPerEpoch} loss {lossValue:G4}", step + 1, stepsPerEpoch);
                        }

                        // Save after every epoch, not just at the end: a long run (especially on
                        // GPU) can be killed, crash, or lose its SSH session partway through, and
                        // without this, all completed work would be unrecoverable.
                        reporter.ReportPhase($"epoch {epoch + 1}/{epochs}: saving checkpoint...");
                        SaveCheckpoint(checkpointDir, imageTower, tagTower, embeddingDim, resolvedInputSize, vocabulary);

                        var stopEarly = false;
                        if (validationIndices.Count == 0)
                        {
                            AnsiConsole.MarkupLine("Not enough images for a validation split — running the full epoch count without early stopping.");
                        }
                        else
                        {
                            var validationBatchCount = Math.Max(1, (int)Math.Ceiling(validationIndices.Count / (double)batchSize));
                            var validationLoss = Evaluate(
                                imageTower, tagTower, loadBatch, imageTagRows, validationIndices, allTagIndices, vocabulary.Records.Count, device, batchSize, resolvedInputSize,
                                onBatchComplete: batchesDone =>
                                    reporter.ReportPhaseProgress($"epoch {epoch + 1}/{epochs}: validating batch {batchesDone}/{validationBatchCount}", batchesDone, validationBatchCount));
                            AnsiConsole.MarkupLineInterpolated($"epoch {epoch + 1}/{epochs} validation loss {validationLoss:G4}");

                            if (earlyStopping.ShouldStop(validationLoss))
                            {
                                AnsiConsole.MarkupLineInterpolated($"[yellow]Early stopping: validation loss stopped improving after epoch {epoch + 1}.[/]");
                                stopEarly = true;
                            }
                        }

                        // Saved once per epoch regardless of which branch ran above, so a
                        // resumed run always picks up this epoch's completed count and
                        // EarlyStopping's latest history (unchanged if there was no
                        // validation split to evaluate against).
                        reporter.ReportPhase($"epoch {epoch + 1}/{epochs}: saving training state...");
                        TrainingState.Save(checkpointDir, new TrainingProgress(epoch + 1, earlyStopping.BestLoss, earlyStopping.EvaluationsSinceImprovement), optimizer);

                        if (stopEarly)
                            break;
                    }

                    reporter.StopPhase();
                }
                finally
                {
                    reporter.Dispose();
                }
            });

        return result;
    }

    private static void SaveCheckpoint(string checkpointDir, ImageTower imageTower, TagTower tagTower, int embeddingDim, int inputSize, TagVocabulary vocabulary)
    {
        var embeddings = TagEmbeddingStore.CreateEmpty(embeddingDim);
        foreach (var row in tagTower.ExtractRows())
            embeddings.AppendRow(row);

        var config = ModelConfig.Default(embeddingDim, inputSize);
        Checkpoint.Save(checkpointDir, imageTower, config, vocabulary, embeddings);
    }

    private static double Evaluate(
        ImageTower imageTower,
        TagTower tagTower,
        Func<IReadOnlyList<int>, (Tensor Pixels, IReadOnlyList<LetterboxBox> Boxes)> loadBatch,
        IReadOnlyList<IReadOnlyList<int>> imageTagRows,
        IReadOnlyList<int> validationIndices,
        Tensor allTagIndices,
        int vocabularySize,
        Device device,
        int batchSize,
        int inputSize,
        Action<int>? onBatchComplete = null)
    {
        using var _ = no_grad();

        using var tagEmbeddings = tagTower.forward(allTagIndices);

        double totalLoss = 0;
        var totalCount = 0;
        var batchesDone = 0;
        for (var offset = 0; offset < validationIndices.Count; offset += batchSize)
        {
            var chunk = validationIndices.Skip(offset).Take(batchSize).ToList();

            var (rawPixelBatch, boxes) = loadBatch(chunk);
            using var pixelBatch = rawPixelBatch.to(device);
            var (_, spatial) = imageTower.forward(pixelBatch);
            using var _spatial = spatial;
            using var spatialMask = SpatialMask.Build(boxes, inputSize, spatial.shape[2], spatial.shape[3], device);
            using var pooled = ImageTower.MaskedPool(spatial, spatialMask);

            var batchTagRows = chunk.Select(i => imageTagRows[i]).ToList();
            using var labels = BatchLabelBuilder.Build(batchTagRows, vocabularySize).to(device);
            using var loss = SigmoidContrastiveLoss.Compute(pooled, tagEmbeddings, labels);

            totalLoss += loss.item<float>() * chunk.Count;
            totalCount += chunk.Count;

            batchesDone++;
            onBatchComplete?.Invoke(batchesDone);
        }

        return totalLoss / totalCount;
    }

    private static (Tensor Pixels, IReadOnlyList<LetterboxBox> Boxes) LoadCacheBatch(PreprocessedDatasetCacheReader cache, IReadOnlyList<int> indices, int inputSize)
    {
        var imageSize = 3 * inputSize * inputSize;
        var flat = new float[indices.Count * imageSize];
        var boxes = new LetterboxBox[indices.Count];
        for (var i = 0; i < indices.Count; i++)
        {
            var image = cache.ReadImage(indices[i]);
            image.Pixels.CopyTo(flat, i * imageSize);
            boxes[i] = image.Content;
        }

        return (tensor(flat, [indices.Count, 3, inputSize, inputSize]), boxes);
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
