using Spectre.Console;
using UnbooruTagger.Core.Dataset;
using UnbooruTagger.Core.Embedding;
using UnbooruTagger.Core.Encoding;
using UnbooruTagger.Core.Scoring;
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
/// Holds out a validation split and stops early once held-out tag ranking quality (mean
/// per-tag AP, see <see cref="TagRankingEvaluator"/>) stops improving, instead of always
/// running the full <paramref name="epochs"/> count.
///
/// By default batches and targets are built deliberately from the corpus's per-tag
/// structure (<see cref="TrainingTargetOptions"/>): each batch is anchored on a few tags'
/// positives and hard negatives (<see cref="TagCentricBatchSampler"/>), and each (image, tag)
/// pair is weighted by how certain the corpus is about it (<see cref="PairWeightBuilder"/>).
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
        double selfSupervisedWeight = 0.1,
        double checkpointIntervalMinutes = 15.0,
        int gradientAccumulationSteps = 1,
        string? warmStartModelPath = null,
        string? warmStartVocabPath = null,
        int localizationTagChunkSize = 4096,
        TrainingTargetOptions? targetOptions = null,
        int ioParallelism = 32,
        int prefetchWindowBatches = 8,
        TowerOptions? tower = null)
    {
        tower ??= TowerOptions.Original;
        targetOptions ??= TrainingTargetOptions.Default;

        if ((manifestPath is null) == (cacheDir is null))
            throw new ArgumentException("Provide exactly one of --manifest or --cache-dir.");

        if (gradientAccumulationSteps < 1)
            throw new ArgumentException("--gradient-accumulation-steps must be at least 1.");

        if (localizationTagChunkSize < 1)
            throw new ArgumentException("--localization-tag-chunk-size must be at least 1.");

        using var cacheReader = cacheDir is not null ? new PreprocessedDatasetCacheReader(cacheDir) : null;
        ContentRating[]? ratings = null;
        int[]? ratingRows = null;

        TagVocabulary vocabulary;
        List<List<int>> imageTagRows;
        int datasetCount;
        int resolvedInputSize;
        Func<IReadOnlyList<int>, (Tensor Pixels, IReadOnlyList<LetterboxBox> Boxes)> loadBatch;
        // Cache-backed runs load through BatchPrefetcher (background, offset-sorted, many
        // reads in flight); manifest runs keep the simple synchronous loadBatch path.
        Func<IReadOnlyList<int>, EncodedImage[]>? readWindow = null;

        if (cacheReader is not null)
        {
            vocabulary = TagVocabulary.Load(Path.Combine(cacheDir!, "tag_vocabulary.json"));
            imageTagRows = cacheReader.ImageTagRows.Select(rows => rows.ToList()).ToList();
            datasetCount = cacheReader.ImageCount;

            // Fill in tags Danbooru's implication rules guarantee (TagImplications); Gelbooru
            // posts don't get them applied at the source.
            var implications = TagImplications.Load(Path.Combine(cacheDir!, TagImplications.FileName), vocabulary);
            if (implications.RuleCount > 0)
            {
                imageTagRows = implications.Complete(imageTagRows, out var addedLabels, out var changedImages);
                AnsiConsole.MarkupLineInterpolated(
                    $"Tag implications: {implications.RuleCount:N0} rules added {addedLabels:N0} missing labels to {changedImages:N0} images.");
            }

            // Content ratings (ratings.bin, from the crawler's export-ratings) become three
            // cumulative labels appended to the vocabulary and each image's tags -- see RatingLabels.
            ratings = targetOptions.Ratings ? ImageRatings.Load(cacheDir!, cacheReader.ImageCount) : null;
            if (ratings is not null)
            {
                ratingRows = RatingLabels.EnsureRows(vocabulary);
                imageTagRows = RatingLabels.Apply(imageTagRows, ratings, ratingRows);
            }
            resolvedInputSize = cacheReader.InputSize;
            loadBatch = indices => LoadCacheBatch(cacheReader, indices, resolvedInputSize);
            readWindow = indices => cacheReader.ReadEncoded(indices, ioParallelism);
        }
        else
        {
            var manifest = DatasetManifest.Load(manifestPath!);
            (vocabulary, imageTagRows) = BuildVocabulary(manifest);
            datasetCount = manifest.Entries.Count;
            resolvedInputSize = inputSize;
            loadBatch = indices => ImageBatchLoader.Load(indices.Select(i => manifest.Entries[i].ImagePath).ToList(), resolvedInputSize);
        }

        // Saved into the checkpoint so export-onnx can ship it with the model bundle and
        // inference suppresses exactly the contradictions training was taught.
        var exclusiveGroups = ExclusiveTagGroups.Load(
            targetOptions.ExclusiveTagsPath ?? Path.Combine(checkpointDir, ExclusiveTagGroups.FileName), vocabulary, fallBackToDefaults: true);

        var device = DeviceSelector.Best();

        ImageTower imageTower;
        TagTower tagTower;
        LogitScaleBias logitScaleBias;
        var startEpoch = 0;
        var resumedProgress = TrainingProgress.Initial;
        var resumingOptimizerState = false;
        var resumingLogitScaleBias = false;
        if (Checkpoint.Exists(checkpointDir))
        {
            // Resume instead of starting over: re-running `train` against the same
            // --checkpoint-dir after a crash/kill previously silently discarded every
            // completed epoch's work (SaveCheckpoint wrote it, but nothing ever read it
            // back), which is the whole point of saving a checkpoint every epoch in the
            // first place.
            var (loadedImageTower, config, checkpointVocabulary, embeddings, loadedLogitScaleBias) = Checkpoint.Load(checkpointDir, device);
            logitScaleBias = loadedLogitScaleBias;
            resumingLogitScaleBias = Checkpoint.HasLogitScaleBias(checkpointDir);
            if (!resumingLogitScaleBias)
                AnsiConsole.MarkupLine("[yellow]Resuming: checkpoint predates the learned logit scale/bias — starting it fresh (Scale=10, Bias=-10) instead of continuing a previously-learned value.[/]");

            // The embedding table's row count is baked into its saved tensor shape, so
            // the vocabulary used from here on must be the exact one that produced it —
            // not a fresh load of the dataset's current vocabulary, which could have
            // grown (e.g. a still-running build-large-cache) since this checkpoint was
            // written.
            if (ratingRows is not null && RatingLabels.FindRows(checkpointVocabulary) is null)
                throw new InvalidOperationException(
                    $"Checkpoint at '{checkpointDir}' predates content-rating labels (the dataset now has ratings.bin). Train into a fresh --checkpoint-dir to learn ratings, or pass --no-ratings to keep resuming this one without them.");

            // Rows are matched to the checkpoint's vocabulary by tag name (VocabularyRemap),
            // so a compacted vocabulary, or rows added by add-tag, line up without any
            // assumption that dataset row N is checkpoint row N.
            var (remappedRows, unmatched) = VocabularyRemap.Apply(vocabulary, imageTagRows, checkpointVocabulary);
            imageTagRows = remappedRows;
            if (unmatched > 0)
                AnsiConsole.MarkupLineInterpolated($"Resuming: {unmatched} dataset tags aren't in the checkpoint's vocabulary (e.g. zero-image rows it dropped) and are ignored.");

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
            // Fresh run: drop vocabulary rows no image carries (stale crawl rows), so the tag
            // table and every per-step [batch, vocabulary] tensor only cover learnable tags.
            if (targetOptions.CompactVocabulary)
            {
                var compact = VocabularyRemap.Compact(vocabulary, imageTagRows);
                AnsiConsole.MarkupLineInterpolated($"Compacting vocabulary: {vocabulary.Records.Count} -> {compact.Records.Count} tags (dropping rows no image carries).");
                (imageTagRows, _) = VocabularyRemap.Apply(vocabulary, imageTagRows, compact);
                vocabulary = compact;
            }

            using var warmStartEmbedder = TagWarmStartEmbedderFactory.Create(warmStartModelPath, warmStartVocabPath, embeddingDim);
            var initialRows = vocabulary.Records
                .Select(record => warmStartEmbedder?.Embed(record.Tag) ?? EmbeddingInit.RandomRow(embeddingDim))
                .ToArray();
            imageTower = new ImageTower(
                embeddingDim, tower.StemChannels, tower.StageChannels, tower.BlocksPerStage, device, tower.FusedStages);
            tagTower = TagTower.CreateFullyTrainable(initialRows, embeddingDim, device);
            logitScaleBias = new LogitScaleBias(device: device);
        }

        // Not a weight change, so it applies on resume too; saved in model_config.json.
        if (targetOptions.LocalizationFromFusedOnly is { } fusedOnly && fusedOnly != imageTower.LocalizationFromFusedOnly)
        {
            imageTower.LocalizationFromFusedOnly = fusedOnly;
            AnsiConsole.MarkupLineInterpolated($"Localization map: {(fusedOnly ? "fused laterals only" : "full spatial map")}.");
        }
        if (targetOptions.LocalizationSkippedLaterals is { } skipped && skipped != imageTower.LocalizationSkippedLaterals)
        {
            imageTower.LocalizationSkippedLaterals = skipped;
            AnsiConsole.MarkupLineInterpolated($"Localization map: leaving out the {skipped} coarsest fused lateral(s); {imageTower.FusedStages - skipped} kept.");
        }

        exclusiveGroups = ExclusiveTagGroups.Parse(exclusiveGroups.SourceText, vocabulary);
        if (ratingRows is not null)
            ratingRows = RatingLabels.FindRows(vocabulary);
        var tagFrequencies = ComputeTagFrequencies(imageTagRows);
        Directory.CreateDirectory(checkpointDir);
        exclusiveGroups.Save(Path.Combine(checkpointDir, ExclusiveTagGroups.FileName));

        // Not checkpointed/resumed: it only shapes how the image tower trains (see the
        // self-supervised loss below), never reaches the exported model, and is cheap
        // enough to relearn from scratch within a handful of steps after a resume.
        var predictionHead = new PredictionHead(embeddingDim, Math.Max(embeddingDim / 4, 32), device);

        var optimizer = optim.Adam(
            imageTower.parameters().Concat(tagTower.parameters()).Concat(predictionHead.parameters()).ToArray(), lr: learningRate);
        if (resumingOptimizerState)
            TrainingState.LoadOptimizerState(checkpointDir, optimizer);

        // Deliberately a separate optimizer from the one above, not folded into the same
        // parameter list — see TrainingState.SaveLogitScaleBiasOptimizerState's doc comment
        // for why that would corrupt resuming an existing checkpoint's momentum.
        var logitScaleBiasOptimizer = optim.Adam(logitScaleBias.parameters().ToArray(), lr: learningRate);
        if (resumingOptimizerState && resumingLogitScaleBias && TrainingState.HasLogitScaleBiasOptimizerState(checkpointDir))
            TrainingState.LoadLogitScaleBiasOptimizerState(checkpointDir, logitScaleBiasOptimizer);

        var allTagIndices = tensor(Enumerable.Range(0, vocabulary.Records.Count).Select(i => (long)i).ToArray(), device: device);

        // Persisted across resumes (TrainingProgress.ValidationSplitSeed) so a mid-epoch
        // resume reconstructs the exact same split instead of re-shuffling -- otherwise
        // a resumed run could train on images the original run held out for validation.
        // Only a genuinely fresh run (or an old checkpoint written before this field
        // existed) picks a new random seed.
        var validationSplitSeed = resumingOptimizerState ? resumedProgress.ValidationSplitSeed : Random.Shared.Next();
        var (trainingIndices, validationIndices) = SplitForValidation(datasetCount, validationFraction, validationSplitSeed);
        var trainingTagRows = trainingIndices.Select(i => imageTagRows[i]).ToList();
        var sampler = new RareTagOversamplingBatchSampler(trainingTagRows, tagFrequencies);

        AnsiConsole.MarkupLine("Building per-tag statistics (positives, implied tags, context and hard-negative sources)...");
        // Site coverage (cross-site agreement + image_sites.bin): a tag only one site's tag
        // system has is no evidence on an image from the other site alone (TagStatistics.IsSiteBlind).
        var tagSites = cacheDir is not null ? TagAgreement.LoadSiteCoverage(Path.Combine(cacheDir, TagAgreement.FileName), vocabulary) : null;
        var imageSites = cacheDir is not null && tagSites is not null ? ImageSites.Load(cacheDir, datasetCount) : null;
        if (tagSites is not null && imageSites is not null)
            AnsiConsole.MarkupLineInterpolated(
                $"Site coverage: {tagSites.Count(s => s != (BooruSites.Danbooru | BooruSites.Gelbooru)):N0} tags are used by only one site; their absence on images from only the other site is ignored.");
        else
            (tagSites, imageSites) = (null, null);
        var statistics = TagStatistics.Build(
            trainingTagRows, vocabulary.Records.Count, impliedThreshold: targetOptions.ImpliedThreshold, seed: validationSplitSeed,
            missingThreshold: targetOptions.MissingThreshold,
            imageSites: imageSites is not null ? trainingIndices.Select(i => imageSites[i]).ToArray() : null, tagSites: tagSites);
        var anyAnchorTag = statistics.Positives.Any(p => p.Length > 0);
        if (targetOptions.TagCentricSampling && !anyAnchorTag)
            AnsiConsole.MarkupLine("[yellow]No training image carries any tag -- falling back to rare-tag sampling.[/]");
        var tagCentricSampler = targetOptions.TagCentricSampling && anyAnchorTag
            ? new TagCentricBatchSampler(statistics, targetOptions.PositivesPerAnchor, targetOptions.HardNegativesPerAnchor,
                contextNegativesPerAnchor: targetOptions.ContextNegativesPerAnchor)
            : null;
        AnsiConsole.MarkupLineInterpolated(
            $"{statistics.ContextSources.Count(c => c.Length > 0)} tags have context sources (P(tag|source) in [[{targetOptions.ImpliedThreshold}, {targetOptions.MissingThreshold})); {targetOptions.ContextNegativesPerAnchor} context negative(s) per anchor, trained {(targetOptions.ContextNegativeWeight > 0 ? $"at full strength, weight {targetOptions.ContextNegativeWeight}, for localizable tags" : "as unverified negatives")}.");
        AnsiConsole.MarkupLineInterpolated(
            $"{statistics.Positives.Count(p => p.Length > 0)} of {vocabulary.Records.Count} tags have training images; {(tagCentricSampler is not null ? $"{tagCentricSampler.AnchorTagCount} anchor tags" : "rare-tag image sampling")}.");

        // Tags with no region to point at (artists, series, meta, ratings, whole-image general
        // tags like solo/upper_body) are never given to the per-location loss -- it could only
        // teach their heatmaps noise.
        // Rules from --localizable-tags, else the checkpoint's saved copy, else the defaults
        // (TagLocalizability); saved into the checkpoint so export ships them to detect.
        var localizability = TagLocalizability.Load(targetOptions.LocalizableTagsPath ?? Path.Combine(checkpointDir, TagLocalizability.FileName));
        localizability.Save(Path.Combine(checkpointDir, TagLocalizability.FileName));
        var localizable = vocabulary.Records.Select(r => localizability.IsLocalizable(r.Tag)).ToArray();
        var pairWeightBuilder = new PairWeightBuilder(statistics, exclusiveGroups, targetOptions.CertainNegativeWeight,
            exhaustiveRows: ratings is not null ? ratingRows : null, contextNegativeWeight: targetOptions.ContextNegativeWeight,
            contextNegativeTags: localizable);

        // Crop consistency trains only tags the classifier is already good at (CropTeacherGate),
        // starting from the last epoch's evaluation and refreshed after every epoch.
        var cropGate = targetOptions.CropConsistencyWeight > 0
            ? CropTeacherGate.Load(Path.Combine(checkpointDir, EvaluationReportFileName), vocabulary, localizable, targetOptions.CropTeacherMinAveragePrecision, targetOptions.CropTeacherMinPositives)
            : CropTeacherGate.None(vocabulary.Records.Count);
        var cropTrusted = tensor(cropGate.Trusted.Select(t => t ? 1f : 0f).ToArray(), device: device);
        if (targetOptions.CropConsistencyWeight > 0)
            AnsiConsole.MarkupLineInterpolated(
                $"Crop consistency: {cropGate.TrustedCount} localizable tags trusted as teachers (held-out AP >= {targetOptions.CropTeacherMinAveragePrecision}, >= {targetOptions.CropTeacherMinPositives} positives){(cropGate.TrustedCount == 0 ? "; off until an epoch evaluation trusts some" : "")}; re-checked after every epoch.");
        if (targetOptions.MosaicsPerStep > 0)
        {
            if (!targetOptions.DeliberateTargets || localizationWeight <= 0)
                throw new ArgumentException("--mosaics-per-step trains the localization loss on deliberate targets; it needs --targets deliberate and --localization-weight > 0.");
            if (batchSize < Mosaic.TilesPerMosaic)
                throw new ArgumentException($"--mosaics-per-step needs at least {Mosaic.TilesPerMosaic} images per batch (--batch-size {batchSize}).");
            AnsiConsole.MarkupLineInterpolated(
                $"Mosaic: {Math.Min(targetOptions.MosaicsPerStep, batchSize / Mosaic.TilesPerMosaic)} canvas(es) of four half-size batch images per step, weight {targetOptions.MosaicWeight} x the localization weight.");
        }

        var earlyStopping = new EarlyStopping(earlyStoppingPatience);
        // Only resume early-stopping history recorded against the same metric -- a best
        // value from the old mean-validation-loss metric (~0.005) would read as an
        // unbeatable 1-mAP and stop the run after `patience` epochs regardless of progress.
        if (resumingOptimizerState && resumedProgress.EarlyStoppingMetric == EarlyStoppingMetricName)
            earlyStopping.Restore(resumedProgress.EarlyStoppingBestLoss, resumedProgress.EarlyStoppingEvaluationsSinceImprovement);
        else if (resumingOptimizerState)
            AnsiConsole.MarkupLine("[yellow]Resuming: early-stopping history was recorded against a different metric -- starting it fresh against held-out mAP.[/]");

        var stepsPerEpoch = Math.Max(1, trainingIndices.Count / batchSize);
        var result = 0;

        if (startEpoch >= epochs)
        {
            AnsiConsole.MarkupLineInterpolated($"[yellow]Checkpoint already completed {startEpoch}/{epochs} epochs -- nothing to do. Pass a larger --epochs to continue training.[/]");
            return result;
        }

        // Steps already done in the epoch we're about to resume into (0 for a fresh run,
        // or for a resume that landed exactly on an epoch boundary). See
        // ResumeStepResolver's own doc comment for the silent-skip bug this avoids.
        var resumeStep = ResumeStepResolver.Resolve(resumingOptimizerState, resumedProgress, stepsPerEpoch);
        var resumedStepsInEpoch = resumeStep.StepsInEpoch;
        if (resumeStep.StepLayoutChanged)
        {
            AnsiConsole.MarkupLineInterpolated(
                $"[yellow]Resuming: this run's step layout ({stepsPerEpoch} steps/epoch) doesn't match what was saved ({resumedProgress.StepsPerEpoch} steps/epoch, or this checkpoint predates tracking it) -- likely --batch-size or the dataset changed across the resume. Restarting progress within epoch {startEpoch + 1} from step 0 instead of a no-longer-comparable step count.[/]");
        }
        var checkpointInterval = checkpointIntervalMinutes > 0 ? TimeSpan.FromMinutes(checkpointIntervalMinutes) : TimeSpan.MaxValue;
        var sinceLastCheckpoint = System.Diagnostics.Stopwatch.StartNew();

        IEnumerable<BatchRequest> TrainingRequests()
        {
            while (true)
            {
                var sampledBatch = tagCentricSampler?.SampleBatch(batchSize);
                var local = sampledBatch is not null ? sampledBatch.Select(x => x.Image).ToArray() : sampler.SampleBatch(batchSize);
                yield return new BatchRequest(local.Select(li => trainingIndices[li]).ToArray(), sampledBatch);
            }
        }

        // The samplers are only ever touched from the prefetcher's thread from here on.
        using var prefetcher = readWindow is not null
            ? new BatchPrefetcher(TrainingRequests(), readWindow, resolvedInputSize, prefetchWindowBatches, 2 * prefetchWindowBatches)
            : null;

        // Global step the run resumes at; the loss history and snapshot evaluations are cut back to it.
        var resumedGlobalStep = (long)startEpoch * stepsPerEpoch + resumedStepsInEpoch;
        var snapshotEvery = targetOptions.EvaluationCommand is not null && validationIndices.Count > 0 ? Math.Max(0, targetOptions.SnapshotEvaluationSteps) : 0;
        if (targetOptions.SnapshotEvaluationSteps > 0 && snapshotEvery == 0)
            AnsiConsole.MarkupLine("[yellow]--snapshot-evaluation-steps needs --evaluation-command and a validation split: no snapshot evaluations.[/]");
        var dashboard = new TrainingDashboard(new TrainingRunInfo(
            $"training · {Path.GetFileName(Path.TrimEndingDirectorySeparator(Path.GetFullPath(checkpointDir)))} · {imageTower.StageChannels[^1]}-channel tower · batch {batchSize}×{gradientAccumulationSteps}",
            epochs, stepsPerEpoch, startEpoch, resumedStepsInEpoch, batchSize, checkpointInterval,
            EvaluationsOffloaded: targetOptions.EvaluationCommand is not null && validationIndices.Count > 0,
            SnapshotEvaluationSteps: snapshotEvery),
            lossHistory: resumingOptimizerState ? LossHistory.Load(checkpointDir, resumedGlobalStep) : null);
        // Every finished evaluation so far, for the trend charts: epoch ends (history/epoch-N), then snapshots.
        for (var e = 1; e <= startEpoch; e++)
            if (EvaluationPoint.Read(Path.Combine(checkpointDir, "history", $"epoch-{e}"), (long)e * stepsPerEpoch, endOfEpoch: true) is { } epochPoint)
                dashboard.AddEvaluationPoint(epochPoint);
        using var snapshots = snapshotEvery > 0 ? new SnapshotEvaluations(checkpointDir, targetOptions.EvaluationCommand!) : null;
        if (snapshots is not null)
        {
            var (finished, relaunched) = snapshots.Resume(resumedGlobalStep);
            foreach (var point in finished.Select(f => f.Point).OfType<EvaluationPoint>())
                dashboard.AddEvaluationPoint(point);
            foreach (var step in relaunched)
                dashboard.SnapshotEvaluationStarted(step);
            AnsiConsole.MarkupLineInterpolated($"Snapshot evaluations every {snapshotEvery:N0} steps: {finished.Count(f => f.Point is not null)} finished so far{(relaunched.Count > 0 ? $", {relaunched.Count} unfinished restarted" : "")}.");
        }
        // Interactive terminals get the live panel; redirected output (a log file, tests)
        // gets a one-line summary every few minutes instead of a panel per refresh.
        var gold = targetOptions.GoldPath is not null ? LocalizationEvaluator.LoadGold(targetOptions.GoldPath) : null;
        if (gold is not null)
            AnsiConsole.MarkupLineInterpolated($"Gold set: {gold.Count} images, scored at every epoch evaluation.");
        // --evaluation-command: epoch evaluations run in another process (e.g. `evaluate` on a
        // second GPU) while training continues; their results are applied as they come back.
        using var offloaded = targetOptions.EvaluationCommand is not null && validationIndices.Count > 0
            ? new OffloadedEvaluations(checkpointDir, targetOptions.EvaluationCommand)
            : null;
        if (offloaded is not null)
            foreach (var pending in offloaded.ResumePending(startEpoch))
            {
                AnsiConsole.MarkupLineInterpolated($"Re-running the offloaded evaluation of epoch {pending}: it never finished before the last stop.");
                dashboard.EvaluationOffloaded(pending);
            }

        // Applies offloaded evaluations that have come back, in epoch order. Returns the epochs
        // applied (to mark once the state reflecting them is saved) and whether early stopping fired.
        (List<int> Applied, bool StopEarly) ApplyOffloaded()
        {
            var applied = new List<int>();
            var stop = false;
            foreach (var result in offloaded!.TakeFinished())
            {
                applied.Add(result.Epoch);
                if (result.Report is null)
                {
                    dashboard.OffloadedEvaluationFailed(result.Epoch);
                    dashboard.Log("[red]" + Markup.Escape($"offloaded evaluation of epoch {result.Epoch} failed ({result.Failure}; see {Path.Combine(result.Directory, "evaluation.log")}): it doesn't count toward early stopping") + "[/]");
                    continue;
                }
                // The top-level copies are what export-onnx and CropTeacherGate.Load read.
                foreach (var name in new[] { EvaluationReportFileName, RatingReportFileName, EvaluateCommandHandler.LocalizationReportFileName, TagThresholds.FileName })
                    if (File.Exists(Path.Combine(result.Directory, name)))
                        File.Copy(Path.Combine(result.Directory, name), Path.Combine(checkpointDir, name), overwrite: true);
                if (targetOptions.CropConsistencyWeight > 0)
                {
                    cropGate = CropTeacherGate.From(result.Report, vocabulary, localizable, targetOptions.CropTeacherMinAveragePrecision, targetOptions.CropTeacherMinPositives);
                    cropTrusted.Dispose();
                    cropTrusted = tensor(cropGate.Trusted.Select(t => t ? 1f : 0f).ToArray(), device: device);
                    dashboard.Log(Markup.Escape($"crop consistency: {cropGate.TrustedCount} tags now trusted as teachers (held-out AP >= {targetOptions.CropTeacherMinAveragePrecision})"));
                }
                dashboard.EvaluationFinished(result.Epoch - 1, result.Report, result.Ratings);
                if (EvaluationPoint.Read(result.Directory, (long)result.Epoch * stepsPerEpoch, endOfEpoch: true) is { } epochPoint)
                    dashboard.AddEvaluationPoint(epochPoint);
                dashboard.Log(Markup.Escape($"epoch {result.Epoch}/{epochs} (offloaded evaluation)"));
                dashboard.Log(DescribeReport(result.Report));
                if (result.Ratings is not null)
                    dashboard.Log(DescribeRatings(result.Ratings));
                if (!stop && earlyStopping.ShouldStop(result.Report.EarlyStoppingValue))
                {
                    dashboard.Log("[yellow]" + Markup.Escape($"Early stopping: held-out mAP stopped improving after epoch {result.Epoch}. Its weights are kept in {offloaded.SnapshotDirectory(result.Epoch - earlyStopping.EvaluationsSinceImprovement)}.") + "[/]");
                    stop = true;
                }
            }
            return (applied, stop);
        }

        // A model-only copy for an evaluation in another process (epoch or step snapshot).
        void WriteSnapshot(string snapshot, int completedEpochs, int stepsInEpoch)
        {
            SaveCheckpoint(snapshot, imageTower, tagTower, logitScaleBias, embeddingDim, resolvedInputSize, vocabulary);
            foreach (var name in new[] { ExclusiveTagGroups.FileName, TagLocalizability.FileName })
                if (File.Exists(Path.Combine(checkpointDir, name)))
                    File.Copy(Path.Combine(checkpointDir, name), Path.Combine(snapshot, name), overwrite: true);
            // evaluate reconstructs the held-out split from this.
            TrainingState.SaveProgress(snapshot, new TrainingProgress(completedEpochs, earlyStopping.BestLoss, earlyStopping.EvaluationsSinceImprovement, stepsInEpoch, validationSplitSeed, stepsPerEpoch, EarlyStoppingMetricName));
        }

        // Snapshot evaluations that came back: onto the trend charts, or logged if they failed.
        void TakeFinishedSnapshots()
        {
            foreach (var finished in snapshots!.TakeFinished())
                if (finished.Point is { } point)
                    dashboard.AddEvaluationPoint(point);
                else
                {
                    dashboard.SnapshotEvaluationFailed(finished.Step);
                    dashboard.Log("[red]" + Markup.Escape($"snapshot evaluation at step {finished.Step:N0} failed ({finished.Failure}; see {Path.Combine(finished.Directory, "evaluation.log")})") + "[/]");
                }
        }

        var stepsThisProcess = 0;
        var cropRandom = new Random();
        var mosaicRandom = new Random();
        long croppedImages = 0;
        void TrainingLoop()
        {
            for (var epoch = startEpoch; epoch < epochs; epoch++)
            {
                var stepStart = epoch == startEpoch ? resumedStepsInEpoch : 0;
                dashboard.EpochStarted(epoch, stepStart);
                var epochWaitStart = prefetcher?.ConsumerWait ?? TimeSpan.Zero;
                var epochClock = System.Diagnostics.Stopwatch.StartNew();
                for (var step = stepStart; step < stepsPerEpoch; step++)
                {
                    // Every tensor allocated in a step (pooled/spatial embeddings, the full
                    // tagTower forward pass over the whole vocabulary, each `loss +=`
                    // reassignment's intermediate) is native/CUDA memory that .NET's GC does
                    // not reclaim promptly. Without this scope those accumulate every step
                    // until VRAM is exhausted; the scope frees them all once `lossValue` has
                    // been extracted as a plain float below.
                    using var stepScope = NewDisposeScope();

                    SampledImage[]? sampled;
                    IReadOnlyList<int> globalBatchIndices;
                    Tensor rawPixelBatch;
                    IReadOnlyList<LetterboxBox> boxes;
                    if (prefetcher is not null)
                    {
                        if (!prefetcher.TryTake(out var loaded))
                            throw new InvalidOperationException("The training batch stream ended unexpectedly.");
                        sampled = loaded.Request.Sampled;
                        globalBatchIndices = loaded.Request.Images;
                        rawPixelBatch = tensor(loaded.Pixels, [loaded.Boxes.Length, 3, resolvedInputSize, resolvedInputSize]);
                        boxes = loaded.Boxes;
                    }
                    else
                    {
                        sampled = tagCentricSampler?.SampleBatch(batchSize);
                        var localBatchIndices = sampled is not null ? sampled.Select(x => x.Image).ToArray() : sampler.SampleBatch(batchSize);
                        globalBatchIndices = localBatchIndices.Select(li => trainingIndices[li]).ToList();
                        (rawPixelBatch, boxes) = loadBatch(globalBatchIndices);
                    }

                    using var pixelBatch = rawPixelBatch.to(device);

                    // Crop consistency's extra forward runs first, without gradients, keeping only
                    // the crops' pooled embeddings ([B, D]). Run after the main forward, its
                    // activations sat on top of the ones held for backward() and ran out of memory.
                    // Only images carrying a trusted tag are cropped (the rest would train
                    // nothing), so the pass costs in proportion to how much the teacher is trusted.
                    CellWindow[]? cropWindows = null;
                    Tensor? cropPooled = null;
                    var cropGrid = resolvedInputSize / imageTower.OutputStride;
                    var cropImages = targetOptions.CropConsistencyWeight > 0 && cropGate.TrustedCount > 0 && localizationWeight > 0 && targetOptions.DeliberateTargets
                        ? globalBatchIndices.Select((g, b) => (g, b)).Where(x => imageTagRows[x.g].Any(r => (uint)r < (uint)cropGate.Trusted.Length && cropGate.Trusted[r])).Select(x => (long)x.b).ToArray()
                        : [];
                    croppedImages += cropImages.Length;
                    if (cropImages.Length > 0)
                    {
                        cropWindows = CropConsistency.SampleWindows(boxes, resolvedInputSize, cropGrid, cropGrid, cropRandom);
                        using (no_grad())
                        using (var cropScope = NewDisposeScope())
                        {
                            var selected = tensor(cropImages, device: device);
                            var crops = CropConsistency.CropPixels(pixelBatch.index_select(0, selected), cropImages.Select(b => cropWindows[b]).ToArray(), cropGrid);
                            var pooledCrops = imageTower.forward(crops).Pooled;
                            cropPooled = zeros([pixelBatch.shape[0], pooledCrops.shape[1]], device: device)
                                .index_copy(0, selected, pooledCrops).MoveToOuterDisposeScope();
                        }
                    }

                    // Classification pools the full map; the localization losses below use
                    // the localization map (the same tensor unless LocalizationFromFusedOnly).
                    var (spatial, localizationMap) = imageTower.ForwardMaps(pixelBatch);
                    var spatialMask = SpatialMask.Build(boxes, resolvedInputSize, spatial.shape[2], spatial.shape[3], device);
                    var pooled = ImageTower.MaskedPool(spatial, spatialMask);
                    var tagEmbeddings = tagTower.forward(allTagIndices);

                    var batchTagRows = globalBatchIndices.Select(g => imageTagRows[g]).ToList();
                    using var labels = BatchLabelBuilder.Build(batchTagRows, vocabulary.Records.Count).to(device);
                    using var cpuTargets = targetOptions.DeliberateTargets
                        ? pairWeightBuilder.Build(batchTagRows, sampled, imageSites is not null ? globalBatchIndices.Select(g => imageSites[g]).ToArray() : null)
                        : null;
                    using var targets = cpuTargets?.To(device);

                    // Gradient accumulation: with gradientAccumulationSteps == 1 this is
                    // exactly the old zero_grad/backward/step-every-micro-batch behavior
                    // (GradientAccumulatorTests.NoAccumulation_... pins that). With it > 1,
                    // zero_grad only happens at a cycle's first micro-batch, and step only
                    // at a cycle's last (or a trailing partial cycle's flush at the epoch's
                    // final micro-batch) -- TorchSharp's own backward() already accumulates
                    // into .grad across calls, so the only extra work is not clearing it
                    // and not stepping on every micro-batch. Must happen before EVERY
                    // backward() call below, including ComputeLocalizedChunked's internal
                    // ones, not just the final one.
                    if (GradientAccumulator.ShouldZeroGrad(step, gradientAccumulationSteps))
                    {
                        optimizer.zero_grad();
                        logitScaleBiasOptimizer.zero_grad();
                    }

                    // Computed and backpropagated FIRST, via its own internal chunked
                    // backward(retain_graph: true) calls -- see ComputeLocalizedChunked's own
                    // doc comment. It must run before the primary/self-supervised loss below:
                    // retain_graph: true keeps the shared image/tag-tower graph alive for what
                    // follows, and only the LAST backward() call of the step (the
                    // primary+self-supervised one below) is allowed to finally release it
                    // (retain_graph: false, the default).
                    var localizedLossValue = 0f;
                    if (localizationWeight > 0 && targets is null)
                    {
                        // Folds in --gradient-accumulation-steps' scale here since this
                        // bypasses the normal "build one loss tensor, divide by
                        // gradientAccumulationSteps, backward once" path below.
                        var localizedGradientScale = (float)(localizationWeight / gradientAccumulationSteps);
                        localizedLossValue = SigmoidContrastiveLoss.ComputeLocalizedChunked(
                            localizationMap, tagEmbeddings, labels, (float)localizationTemperature,
                            localizationTagChunkSize, localizedGradientScale, spatialMask, logitScaleBias);
                    }

                    var loss = targets is not null
                        ? SigmoidContrastiveLoss.ComputeWeighted(pooled, tagEmbeddings, targets, logitScaleBias, targetOptions.NegativeGamma, targetOptions.NegativeClip)
                        : SigmoidContrastiveLoss.Compute(pooled, tagEmbeddings, labels, logitScaleBias);
                    // The whole-image tag term alone, for the dashboard's loss breakdown.
                    var tagLoss = loss;
                    Tensor? sparseLocalized = null;
                    Tensor? cropTerm = null;
                    Tensor? mosaicTerm = null;

                    // Deliberate targets: the per-location loss only on pairs where "where is
                    // it?" has an answer (LocalizationPairSelector) -- ~100 pairs per image
                    // instead of the whole vocabulary, so it rides in the same backward pass.
                    if (localizationWeight > 0 && cpuTargets is not null)
                    {
                        float[] wholeImageLogits;
                        using (no_grad())
                            wholeImageLogits = pooled.matmul(tagEmbeddings.t()).cpu().data<float>().ToArray();
                        using var pairs = LocalizationPairSelector.Select(cpuTargets, wholeImageLogits, localizable, targetOptions.LocalizationMinedNegatives);
                        using var devicePairs = pairs.To(device);
                        var localized = SigmoidContrastiveLoss.ComputeLocalizedPairs(
                            localizationMap, tagEmbeddings, devicePairs, (float)localizationTemperature, spatialMask, logitScaleBias,
                            targetOptions.NegativeGamma, targetOptions.NegativeClip,
                            targetOptions.LocalizationBackgroundWeight, targetOptions.LocalizationBackgroundFraction);
                        loss += localizationWeight * localized;
                        sparseLocalized = localized;

                        // Crop consistency: the classifier, shown a crop without gradients, says
                        // whether each positive tag is inside that window; the localization map
                        // pooled over the window learns to agree (ComputeWindowConsistency).
                        if (cropWindows is not null && cropPooled is not null)
                        {
                            Tensor teacher;
                            using (no_grad())
                            {
                                var pairEmbeddings = tagEmbeddings.detach().index_select(0, devicePairs.TagRows.flatten())
                                    .reshape([devicePairs.TagRows.shape[0], devicePairs.TagRows.shape[1], imageTower.EmbeddingDim]);
                                teacher = logitScaleBias.forward(pairEmbeddings.bmm(cropPooled.unsqueeze(2)).squeeze(2));
                            }

                            if (localizationMap.shape[2] != cropGrid)
                                throw new InvalidOperationException($"Crop windows assume a {cropGrid}-cell map but the localization map has {localizationMap.shape[2]}.");
                            using var windowMask = CropConsistency.WindowMask(cropWindows, cropGrid, cropGrid, device);
                            using var pairTrusted = cropTrusted.index_select(0, devicePairs.TagRows.flatten()).reshape(devicePairs.TagRows.shape);
                            cropTerm = SigmoidContrastiveLoss.ComputeWindowConsistency(
                                localizationMap, tagEmbeddings, devicePairs, windowMask, teacher, (float)localizationTemperature, logitScaleBias,
                                pairTrusted, targetOptions.CropTeacherConfidence);
                            loss += targetOptions.CropConsistencyWeight * cropTerm;
                        }

                        // Mosaic: four of the batch's images at half size on one canvas; each tile is
                        // a bag of its own, and the other tiles' tags must not light up in it (Mosaic).
                        var tileImages = targetOptions.MosaicsPerStep > 0
                            ? Mosaic.PickTiles((int)pixelBatch.shape[0], targetOptions.MosaicsPerStep, mosaicRandom)
                            : [];
                        if (tileImages.Length > 0)
                        {
                            var canvases = Mosaic.Compose(pixelBatch, tileImages, resolvedInputSize);
                            var (_, mosaicMap) = imageTower.ForwardMaps(canvases);
                            var tileBoxes = tileImages.Select((image, k) => Mosaic.TileBox(boxes[image], k % Mosaic.TilesPerMosaic, resolvedInputSize)).ToArray();
                            using var tileMask = SpatialMask.Build(tileBoxes, resolvedInputSize, mosaicMap.shape[2], mosaicMap.shape[3], device);
                            using var cpuTilePairs = Mosaic.TilePairs(pairs, cpuTargets, tileImages);
                            using var tilePairs = cpuTilePairs.To(device);
                            using var canvasOfTile = tensor(Enumerable.Range(0, tileImages.Length).Select(k => (long)(k / Mosaic.TilesPerMosaic)).ToArray(), device: device);
                            mosaicTerm = SigmoidContrastiveLoss.ComputeLocalizedPairs(
                                mosaicMap.index_select(0, canvasOfTile), tagEmbeddings, tilePairs, (float)localizationTemperature, tileMask, logitScaleBias,
                                targetOptions.NegativeGamma, targetOptions.NegativeClip);
                            loss += localizationWeight * targetOptions.MosaicWeight * mosaicTerm;
                        }
                    }

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

                    // The step's final backward() call -- retain_graph defaults to false here,
                    // finally releasing the graph ComputeLocalizedChunked's calls above kept
                    // alive. Dividing by gradientAccumulationSteps before backward() keeps the
                    // accumulated gradient's scale equivalent to one mean-reduced loss over the
                    // full effective batch (batchSize * gradientAccumulationSteps) instead of
                    // summing gradientAccumulationSteps independently-normalized micro-batch
                    // gradients, which would be gradientAccumulationSteps times too large.
                    (loss / gradientAccumulationSteps).backward();

                    if (GradientAccumulator.ShouldStep(step, gradientAccumulationSteps, stepsPerEpoch))
                    {
                        if (targetOptions.CosineSchedule)
                        {
                            var rate = LearningRateSchedule.At(
                                (long)epoch * stepsPerEpoch + step, (long)epochs * stepsPerEpoch, learningRate, targetOptions.WarmupSteps);
                            foreach (var group in optimizer.ParamGroups)
                                group.LearningRate = rate;
                            foreach (var group in logitScaleBiasOptimizer.ParamGroups)
                                group.LearningRate = rate;
                        }

                        optimizer.step();
                        logitScaleBiasOptimizer.step();
                    }

                    // Unscaled -- reports the real loss magnitude every micro-batch (matching
                    // pre-accumulation logs) rather than the backward()-only scaled-down value.
                    // Reconstructs the same combined total the old single summed `loss` tensor
                    // used to report, now that the localization term is computed (and already
                    // backpropagated) separately from `loss`.
                    var lossValue = loss.item<float>() + (float)(localizationWeight * localizedLossValue);
                    if (float.IsNaN(lossValue))
                    {
                        dashboard.Log("[red]" + Markup.Escape($"epoch {epoch + 1}/{epochs} step {step + 1}/{stepsPerEpoch}: {NaNGuard.Message}") + "[/]");
                        result = 1;
                        dashboard.Finished("stopped: loss diverged to NaN");
                        return;
                    }

                    // How much of each step went to waiting on data vs. everything else
                    // (GPU compute, target building) -- the number that says whether
                    // loading or compute is the bottleneck.
                    var stepsThisEpoch = step + 1 - stepStart;
                    var waitMs = ((prefetcher?.ConsumerWait ?? TimeSpan.Zero) - epochWaitStart).TotalMilliseconds / stepsThisEpoch;
                    var stepMs = epochClock.Elapsed.TotalMilliseconds / stepsThisEpoch;
                    var localizationShown = sparseLocalized is not null
                        ? (float)(localizationWeight * sparseLocalized.item<float>())
                        : (float)(localizationWeight * localizedLossValue);
                    if (cropTerm is not null)
                        localizationShown += (float)(targetOptions.CropConsistencyWeight * cropTerm.item<float>());
                    if (mosaicTerm is not null)
                        localizationShown += (float)(localizationWeight * targetOptions.MosaicWeight * mosaicTerm.item<float>());
                    var currentRate = targetOptions.CosineSchedule
                        ? LearningRateSchedule.At((long)epoch * stepsPerEpoch + step, (long)epochs * stepsPerEpoch, learningRate, targetOptions.WarmupSteps)
                        : learningRate;
                    dashboard.StepCompleted(step + 1, lossValue, tagLoss.item<float>(), localizationShown, currentRate, stepMs - waitMs, waitMs);
                    if (targetOptions.CropConsistencyWeight > 0)
                        dashboard.CropTeacher(cropGate.TrustedCount, croppedImages / (double)((stepsThisProcess + 1L) * batchSize));

                    if (snapshots is not null)
                    {
                        TakeFinishedSnapshots();
                        // Epoch ends get their own evaluation below.
                        var globalStep = (long)epoch * stepsPerEpoch + step + 1;
                        if (globalStep % snapshotEvery == 0 && step + 1 < stepsPerEpoch)
                        {
                            dashboard.Phase($"snapshotting step {globalStep:N0} for evaluation");
                            snapshots.Launch(globalStep, snapshot => WriteSnapshot(snapshot, epoch, step + 1));
                            dashboard.SnapshotEvaluationStarted(globalStep);
                        }
                    }

                    // Mid-epoch safety net: an epoch can take multiple days, so waiting
                    // for the epoch-boundary save below to be the only recovery point
                    // would mean a crash loses up to a full epoch of work. Same two
                    // artifacts as the epoch-boundary save, just with CompletedEpochs
                    // still at the *previous* completed count and CompletedStepsInEpoch
                    // recording how far into this epoch we got.
                    stepsThisProcess++;
                    var reachedMaxSteps = targetOptions.MaxSteps > 0 && stepsThisProcess >= targetOptions.MaxSteps;
                    if (sinceLastCheckpoint.Elapsed >= checkpointInterval || reachedMaxSteps)
                    {
                        var (appliedOffloaded, offloadedStop) = offloaded is not null ? ApplyOffloaded() : ([], false);
                        dashboard.Phase("saving checkpoint");
                        SaveCheckpoint(checkpointDir, imageTower, tagTower, logitScaleBias, embeddingDim, resolvedInputSize, vocabulary);
                        TrainingState.Save(
                            checkpointDir,
                            new TrainingProgress(epoch, earlyStopping.BestLoss, earlyStopping.EvaluationsSinceImprovement, step + 1, validationSplitSeed, stepsPerEpoch, EarlyStoppingMetricName),
                            optimizer);
                        TrainingState.SaveLogitScaleBiasOptimizerState(checkpointDir, logitScaleBiasOptimizer);
                        dashboard.SaveLossHistory(checkpointDir);
                        sinceLastCheckpoint.Restart();
                        dashboard.CheckpointSaved();
                        foreach (var e in appliedOffloaded)
                            offloaded!.MarkApplied(e);
                        if (offloadedStop)
                        {
                            dashboard.Finished("early stopping (offloaded evaluation)");
                            return;
                        }
                    }

                    if (reachedMaxSteps)
                    {
                        dashboard.Finished($"stopped after --max-steps {targetOptions.MaxSteps}");
                        return;
                    }
                }

                // Save after every epoch, not just at the end: a long run (especially on
                // GPU) can be killed, crash, or lose its SSH session partway through, and
                // without this, all completed work would be unrecoverable.
                dashboard.Phase($"end of epoch {epoch + 1}: saving checkpoint");
                SaveCheckpoint(checkpointDir, imageTower, tagTower, logitScaleBias, embeddingDim, resolvedInputSize, vocabulary);
                sinceLastCheckpoint.Restart();
                dashboard.CheckpointSaved();

                var stopEarly = false;
                List<int> appliedAtEpochEnd = [];
                if (validationIndices.Count == 0)
                {
                    dashboard.Log("Not enough images for a validation split — running the full epoch count without early stopping.");
                }
                else if (offloaded is not null)
                {
                    var completedEpoch = epoch + 1;
                    dashboard.Phase($"end of epoch {completedEpoch}: snapshotting for the offloaded evaluation");
                    offloaded.Launch(completedEpoch, snapshot => WriteSnapshot(snapshot, completedEpoch, 0));
                    dashboard.EvaluationOffloaded(completedEpoch);
                    dashboard.Log(Markup.Escape($"epoch {completedEpoch}: evaluation started in the background ({targetOptions.EvaluationCommand}); training continues"));

                    // The last epoch has nothing left to overlap with: wait, so the run ends with its reports.
                    if (epoch == epochs - 1)
                    {
                        dashboard.Phase($"end of epoch {completedEpoch}: waiting for the offloaded evaluations");
                        offloaded.WaitAll();
                        if (snapshots is not null)
                        {
                            snapshots.WaitAll();
                            TakeFinishedSnapshots();
                        }
                    }
                    (appliedAtEpochEnd, stopEarly) = ApplyOffloaded();
                }
                else
                {
                    var heldOut = targetOptions.ValidationMaxImages > 0 && validationIndices.Count > targetOptions.ValidationMaxImages
                        ? validationIndices.Take(targetOptions.ValidationMaxImages).ToList()
                        : validationIndices;
                    var validationBatchCount = Math.Max(1, (int)Math.Ceiling(heldOut.Count / (double)batchSize));
                    dashboard.EvaluationStarted();
                    var (validationLoss, heldOutPooled) = Evaluate(
                        imageTower, tagTower, logitScaleBias, loadBatch, imageTagRows, heldOut, allTagIndices, vocabulary.Records.Count, device, batchSize, resolvedInputSize,
                        readWindow: readWindow,
                        onBatchComplete: batchesDone =>
                            dashboard.PhaseProgress($"end of epoch {epoch + 1}: embedding held-out images (batches)", batchesDone, validationBatchCount));

                    dashboard.Phase($"end of epoch {epoch + 1}: scoring held-out tag rankings and ratings");
                    var report = EvaluateRankings(
                        heldOutPooled, heldOut.Select(i => imageTagRows[i]).ToList(), tagTower, allTagIndices, logitScaleBias,
                        vocabulary, statistics, exclusiveGroups, device, validationSplitSeed,
                        imageSites is not null ? heldOut.Select(i => imageSites[i]).ToArray() : null);
                    var ratingReport = ratings is not null && ratingRows is not null
                        ? EvaluateRatings(heldOutPooled, heldOut.Select(i => ratings[i]).ToList(), tagTower, allTagIndices, logitScaleBias, ratingRows, imageTower.EmbeddingDim)
                        : null;
                    WriteEvaluationArtifacts(checkpointDir, report, ratingReport);
                    ArchiveEpochReports(checkpointDir, epoch + 1);
                    // Diagnostic only: a failure here is logged, never allowed to stop training.
                    if (gold is { Count: > 0 } && readWindow is not null)
                    try
                    {
                        dashboard.Phase($"end of epoch {epoch + 1}: scoring gold-set localization and the crop teacher");
                        float[] tagEmbeddingArray;
                        using (no_grad())
                        using (var embeddingsNow = tagTower.forward(allTagIndices))
                            tagEmbeddingArray = embeddingsNow.cpu().data<float>().ToArray();
                        var localization = new LocalizationEvaluator(
                                imageTower, tagEmbeddingArray, imageTower.EmbeddingDim, logitScaleBias.ToLogitScaling(), readWindow, resolvedInputSize, device)
                            .Evaluate(vocabulary, imageTagRows, heldOut, gold, localizability, seed: validationSplitSeed, labelFreeChecks: false);
                        EvaluateCommandHandler.WriteLocalizationReport(checkpointDir, localization);
                        ArchiveEpochReports(checkpointDir, epoch + 1);
                        foreach (var g in localization.Gold.Where(g => g.Name == "all"))
                            dashboard.Log(EvaluateCommandHandler.DescribeGold(g));
                        foreach (var t in (localization.Teacher ?? []).Where(t => t.Name == "all"))
                            dashboard.Log(EvaluateCommandHandler.DescribeTeacher(t));
                    }
                    catch (Exception e)
                    {
                        dashboard.Log("[red]" + Markup.Escape($"gold-set scoring failed (training continues): {e.GetType().Name}: {e.Message}") + "[/]");
                    }
                    if (targetOptions.CropConsistencyWeight > 0)
                    {
                        cropGate = CropTeacherGate.From(report, vocabulary, localizable, targetOptions.CropTeacherMinAveragePrecision, targetOptions.CropTeacherMinPositives);
                        cropTrusted.Dispose();
                        cropTrusted = tensor(cropGate.Trusted.Select(t => t ? 1f : 0f).ToArray(), device: device);
                        dashboard.Log(Markup.Escape($"crop consistency: {cropGate.TrustedCount} tags now trusted as teachers (held-out AP >= {targetOptions.CropTeacherMinAveragePrecision})"));
                    }
                    dashboard.EvaluationFinished(epoch, report, ratingReport);
                    if (EvaluationPoint.Read(Path.Combine(checkpointDir, "history", $"epoch-{epoch + 1}"), (long)(epoch + 1) * stepsPerEpoch, endOfEpoch: true) is { } epochPoint)
                        dashboard.AddEvaluationPoint(epochPoint);
                    dashboard.Log(Markup.Escape($"epoch {epoch + 1}/{epochs} validation loss {validationLoss:G4}"));
                    dashboard.Log(DescribeReport(report));
                    if (ratingReport is not null)
                        dashboard.Log(DescribeRatings(ratingReport));

                    if (earlyStopping.ShouldStop(report.EarlyStoppingValue))
                    {
                        dashboard.Log("[yellow]" + Markup.Escape($"Early stopping: held-out mAP stopped improving after epoch {epoch + 1}.") + "[/]");
                        stopEarly = true;
                    }
                }

                // Saved once per epoch regardless of which branch ran above, so a
                // resumed run always picks up this epoch's completed count and
                // EarlyStopping's latest history (unchanged if there was no
                // validation split to evaluate against).
                dashboard.Phase($"end of epoch {epoch + 1}: saving training state");
                TrainingState.Save(checkpointDir, new TrainingProgress(epoch + 1, earlyStopping.BestLoss, earlyStopping.EvaluationsSinceImprovement, CompletedStepsInEpoch: 0, validationSplitSeed, stepsPerEpoch, EarlyStoppingMetricName), optimizer);
                TrainingState.SaveLogitScaleBiasOptimizerState(checkpointDir, logitScaleBiasOptimizer);
                dashboard.SaveLossHistory(checkpointDir);
                foreach (var e in appliedAtEpochEnd)
                    offloaded!.MarkApplied(e);

                if (stopEarly)
                    break;
            }

            dashboard.Finished("finished");
        }

        var references = new List<(string, EvaluationPoint)>();
        foreach (var reference in targetOptions.ChartReferences ?? [])
        {
            var split = reference.IndexOf('=');
            if (split > 0 && EvaluationPoint.Read(reference[(split + 1)..], 0, endOfEpoch: true) is { } point)
                references.Add((reference[..split], point));
            else
                AnsiConsole.MarkupLineInterpolated($"[yellow]--chart-reference '{reference}': expected label=directory with an evaluation_report.json; ignored.[/]");
        }
        dashboard.SetChartReferences(references);

        if (AnsiConsole.Profile.Capabilities.Interactive && !targetOptions.TextDashboard && TerminalGraphics.SupportsSixel())
        {
            using (new DashboardScreen(dashboard, refresh: TimeSpan.FromSeconds(1)))
                TrainingLoop();
            AnsiConsole.Write(dashboard.Render());
            foreach (var line in dashboard.Events)
                AnsiConsole.MarkupLine(line);
        }
        else if (AnsiConsole.Profile.Capabilities.Interactive)
        {
            AnsiConsole.Live(dashboard.Render())
                .AutoClear(false)
                .Overflow(VerticalOverflow.Ellipsis)
                .Start(ctx =>
                {
                    using (dashboard.Attach(ctx))
                        TrainingLoop();
                    ctx.UpdateTarget(dashboard.Render());
                    ctx.Refresh();
                });

            // The panel shows only recent events; scrollback gets all of them once it stops.
            foreach (var line in dashboard.Events)
                AnsiConsole.MarkupLine(line);
        }
        else
        {
            using (dashboard.AttachLog(TimeSpan.FromMinutes(5)))
                TrainingLoop();
            AnsiConsole.MarkupLine(dashboard.Summary());
        }

        return result;
    }

    private static void SaveCheckpoint(string checkpointDir, ImageTower imageTower, TagTower tagTower, LogitScaleBias logitScaleBias, int embeddingDim, int inputSize, TagVocabulary vocabulary)
    {
        var embeddings = TagEmbeddingStore.CreateEmpty(embeddingDim);
        foreach (var row in tagTower.ExtractRows())
            embeddings.AppendRow(row);

        var config = ModelConfig.From(imageTower, inputSize);
        Checkpoint.Save(checkpointDir, imageTower, config, vocabulary, embeddings, logitScaleBias);
    }

    internal const string EarlyStoppingMetricName = "1-mAP";

    internal static EvaluationReport EvaluateRankings(
        float[] heldOutPooled, IReadOnlyList<IReadOnlyList<int>> heldOutTags, TagTower tagTower, Tensor allTagIndices, LogitScaleBias logitScaleBias,
        TagVocabulary vocabulary, TagStatistics statistics, ExclusiveTagGroups exclusiveGroups, Device device, int seed,
        IReadOnlyList<BooruSites>? heldOutSites = null)
    {
        float[] tagEmbeddings;
        using (no_grad())
        using (var embeddings = tagTower.forward(allTagIndices))
            tagEmbeddings = embeddings.cpu().data<float>().ToArray();

        return TagRankingEvaluator.Evaluate(
            heldOutPooled, heldOutTags, tagEmbeddings, (int)(tagEmbeddings.Length / vocabulary.Records.Count), logitScaleBias.ToLogitScaling(),
            vocabulary, statistics, exclusiveGroups, device, seed: seed, imageSites: heldOutSites);
    }

    internal const string EvaluationReportFileName = "evaluation_report.json";


    /// <summary>Copies this epoch's reports into history/epoch-N/ (the top-level copies are overwritten every epoch).</summary>
    private static void ArchiveEpochReports(string checkpointDir, int epoch)
    {
        var target = Path.Combine(checkpointDir, "history", $"epoch-{epoch}");
        Directory.CreateDirectory(target);
        foreach (var name in new[] { EvaluationReportFileName, RatingReportFileName, EvaluateCommandHandler.LocalizationReportFileName, TagThresholds.FileName })
        {
            var source = Path.Combine(checkpointDir, name);
            if (File.Exists(source))
                File.Copy(source, Path.Combine(target, name), overwrite: true);
        }
    }

    /// <summary>Recall each cumulative rating label's cutoff must reach: ≥sensitive, ≥questionable, explicit. Missing questionable/explicit content is the costly error.</summary>
    internal static readonly double[] RatingTargetRecall = [0.90, 0.97, 0.97];

    internal static RatingReport EvaluateRatings(
        float[] heldOutPooled, IReadOnlyList<ContentRating> heldOutRatings, TagTower tagTower, Tensor allTagIndices, LogitScaleBias logitScaleBias,
        int[] ratingRows, int dim)
    {
        float[] tagEmbeddings;
        using (no_grad())
        using (var embeddings = tagTower.forward(allTagIndices))
            tagEmbeddings = embeddings.cpu().data<float>().ToArray();
        return RatingEvaluator.Evaluate(heldOutPooled, dim, heldOutRatings, tagEmbeddings, ratingRows, logitScaleBias.ToLogitScaling(), RatingTargetRecall);
    }

    internal const string RatingReportFileName = "rating_report.json";

    internal static string DescribeRatings(RatingReport r) =>
        Markup.Escape(
            $"ratings ({r.Images} held-out images): accuracy {r.Accuracy:P1} | questionable/explicit called general {r.ExplicitOrQuestionableCalledGeneral:P2} | general called questionable/explicit {r.GeneralCalledExplicitOrQuestionable:P2} | " +
            $"cutoffs {string.Join(", ", r.Thresholds.Select(t => t.ToString("F3")))} at recall {string.Join(", ", r.Recall.Select(x => x.ToString("P1")))}, precision {string.Join(", ", r.Precision.Select(x => x.ToString("P1")))}");

    /// <summary>Writes the calibrated per-tag thresholds (picked up by export-onnx) and the full per-tag report, worst tags first. Rating labels use their recall-targeted cutoffs instead of best-F1 ones.</summary>
    internal static void WriteEvaluationArtifacts(string directory, EvaluationReport report, RatingReport? ratingReport = null)
    {
        var thresholds = new Dictionary<string, float>(TagRankingEvaluator.ToThresholds(report).ByTag);
        if (ratingReport is not null)
        {
            for (var k = 0; k < ImageRatings.CumulativeTags.Length; k++)
                thresholds[ImageRatings.CumulativeTags[k]] = ratingReport.Thresholds[k];
            File.WriteAllText(Path.Combine(directory, RatingReportFileName),
                System.Text.Json.JsonSerializer.Serialize(ratingReport, new System.Text.Json.JsonSerializerOptions { WriteIndented = true }));
        }

        new TagThresholds(thresholds).Save(Path.Combine(directory, TagThresholds.FileName));
        File.WriteAllText(
            Path.Combine(directory, EvaluationReportFileName),
            System.Text.Json.JsonSerializer.Serialize(report, new System.Text.Json.JsonSerializerOptions
            {
                WriteIndented = true,
                NumberHandling = System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals
            }));
    }

    internal static string DescribeReport(EvaluationReport report) =>
        Markup.Escape(
            $"held-out: {report.Images} images, {report.EvaluatedTags} tags | mAP {report.MeanAveragePrecision:F4}, hard-negative mAP {report.MeanHardAveragePrecision:F4}, context mAP {report.MeanContextAveragePrecision:F4}, mean best-F1 {report.MeanBestF1:F4}, mean held-out F1 at calibrated thresholds {report.MeanHeldOutF1:F4} | " +
            $"image-level at calibrated thresholds: precision {report.MicroPrecision:F4}, recall {report.MicroRecall:F4}, F1 {report.MicroF1:F4}, R-precision {report.RPrecision:F4}, exclusive-tag violations {report.ExclusiveViolationRate:P2}" +
            (report.ContextReliance is { } c
                ? $" | context reliance: recall gap {c.MacroRecallGap:+0.0%;-0.0%} ({c.RecallGapTags} tags), false-positive gap {c.MacroFalsePositiveGap:+0.0%;-0.0%} ({c.FalsePositiveGapTags} tags)"
                : ""));

    internal static (double Loss, float[] Pooled) Evaluate(
        ImageTower imageTower,
        TagTower tagTower,
        LogitScaleBias logitScaleBias,
        Func<IReadOnlyList<int>, (Tensor Pixels, IReadOnlyList<LetterboxBox> Boxes)> loadBatch,
        IReadOnlyList<IReadOnlyList<int>> imageTagRows,
        IReadOnlyList<int> validationIndices,
        Tensor allTagIndices,
        int vocabularySize,
        Device device,
        int batchSize,
        int inputSize,
        Action<int>? onBatchComplete = null,
        Func<IReadOnlyList<int>, EncodedImage[]>? readWindow = null)
    {
        using var _ = no_grad();

        // Same background, offset-sorted loading as training when the data comes from a
        // cache. Validation order doesn't matter (embeddings are written back by position),
        // so a bigger window buys more read locality.
        using var prefetcher = readWindow is not null
            ? new BatchPrefetcher(
                Enumerable.Range(0, (validationIndices.Count + batchSize - 1) / batchSize)
                    .Select(b => new BatchRequest(validationIndices.Skip(b * batchSize).Take(batchSize).ToArray())),
                readWindow, inputSize, windowBatches: 16, capacityBatches: 32)
            : null;

        using var tagEmbeddings = tagTower.forward(allTagIndices);

        double totalLoss = 0;
        var totalCount = 0;
        var pooledAll = new float[validationIndices.Count * imageTower.EmbeddingDim];
        var batchesDone = 0;
        for (var offset = 0; offset < validationIndices.Count; offset += batchSize)
        {
            // Same reasoning as the training loop's stepScope (see there): without this,
            // every validation batch's tensors (including the discarded pooled global
            // embedding from imageTower.forward below) leak native/CUDA memory that
            // outlives this method, across every epoch's validation pass.
            using var batchScope = NewDisposeScope();

            var chunk = validationIndices.Skip(offset).Take(batchSize).ToList();

            Tensor rawPixelBatch;
            IReadOnlyList<LetterboxBox> boxes;
            if (prefetcher is not null)
            {
                if (!prefetcher.TryTake(out var loaded))
                    throw new InvalidOperationException("The validation batch stream ended early.");
                rawPixelBatch = tensor(loaded.Pixels, [loaded.Boxes.Length, 3, inputSize, inputSize]);
                boxes = loaded.Boxes;
            }
            else
            {
                (rawPixelBatch, boxes) = loadBatch(chunk);
            }

            using var pixelBatch = rawPixelBatch.to(device);
            var (_, spatial) = imageTower.forward(pixelBatch);
            using var _spatial = spatial;
            using var spatialMask = SpatialMask.Build(boxes, inputSize, spatial.shape[2], spatial.shape[3], device);
            using var pooled = ImageTower.MaskedPool(spatial, spatialMask);

            var batchTagRows = chunk.Select(i => imageTagRows[i]).ToList();
            using var labels = BatchLabelBuilder.Build(batchTagRows, vocabularySize).to(device);
            using var loss = SigmoidContrastiveLoss.Compute(pooled, tagEmbeddings, labels, logitScaleBias);
            pooled.cpu().data<float>().ToArray().CopyTo(pooledAll, offset * imageTower.EmbeddingDim);

            totalLoss += loss.item<float>() * chunk.Count;
            totalCount += chunk.Count;

            batchesDone++;
            onBatchComplete?.Invoke(batchesDone);
        }

        return (totalLoss / totalCount, pooledAll);
    }

    internal static (Tensor Pixels, IReadOnlyList<LetterboxBox> Boxes) LoadCacheBatch(PreprocessedDatasetCacheReader cache, IReadOnlyList<int> indices, int inputSize)
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

    /// <summary>
    /// Shuffles all dataset indices once and carves off a validation slice, so the same
    /// held-out images are used for every epoch's evaluation. Seeded (rather than
    /// <see cref="Random.Shared"/>) and the seed is persisted in <see cref="TrainingProgress"/>
    /// so a resumed run reconstructs the identical split instead of re-shuffling.
    /// </summary>
    internal static (List<int> Training, List<int> Validation) SplitForValidation(int datasetCount, double validationFraction, int seed)
    {
        var indices = Enumerable.Range(0, datasetCount).ToList();
        var random = new Random(seed);
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
