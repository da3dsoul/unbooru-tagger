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
        // Reads at a given canvas size, when that can differ from resolvedInputSize (--train-scales) or images need rescaling.
        Func<IReadOnlyList<int>, int, EncodedImage[]>? readWindowAt = null;
        // Replaced whole (never changed in place) when a pending store appears, so other threads always see a complete list.
        HiResImageStore[] hiResStores = [];
        // --hires-store directories a pull hasn't created yet (no store.json): opened at a store refresh once they exist.
        var pendingStores = new List<string>();
        var trainingScales = targetOptions.TrainScales is { } scaleSpec
            ? new TrainingScales(TrainingScales.Parse(scaleSpec), Random.Shared.Next()) // batch sampling isn't seeded either
            : null;
        // Reviewed tag parentage (TagParentage): --tag-parentage, else the checkpoint's saved copy.
        var parentagePath = TagParentage.ResolvePath(targetOptions.TagParentagePath, checkpointDir);
        if (parentagePath is not null && targetOptions.TagParentagePath is not null && !File.Exists(parentagePath))
            throw new FileNotFoundException($"--tag-parentage: '{parentagePath}' doesn't exist.");
        var parentage = TagParentage.Load(parentagePath);

        if (cacheReader is not null)
        {
            vocabulary = TagVocabulary.Load(Path.Combine(cacheDir!, "tag_vocabulary.json"));
            imageTagRows = cacheReader.ImageTagRows.Select(rows => rows.ToList()).ToList();
            datasetCount = cacheReader.ImageCount;

            // Fill in tags Danbooru's implication rules guarantee (TagImplications); Gelbooru
            // posts don't get them applied at the source. The reviewed is-a and implies rules
            // complete the same way (boots → shoes).
            var implications = TagImplications.Load(Path.Combine(cacheDir!, TagImplications.FileName), vocabulary, parentage.CompletionPairs);
            if (parentage.RuleCount > 0)
                AnsiConsole.MarkupLineInterpolated(
                    $"Tag parentage: {parentage.IsA.Count} is-a, {parentage.Implies.Count} implies and {parentage.Excludes.Count} excludes rules from '{parentagePath}'.");
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
            resolvedInputSize = targetOptions.TrainInputSize ?? cacheReader.InputSize;
            // Kept open for the whole run (the prefetcher reads them to the end); a pull still writing one is picked up as it goes.
            var storeDirectories = (targetOptions.HiResStore ?? "").Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
            pendingStores.AddRange(storeDirectories.Where(directory => !HiResImageStore.Exists(directory)));
            hiResStores = storeDirectories.Where(HiResImageStore.Exists).Select(HiResImageStore.Open).OrderBy(store => store.LongSide).ToArray();
            var stores = hiResStores;
            var cacheSize = cacheReader.InputSize;
            if (trainingScales is null && resolvedInputSize == cacheSize && stores.Length == 0 && pendingStores.Count == 0)
            {
                loadBatch = indices => LoadCacheBatch(cacheReader, indices, resolvedInputSize);
                readWindow = indices => cacheReader.ReadEncoded(indices, ioParallelism);
            }
            else
            {
                // Training at another input size (e.g. fine-tuning at 1024) or several (--train-scales): rows a hi-res store holds
                // come from it, the rest are the cached images rescaled (ImageResize).
                readWindowAt = (indices, size) => size == cacheSize && hiResStores.Length == 0
                    ? cacheReader.ReadEncoded(indices, ioParallelism)
                    : ImageResize.Adapt(cacheReader.ReadEncoded(indices, ioParallelism), indices, cacheSize, size, hiResStores);
                var at = readWindowAt;
                readWindow = indices => at(indices, resolvedInputSize);
                loadBatch = indices => LoadEncodedBatch(at(indices, resolvedInputSize), resolvedInputSize);
                AnsiConsole.MarkupLineInterpolated(
                    $"Input size {(trainingScales is null ? $"{resolvedInputSize}px" : $"mixed ({trainingScales}); checkpoints keep {resolvedInputSize}px")} (the cache's is {cacheSize}px); {(stores.Length == 0 ? "no hi-res store, every image rescaled" : string.Join(", ", stores.Select(store => $"{store.Count:N0} rows from the hi-res store at {store.LongSide}px ({store.Directory})")) + ", the rest rescaled")}{(pendingStores.Count > 0 ? $"; waiting for {string.Join(", ", pendingStores)}" : "")}.");
            }
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
                embeddingDim, tower.StemChannels, tower.StageChannels, tower.BlocksPerStage, device, tower.FusedStages, targetOptions.ClassificationGlimpses);
            tagTower = TagTower.CreateFullyTrainable(initialRows, embeddingDim, device);
            logitScaleBias = new LogitScaleBias(device: device);
        }

        if (targetOptions.ClassificationGlimpses && !imageTower.ClassificationGlimpses)
            throw new ArgumentException("--classification-glimpses adds weights (the zoom codes), so it starts a fresh run; this checkpoint was trained without it.");

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
        if (targetOptions.Bf16Compute)
        {
            imageTower.ComputeType = ScalarType.BFloat16;
            AnsiConsole.MarkupLine("Image tower computes in bfloat16 (fp32 weights, gradients and outputs).");
        }
        if (imageTower.ClassificationGlimpses)
            imageTower.GlimpseCount = targetOptions.GlimpsesPerImage;
        // NHWC memory order on the device (ImageTower.ChannelsLast; same maps): bf16 convolutions there come back
        // channels-last, and group norm used to put each block back in NCHW for its MLP to copy out again.
        imageTower.ChannelsLast = device.type != TorchSharp.DeviceType.CPU;
        SigmoidContrastiveLoss.HillNegatives = targetOptions.HillNegatives;
        if (targetOptions.HillNegatives)
            AnsiConsole.MarkupLine("Unverified negatives: the Hill loss (1.5 - p) p^2, in place of the asymmetric loss.");
        // Pooling just being turned on: its anneal (if any) counts from the step the run resumes at, below.
        var poolingTurnedOn = targetOptions.ClassificationPoolingTemperature is > 0 && imageTower.ClassificationPoolingTemperature <= 0;
        if (targetOptions.ClassificationPoolingTemperature is { } poolingTemperature && poolingTemperature != imageTower.ClassificationPoolingTemperature)
        {
            imageTower.ClassificationPoolingTemperature = poolingTemperature;
            if (poolingTemperature <= 0)
                imageTower.ClassificationPoolingSchedule = null;
        }
        if (targetOptions.ClassificationPoolingTopK is { } topK)
            imageTower.ClassificationPoolingTopK = Math.Max(0, topK);
        if (targetOptions.ClassificationPoolingBottomWeight is { } bottomWeight)
            imageTower.ClassificationPoolingBottomWeight = Math.Max(0, bottomWeight);
        if (targetOptions.GlimpsePoolZooms is { } poolZooms)
            imageTower.GlimpsePoolZooms = Math.Clamp(poolZooms, 1, Glimpses.Zooms.Length);
        if (targetOptions.GlimpsePoolMinZoom is { } poolMinZoom)
            imageTower.GlimpsePoolMinZoom = Math.Clamp(poolMinZoom, 0, imageTower.GlimpsePoolZooms - 1);

        exclusiveGroups = ExclusiveTagGroups.Parse(exclusiveGroups.SourceText, vocabulary);
        if (ratingRows is not null)
            ratingRows = RatingLabels.FindRows(vocabulary);
        var tagFrequencies = ComputeTagFrequencies(imageTagRows);
        Directory.CreateDirectory(checkpointDir);
        exclusiveGroups.Save(Path.Combine(checkpointDir, ExclusiveTagGroups.FileName));
        var savedParentage = Path.Combine(checkpointDir, TagParentage.FileName);
        if (parentage.RuleCount > 0 && Path.GetFullPath(parentagePath!) != Path.GetFullPath(savedParentage))
            File.Copy(parentagePath!, savedParentage, overwrite: true);

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
        // A loaded optimizer keeps the rate it was saved with; a constant schedule means --lr from here on (a fine-tune arm at a
        // fixed rate, whatever the cosine schedule had reached).
        if (!targetOptions.CosineSchedule)
            foreach (var group in optimizer.ParamGroups.Concat(logitScaleBiasOptimizer.ParamGroups))
                group.LearningRate = learningRate;

        // A separate glimpse tower (ImageTower.GlimpseTower): made from a copy of this tower's weights the first time, with its own
        // optimizer, fresh then (the shared tower's moments were built from both paths' gradients and its momentum points at the
        // solution being left) and warmed up over GlimpseTowerWarmupSteps steps into the run's schedule.
        var glimpseTowerMade = false;
        if (targetOptions.SeparateGlimpseTower && imageTower.ClassificationGlimpses && imageTower.GlimpseTower is null)
        {
            imageTower.GlimpseTower = imageTower.CopyForGlimpses();
            glimpseTowerMade = true;
        }
        var glimpseTowerOptimizer = imageTower.GlimpseTower is { } separateGlimpses ? optim.Adam(separateGlimpses.parameters().ToArray(), lr: learningRate) : null;
        if (glimpseTowerOptimizer is not null && !glimpseTowerMade && resumingOptimizerState && TrainingState.HasGlimpseTowerOptimizerState(checkpointDir))
            TrainingState.LoadGlimpseTowerOptimizerState(checkpointDir, glimpseTowerOptimizer);
        if (imageTower.GlimpseTower is not null)
            AnsiConsole.MarkupLine(glimpseTowerMade
                ? $"Glimpses: a separate tower, copied from this one, fresh optimizer warmed up over {GlimpseTowerWarmupSteps:N0} steps, at {targetOptions.GlimpseTowerLearningRateScale}x the run's learning rate."
                : $"Glimpses: the checkpoint's separate glimpse tower, at {targetOptions.GlimpseTowerLearningRateScale}x the run's learning rate.");
        long glimpseTowerSteps = 0;

        // --fine-localization: a stride-2 level under the localization map (ImageTower.Fine), made the first time with a zero
        // output and its own optimizer, warmed up like the glimpse tower's; a checkpoint that has one keeps training it.
        var fineMade = false;
        if (targetOptions.FineLocalization && imageTower.Fine is null)
        {
            imageTower.Fine = imageTower.NewFineLocalization(device);
            fineMade = true;
        }
        var fineOptimizer = imageTower.Fine is { } fineLevel ? optim.Adam(fineLevel.parameters().ToArray(), lr: learningRate) : null;
        if (fineOptimizer is not null && !fineMade && resumingOptimizerState && TrainingState.HasFineLocalizationOptimizerState(checkpointDir))
            TrainingState.LoadFineLocalizationOptimizerState(checkpointDir, fineOptimizer);
        if (imageTower.Fine is not null)
            AnsiConsole.MarkupLine(fineMade
                ? $"Fine localization: a new stride-2 level under the localization map ({imageTower.Fine.Channels} channels, zero output at first), its own optimizer warmed up over {GlimpseTowerWarmupSteps:N0} steps; maps are now {resolvedInputSize / imageTower.OutputStride}x{resolvedInputSize / imageTower.OutputStride}."
                : $"Fine localization: the checkpoint's stride-2 level; maps are {resolvedInputSize / imageTower.OutputStride}x{resolvedInputSize / imageTower.OutputStride}.");
        long fineSteps = 0;
        // Training scores the fine map from its parts (FineLocalizationMap), which only the localization and mosaic losses read;
        // everything else that reads the localization map would need it built at 256x256 x the embedding width.
        if (imageTower.Fine is not null)
        {
            var readers = new (bool On, string Option)[]
            {
                (!targetOptions.DeliberateTargets, "--targets legacy"),
                (imageTower.ClassificationGlimpses, "--classification-glimpses"),
                (imageTower.ClassificationPoolingTemperature > 0, "--classification-pooling-temperature"),
                (targetOptions.MosaicClassificationWeight > 0, "--mosaic-classification-weight"),
                (targetOptions.MosaicLargeGrid > 0, "--mosaic-large-grid"),
                (targetOptions.LocalizationBackgroundWeight > 0, "--localization-background-weight"),
                (targetOptions.CropConsistencyWeight > 0, "--crop-consistency-weight"),
                (targetOptions.ErasurePairsPerStep > 0, "--erasure-pairs-per-step"),
                (targetOptions.TwinPaste > 0, "--twin-paste"),
                (targetOptions.SelfMapWeight > 0, "--self-map-weight"),
                (targetOptions.CharacterExtentWeight > 0, "--character-extent-weight"),
                (targetOptions.CharacterHideWeight > 0, "--character-hide-weight"),
            };
            if (readers.FirstOrDefault(r => r.On) is { On: true } reader)
                throw new ArgumentException($"--fine-localization doesn't support {reader.Option}: it reads the localization map, which training keeps in parts with a fine level.");
        }

        var allTagIndices = tensor(Enumerable.Range(0, vocabulary.Records.Count).Select(i => (long)i).ToArray(), device: device);

        // Persisted across resumes (TrainingProgress.ValidationSplitSeed) so a mid-epoch
        // resume reconstructs the exact same split instead of re-shuffling -- otherwise
        // a resumed run could train on images the original run held out for validation.
        // Only a genuinely fresh run (or an old checkpoint written before this field
        // existed) picks a new random seed, unless --validation-split-seed names one.
        if (resumingOptimizerState && targetOptions.ValidationSplitSeed is { } asked && asked != resumedProgress.ValidationSplitSeed)
            throw new InvalidOperationException(
                $"--validation-split-seed {asked} doesn't match the checkpoint's {resumedProgress.ValidationSplitSeed}: a resume can't change which images are held out.");
        var validationSplitSeed = resumingOptimizerState ? resumedProgress.ValidationSplitSeed : targetOptions.ValidationSplitSeed ?? Random.Shared.Next();
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
        var families = parentage.RuleCount > 0 && cacheDir is not null
            ? parentage.Bind(vocabulary, TagImplications.LoadPairs(Path.Combine(cacheDir, TagImplications.FileName)))
            : null;
        if (families is not null)
            AnsiConsole.MarkupLineInterpolated(
                $"Tag families: {families.FamilyRows.Count():N0} tags are kinds of a reviewed parent (unknown, not negative, where the parent is tagged without a kind); {Enumerable.Range(0, vocabulary.Records.Count).Count(t => families.Excluders(t).Count > 0):N0} tags are ruled out by an excludes rule.");
        var statistics = TagStatistics.Build(
            trainingTagRows, vocabulary.Records.Count, impliedThreshold: targetOptions.ImpliedThreshold, seed: validationSplitSeed,
            missingThreshold: targetOptions.MissingThreshold,
            imageSites: imageSites is not null ? trainingIndices.Select(i => imageSites[i]).ToArray() : null, tagSites: tagSites,
            families: families);
        var anyAnchorTag = statistics.Positives.Any(p => p.Length > 0);
        if (targetOptions.TagCentricSampling && !anyAnchorTag)
            AnsiConsole.MarkupLine("[yellow]No training image carries any tag -- falling back to rare-tag sampling.[/]");
        // Out-of-context positives: localizable general tags (the rules are loaded again, and saved, below).
        var outOfContextRules = TagLocalizability.Load(targetOptions.LocalizableTagsPath ?? Path.Combine(checkpointDir, TagLocalizability.FileName));
        var outOfContextTags = vocabulary.Records.Select(r => ContextRelianceEvaluator.IsGeneralTag(r.Tag) && outOfContextRules.IsLocalizable(r.Tag)).ToArray();
        ContextPriorTable? outOfContextPriors = null;
        if (targetOptions.OutOfContextShare > 0 || targetOptions.OutOfContextAll || targetOptions.GlimpseFeatureSplit)
        {
            outOfContextPriors = ContextPriorTable.Build(statistics, vocabulary.Records.Select(r => ContextRelianceEvaluator.IsContextSource(r.Tag)).ToArray(), outOfContextTags,
                seed: validationSplitSeed);
            if (targetOptions.OutOfContextShare > 0 || targetOptions.OutOfContextAll)
                AnsiConsole.MarkupLineInterpolated(
                $"Out-of-context positives: {targetOptions.OutOfContextShare:P0} of a localizable general anchor's positives are the lowest-prior of 16 (kept below a prior of {targetOptions.OutOfContextMaxPrior}){(targetOptions.OutOfContextAll ? ", and every localizable general positive below that prior" : "")}, weight {(targetOptions.OutOfContextWeight > 0 ? targetOptions.OutOfContextWeight : 1)}; priors from {outOfContextPriors.SourceCount:N0} characters/artists/series.");
        }
        var tagCentricSampler = targetOptions.TagCentricSampling && anyAnchorTag
            ? new TagCentricBatchSampler(statistics, targetOptions.PositivesPerAnchor, targetOptions.HardNegativesPerAnchor,
                contextNegativesPerAnchor: targetOptions.ContextNegativesPerAnchor)
            {
                FocusShare = targetOptions.FocusShare,
                OutOfContextPriors = outOfContextPriors,
                OutOfContextTags = outOfContextTags,
                OutOfContextShare = targetOptions.OutOfContextShare,
                OutOfContextMaxPrior = targetOptions.OutOfContextMaxPrior,
                ExcluderShare = targetOptions.ExcluderShare,
                ContextPairsPerBatch = targetOptions.ContextPairsPerBatch,
                ExclusivePerBatch = targetOptions.GlimpseExclusivePerBatch,
                ExclusivePriors = targetOptions.GlimpseFeatureSplit ? outOfContextPriors : null,
                ExclusiveMaxPrior = FeatureSplitExclusiveMaxPrior,
                TwinEvery = targetOptions.TwinEvery,
                TwinTags = targetOptions.TwinEvery > 0
                    ? Enumerable.Range(0, vocabulary.Records.Count).Where(t => outOfContextTags[t] && statistics.Positives[t].Length >= TwinMinPositives).ToArray()
                    : null,
                CharacterTags = vocabulary.Records.OrderBy(r => r.RowIndex).Select(r => r.Tag.StartsWith("character:", StringComparison.Ordinal)).ToArray(),
            }
            : null;
        if (targetOptions.TwinEvery > 0)
            AnsiConsole.MarkupLineInterpolated(
                $"Matched twins: every {targetOptions.TwinEvery} batch(es), one image with a localizable general tag ({outOfContextTags.Where((e, t) => e && statistics.Positives[t].Length >= TwinMinPositives).Count():N0} tags) and twins without it whose tags overlap it most; the half of its glimpses most like {TwinWindows.Matches} twins' are a \"no\" for the tag (weight {targetOptions.TwinWeight}).");
        if (targetOptions.ContextPairsPerBatch > 0)
            AnsiConsole.MarkupLineInterpolated(
                $"Context pairs: {targetOptions.ContextPairsPerBatch} per batch, anchored on localizable tags with context sources (two images of one source, with and without the tag); with glimpses, the one with it must score {targetOptions.ContextPairMargin} logit(s) above the one without (weight {targetOptions.ContextPairWeight}).");
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
        // Context pairs anchor on localizable tags only: their hinge trains glimpse scores, which only localizable tags have.
        if (tagCentricSampler is not null)
            tagCentricSampler.ContextPairTags = Enumerable.Range(0, localizable.Length).Where(t => localizable[t]).ToArray();
        // Feature split (Singh et al. 2020) on the glimpse embedding, for localizable general tags: their exclusive positives (and
        // with GlimpseFeatureSplitNegatives, context negatives) score from the object half, judged by each image's own prior
        // (the highest P(tag | its character/artist/series), outOfContextPriors). TagStatistics.ContextSources keeps only a tag's
        // 8 strongest sources, so "none of them" mostly meant a glove-wearing character outside the top 8: identity said "yes".
        if (targetOptions.MosaicGlimpseWeight > 0 && imageTower.ClassificationGlimpses)
            AnsiConsole.MarkupLineInterpolated(
                $"Mosaic glimpses: each canvas gets {imageTower.GlimpseCount} glimpses of its own; up to {MosaicGlimpseTags} of its tiles' localizable tags are a \"no\" in every window on tiles that say no and a \"yes\" pooled over their own tile (weight {targetOptions.MosaicGlimpseWeight}).");
        if (targetOptions.GlimpseFeatureSplit && imageTower.ClassificationGlimpses && imageTower.GlimpseObjectDims <= 0)
            imageTower.GlimpseObjectDims = imageTower.EmbeddingDim / 2;
        // Only tags with a real context bias (some tag carries them with P >= ImpliedThreshold: TagStatistics.ContextSources), as
        // Singh splits only the biased categories: for the rest nearly every positive is "exclusive" and the split means nothing.
        var featureSplitTags = targetOptions.GlimpseFeatureSplit && imageTower.GlimpseObjectDims > 0
            ? outOfContextTags.Select((eligible, t) => eligible && statistics.ContextSources[t].Length > 0).ToArray()
            : null;
        Tensor? glimpseContextMean = featureSplitTags is not null && imageTower.GlimpseContextMean is { } savedMean
            ? tensor(savedMean, device: device).DetachFromDisposeScope()
            : null;
        long splitPositives = 0, splitExclusive = 0, splitNegatives = 0;
        long pastesMade = 0, pastesGated = 0;
        long selfMapSteps = 0, selfMapNegativeTags = 0;
        if (featureSplitTags is not null)
            AnsiConsole.MarkupLineInterpolated(
                $"Glimpse feature split: dimensions 0-{imageTower.GlimpseObjectDims - 1} of {imageTower.EmbeddingDim} are the object half; {featureSplitTags.Count(t => t):N0} context-biased localizable general tags score their exclusive positives (a known character/artist/series, prior below {FeatureSplitExclusiveMaxPrior}; weight {targetOptions.GlimpseFeatureSplitWeight}){(targetOptions.GlimpseFeatureSplitNegatives ? $" and context negatives (prior at least {FeatureSplitContextMinPrior})" : "")} from it, the context half held at its running mean{(targetOptions.GlimpseExclusivePerBatch > 0 ? $"; {targetOptions.GlimpseExclusivePerBatch} images per batch are context negatives and exclusive positives" : "")}.");
        // Classification pooling: localizable tags are scored from the map's cells (ClassificationPooling).
        var classificationPooledRowArray = imageTower.ClassificationPoolingTemperature > 0
            ? Enumerable.Range(0, localizable.Length).Where(t => localizable[t]).Select(t => (long)t).ToArray()
            : [];
        using var classificationPooledRows = tensor(classificationPooledRowArray, device: device);
        // Each vocabulary row's column among the pooled rows (-1 = not pooled), for framed-positive pooling.
        var pooledColumnArray = Enumerable.Repeat(-1L, localizable.Length).ToArray();
        for (var c = 0; c < classificationPooledRowArray.Length; c++)
            pooledColumnArray[classificationPooledRowArray[c]] = c;
        using var pooledColumnOf = tensor(pooledColumnArray, device: device);
        using var localizableMask = imageTower.ClassificationGlimpses ? tensor(localizable.Select(l => l ? 1f : 0f).ToArray(), device: device) : null;
        using var locality = imageTower.ClassificationGlimpses && targetOptions.LocalityNegativeWeight > 0
            ? new LocalityNegatives(device)
            {
                MinObservations = targetOptions.LocalityMinObservations,
                ConfidenceGate = targetOptions.LocalityConfidenceGate,
                PerPair = targetOptions.LocalityPerPair,
                DetachScaling = targetOptions.LocalityDetachScaling,
            }
            : null;
        if (locality is not null)
        {
            var loaded = targetOptions.LocalityPriorPath is { } priorPath
                ? locality.LoadPrior(priorPath, vocabulary.Records.ToDictionary(r => r.Tag, r => r.RowIndex, StringComparer.Ordinal))
                : 0;
            AnsiConsole.MarkupLineInterpolated(
                $"Locality negatives: weight {targetOptions.LocalityNegativeWeight}{(targetOptions.LocalityPerPair ? " per pair" : "")}{(targetOptions.LocalityDetachScaling ? ", scale and bias detached" : "")}{(targetOptions.LocalityConfidenceGate > 0 ? $", calls of at least {targetOptions.LocalityConfidenceGate}" : "")}, a pair counts after {targetOptions.LocalityMinObservations} weighted observations; {(loaded > 0 ? $"{loaded:N0} pairs from '{targetOptions.LocalityPriorPath}'" : "learned from nothing")} (not saved with the checkpoint).");
        }
        if (imageTower.ClassificationGlimpses && imageTower.ClassificationPoolingTemperature <= 0)
            throw new ArgumentException("--classification-glimpses pools glimpses at --classification-pooling-temperature, which must be above 0.");
        if (imageTower.ClassificationPoolingTemperature > 0)
        {
            if (!targetOptions.DeliberateTargets)
                throw new ArgumentException("--classification-pooling-temperature needs --targets deliberate.");
            if (imageTower.ClassificationGlimpses)
                AnsiConsole.MarkupLineInterpolated(
                    $"Classification glimpses: {classificationPooledRows.shape[0]:N0} localizable tags scored by {(imageTower.ClassificationPoolingTopK > 0 ? $"the mean of the top {imageTower.ClassificationPoolingTopK}{(imageTower.ClassificationPoolingBottomWeight > 0 ? $" + {imageTower.ClassificationPoolingBottomWeight} x the bottom {imageTower.ClassificationPoolingTopK}" : "")}" : "a log-sum-exp")} over {Glimpses.Size}px glimpses{(imageTower.GlimpsePoolZooms < Glimpses.Zooms.Length || imageTower.GlimpsePoolMinZoom > 0 ? $" (only zooms {string.Join("/", Glimpses.Zooms.Take(imageTower.GlimpsePoolZooms).Skip(imageTower.GlimpsePoolMinZoom))} score)" : "")}{(targetOptions.GlimpseFramedPositivePooling ? "; label-framed positives pooled over their own windows" : "")} (zooms {string.Join("/", Glimpses.Zooms)}; {imageTower.GlimpseCount} per image, chosen by the model from its localization map) at temperature {imageTower.ClassificationPoolingTemperature}; unframed positives weighted {targetOptions.GlimpseUnframedPositiveWeight}; the rest from the whole image.");
            else
            AnsiConsole.MarkupLineInterpolated(
                $"Classification pooling: {classificationPooledRows.shape[0]:N0} localizable tags scored by a log-sum-exp over a {ClassificationPooling.Grid}x{ClassificationPooling.Grid} grid of the map's cells at temperature {imageTower.ClassificationPoolingTemperature} (trained each step on the batch's positives, full-strength negatives and {targetOptions.ClassificationPoolingMinedNegatives} highest-scoring negatives per image); the rest from the averaged embedding.");
        }
        ContextPrior? contextPrior = null;
        if (targetOptions.ContextPriorWeight > 0 && targetOptions.DeliberateTargets)
        {
            contextPrior = ContextPrior.Build(statistics,
                vocabulary.Records.Select(r => ContextRelianceEvaluator.IsContextSource(r.Tag)).ToArray(),
                vocabulary.Records.Select((r, i) => localizable[i] && ContextRelianceEvaluator.IsGeneralTag(r.Tag)).ToArray(),
                targetOptions.ContextPriorWeight);
            AnsiConsole.MarkupLineInterpolated(
                $"Context prior: weight {targetOptions.ContextPriorWeight}, {contextPrior.PairCount:N0} (source, tag) offsets from {contextPrior.SourceCount:N0} characters/artists/series; training logits only.");
        }
        var pairWeightBuilder = new PairWeightBuilder(statistics, exclusiveGroups, targetOptions.CertainNegativeWeight,
            exhaustiveRows: ratings is not null ? ratingRows : null, contextNegativeWeight: targetOptions.ContextNegativeWeight,
            contextNegativeTags: localizable, contextPrior: contextPrior, hardNegativeSourceWeight: targetOptions.HardNegativeSourceWeight,
            outOfContextPositiveWeight: targetOptions.OutOfContextWeight, excludedNegativeWeight: targetOptions.ExcludedNegativeWeight,
            outOfContextPriors: targetOptions.OutOfContextAll ? outOfContextPriors : null, outOfContextTags: outOfContextTags,
            outOfContextMaxPrior: (float)targetOptions.OutOfContextMaxPrior);
        if (targetOptions.ExcluderShare > 0 || targetOptions.ExcludedNegativeWeight > 0)
            AnsiConsole.MarkupLineInterpolated(
                $"Excludes rules: {targetOptions.ExcluderShare:P0} of anchors are tags an excludes rule rules out, each with two excluder images (no_gloves for gloves) as certain negatives; ruled-out tags weigh {Math.Max(targetOptions.CertainNegativeWeight, targetOptions.ExcludedNegativeWeight)}.");
        if (targetOptions.HardNegativeSourceWeight > 0)
            AnsiConsole.MarkupLineInterpolated(
                $"Hard-negative sources: {statistics.HardNegativeSources.Where((h, t) => h.Length > 0 && localizable[t]).Count():N0} localizable tags have them; an image with a source but not the tag is a full-strength negative, weight {targetOptions.HardNegativeSourceWeight}.");

        // Focus tags: the ones the latest epoch evaluation found answered from context, re-picked
        // after every epoch evaluation (FocusTags).
        var focusEligible = vocabulary.Records.Select((r, i) => localizable[i] && ContextRelianceEvaluator.IsGeneralTag(r.Tag)).ToArray();
        IReadOnlySet<int> focusSet = new HashSet<int>();
        string? RefreshFocus()
        {
            if (targetOptions.FocusShare <= 0 || tagCentricSampler is null)
                return null;
            if (targetOptions.FocusTagNames is { Count: > 0 } named)
            {
                // --focus-tags: a fixed list instead of the evaluation's pick (e.g. the tags a hi-res store was pulled for).
                var rowOf = vocabulary.Records.ToDictionary(r => r.Tag, r => r.RowIndex);
                tagCentricSampler.Focus = named.Select(t => rowOf.TryGetValue(t, out var row) ? row : throw new ArgumentException($"--focus-tags: '{t}' isn't in the vocabulary.")).ToArray();
                focusSet = tagCentricSampler.Focus.ToHashSet();
                return Markup.Escape($"focus: {focusSet.Count} named tags get {targetOptions.FocusShare:P0} of the anchors ({string.Join(", ", named)})");
            }
            var focus = FocusTags.Load(checkpointDir, EvaluationReportFileName, EvaluateCommandHandler.LocalizationReportFileName, vocabulary, focusEligible);
            tagCentricSampler.Focus = focus.Rows;
            focusSet = tagCentricSampler.Focus.ToHashSet();
            var shown = string.Join(", ", focusSet.Take(12).Select(t => vocabulary.Records[t].Tag)) + (focusSet.Count > 12 ? ", ..." : "");
            return Markup.Escape($"focus: {focusSet.Count} tags answered from context get {targetOptions.FocusShare:P0} of the anchors{(focusSet.Count > 0 ? $" ({shown})" : "; none until an epoch evaluation picks some")}");
        }

        if (targetOptions.FocusShare > 0 && tagCentricSampler is null)
            throw new ArgumentException("--focus-share draws anchors for tag-centric sampling; it needs --sampling tag-centric.");
        if (RefreshFocus() is { } focusMessage)
            AnsiConsole.MarkupLine(char.ToUpperInvariant(focusMessage[0]) + focusMessage[1..] + "; re-picked after every epoch evaluation.");
        if (targetOptions.ErasurePairsPerStep > 0)
        {
            if (!targetOptions.DeliberateTargets || localizationWeight <= 0)
                throw new ArgumentException("--erasure-pairs-per-step picks its pairs and regions from the localization loss's; it needs --targets deliberate and --localization-weight > 0.");
            if (targetOptions.ErasureEvery < 1)
                throw new ArgumentException("--erasure-every must be at least 1.");
            AnsiConsole.MarkupLine(Markup.Escape(
                $"Erased views: {targetOptions.ErasurePairsPerStep} called localizable positive(s) " + (targetOptions.ErasureEvery == 1 ? "per step" : $"every {targetOptions.ErasureEvery} steps") + $", the region their map points at blanked vs. a same-size control (at most {ErasureContrast.MaxAreaFraction:P0} of the image); " +
                (targetOptions.ErasureLoss switch { ErasureLoss.Hinge => $"hinge, margin {targetOptions.ErasureMargin}", ErasureLoss.Negative => "object-blanked negative, control-blanked positive", _ => "object-blanked negative only" }) +
                $", weight {targetOptions.ErasureWeight}" + (targetOptions.ErasureAnchorWeight > 0 ? $", control held to the unblanked score at weight {targetOptions.ErasureAnchorWeight}" : "") + $", filled with {string.Join("/", targetOptions.ErasureFills).ToLowerInvariant()}."));
        }

        var isCharacter = vocabulary.Records.OrderBy(r => r.RowIndex).Select(r => TagScorer.IsCharacter(r.Tag)).ToArray();
        var soloRow = vocabulary.TryGet("solo", out var soloRecord) ? soloRecord.RowIndex : -1;
        if (targetOptions.CharacterExtentWeight > 0 || targetOptions.CharacterHideWeight > 0)
        {
            if (!targetOptions.DeliberateTargets || localizationWeight <= 0 || imageTower.ClassificationGlimpses)
                throw new ArgumentException("--character-extent-weight / --character-hide-weight train the localization map: they need deliberate targets, --localization-weight > 0 and no --classification-glimpses.");
            if (soloRow < 0)
                throw new ArgumentException("--character-extent-weight / --character-hide-weight pick images labelled solo, which isn't in the vocabulary.");
            if (targetOptions.CharacterExtentWeight > 0)
                AnsiConsole.MarkupLine(Markup.Escape(
                    $"Character extent: on images labelled solo with one character, the cells within {CharacterExtent.FigureBand} log-odds of one of its other labelled localizable tags' peaks are the figure; " +
                    $"the character's map is pushed to hold them within {CharacterExtent.ExtentBand} of its peak and the rest below (a {CharacterExtent.IgnoreRing}-cell ring ignored), weight {targetOptions.CharacterExtentWeight}."));
            if (targetOptions.CharacterHideWeight > 0)
                AnsiConsole.MarkupLine(Markup.Escape(
                    $"Character hide-and-seek: up to {targetOptions.CharacterHidePerStep} solo one-character images per micro-batch with the region the character's map points at blanked (at most {ErasureContrast.MaxAreaFraction:P0} of the image), the character a positive bag on that view's map, weight {targetOptions.CharacterHideWeight}."));
        }
        long characterExtentImages = 0, characterHiddenImages = 0;
        var characterRandom = new Random();

        if (targetOptions.TwinPaste > 0 && targetOptions.TwinEvery <= 0)
            throw new ArgumentException("--twin-paste pastes into matched twins; it needs --twin-every > 0.");
        if (targetOptions.SelfMapWeight > 0)
        {
            if (!targetOptions.DeliberateTargets || imageTower.ClassificationGlimpses)
                throw new ArgumentException("--self-map-weight trains a whole-image run's windows from its own map: it needs deliberate targets and a checkpoint without --classification-glimpses.");
            AnsiConsole.MarkupLineInterpolated(
                $"Self-map windows: for up to {SelfMapWindows.Positives} localizable positives per image, the 64/128px windows at the tag's map peak are a \"yes\" and {SelfMapWindows.NoWindows} of its bottom {SelfMapWindows.NoQuantile:P0} 64px windows a \"no\", per window (weight {targetOptions.SelfMapWeight}{(targetOptions.SelfMapDetachScaling ? ", no gradient into the logit scale/bias" : "")}).");
            if (targetOptions.SelfMapNegatives != "none")
                AnsiConsole.MarkupLineInterpolated(
                    $"Self-map negatives: for up to {SelfMapWindows.Positives} localizable tags per image that it is a certain negative for ({(targetOptions.SelfMapNegatives == "all" ? "context negatives included" : "context negatives left out")}), the 64/128px windows at the tag's map peak are a \"no\" (weight {targetOptions.SelfMapNegativeWeight}).");
        }
        else if (targetOptions.SelfMapNegatives != "none")
            throw new ArgumentException("--self-map-negatives adds negative images to the self-map windows: it needs --self-map-weight > 0.");
        if (targetOptions.MosaicGlimpseWeight > 0 && (targetOptions.MosaicsPerStep <= 0 || targetOptions.MosaicFullSizeTiles))
            throw new ArgumentException("--mosaic-glimpse-weight labels glimpses on half-size mosaic canvases; it needs --mosaics-per-step > 0 without --mosaic-full-size.");
        if (targetOptions.MosaicClassificationWeight > 0 && targetOptions.MosaicsPerStep <= 0)
            throw new ArgumentException("--mosaic-classification-weight classifies mosaic tiles; it needs --mosaics-per-step > 0.");
        var mosaicTileSize = targetOptions.MosaicFullSizeTiles ? resolvedInputSize : resolvedInputSize / 2;

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
                $"Mosaic: {Math.Min(targetOptions.MosaicsPerStep, batchSize / Mosaic.TilesPerMosaic)} canvas(es) of four {(targetOptions.MosaicFullSizeTiles ? "full-size" : "half-size")} batch images per step, weight {targetOptions.MosaicWeight} x the localization weight{(targetOptions.MosaicClassificationWeight > 0 ? $", each tile also classified (weight {targetOptions.MosaicClassificationWeight})" : "")}.");
        }
        if (targetOptions.MosaicLargeGrid > 0)
        {
            if (targetOptions.MosaicsPerStep <= 0 || targetOptions.MosaicFullSizeTiles || targetOptions.MosaicGlimpseWeight > 0 || targetOptions.MosaicClassificationWeight > 0)
                throw new ArgumentException("--mosaic-large-grid replaces some steps' half-size 2x2 mosaics: it needs --mosaics-per-step > 0, without --mosaic-full-size, --mosaic-glimpse-weight or --mosaic-classification-weight.");
            if (targetOptions.MosaicLargeGrid < 3 || batchSize < Mosaic.TilesOf(targetOptions.MosaicLargeGrid) || targetOptions.MosaicLargeGridEvery < 1 || resolvedInputSize % 4 != 0)
                throw new ArgumentException($"--mosaic-large-grid {targetOptions.MosaicLargeGrid}: needs a grid of 3 or more, --batch-size >= {Mosaic.TilesOf(targetOptions.MosaicLargeGrid)} (is {batchSize}), --mosaic-large-grid-every >= 1 and an input size divisible by 4.");
            AnsiConsole.MarkupLineInterpolated(
                $"Large mosaics: every {targetOptions.MosaicLargeGridEvery} step(s), one {targetOptions.MosaicLargeGrid}x{targetOptions.MosaicLargeGrid} canvas of batch images shrunk 4x ({targetOptions.MosaicLargeGrid * resolvedInputSize / 4}px) instead of the 2x2 mosaics.");
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

        Dictionary<int, int>? globalToLocal = null;
        var storedRandom = new Random();
        // Messages from the prefetcher's thread, logged by the training loop (the dashboard is the loop's).
        var prefetcherLog = new System.Collections.Concurrent.ConcurrentQueue<string>();
        IEnumerable<BatchRequest> TrainingRequests()
        {
            // Runs on the prefetcher's thread, between its reads, so refreshing the stores here never overlaps a training read
            // (HiResImageStore.Refresh swaps its index, so the in-process evaluation's reads are safe too).
            var refreshed = System.Diagnostics.Stopwatch.StartNew();
            Dictionary<int, int[]>? storedPools = null;
            while (true)
            {
                if ((hiResStores.Length > 0 || pendingStores.Count > 0) && refreshed.Elapsed >= StoreRefreshInterval)
                {
                    refreshed.Restart();
                    var added = hiResStores.Sum(store => store.Refresh());
                    foreach (var directory in pendingStores.Where(HiResImageStore.Exists).ToList())
                    {
                        var opened = HiResImageStore.Open(directory);
                        pendingStores.Remove(directory);
                        hiResStores = [.. hiResStores.Append(opened).OrderBy(store => store.LongSide)];
                        added += opened.Count;
                        prefetcherLog.Enqueue(Markup.Escape($"hi-res store opened: {directory} ({opened.LongSide}px, {opened.Count:N0} rows)"));
                    }
                    if (added > 0)
                    {
                        storedPools = null;
                        prefetcherLog.Enqueue(Markup.Escape($"hi-res stores: {added:N0} rows added or skipped ({string.Join(", ", hiResStores.Select(store => $"{store.LongSide}px {store.Count:N0}"))})"));
                    }
                }

                // A scale trained only on stored rows is skipped while no store holds any at its size (a 2048 store not pulled yet):
                // the cache enlarged 4x would show nothing new.
                storedPools ??= trainingScales is null ? [] : StoredPools();
                var scale = trainingScales?.Next();
                for (var redraw = 0; scale is { StoredShare: >= 1 } && storedPools[scale.Size].Length == 0 && redraw < 100; redraw++)
                    scale = trainingScales!.Next();
                var count = scale?.Batch ?? batchSize;
                var stored = Array.Empty<int>();
                if (scale is { StoredShare: > 0 } && storedPools[scale.Size] is { Length: > 0 } pool)
                    stored = Enumerable.Range(0, trainingScales!.StoredCount(scale)).Select(_ => pool[storedRandom.Next(pool.Length)]).ToArray();

                var sampledCount = count - stored.Length;
                var sampledBatch = sampledCount > 0 ? tagCentricSampler?.SampleBatch(sampledCount) : tagCentricSampler is null ? null : [];
                var local = sampledBatch is not null ? sampledBatch.Select(x => x.Image).ToArray() : sampledCount > 0 ? sampler.SampleBatch(sampledCount) : [];
                if (stored.Length > 0)
                {
                    // Stored rows go last, so a batch still starts the way the sampler built it (twins, context pairs).
                    local = [.. local, .. stored];
                    if (sampledBatch is not null)
                        sampledBatch = [.. sampledBatch, .. stored.Select(image => new SampledImage(image, AnchorTag: -1, IsHardNegative: false))];
                }
                var images = local.Select(li => trainingIndices[li]).ToArray();
                var fromStore = hiResStores.Length == 0 ? 0 : images.Count(row => ImageResize.Pick(hiResStores, row, scale?.Size ?? resolvedInputSize) is not null);
                yield return new BatchRequest(images, sampledBatch, scale?.Size ?? 0, fromStore, stored.Length);
            }

            // Per scale with a stored share: the training images (local indices) a store holds at that size or larger.
            Dictionary<int, int[]> StoredPools()
            {
                globalToLocal ??= trainingIndices.Select((global, local) => (global, local)).ToDictionary(x => x.global, x => x.local);
                return trainingScales!.Scales.Where(s => s.StoredShare > 0).ToDictionary(s => s.Size, s => hiResStores
                    .Where(store => store.LongSide >= s.Size).SelectMany(store => store.Rows).Distinct()
                    .Select(row => globalToLocal.TryGetValue(row, out var local) ? local : -1).Where(local => local >= 0).ToArray());
            }
        }

        // The samplers are only ever touched from the prefetcher's thread from here on.
        using var prefetcher = readWindow is not null
            ? readWindowAt is not null
                ? new BatchPrefetcher(TrainingRequests(), readWindowAt, resolvedInputSize, prefetchWindowBatches, 2 * prefetchWindowBatches,
                    pin: device.type != TorchSharp.DeviceType.CPU)
                : new BatchPrefetcher(TrainingRequests(), readWindow, resolvedInputSize, prefetchWindowBatches, 2 * prefetchWindowBatches,
                pin: device.type != TorchSharp.DeviceType.CPU)
            : null;

        // Global step the run resumes at; the loss history and snapshot evaluations are cut back to it.
        var resumedGlobalStep = (long)startEpoch * stepsPerEpoch + resumedStepsInEpoch;
        if (targetOptions.ClassificationPoolingTemperature is > 0 and var poolingTarget)
            imageTower.ClassificationPoolingSchedule = targetOptions.ClassificationPoolingAnnealSteps > 0 && targetOptions.ClassificationPoolingStartTemperature is > 0 and var poolingStart
                ? new ClassificationPooling.Schedule(
                    poolingTurnedOn || imageTower.ClassificationPoolingSchedule is null ? resumedGlobalStep : imageTower.ClassificationPoolingSchedule.StartStep,
                    poolingStart, poolingTarget, targetOptions.ClassificationPoolingAnnealSteps)
                : null;
        if (imageTower.ClassificationPoolingSchedule is { } poolingSchedule)
        {
            imageTower.ClassificationPoolingTemperature = poolingSchedule.At(resumedGlobalStep);
            AnsiConsole.MarkupLineInterpolated(
                $"Classification pooling temperature annealed from {poolingSchedule.StartTemperature} to {poolingSchedule.TargetTemperature} over {poolingSchedule.AnnealSteps:N0} steps from step {poolingSchedule.StartStep:N0}: {imageTower.ClassificationPoolingTemperature:0.###} now.");
        }
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
        if (gold is not null && families is not null)
            gold = LocalizationEvaluator.ExpandFamilies(gold, families, vocabulary);
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
                if (RefreshFocus() is { } focusMessage) dashboard.Log(focusMessage);
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
            foreach (var name in new[] { ExclusiveTagGroups.FileName, TagLocalizability.FileName, TagParentage.FileName })
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
        var erasureRandom = new Random();
        // Steps' losses waiting to be read back for the dashboard (see FlushSteps).
        const int StepReadEvery = 5;
        var pendingSteps = new List<(int Step, Tensor Losses, int Size)>();
        // What mixed scales actually did (null without --train-scales), summarized in the log every ScaleSummaryEvery steps.
        var scaleStats = trainingScales is not null ? new ScaleStats(trainingScales.Scales) : null;
        var flushClock = System.Diagnostics.Stopwatch.StartNew();
        var scaleSummaryStores = hiResStores.ToDictionary(store => store, store => store.Count);
        string StoreLine() => hiResStores.Length == 0 ? pendingStores.Count > 0 ? "stores pending" : "no stores" : "stores " + string.Join(", ", hiResStores.Select(store =>
        {
            var added = store.Count - scaleSummaryStores.GetValueOrDefault(store);
            scaleSummaryStores[store] = store.Count;
            return $"{store.LongSide}px {store.Count:N0}{(added > 0 ? $" (+{added:N0})" : "")}";
        }));
        // Reads the pending steps' losses back in one go and reports them (their progress went to the dashboard as each step
        // ended); returns the first step whose loss was NaN, if any.
        int? FlushSteps()
        {
            if (pendingSteps.Count == 0)
                return null;
            float[] values;
            using (var all = stack(pendingSteps.Select(p => p.Losses).ToArray()))
                values = all.cpu().data<float>().ToArray();
            // That read waited for the device to finish everything queued so far, uploads included.
            DeviceUpload.Release();
            if (scaleStats is not null)
            {
                scaleStats.Interval(flushClock.Elapsed.TotalMilliseconds, DeviceMemory.ProcessVramMb());
                flushClock.Restart();
                for (var k = 0; k < pendingSteps.Count; k++)
                    scaleStats.Loss(pendingSteps[k].Size, values[3 * k]);
            }
            int? nan = null;
            for (var k = 0; k < pendingSteps.Count; k++)
            {
                var (pendingStep, losses, _) = pendingSteps[k];
                losses.Dispose();
                if (nan is not null)
                    continue;
                if (float.IsNaN(values[3 * k]))
                {
                    nan = pendingStep;
                    continue;
                }
                dashboard.StepLosses(pendingStep, values[3 * k], values[3 * k + 1], values[3 * k + 2]);
            }
            pendingSteps.Clear();
            return nan;
        }

        // Labelled localizable positives the glimpse loss saw, how many of them no chosen window framed, and the chosen
        // windows' zooms: counted on the device, read at checkpoint saves.
        long glimpsePositives = 0;
        using var glimpseUnframed = zeros(1, dtype: ScalarType.Int64, device: device).DetachFromDisposeScope();
        using var glimpseZooms = zeros(Glimpses.Zooms.Length, dtype: ScalarType.Int64, device: device).DetachFromDisposeScope();
        var glimpseRandom = new Random();
        long erasureTotal = 0, erasureOnOther = 0;
        long croppedImages = 0;
        // Ctrl+C stops at the end of the current step with a checkpoint saved; a second Ctrl+C quits at once.
        // Being killed mid-step left GPU copies in flight on the Arc: the copy engine faulted on the freed
        // memory, and the next process to load a checkpoint onto the GPU hung until it was killed.
        using var stopRequest = new CancellationTokenSource();
        Console.CancelKeyPress += (_, e) =>
        {
            if (stopRequest.IsCancellationRequested)
                return;
            e.Cancel = true;
            stopRequest.Cancel();
            dashboard.Log("[yellow]Ctrl+C: stopping after this step, with a checkpoint saved. Ctrl+C again quits at once.[/]");
        };

        void TrainingLoop()
        {
            var profiler = StepProfiler.FromEnvironment(checkpointDir);
            // The stem's first parameter: its gradient is the last backward() computes.
            var profiledParameter = profiler is not null ? imageTower.parameters().First() : null;
            for (var epoch = startEpoch; epoch < epochs; epoch++)
            {
                var stepStart = epoch == startEpoch ? resumedStepsInEpoch : 0;
                dashboard.EpochStarted(epoch, stepStart);
                var epochWaitStart = prefetcher?.ConsumerWait ?? TimeSpan.Zero;
                var epochClock = System.Diagnostics.Stopwatch.StartNew();
                for (var step = stepStart; step < stepsPerEpoch; step++)
                {
                    profiler?.StepStarted();
                    while (prefetcherLog.TryDequeue(out var prefetcherMessage))
                        dashboard.Log(prefetcherMessage);
                    // Every tensor allocated in a step (pooled/spatial embeddings, the full
                    // tagTower forward pass over the whole vocabulary, each `loss +=`
                    // reassignment's intermediate) is native/CUDA memory that .NET's GC does
                    // not reclaim promptly. Without this scope those accumulate every step
                    // until VRAM is exhausted; the scope frees them all once `lossValue` has
                    // been extracted as a plain float below.
                    using var stepScope = NewDisposeScope();
                    if (imageTower.ClassificationPoolingSchedule is { } schedule)
                        imageTower.ClassificationPoolingTemperature = schedule.At((long)epoch * stepsPerEpoch + step);

                    SampledImage[]? sampled;
                    BatchRequest? stepRequest = null;
                    IReadOnlyList<int> globalBatchIndices;
                    Tensor? rawPixelBatch = null, pinnedPixelBatch = null;
                    IReadOnlyList<LetterboxBox> boxes;
                    if (prefetcher is not null)
                    {
                        if (!prefetcher.TryTake(out var loaded))
                            throw new InvalidOperationException("The training batch stream ended unexpectedly.");
                        stepRequest = loaded.Request;
                        sampled = loaded.Request.Sampled;
                        globalBatchIndices = loaded.Request.Images;
                        if (loaded.Pinned is not null)
                            pinnedPixelBatch = loaded.Pinned;
                        else
                        {
                            var loadedSize = loaded.Request.InputSize > 0 ? loaded.Request.InputSize : resolvedInputSize;
                            rawPixelBatch = tensor(loaded.Pixels, [loaded.Boxes.Length, 3, loadedSize, loadedSize]);
                        }
                        boxes = loaded.Boxes;
                    }
                    else
                    {
                        sampled = tagCentricSampler?.SampleBatch(batchSize);
                        var localBatchIndices = sampled is not null ? sampled.Select(x => x.Image).ToArray() : sampler.SampleBatch(batchSize);
                        globalBatchIndices = localBatchIndices.Select(li => trainingIndices[li]).ToList();
                        (rawPixelBatch, boxes) = loadBatch(globalBatchIndices);
                    }

                    using var pixelBatch = pinnedPixelBatch is not null ? DeviceUpload.UploadPinned(pinnedPixelBatch, device) : rawPixelBatch!.Upload(device);
                    profiler?.Mark("load batch", pixelBatch);
                    // The batch's canvas: the run's input size, or with --train-scales the size this micro-batch was drawn at.
                    var stepSize = (int)pixelBatch.shape[2];
                    var stepMosaicTileSize = targetOptions.MosaicFullSizeTiles ? stepSize : stepSize / 2;
                    // Each image counts the same in the accumulated gradient whatever its micro-batch's size (1 without --train-scales).
                    var batchShare = (double)pixelBatch.shape[0] / batchSize;

                    // Pasted twins (TwinPaste): the positive's region at its tag's map peak into one twin (which then carries the tag),
                    // a region from elsewhere into another (which still doesn't). The peak comes from a no-gradient pass of the positive alone.
                    Dictionary<int, int>? pastedTags = null;
                    if (targetOptions.TwinPaste > 0 && sampled is { Length: > 0 } && sampled[0].IsTwin)
                    {
                        var anchorTag = sampled[0].AnchorTag;
                        var twinRows = Enumerable.Range(1, sampled.Length - 1).Where(i => sampled[i].IsTwin).ToArray();
                        TwinPaste.Result pasteResult;
                        using (no_grad())
                        using (NewDisposeScope())
                        {
                            var positiveMask = SpatialMask.Build([boxes[0]], stepSize, stepSize / imageTower.OutputStride, stepSize / imageTower.OutputStride, device);
                            var (_, positiveMap) = imageTower.ForwardPooled(pixelBatch.narrow(0, 0, 1), positiveMask);
                            using var anchorRow = tensor(new long[] { anchorTag }, device: device);
                            var tagRow = tagTower.forward(anchorRow)[0];
                            pasteResult = TwinPaste.Apply(pixelBatch, boxes, positiveMap, positiveMask, tagRow, logitScaleBias.ToLogitScaling(), 0, twinRows,
                                targetOptions.TwinPaste, imageTower.OutputStride, mosaicRandom);
                        }
                        pastesGated += pasteResult.Gated ? 1 : 0;
                        pastesMade += pasteResult.Pasted.Length;
                        pastedTags = pasteResult.Pasted.ToDictionary(i => i, _ => anchorTag);
                        // The controls are certain negatives for the tag, the same as a hard negative.
                        foreach (var control in pasteResult.Controls)
                            sampled[control] = sampled[control] with { IsHardNegative = true };
                        profiler?.Mark("twin paste", pixelBatch);
                    }

                    // Crop consistency's extra forward runs first, without gradients, keeping only
                    // the crops' pooled embeddings ([B, D]). Run after the main forward, its
                    // activations sat on top of the ones held for backward() and ran out of memory.
                    // Only images carrying a trusted tag are cropped (the rest would train
                    // nothing), so the pass costs in proportion to how much the teacher is trusted.
                    CellWindow[]? cropWindows = null;
                    Tensor? cropPooled = null;
                    var cropGrid = stepSize / imageTower.OutputStride;
                    var cropImages = targetOptions.CropConsistencyWeight > 0 && cropGate.TrustedCount > 0 && localizationWeight > 0 && targetOptions.DeliberateTargets
                        ? globalBatchIndices.Select((g, b) => (g, b)).Where(x => imageTagRows[x.g].Any(r => (uint)r < (uint)cropGate.Trusted.Length && cropGate.Trusted[r])).Select(x => (long)x.b).ToArray()
                        : [];
                    croppedImages += cropImages.Length;
                    if (cropImages.Length > 0)
                    {
                        cropWindows = CropConsistency.SampleWindows(boxes, stepSize, cropGrid, cropGrid, cropRandom);
                        using (no_grad())
                        using (var cropScope = NewDisposeScope())
                        {
                            var selected = tensor(cropImages, device: device);
                            var crops = CropConsistency.CropPixels(pixelBatch.index_select(0, selected), cropImages.Select(b => cropWindows[b]).ToArray(), cropGrid);
                            var pooledCrops = imageTower.forward(crops).Pooled;
                            cropPooled = zeros([pixelBatch.shape[0], pooledCrops.shape[1]], device: device)
                                .index_copy(0, selected, pooledCrops).MoveToOuterDisposeScope();
                        }
                        profiler?.Mark("crop teacher forward", cropPooled);
                    }

                    // Classification pools the full map; the localization losses below use
                    // the localization map (the same tensor unless LocalizationFromFusedOnly). Unless
                    // something below needs the full map itself, ForwardPooled computes its masked
                    // pool without building it ([batch, 512, 128, 128] per add at 512px).
                    var needsFullMap = (imageTower.ClassificationPoolingTemperature > 0 && !imageTower.ClassificationGlimpses) || targetOptions.MosaicClassificationWeight > 0;
                    var mapSize = stepSize / imageTower.OutputStride;
                    var spatialMask = SpatialMask.Build(boxes, stepSize, mapSize, mapSize, device);
                    Tensor? spatial = null;
                    Tensor localizationMap, pooled;
                    // With a fine level, the map in its parts; localizationMap is then its coarse part, which nothing reads
                    // (the options that would are refused at startup).
                    FineLocalizationMap? fineMap = null;
                    if (needsFullMap)
                    {
                        (spatial, localizationMap) = imageTower.ForwardMaps(pixelBatch);
                        pooled = ImageTower.MaskedPool(spatial, spatialMask);
                    }
                    else if (imageTower.Fine is not null)
                    {
                        (pooled, fineMap) = imageTower.ForwardPooledFine(pixelBatch, spatialMask);
                        localizationMap = fineMap.Coarse;
                    }
                    else
                        (pooled, localizationMap) = imageTower.ForwardPooled(pixelBatch, spatialMask);
                    profiler?.Mark("forward", pooled);
                    var tagEmbeddings = tagTower.forward(allTagIndices);
                    profiler?.Mark("tag tower forward", tagEmbeddings);

                    var batchTagRows = globalBatchIndices.Select((g, b) => pastedTags is not null && pastedTags.TryGetValue(b, out var pastedTag)
                        ? (IReadOnlyList<int>)imageTagRows[g].Append(pastedTag).ToArray() : imageTagRows[g]).ToList();
                    using var labels = BatchLabelBuilder.Build(batchTagRows, vocabulary.Records.Count).Upload(device);
                    using var cpuTargets = targetOptions.DeliberateTargets
                        ? pairWeightBuilder.Build(batchTagRows, sampled, imageSites is not null ? globalBatchIndices.Select(g => imageSites[g]).ToArray() : null)
                        : null;
                    using var targets = cpuTargets?.To(device);
                    profiler?.Mark("targets (CPU) + upload", targets?.Weights ?? labels);

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
                        glimpseTowerOptimizer?.zero_grad();
                        fineOptimizer?.zero_grad();
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
                        var localizedGradientScale = (float)(localizationWeight * batchShare / gradientAccumulationSteps);
                        localizedLossValue = SigmoidContrastiveLoss.ComputeLocalizedChunked(
                            localizationMap, tagEmbeddings, labels, (float)localizationTemperature,
                            localizationTagChunkSize, localizedGradientScale, spatialMask, logitScaleBias);
                    }

                    Tensor loss;
                    // With per-tag pooling, the classifier's own scores (raw, detached): what the localization loss's
                    // highest-scoring wrong tags are ranked by. The averaged embedding's are never trained for those tags.
                    Tensor? classifierScores = null;
                    Tensor? twinTerm = null;
                    if (targets is not null && imageTower.ClassificationPoolingTemperature > 0 && imageTower.ClassificationGlimpses)
                    {
                        // Glimpses: the windows the model picks from its own localization map, as evaluation does. A positive
                        // none of them frames is left to the localization loss (GlimpseUnframedPositiveWeight).
                        // Early in a fresh run the map can't point yet, so the model's own choice would rarely frame a positive
                        // and the glimpse classifier would see almost none: a share of the windows (half at step 0, none after
                        // GlimpseLabelFramingSteps) frames labelled positives first.
                        // GlimpseLabelFramingFloor keeps some of them for good, so a tag the proposals rarely suspect still gets looked at.
                        var labelShare = Math.Max((double)targetOptions.GlimpseLabelFramingFloor / imageTower.GlimpseCount,
                            targetOptions.GlimpseLabelFramingSteps > 0
                                ? 0.5 * Math.Max(0, 1 - ((double)epoch * stepsPerEpoch + step) / targetOptions.GlimpseLabelFramingSteps)
                                : 0);
                        // Everything below stays on the device: no step waits for a read back (labels and boxes go up, nothing comes down).
                        using var contentBoxes = Glimpses.ContentBoxes(boxes, device);
                        Tensor? firstTags = null, firstValid = null, firstRows = null;
                        var labelTags = (int)Math.Round(labelShare * imageTower.GlimpseCount + Glimpses.Zooms.Length - 1) / Glimpses.Zooms.Length;
                        if (labelTags > 0)
                        {
                            var picked = LocalizablePositives(cpuTargets!, localizable).Select(p => p.OrderBy(_ => glimpseRandom.Next()).Take(labelTags).ToList()).ToArray();
                            if (picked.Any(p => p.Count > 0))
                            {
                                var up = PaddedRows.Upload(picked, vocabulary.Records.Count, device);
                                firstTags = tagEmbeddings.index_select(0, up.Rows.flatten()).reshape(up.Rows.shape[0], up.Rows.shape[1], -1);
                                firstValid = up.Valid;
                                firstRows = up.Rows;
                            }
                        }
                        // What the image might hold, from the whole image (context included): it picks where to look, and
                        // learns from the labels (GlimpseProposalWeight); the tags' scores come from the glimpses.
                        var proposals = imageTower.Proposals(pooled);
                        using var choice = Glimpses.Propose(proposals, localizationMap, tagEmbeddings.index_select(0, classificationPooledRows), spatialMask, contentBoxes,
                            stepSize, imageTower.OutputStride, imageTower.GlimpseCount, firstTags: firstTags, firstValid: firstValid);
                        profiler?.Mark("glimpse selection", choice.Valid);
                        var (cells, validCells) = Glimpses.Embed(imageTower, pixelBatch, contentBoxes, choice);
                        profiler?.Mark("glimpse forward", cells);
                        var glimpseWeightTensor = WeightUnframedPositives(cpuTargets!, targets, localizable, localizationMap, tagEmbeddings, spatialMask, contentBoxes,
                            stepSize, imageTower.OutputStride, choice, targetOptions.GlimpseUnframedPositiveWeight, glimpseUnframed, ref glimpsePositives);
                        using (no_grad())
                            glimpseZooms.add_(Glimpses.ZoomCounts(choice));
                        profiler?.Mark("glimpse framing", glimpseWeightTensor);
                        // Framed-positive pooling: a labelled positive the label framing looked at is scored from the close windows
                        // (below 256px, and within the pooled zooms) that frame it, so those windows get its "yes" (WSDDN/OICR's
                        // asymmetry: positives on their own regions, negatives on every window). The 256px window holds a quarter of
                        // the image and with it the context, and was never the most readable size on the gold probes. Everything
                        // else pools every window; so does a positive no close window frames.
                        Tensor? windowMask = null;
                        if (targetOptions.GlimpseFramedPositivePooling && firstRows is not null)
                        {
                            using var _ = no_grad();
                            var (framedX, framedY) = Glimpses.Peaks(localizationMap, firstTags!, spatialMask, imageTower.OutputStride);
                            var close = validCells.logical_and(choice.Zoom.lt(Glimpses.Zooms.Length - 1)); // [B, K]
                            var framing = Glimpses.FramingWindows(choice, framedX, framedY, contentBoxes, stepSize).logical_and(close.unsqueeze(1)); // [B, P, K]
                            windowMask = validCells.unsqueeze(1).expand(-1, classificationPooledRows.shape[0], -1).clone(); // [B, L, K]
                            var columns = pooledColumnOf.index_select(0, firstRows.flatten()).view_as(firstRows); // [B, P]
                            var use = firstValid!.logical_and(columns.ge(0)).logical_and(framing.any(2));
                            var images = arange(columns.shape[0], device: device).unsqueeze(1).expand_as(columns);
                            windowMask.index_put_(framing.index(TensorIndex.Tensor(use)), TensorIndex.Tensor(images.index(TensorIndex.Tensor(use))), TensorIndex.Tensor(columns.index(TensorIndex.Tensor(use))));
                        }
                        var raw = ClassificationPooling.RawScores(pooled, cells, validCells, tagEmbeddings, classificationPooledRows, imageTower.ClassificationPoolingTemperature, imageTower.ClassificationPoolingTopK, imageTower.ClassificationPoolingBottomWeight, windowMask);
                        classifierScores = raw.detach();
                        // Feature split (Singh et al. 2020): the chosen pairs score with the context half held at its running mean
                        // and the tags' context half detached, so their gradient reaches the object half only. Mining and every
                        // other pair use the whole embedding, as evaluation does by default.
                        if (featureSplitTags is not null)
                        {
                            var objectDims = imageTower.GlimpseObjectDims;
                            var contextDims = cells.shape[1] - objectDims;
                            using (no_grad())
                            {
                                var real = validCells.unsqueeze(1).to_type(cells.dtype);
                                var batchMean = (cells.narrow(1, objectDims, contextDims) * real).sum([0, 2]) / real.sum().clamp_min(1f);
                                if (glimpseContextMean is null)
                                    glimpseContextMean = batchMean.clone().DetachFromDisposeScope();
                                else
                                    glimpseContextMean.mul_(0.9f).add_(batchMean, alpha: 0.1f);
                            }
                            var split = FeatureSplitPairs(cpuTargets!, batchTagRows, featureSplitTags, outOfContextPriors!, targetOptions.GlimpseFeatureSplitNegatives,
                                ref splitPositives, ref splitExclusive, ref splitNegatives);
                            if (split is not null)
                            {
                                using var splitMask = tensor(split.Value.Mask, [split.Value.Batch, split.Value.Vocabulary]).Upload(device);
                                var splitTags = cat([tagEmbeddings.narrow(1, 0, objectDims), tagEmbeddings.narrow(1, objectDims, contextDims).detach()], 1);
                                var splitRaw = ClassificationPooling.RawScores(pooled, Glimpses.ObjectOnly(cells, objectDims, glimpseContextMean!), validCells, splitTags,
                                    classificationPooledRows, imageTower.ClassificationPoolingTemperature, imageTower.ClassificationPoolingTopK, imageTower.ClassificationPoolingBottomWeight, windowMask);
                                raw = where(splitMask, splitRaw, raw);
                                if (targetOptions.GlimpseFeatureSplitWeight != 1f)
                                    glimpseWeightTensor = glimpseWeightTensor * where(splitMask.logical_and(targets.Labels.gt(0)),
                                        full_like(glimpseWeightTensor, targetOptions.GlimpseFeatureSplitWeight), ones_like(glimpseWeightTensor));
                            }
                        }
                        windowMask?.Dispose();
                        loss = SigmoidContrastiveLoss.ComputeWeightedScores(raw,
                            new PairTargets(targets.Labels, glimpseWeightTensor, targets.FocalMask, targets.PositiveWeightSum, targets.LogitOffsets),
                            logitScaleBias, targetOptions.NegativeGamma, targetOptions.NegativeClip);
                        // Matched twins: the positive's windows most like its twins' are a "no" for the tag (TwinWindows).
                        if (targetOptions.TwinWeight > 0 && sampled is not null && Array.FindIndex(sampled, x => x.IsTwin) is var twinPositive and >= 0)
                        {
                            var twinRows = Enumerable.Range(0, sampled.Length).Where(i => sampled[i].IsTwin && i != twinPositive).Select(i => (long)i).ToList();
                            if (twinRows.Count >= TwinWindows.Matches)
                            {
                                var padded = Enumerable.Range(0, (int)pixelBatch.shape[0] - 1).Select(i => i < twinRows.Count ? twinRows[i] : twinRows[0]).ToArray();
                                using var twins = tensor(padded).Upload(device);
                                using var twinValid = tensor(Enumerable.Range(0, padded.Length).Select(i => i < twinRows.Count).ToArray()).Upload(device);
                                var tag = tagEmbeddings[sampled[twinPositive].AnchorTag];
                                var (peakX, peakY) = Glimpses.Peaks(localizationMap.narrow(0, twinPositive, 1), tag.detach().view(1, 1, -1), spatialMask.narrow(0, twinPositive, 1), imageTower.OutputStride);
                                using var no = TwinWindows.NoWindows(cells, validCells, twinPositive, twins, twinValid, choice, peakX, peakY);
                                twinTerm = TwinWindows.Term(cells, twinPositive, tag, no, logitScaleBias, targetOptions.TwinDetachScaling);
                                loss = loss + targetOptions.TwinWeight * twinTerm;
                            }
                        }
                        // Context pairs (TagCentricBatchSampler.ContextPairsPerAnchor): the image with the tag must outscore the one of
                        // the same character/artist/series without it, by a margin in logits; identity is in both, so it can't pay.
                        if (targetOptions.ContextPairWeight > 0 && sampled is not null)
                        {
                            var pairs = sampled.Select((s, i) => (s, i)).Where(x => x.s.ContextPair >= 0 && localizable[x.s.AnchorTag])
                                .GroupBy(x => x.s.ContextPair).Where(g => g.Count() == 2)
                                .Select(g => (With: g.First(x => !x.s.IsContextNegative).i, Without: g.First(x => x.s.IsContextNegative).i, Tag: g.First().s.AnchorTag))
                                .ToArray();
                            if (pairs.Length > 0)
                            {
                                using var withRows = tensor(pairs.Select(p => (long)p.With).ToArray()).Upload(device);
                                using var withoutRows = tensor(pairs.Select(p => (long)p.Without).ToArray()).Upload(device);
                                using var pairTags = tensor(pairs.Select(p => (long)p.Tag).ToArray()).Upload(device).unsqueeze(1);
                                var withLogits = logitScaleBias.forward(raw.index_select(0, withRows).gather(1, pairTags));
                                var withoutLogits = logitScaleBias.forward(raw.index_select(0, withoutRows).gather(1, pairTags));
                                loss = loss + targetOptions.ContextPairWeight * nn.functional.relu(targetOptions.ContextPairMargin - (withLogits - withoutLogits)).mean();
                            }
                        }
                        if (targetOptions.GlimpseProposalWeight > 0)
                        {
                            // The proposals learn every localizable tag from the labels, as the whole-image classifier does the rest.
                            using var proposalWeights = targets.Weights * localizableMask!;
                            // Optionally without a gradient into the shared tag embeddings, which the localization map scores with:
                            // then only the proposal head learns the context (GlimpseProposalDetachTags).
                            var proposalTags = targetOptions.GlimpseProposalDetachTags ? tagEmbeddings.detach() : tagEmbeddings;
                            loss = loss + targetOptions.GlimpseProposalWeight * SigmoidContrastiveLoss.ComputeWeightedScores(proposals.matmul(proposalTags.t()),
                                new PairTargets(targets.Labels, proposalWeights, targets.FocalMask, targets.PositiveWeightSum, targets.LogitOffsets),
                                logitScaleBias, targetOptions.NegativeGamma, targetOptions.NegativeClip);
                        }
                        if (locality is not null)
                        {
                            var positiveRows = LocalityNegatives.Cap(LocalizablePositives(cpuTargets!, localizable), glimpseRandom);
                            if (positiveRows.Any(p => p.Count > 1))
                            {
                                var up = PaddedRows.Upload(positiveRows, vocabulary.Records.Count, device, padTo: LocalityNegatives.PositivesPerImage);
                                loss = loss + targetOptions.LocalityNegativeWeight * locality.Step(cells, choice, up.Rows, up.Valid, tagEmbeddings, logitScaleBias);
                            }
                        }
                        profiler?.Mark("glimpses", loss);
                    }
                    else if (targets is not null && imageTower.ClassificationPoolingTemperature > 0)
                    {
                        var (cells, validCells) = ClassificationPooling.Cells(spatial!, spatialMask);
                        using var trainedRows = ClassificationPooling.TrainedRows(cells, validCells, tagEmbeddings, classificationPooledRowArray, cpuTargets!,
                            imageTower.ClassificationPoolingTemperature, targetOptions.ClassificationPoolingMinedNegatives, imageTower.ClassificationPoolingTopK, imageTower.ClassificationPoolingBottomWeight);
                        var raw = ClassificationPooling.RawScores(pooled, cells, validCells, tagEmbeddings, trainedRows, imageTower.ClassificationPoolingTemperature, imageTower.ClassificationPoolingTopK, imageTower.ClassificationPoolingBottomWeight);
                        classifierScores = raw.detach();
                        loss = SigmoidContrastiveLoss.ComputeWeightedScores(raw,
                            targets, logitScaleBias, targetOptions.NegativeGamma, targetOptions.NegativeClip);
                    }
                    else
                        loss = targets is not null
                            ? SigmoidContrastiveLoss.ComputeWeighted(pooled, tagEmbeddings, targets, logitScaleBias, targetOptions.NegativeGamma, targetOptions.NegativeClip)
                            : SigmoidContrastiveLoss.Compute(pooled, tagEmbeddings, labels, logitScaleBias);
                    profiler?.Mark("tag loss", loss);
                    // The whole-image tag term alone, for the dashboard's loss breakdown (with the twin term, which is in it).
                    var tagLoss = loss;
                    // Self-map windows (SelfMapWindows): windows at a positive's map peak "yes", from its map's bottom 40% "no", per window.
                    if (targetOptions.SelfMapWeight > 0 && cpuTargets is not null && !imageTower.ClassificationGlimpses)
                    {
                        var picked = LocalizablePositives(cpuTargets, localizable).Select(p => p.OrderBy(_ => glimpseRandom.Next()).Take(SelfMapWindows.Positives).ToList()).ToArray();
                        // --self-map-negatives: tags the image certainly doesn't carry, looked at where their map peaks.
                        var negatives = targetOptions.SelfMapNegatives != "none"
                            ? LocalizableCertainNegatives(cpuTargets, localizable, targetOptions.SelfMapNegatives == "all", targetOptions.ContextNegativeWeight, targetOptions.CertainNegativeWeight)
                                .Select(p => p.OrderBy(_ => glimpseRandom.Next()).Take(SelfMapWindows.Positives).ToList()).ToArray()
                            : null;
                        if (picked.Any(p => p.Count > 0) || negatives?.Any(p => p.Count > 0) == true)
                        {
                            var up = PaddedRows.Upload(picked, vocabulary.Records.Count, device, padTo: SelfMapWindows.Positives);
                            var down = negatives is not null ? PaddedRows.Upload(negatives, vocabulary.Records.Count, device, padTo: SelfMapWindows.Positives) : null;
                            var rows = down is not null ? cat([up.Rows, down.Rows], 1) : up.Rows;
                            var rowsValid = down is not null ? cat([up.Valid, down.Valid], 1) : up.Valid;
                            var slotTags = tagEmbeddings.index_select(0, rows.flatten()).reshape(rows.shape[0], rows.shape[1], -1);
                            using var contentBoxes = Glimpses.ContentBoxes(boxes, device);
                            var (choice, windowLabels) = SelfMapWindows.Windows(localizationMap, spatialMask, slotTags, rowsValid, contentBoxes, stepSize, imageTower.OutputStride);
                            using (choice)
                            using (windowLabels)
                            {
                                var (windowCells, _) = Glimpses.Embed(imageTower, pixelBatch, contentBoxes, choice, requireZoomCodes: false);
                                var positiveWindows = SelfMapWindows.Positives * SelfMapWindows.PerPositive;
                                loss = loss + targetOptions.SelfMapWeight * SelfMapWindows.Term(windowCells.narrow(2, 0, positiveWindows), slotTags.narrow(1, 0, SelfMapWindows.Positives),
                                    windowLabels.narrow(1, 0, positiveWindows), choice.Valid.narrow(1, 0, positiveWindows), logitScaleBias, targetOptions.SelfMapDetachScaling);
                                if (down is not null)
                                {
                                    // Only the peak windows of a negative slot count; its bottom windows say nothing new.
                                    using var peakOnly = SelfMapWindows.PeakWindowMask(SelfMapWindows.Positives, device);
                                    var negativeValid = choice.Valid.narrow(1, positiveWindows, positiveWindows).logical_and(peakOnly);
                                    loss = loss + targetOptions.SelfMapNegativeWeight * SelfMapWindows.Term(windowCells.narrow(2, positiveWindows, positiveWindows),
                                        slotTags.narrow(1, SelfMapWindows.Positives, SelfMapWindows.Positives), zeros_like(windowLabels.narrow(1, 0, positiveWindows)), negativeValid,
                                        logitScaleBias, targetOptions.SelfMapDetachScaling);
                                    selfMapNegativeTags += negatives!.Sum(p => p.Count);
                                }
                            }
                            selfMapSteps++;
                        }
                        profiler?.Mark("self-map windows", loss);
                    }
                    Tensor? sparseLocalized = null;
                    Tensor? cropTerm = null;
                    Tensor? mosaicTerm = null;
                    Tensor? tileClassificationTerm = null;
                    Tensor? mosaicGlimpseTerm = null;
                    Tensor? erasureTerm = null;

                    // Deliberate targets: the per-location loss only on pairs where "where is
                    // it?" has an answer (LocalizationPairSelector) -- ~100 pairs per image
                    // instead of the whole vocabulary, so it rides in the same backward pass.
                    if (localizationWeight > 0 && cpuTargets is not null)
                    {
                        // Erased views read the scores back anyway; otherwise the pairs are picked on the device, so the step
                        // doesn't wait for the forward to finish here.
                        var erasureThisStep = targetOptions.ErasurePairsPerStep > 0 && stepsThisProcess % targetOptions.ErasureEvery == 0;
                        Tensor wholeImageScores;
                        using (no_grad())
                            wholeImageScores = (classifierScores ?? pooled.matmul(tagEmbeddings.t())).detach();
                        var wholeImageLogits = erasureThisStep ? wholeImageScores.cpu().data<float>().ToArray() : [];
                        using var cpuPairs = erasureThisStep ? LocalizationPairSelector.Select(cpuTargets, wholeImageLogits, localizable, targetOptions.LocalizationMinedNegatives) : null;
                        List<int>[] labelledPairs = [];
                        using var devicePairs = cpuPairs?.To(device)
                            ?? LocalizationPairSelector.SelectOnDevice(cpuTargets, targets!, wholeImageScores, localizable, targetOptions.LocalizationMinedNegatives, out labelledPairs);
                        profiler?.Mark("localization pair selection", devicePairs.TagRows);

                        // Erased views: the occlusion check as a loss (ErasureContrast). The region comes from the
                        // tag's own map, detached, so blanking can't teach the map to dodge it. Picked here, while
                        // the GPU has only the forward behind it, so reading the maps back doesn't wait on the
                        // losses queued below; the views then go through the tower with the mosaic canvases.
                        var erasurePairs = new List<ErasurePair>();
                        Tensor? erasureVariants = null, erasureMask = null, variantPooled = null;
                        if (erasureThisStep)
                        {
                            var candidates = ErasureContrast.Candidates(cpuTargets, wholeImageLogits, logitScaleBias.Scale, logitScaleBias.Bias, localizable,
                                targetOptions.ErasurePairsPerStep, erasureRandom);
                            if (candidates.Count > 0)
                            {
                                // Each candidate's map, then its other tag's (a zero map when it has none, unused).
                                float[] heat;
                                using (no_grad())
                                {
                                    using var images = tensor(candidates.SelectMany(c => new long[] { c.Image, c.Image }).ToArray(), device: device);
                                    using var rows = tensor(candidates.SelectMany(c => new long[] { c.Tag, Math.Max(0, c.Other) }).ToArray(), device: device);
                                    var maps = localizationMap.detach().index_select(0, images);
                                    var embeddings = tagEmbeddings.detach().index_select(0, rows);
                                    heat = (maps * embeddings.unsqueeze(2).unsqueeze(3)).sum(1).to_type(ScalarType.Float32).cpu().data<float>().ToArray();
                                }

                                var grid = (int)localizationMap.shape[2];
                                var cells = grid * grid;
                                var cellSize = stepSize / grid;
                                using var cpuMask = SpatialMask.Build(boxes, stepSize, grid, grid);
                                var validCells = cpuMask.data<float>().ToArray();
                                for (var k = 0; k < candidates.Count; k++)
                                {
                                    var (image, tag, other) = candidates[k];
                                    var content = new[] { boxes[image].X, boxes[image].Y, boxes[image].Width, boxes[image].Height };
                                    var valid = validCells.AsSpan(image * cells, cells).ToArray().Select(v => v > 0).ToArray();
                                    if (ErasureContrast.Region(heat.AsSpan(2 * k * cells, cells).ToArray(), valid, grid, cellSize, content) is not { } region)
                                        continue;
                                    var control = other >= 0 ? ErasureContrast.ControlOnOther(heat.AsSpan((2 * k + 1) * cells, cells).ToArray(), valid, grid, cellSize, content, region) : null;
                                    if (control is not null)
                                        erasureOnOther++;
                                    else if (LocalizationEvaluator.ControlRectangles([region], content, erasureRandom) is [var placed])
                                        control = placed;
                                    if (control is not null)
                                    {
                                        erasurePairs.Add(new(image, tag, region, control));
                                        erasureTotal++;
                                    }
                                }
                            }

                            if (erasurePairs.Count > 0)
                            {
                                erasureVariants = ErasureContrast.Variants(pixelBatch, erasurePairs, stepSize,
                                    erasurePairs.Select(_ => targetOptions.ErasureFills[erasureRandom.Next(targetOptions.ErasureFills.Count)]).ToList());
                                using var variantRows = tensor(erasurePairs.SelectMany(p => new long[] { p.Image, p.Image }).ToArray(), device: device);
                                erasureMask = spatialMask.index_select(0, variantRows);
                            }

                            profiler?.Mark("erasure selection", erasureVariants);
                        }

                        var localized = fineMap is not null
                            ? SigmoidContrastiveLoss.ComputeLocalizedPairs(fineMap, tagEmbeddings, devicePairs, (float)localizationTemperature, spatialMask, logitScaleBias,
                                targetOptions.NegativeGamma, targetOptions.NegativeClip)
                            : SigmoidContrastiveLoss.ComputeLocalizedPairs(
                                localizationMap, tagEmbeddings, devicePairs, (float)localizationTemperature, spatialMask, logitScaleBias,
                                targetOptions.NegativeGamma, targetOptions.NegativeClip,
                                targetOptions.LocalizationBackgroundWeight, targetOptions.LocalizationBackgroundFraction);
                        loss += localizationWeight * localized;
                        sparseLocalized = localized;
                        profiler?.Mark("localization loss", loss);

                        // Characters' maps cover the character, not only the face (CharacterExtent).
                        if (targetOptions.CharacterExtentWeight > 0)
                        {
                            var extentImages = CharacterExtent.Select(cpuTargets, localizable, isCharacter, soloRow, max: 0, needOthers: true, characterRandom);
                            if (extentImages.Count > 0)
                            {
                                loss += targetOptions.CharacterExtentWeight * CharacterExtent.ExtentTerm(localizationMap, spatialMask, tagEmbeddings, logitScaleBias, extentImages);
                                characterExtentImages += extentImages.Count;
                            }
                            profiler?.Mark("character extent", loss);
                        }
                        if (targetOptions.CharacterHideWeight > 0)
                        {
                            var hideImages = CharacterExtent.Select(cpuTargets, localizable, isCharacter, soloRow, targetOptions.CharacterHidePerStep, needOthers: false, characterRandom);
                            if (hideImages.Count > 0)
                            {
                                // The region comes from the character's own map, detached (read back: the rectangles are cut on the CPU).
                                var grid = (int)localizationMap.shape[2];
                                var cells = grid * grid;
                                var cellSize = stepSize / grid;
                                float[] heat;
                                using (no_grad())
                                {
                                    using var images = tensor(hideImages.Select(h => (long)h.Image).ToArray(), device: device);
                                    using var rows = tensor(hideImages.Select(h => (long)h.Character).ToArray(), device: device);
                                    heat = (localizationMap.detach().index_select(0, images) * tagEmbeddings.detach().index_select(0, rows).unsqueeze(2).unsqueeze(3))
                                        .sum(1).to_type(ScalarType.Float32).cpu().data<float>().ToArray();
                                }
                                using var cpuMask = SpatialMask.Build(boxes, stepSize, grid, grid);
                                var validCells = cpuMask.data<float>().ToArray();
                                var kept = new List<CharacterExtentImage>();
                                var regions = new List<int[]>();
                                for (var k = 0; k < hideImages.Count; k++)
                                {
                                    var image = hideImages[k].Image;
                                    var content = new[] { boxes[image].X, boxes[image].Y, boxes[image].Width, boxes[image].Height };
                                    var valid = validCells.AsSpan(image * cells, cells).ToArray().Select(v => v > 0).ToArray();
                                    if (ErasureContrast.Region(heat.AsSpan(k * cells, cells).ToArray(), valid, grid, cellSize, content) is { } region)
                                    {
                                        kept.Add(hideImages[k]);
                                        regions.Add(region);
                                    }
                                }
                                if (kept.Count > 0)
                                {
                                    using var hidden = CharacterExtent.Hidden(pixelBatch, kept, regions, stepSize,
                                        kept.Select(_ => OcclusionFills.Training[characterRandom.Next(OcclusionFills.Training.Length)]).ToList());
                                    using var hiddenRows = tensor(kept.Select(h => (long)h.Image).ToArray(), device: device);
                                    using var hiddenMask = spatialMask.index_select(0, hiddenRows);
                                    var (_, hiddenMap) = imageTower.ForwardPooled(hidden, hiddenMask);
                                    loss += targetOptions.CharacterHideWeight * CharacterExtent.HideTerm(hiddenMap, hiddenMask, tagEmbeddings, logitScaleBias, kept, (float)localizationTemperature);
                                    characterHiddenImages += kept.Count;
                                }
                            }
                            profiler?.Mark("character hide-and-seek", loss);
                        }

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
                        // Their own pass, not rows of the batch's: measured on the B70, one forward+backward of
                        // 12 images ran ~5% slower than one of 10 plus one of 2 (241 vs 230 ms/step).
                        // --mosaic-large-grid: on some steps one N x N canvas of quarter-size images instead.
                        var mosaicGrid = targetOptions.MosaicLargeGrid > 0 && stepsThisProcess % targetOptions.MosaicLargeGridEvery == 0 ? targetOptions.MosaicLargeGrid : 2;
                        var stepTileSize = mosaicGrid == 2 ? stepMosaicTileSize : stepSize / 4;
                        var tileImages = targetOptions.MosaicsPerStep > 0
                            ? Mosaic.PickTiles((int)pixelBatch.shape[0], mosaicGrid == 2 ? targetOptions.MosaicsPerStep : 1, mosaicRandom, sampled, focusSet, mosaicGrid)
                            : [];
                        if (tileImages.Length > 0)
                        {
                            var canvases = Mosaic.Compose(pixelBatch, tileImages, stepSize, stepTileSize, mosaicGrid);
                            profiler?.Mark("mosaic compose", canvases);
                            Tensor? mosaicSpatial = null;
                            Tensor mosaicMap;
                            Tensor? mosaicPooled = null;
                            FineLocalizationMap? mosaicFine = null;
                            if (needsFullMap)
                                (mosaicSpatial, mosaicMap) = imageTower.ForwardMaps(canvases);
                            else if (fineMap is not null)
                            {
                                // Erased views are refused with a fine level, so the canvases go alone.
                                using var canvasMask = ones(canvases.shape[0], 1, canvases.shape[2] / imageTower.OutputStride, canvases.shape[3] / imageTower.OutputStride, device: device);
                                (mosaicPooled, mosaicFine) = imageTower.ForwardPooledFine(canvases, canvasMask);
                                mosaicMap = mosaicFine.Coarse;
                            }
                            else
                            {
                                // Only the canvases' localization map is used; their pooled rows aren't. Erased views
                                // (pooled rows only) share the pass: one forward of 4 images is cheaper than two of 2.
                                using var canvasMask = ones(canvases.shape[0], 1, canvases.shape[2] / imageTower.OutputStride, canvases.shape[3] / imageTower.OutputStride, device: device);
                                if (erasureVariants is not null && erasureVariants.shape[2] == canvases.shape[2])
                                {
                                    var canvasCount = canvases.shape[0];
                                    using var input = cat([canvases, erasureVariants], 0);
                                    using var inputMask = cat([canvasMask, erasureMask!], 0);
                                    var (sharedPooled, sharedMap) = imageTower.ForwardPooled(input, inputMask);
                                    mosaicMap = sharedMap.narrow(0, 0, canvasCount);
                                    mosaicPooled = sharedPooled.narrow(0, 0, canvasCount);
                                    variantPooled = sharedPooled.narrow(0, canvasCount, erasureVariants.shape[0]);
                                }
                                else
                                    (mosaicPooled, mosaicMap) = imageTower.ForwardPooled(canvases, canvasMask);
                            }
                            profiler?.Mark("mosaic forward", mosaicMap);
                            var tileBoxes = tileImages.Select((image, k) => Mosaic.TileBox(boxes[image], k % Mosaic.TilesOf(mosaicGrid), stepSize, stepTileSize, mosaicGrid)).ToArray();
                            var mapSide = mosaicFine?.Features.shape[2] ?? mosaicMap.shape[2];
                            using var tileMask = SpatialMask.Build(tileBoxes, mosaicGrid * stepTileSize, mapSide, mapSide, device);
                            using var tilePairs = cpuPairs is not null
                                ? Mosaic.TilePairs(cpuPairs, cpuTargets, tileImages, mosaicGrid).To(device)
                                : Mosaic.TilePairsOnDevice(devicePairs, labelledPairs, cpuTargets, tileImages, mosaicGrid);
                            using var canvasOfTile = tensor(Enumerable.Range(0, tileImages.Length).Select(k => (long)(k / Mosaic.TilesOf(mosaicGrid))).ToArray()).Upload(device);
                            if (mosaicFine is not null)
                            {
                                // Each tile's own quadrant of its canvas's map: the tile mask is zero outside it, so the loss is the same
                                // over a quarter of the locations.
                                var (tileMaps, tileCropMask) = Mosaic.TileCrops(mosaicFine, tileMask, mosaicGrid);
                                mosaicTerm = SigmoidContrastiveLoss.ComputeLocalizedPairs(tileMaps, tagEmbeddings, tilePairs, (float)localizationTemperature, tileCropMask, logitScaleBias,
                                    targetOptions.NegativeGamma, targetOptions.NegativeClip);
                            }
                            else
                                mosaicTerm = SigmoidContrastiveLoss.ComputeLocalizedPairs(
                                    mosaicMap.index_select(0, canvasOfTile), tagEmbeddings, tilePairs, (float)localizationTemperature, tileMask, logitScaleBias,
                                    targetOptions.NegativeGamma, targetOptions.NegativeClip);
                            loss += localizationWeight * targetOptions.MosaicWeight * mosaicTerm;
                            profiler?.Mark("mosaic loss", loss);

                            // Mosaic glimpse labels: the canvases get glimpses where the model chooses to look on them, and a tag one tile
                            // carries is a "no" in every window on tiles whose image says no, window by window (Mosaic.GlimpseTerm).
                            if (targetOptions.MosaicGlimpseWeight > 0 && mosaicPooled is not null && imageTower.ClassificationGlimpses)
                            {
                                var canvasCount = (int)canvases.shape[0];
                                var canvasSize = (int)canvases.shape[2];
                                var whole = Enumerable.Repeat(new LetterboxBox(0, 0, canvasSize, canvasSize), canvasCount).ToArray();
                                using var canvasContent = Glimpses.ContentBoxes(whole, device);
                                using var canvasMapMask = ones(canvasCount, 1, mosaicMap.shape[2], mosaicMap.shape[3], device: device);
                                using var canvasChoice = Glimpses.Propose(imageTower.Proposals(mosaicPooled), mosaicMap, tagEmbeddings.index_select(0, classificationPooledRows),
                                    canvasMapMask, canvasContent, canvasSize, imageTower.OutputStride, imageTower.GlimpseCount);
                                var (canvasCells, _) = Glimpses.Embed(imageTower, canvases, canvasContent, canvasChoice);
                                var (tileRows, tileValid, tileLabelValues) = Mosaic.GlimpseTileLabels(cpuTargets, tileImages, localizable, MosaicGlimpseTags, mosaicRandom);
                                using var rows = tensor(tileRows).Upload(device);
                                using var rowsValid = tensor(tileValid, [canvasCount, MosaicGlimpseTags]).Upload(device);
                                using var tileLabelTensor = tensor(tileLabelValues, [canvasCount, Mosaic.TilesPerMosaic, MosaicGlimpseTags]).Upload(device);
                                var (overlaps, inside) = Mosaic.WindowTiles(canvasChoice, stepMosaicTileSize);
                                mosaicGlimpseTerm = Mosaic.GlimpseTerm(canvasCells, tagEmbeddings.index_select(0, rows).reshape(canvasCount, MosaicGlimpseTags, -1), rowsValid,
                                    tileLabelTensor, overlaps, inside, imageTower.ClassificationPoolingTemperature, logitScaleBias);
                                overlaps.Dispose();
                                inside.Dispose();
                                loss += targetOptions.MosaicGlimpseWeight * mosaicGlimpseTerm;
                                profiler?.Mark("mosaic glimpses", loss);
                            }

                            // Each tile classified from the canvas's full map pooled over it, on its
                            // image's own targets: the neighbouring tiles are in view of the deeper
                            // layers, so only the tile's own pixels give the right answer.
                            if (targetOptions.MosaicClassificationWeight > 0)
                            {
                                // needsFullMap: mosaicSpatial is set.
                                using var tileSpatialMask = mosaicSpatial!.shape[2] == mosaicMap.shape[2]
                                    ? tileMask.alias()
                                    : SpatialMask.Build(tileBoxes, 2 * stepMosaicTileSize, mosaicSpatial.shape[2], mosaicSpatial.shape[3], device);
                                var tilePooled = ImageTower.MaskedPool(mosaicSpatial.index_select(0, canvasOfTile), tileSpatialMask);
                                using var cpuTileTargets = Mosaic.TileTargets(cpuTargets, tileImages);
                                using var tileTargets = cpuTileTargets.To(device);
                                tileClassificationTerm = SigmoidContrastiveLoss.ComputeWeighted(
                                    tilePooled, tagEmbeddings, tileTargets, logitScaleBias, targetOptions.NegativeGamma, targetOptions.NegativeClip);
                                loss += targetOptions.MosaicClassificationWeight * tileClassificationTerm;
                            }
                        }

                        if (erasureVariants is not null)
                        {
                            // No mosaic pass to ride along with.
                            variantPooled ??= imageTower.ForwardPooled(erasureVariants, erasureMask!).Pooled;
                            var unblanked = erasurePairs.Select(p => wholeImageLogits[p.Image * tagEmbeddings.shape[0] + p.Tag] * logitScaleBias.Scale + logitScaleBias.Bias).ToArray();
                            erasureTerm = ErasureContrast.Compute(variantPooled, tagEmbeddings, erasurePairs, logitScaleBias, targetOptions.ErasureLoss, targetOptions.ErasureMargin,
                                unblanked, targetOptions.ErasureAnchorWeight);
                            loss += targetOptions.ErasureWeight * erasureTerm;
                            profiler?.Mark("erasure loss", loss);
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
                    (loss * (float)(batchShare / gradientAccumulationSteps)).backward();
                    profiler?.Mark("backward", profiledParameter?.grad);

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
                        if (glimpseTowerOptimizer is not null)
                        {
                            glimpseTowerSteps++;
                            var rate = optimizer.ParamGroups.First().LearningRate;
                            var warm = glimpseTowerMade ? Math.Min(1.0, (double)glimpseTowerSteps / GlimpseTowerWarmupSteps) : 1.0;
                            foreach (var group in glimpseTowerOptimizer.ParamGroups)
                                group.LearningRate = rate * warm * targetOptions.GlimpseTowerLearningRateScale;
                        }

                        if (fineOptimizer is not null)
                        {
                            fineSteps++;
                            var rate = optimizer.ParamGroups.First().LearningRate;
                            var warm = fineMade ? Math.Min(1.0, (double)fineSteps / GlimpseTowerWarmupSteps) : 1.0;
                            foreach (var group in fineOptimizer.ParamGroups)
                                group.LearningRate = rate * warm;
                        }

                        optimizer.step();
                        logitScaleBiasOptimizer.step();
                        glimpseTowerOptimizer?.step();
                        fineOptimizer?.step();
                        profiler?.Mark("optimizer step", profiledParameter);
                    }

                    // Unscaled -- reports the real loss magnitude every micro-batch (matching
                    // pre-accumulation logs) rather than the backward()-only scaled-down value.
                    // Reconstructs the same combined total the old single summed `loss` tensor
                    // used to report, now that the localization term is computed (and already
                    // backpropagated) separately from `loss`. Kept on the device and read every
                    // StepReadEvery steps (and before a save or a stop): reading them every step made
                    // the device finish the step and sit idle while the next one was set up.
                    Tensor stepLosses;
                    using (no_grad())
                    {
                        var localizationShown = sparseLocalized is not null
                            ? localizationWeight * sparseLocalized.detach()
                            : zeros(1, device: device).squeeze() + localizationWeight * localizedLossValue;
                        if (cropTerm is not null)
                            localizationShown = localizationShown + targetOptions.CropConsistencyWeight * cropTerm.detach();
                        if (mosaicTerm is not null)
                            localizationShown = localizationShown + localizationWeight * targetOptions.MosaicWeight * mosaicTerm.detach();
                        var tagShown = tagLoss.detach();
                        if (mosaicGlimpseTerm is not null)
                            tagShown = tagShown + targetOptions.MosaicGlimpseWeight * mosaicGlimpseTerm.detach();
                        if (tileClassificationTerm is not null)
                            tagShown = tagShown + targetOptions.MosaicClassificationWeight * tileClassificationTerm.detach();
                        if (erasureTerm is not null)
                            tagShown = tagShown + targetOptions.ErasureWeight * erasureTerm.detach();
                        stepLosses = stack([loss.detach() + localizationWeight * localizedLossValue, tagShown, localizationShown])
                            .to_type(ScalarType.Float32).DetachFromDisposeScope();
                    }

                    // How much of each step went to waiting on data vs. everything else
                    // (GPU compute, target building) -- the number that says whether
                    // loading or compute is the bottleneck.
                    var stepsThisEpoch = step + 1 - stepStart;
                    var waitMs = ((prefetcher?.ConsumerWait ?? TimeSpan.Zero) - epochWaitStart).TotalMilliseconds / stepsThisEpoch;
                    var stepMs = epochClock.Elapsed.TotalMilliseconds / stepsThisEpoch;
                    var currentRate = targetOptions.CosineSchedule
                        ? LearningRateSchedule.At((long)epoch * stepsPerEpoch + step, (long)epochs * stepsPerEpoch, learningRate, targetOptions.WarmupSteps)
                        : learningRate;
                    pendingSteps.Add((step + 1, stepLosses, stepSize));
                    scaleStats?.Step(stepSize, (int)pixelBatch.shape[0], stepRequest?.FromStore ?? 0, stepRequest?.DrawnStored ?? 0);
                    if (scaleStats is not null && (stepsThisProcess + 1) % ScaleSummaryEvery == 0 && scaleStats.TakeSummary(StoreLine()) is { } scaleSummary)
                        dashboard.Log(Markup.Escape(scaleSummary));
                    dashboard.StepProgress(step + 1, currentRate, stepMs - waitMs, waitMs);
                    var flushNow = pendingSteps.Count >= StepReadEvery || profiler is not null || step + 1 == stepsPerEpoch
                                   || sinceLastCheckpoint.Elapsed >= checkpointInterval || stopRequest.IsCancellationRequested
                                   || (targetOptions.MaxSteps > 0 && stepsThisProcess + 1 >= targetOptions.MaxSteps);
                    if (flushNow && glimpseContextMean is not null)
                        imageTower.GlimpseContextMean = glimpseContextMean.cpu().data<float>().ToArray();
                    if (flushNow && FlushSteps() is { } nanStep)
                    {
                        dashboard.Log("[red]" + Markup.Escape($"epoch {epoch + 1}/{epochs} step {nanStep}/{stepsPerEpoch}: {NaNGuard.Message}") + "[/]");
                        result = 1;
                        dashboard.Finished("stopped: loss diverged to NaN");
                        return;
                    }
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
                    var stopping = reachedMaxSteps || stopRequest.IsCancellationRequested;
                    if (sinceLastCheckpoint.Elapsed >= checkpointInterval || stopping)
                    {
                        var (appliedOffloaded, offloadedStop) = offloaded is not null ? ApplyOffloaded() : ([], false);
                        dashboard.Phase("saving checkpoint");
                        StagedSave.Run(checkpointDir, staging =>
                        {
                            SaveCheckpoint(staging, imageTower, tagTower, logitScaleBias, embeddingDim, resolvedInputSize, vocabulary);
                            TrainingState.Save(
                                staging,
                                new TrainingProgress(epoch, earlyStopping.BestLoss, earlyStopping.EvaluationsSinceImprovement, step + 1, validationSplitSeed, stepsPerEpoch, EarlyStoppingMetricName),
                                optimizer);
                            TrainingState.SaveLogitScaleBiasOptimizerState(staging, logitScaleBiasOptimizer);
                            if (glimpseTowerOptimizer is not null)
                                TrainingState.SaveGlimpseTowerOptimizerState(staging, glimpseTowerOptimizer);
                            if (fineOptimizer is not null)
                                TrainingState.SaveFineLocalizationOptimizerState(staging, fineOptimizer);
                            dashboard.SaveLossHistory(staging);
                        });
                        sinceLastCheckpoint.Restart();
                        dashboard.CheckpointSaved();
                        if (LocalizationPairSelector.WidestSeen > 0)
                            dashboard.Log(Markup.Escape($"localization pairs: widest row {LocalizationPairSelector.WidestSeen} so far (rows padded to multiples of {LocalizationPairSelector.WidthMultiple}; {LocalizationPairSelector.CappedImages:N0} image or tile rows sampled down to the cap)"));
                        if (erasureTotal > 0)
                            dashboard.Log(Markup.Escape($"erased views: {erasureTotal:N0} pairs so far, {(double)erasureOnOther / erasureTotal:P0} with the control on another tag"));
                        if (targetOptions.CharacterExtentWeight > 0 || targetOptions.CharacterHideWeight > 0)
                            dashboard.Log(Markup.Escape($"characters: {characterExtentImages:N0} solo images trained for extent, {characterHiddenImages:N0} hidden views so far"));
                        if (targetOptions.SelfMapWeight > 0)
                            dashboard.Log(Markup.Escape($"self-map windows: {selfMapSteps:N0} steps with window labels from the map (weight {targetOptions.SelfMapWeight}){(targetOptions.SelfMapNegatives != "none" ? $", {selfMapNegativeTags:N0} negative tags looked at" : "")}"));
                        if (tagCentricSampler is { TwinEvery: > 0 } twinSampler)
                            dashboard.Log(Markup.Escape($"twins: {twinSampler.TwinBatches:N0} twin batches (mean Jaccard {twinSampler.TwinJaccardSum / Math.Max(1, twinSampler.TwinBatches):F2}), {twinSampler.TwinMisses:N0} misses{(targetOptions.TwinPaste > 0 ? $"; {pastesMade:N0} pastes, {pastesGated:N0} batches gated (no confident peak)" : "")}"));
                        if (glimpsePositives > 0)
                        {
                            var zooms = glimpseZooms.cpu().data<long>().ToArray();
                            if (featureSplitTags is not null && splitPositives > 0)
                                dashboard.Log(Markup.Escape($"feature split: {(double)splitExclusive / splitPositives:P1} of {splitPositives:N0} split-tag positives were exclusive; {splitNegatives:N0} context negatives split"));
                            dashboard.Log(Markup.Escape($"glimpses: {(double)glimpseUnframed.item<long>() / glimpsePositives:P1} of {glimpsePositives:N0} labelled positives since the restart framed by none of the chosen windows; chosen windows " +
                                string.Join(", ", zooms.Select((n, z) => $"{Glimpses.SideOf(z)}px {(double)n / zooms.Sum():P0}"))));
                        }
                        if (locality is not null)
                        {
                            var (observed, negatives, negativeWeight) = locality.TakeStats();
                            dashboard.Log(Markup.Escape($"locality: {observed:N0} pair observations since the last save, {negatives:N0} negatives (mean weight {(negatives > 0 ? negativeWeight / negatives : 0):F2})"));
                        }
                        foreach (var e in appliedOffloaded)
                            offloaded!.MarkApplied(e);
                        if (offloadedStop)
                        {
                            dashboard.Finished("early stopping (offloaded evaluation)");
                            return;
                        }
                    }

                    profiler?.Mark("bookkeeping (losses, dashboard, saves)");
                    if (profiler?.StepFinished() == true)
                        dashboard.Log(Markup.Escape($"step profile written to {Path.Combine(checkpointDir, "step_profile.txt")}"));
                    if (stopping)
                    {
                        dashboard.Finished(reachedMaxSteps ? $"stopped after --max-steps {targetOptions.MaxSteps}" : "stopped by Ctrl+C, checkpoint saved");
                        return;
                    }
                }

                // Save after every epoch, not just at the end: a long run (especially on
                // GPU) can be killed, crash, or lose its SSH session partway through, and
                // without this, all completed work would be unrecoverable.
                dashboard.Phase($"end of epoch {epoch + 1}: saving checkpoint");
                StagedSave.Run(checkpointDir, staging => SaveCheckpoint(staging, imageTower, tagTower, logitScaleBias, embeddingDim, resolvedInputSize, vocabulary));
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
                    var (validationLoss, heldOutPooled, heldOutScores) = Evaluate(
                        imageTower, tagTower, logitScaleBias, loadBatch, imageTagRows, heldOut, allTagIndices, vocabulary.Records.Count, device, batchSize, resolvedInputSize,
                        readWindow: readWindow, classificationPooledRows: classificationPooledRows,
                        onBatchComplete: batchesDone =>
                            dashboard.PhaseProgress($"end of epoch {epoch + 1}: embedding held-out images (batches)", batchesDone, validationBatchCount));

                    dashboard.Phase($"end of epoch {epoch + 1}: scoring held-out tag rankings and ratings");
                    var report = EvaluateRankings(
                        heldOutPooled, heldOut.Select(i => imageTagRows[i]).ToList(), tagTower, allTagIndices, logitScaleBias,
                        vocabulary, statistics, exclusiveGroups, device, validationSplitSeed,
                        imageSites is not null ? heldOut.Select(i => imageSites[i]).ToArray() : null, heldOutScores);
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
                            {
                                MapScoreTemperature = localizationWeight > 0 ? (float)localizationTemperature : null,
                                ClassificationPooledTags = localizable,
                            }
                            .Evaluate(vocabulary, imageTagRows, heldOut, gold, localizability, seed: validationSplitSeed, labelFreeChecks: false);
                        EvaluateCommandHandler.WriteLocalizationReport(checkpointDir, localization);
                        ArchiveEpochReports(checkpointDir, epoch + 1);
                        foreach (var g in localization.Gold.Where(g => g.Name == "all"))
                            dashboard.Log(EvaluateCommandHandler.DescribeGold(g));
                        foreach (var t in (localization.Teacher ?? []).Where(t => t.Name == "all"))
                            dashboard.Log(EvaluateCommandHandler.DescribeTeacher(t));
                        foreach (var f in (localization.GlimpseFunnel ?? []).Where(f => f.Name == "all"))
                            dashboard.Log(EvaluateCommandHandler.DescribeFunnel(f));
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
                    if (RefreshFocus() is { } focusMessage) dashboard.Log(focusMessage);
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
                StagedSave.Run(checkpointDir, staging =>
                {
                    TrainingState.Save(staging, new TrainingProgress(epoch + 1, earlyStopping.BestLoss, earlyStopping.EvaluationsSinceImprovement, CompletedStepsInEpoch: 0, validationSplitSeed, stepsPerEpoch, EarlyStoppingMetricName), optimizer);
                    TrainingState.SaveLogitScaleBiasOptimizerState(staging, logitScaleBiasOptimizer);
                    if (glimpseTowerOptimizer is not null)
                        TrainingState.SaveGlimpseTowerOptimizerState(staging, glimpseTowerOptimizer);
                    if (fineOptimizer is not null)
                        TrainingState.SaveFineLocalizationOptimizerState(staging, fineOptimizer);
                    dashboard.SaveLossHistory(staging);
                });
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
        var comparisons = new List<(string, string)>();
        foreach (var comparison in targetOptions.ChartComparisons ?? [])
        {
            var split = comparison.IndexOf('=');
            if (split > 0 && Directory.Exists(comparison[(split + 1)..]))
                comparisons.Add((comparison[..split], comparison[(split + 1)..]));
            else
                AnsiConsole.MarkupLineInterpolated($"[yellow]--chart-compare '{comparison}': expected label=checkpoint directory; ignored.[/]");
        }
        dashboard.SetChartComparisons(comparisons);

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
        IReadOnlyList<BooruSites>? heldOutSites = null, float[]? heldOutScores = null)
    {
        using var logits = HeldOutLogits(heldOutPooled, heldOutScores, heldOutTags.Count, tagTower, allTagIndices, logitScaleBias);
        return TagRankingEvaluator.Evaluate(logits, heldOutTags, vocabulary, statistics, exclusiveGroups, device, seed: seed, imageSites: heldOutSites);
    }

    /// <summary>The evaluators' logits: <paramref name="heldOutScores"/> (raw [images × tags]) when the classifier pools per tag, else from the pooled embeddings.</summary>
    internal static ImageTagLogits HeldOutLogits(
        float[] heldOutPooled, float[]? heldOutScores, int imageCount, TagTower tagTower, Tensor allTagIndices, LogitScaleBias logitScaleBias)
    {
        if (heldOutScores is not null)
            return ImageTagLogits.Dense(heldOutScores, imageCount, logitScaleBias.ToLogitScaling());
        float[] tagEmbeddings;
        using (no_grad())
        using (var embeddings = tagTower.forward(allTagIndices))
            tagEmbeddings = embeddings.cpu().data<float>().ToArray();
        return ImageTagLogits.Linear(heldOutPooled, tagEmbeddings, (int)(tagEmbeddings.Length / allTagIndices.shape[0]), logitScaleBias.ToLogitScaling());
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

    /// <summary>
    /// The glimpse loss's weights ([B, V], on the device): <paramref name="targets"/>' own, except a labelled localizable
    /// positive none of its image's chosen windows frames gets <paramref name="unframedWeight"/>. Framed means a chosen window
    /// is a close look of the glimpse sensor at that tag's own map peak (<see cref="Glimpses.Framed"/>): a 256px window merely
    /// containing a 20px object shows it at 5px. The classifier never saw an unframed one; trained "yes" anyway, it would
    /// learn the tag from whatever the windows did show (the character). Counts the unframed into
    /// <paramref name="unframedCount"/> on the device; the positives into <paramref name="positiveCount"/>.
    /// </summary>
    private static Tensor WeightUnframedPositives(
        PairTargets cpuTargets, PairTargets targets, bool[] localizable, Tensor localizationMap, Tensor tagEmbeddings, Tensor mapMask, Tensor contentBoxes,
        int inputSize, int outputStride, GlimpseChoice choice, float unframedWeight, Tensor unframedCount, ref long positiveCount)
    {
        var positives = LocalizablePositives(cpuTargets, localizable);
        if (positives.All(p => p.Count == 0))
            return targets.Weights.clone();
        using var _ = no_grad();
        var up = PaddedRows.Upload(positives, (int)targets.Weights.shape[1], targets.Weights.device);
        var (x, y) = Glimpses.Peaks(localizationMap, tagEmbeddings.detach().index_select(0, up.Rows.flatten()).reshape(up.Rows.shape[0], up.Rows.shape[1], -1),
            mapMask, outputStride);
        var framed = Glimpses.Framed(choice, x, y, contentBoxes, inputSize).flatten().index_select(0, up.Real);
        unframedCount.add_(framed.logical_not().sum());
        positiveCount += up.RealCount;
        var factor = framed.to_type(ScalarType.Float32) * (1 - unframedWeight) + unframedWeight;
        var weights = targets.Weights.flatten();
        return weights.index_copy(0, up.Linear, weights.index_select(0, up.Linear) * factor).view_as(targets.Weights);
    }

    /// <summary>
    /// Per-image vocabulary rows, uploaded once: [B, P] padded with row 0, which are real, and for the real ones their
    /// positions in the padded [B * P] and in a flattened [B, V]. With <paramref name="padTo"/>, P is rounded up to a multiple
    /// of it, so a matrix multiply over the rows sees a few shapes instead of a new one most steps (on the B70 each new shape
    /// compiles oneDNN kernels; the locality negatives' varying P ran the driver out of resources, "VM worker error: -12").
    /// </summary>
    /// <summary>A feature-split exclusive positive's highest prior (the context-reliance evaluation's low band).</summary>
    private const float FeatureSplitExclusiveMaxPrior = 0.2f;

    /// <summary>Tags per mosaic canvas the glimpse term labels (<c>--mosaic-glimpse-weight</c>); fixed, so its shapes don't change between steps.</summary>
    private const int MosaicGlimpseTags = 48;

    /// <summary>Training images a tag needs to anchor twin batches (<c>--twin-every</c>).</summary>
    private const int TwinMinPositives = 50;

    /// <summary>A feature-split context negative's lowest prior (the band where context makes a tag likely).</summary>
    private const float FeatureSplitContextMinPrior = 0.6f;

    /// <summary>
    /// The feature split's pairs, [B, V] row-major (true = score from the object half), for <paramref name="splitTags"/>: a
    /// positive whose image has a known character/artist/series that rarely carries the tag (<paramref name="priors"/> below
    /// <see cref="FeatureSplitExclusiveMaxPrior"/>: exclusive), and with <paramref name="negatives"/>, a trained negative where
    /// one usually does (at least <see cref="FeatureSplitContextMinPrior"/>: a context negative). Images with no known source,
    /// and priors in between, teach nothing against identity and keep the whole embedding. Null when none.
    /// </summary>
    public static (bool[] Mask, int Batch, int Vocabulary)? FeatureSplitPairs(PairTargets targets, IReadOnlyList<IReadOnlyList<int>> imageTags, IReadOnlyList<bool> splitTags,
        ContextPriorTable priors, bool negatives, ref long positiveCount, ref long exclusiveCount, ref long negativeCount)
    {
        var batch = (int)targets.Labels.shape[0];
        var vocabulary = (int)targets.Labels.shape[1];
        var labels = targets.Labels.data<float>().ToArray();
        var weights = targets.Weights.data<float>().ToArray();
        var mask = new bool[batch * vocabulary];
        var any = false;
        for (var i = 0; i < batch; i++)
        {
            var row = i * vocabulary;
            var tags = imageTags[i];
            var known = priors.HasSource(tags);
            for (var t = 0; t < vocabulary && t < splitTags.Count; t++)
            {
                var at = row + t;
                if (!splitTags[t] || labels[at] <= 0 || weights[at] <= 0)
                    continue;
                positiveCount++;
                if (known && priors.Prior(tags, t) < FeatureSplitExclusiveMaxPrior)
                {
                    exclusiveCount++;
                    mask[at] = any = true;
                }
            }
            if (!negatives)
                continue;
            foreach (var t in priors.LikelyTags(tags, FeatureSplitContextMinPrior))
            {
                var at = row + t;
                if (t >= vocabulary || mask[at] || labels[at] > 0 || weights[at] <= 0)
                    continue;
                negativeCount++;
                mask[at] = any = true;
            }
        }
        return any ? (mask, batch, vocabulary) : null;
    }

    /// <summary>Steps over which a newly made separate glimpse tower's learning rate rises from 0 to the run's.</summary>
    private const int GlimpseTowerWarmupSteps = 1000;

    /// <summary>How often the hi-res stores are re-read for rows a pull has appended since (<see cref="HiResImageStore.Refresh"/>).</summary>
    private static readonly TimeSpan StoreRefreshInterval = TimeSpan.FromMinutes(5);

    /// <summary>Steps between the mixed-scale summaries in the log (<see cref="ScaleStats"/>): about 7 minutes at ~200 ms/step.</summary>
    private const int ScaleSummaryEvery = 2000;

    private sealed record PaddedRows(Tensor Rows, Tensor Valid, Tensor Real, Tensor Linear, int RealCount)
    {
        public static PaddedRows Upload(IReadOnlyList<List<int>> rows, int vocabularySize, Device device, int padTo = 1)
        {
            var width = Math.Max(1, rows.Max(r => r.Count));
            width = (width + padTo - 1) / padTo * padTo;
            var padded = new long[rows.Count * width];
            var valid = new bool[rows.Count * width];
            var real = new List<long>();
            var linear = new List<long>();
            for (var i = 0; i < rows.Count; i++)
                for (var j = 0; j < rows[i].Count; j++)
                {
                    padded[i * width + j] = rows[i][j];
                    valid[i * width + j] = true;
                    real.Add(i * width + j);
                    linear.Add((long)i * vocabularySize + rows[i][j]);
                }
            return new PaddedRows(tensor(padded, [rows.Count, width]).Upload(device), tensor(valid, [rows.Count, width]).Upload(device),
                tensor(real.ToArray()).Upload(device), tensor(linear.ToArray()).Upload(device), real.Count);
        }
    }

    /// <summary>Each image's weighted, labelled localizable positives (vocabulary rows).</summary>
    /// <summary>
    /// Per image, the localizable tags it is a certain negative for (full-strength BCE: label -1, weight > 0, no focal damping):
    /// hard negatives, exclusive groups, excludes rules, and, with <paramref name="includeContext"/>, context negatives (told
    /// apart by their weight, <paramref name="contextWeight"/>, when it differs from <paramref name="certainWeight"/>).
    /// </summary>
    private static List<int>[] LocalizableCertainNegatives(PairTargets targets, bool[] localizable, bool includeContext, float contextWeight, float certainWeight)
    {
        var batch = (int)targets.Labels.shape[0];
        var vocabulary = (int)targets.Labels.shape[1];
        var labels = targets.Labels.data<float>().ToArray();
        var weights = targets.Weights.data<float>().ToArray();
        var focal = targets.FocalMask.data<float>().ToArray();
        bool Context(float w) => contextWeight > 0 && contextWeight != certainWeight && w == contextWeight;
        return Enumerable.Range(0, batch)
            .Select(i => Enumerable.Range(0, vocabulary).Where(t =>
            {
                var k = i * vocabulary + t;
                return localizable[t] && labels[k] < 0 && weights[k] > 0 && focal[k] == 0 && (includeContext || !Context(weights[k]));
            }).ToList())
            .ToArray();
    }

    private static List<int>[] LocalizablePositives(PairTargets targets, bool[] localizable)
    {
        var batch = (int)targets.Labels.shape[0];
        var vocabulary = (int)targets.Labels.shape[1];
        var labels = targets.Labels.data<float>().ToArray();
        var weights = targets.Weights.data<float>().ToArray();
        return Enumerable.Range(0, batch)
            .Select(i => Enumerable.Range(0, vocabulary).Where(t => localizable[t] && labels[i * vocabulary + t] > 0 && weights[i * vocabulary + t] > 0).ToList())
            .ToArray();
    }

    internal static string DescribeReport(EvaluationReport report) =>
        Markup.Escape(
            $"held-out: {report.Images} images, {report.EvaluatedTags} tags | mAP {report.MeanAveragePrecision:F4}, hard-negative mAP {report.MeanHardAveragePrecision:F4}, context mAP {report.MeanContextAveragePrecision:F4}, mean best-F1 {report.MeanBestF1:F4}, mean held-out F1 at calibrated thresholds {report.MeanHeldOutF1:F4} | " +
            $"image-level at calibrated thresholds: precision {report.MicroPrecision:F4}, recall {report.MicroRecall:F4}, F1 {report.MicroF1:F4}, R-precision {report.RPrecision:F4}, exclusive-tag violations {report.ExclusiveViolationRate:P2}" +
            (report.ContextReliance is { } c
                ? $" | context reliance: recall gap {c.MacroRecallGap:+0.0%;-0.0%} ({c.RecallGapTags} tags), false-positive gap {c.MacroFalsePositiveGap:+0.0%;-0.0%} ({c.FalsePositiveGapTags} tags)"
                : ""));

    /// <param name="classificationPooledRows">With <see cref="ImageTower.ClassificationPoolingTemperature"/> &gt; 0, the rows scored per tag (<see cref="ClassificationPooling"/>); the result then also carries every image's raw scores for every tag.</param>
    internal static (double Loss, float[] Pooled, float[]? Scores) Evaluate(
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
        Func<IReadOnlyList<int>, EncodedImage[]>? readWindow = null,
        Tensor? classificationPooledRows = null)
    {
        using var _ = no_grad();
        var perTag = imageTower.ClassificationPoolingTemperature > 0 && classificationPooledRows is not null;
        var scoresAll = perTag ? new float[(long)validationIndices.Count * vocabularySize] : null;

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
        // The localizable tags, which a glimpse tower's window choice ranks (Glimpses.Look).
        using var localizableTags = perTag && imageTower.ClassificationGlimpses ? tagEmbeddings.index_select(0, classificationPooledRows!) : null;

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
            // Without per-tag pooling only the pooled embedding is needed: ForwardPooled skips building the full map.
            var mapSize = inputSize / imageTower.OutputStride;
            using var spatialMask = SpatialMask.Build(boxes, inputSize, mapSize, mapSize, device);
            Tensor? spatial = null, localizationMap = null;
            Tensor pooled;
            if (scoresAll is not null && !imageTower.ClassificationGlimpses)
            {
                spatial = imageTower.forward(pixelBatch).Item2;
                pooled = ImageTower.MaskedPool(spatial, spatialMask);
            }
            else if (imageTower.Fine is not null && !imageTower.ClassificationGlimpses)
                // The pooled embedding alone: a stride-2 localization map (and its branch) for 32 images is ~6 GB, and nothing
                // here reads it. Only with the fine level, so other towers' numbers stay identical.
                pooled = imageTower.ForwardPooledOnly(pixelBatch, spatialMask);
            else
                (pooled, localizationMap) = imageTower.ForwardPooled(pixelBatch, spatialMask);

            var batchTagRows = chunk.Select(i => imageTagRows[i]).ToList();
            using var labels = BatchLabelBuilder.Build(batchTagRows, vocabularySize).to(device);
            using var loss = SigmoidContrastiveLoss.Compute(pooled, tagEmbeddings, labels, logitScaleBias);
            pooled.cpu().data<float>().ToArray().CopyTo(pooledAll, offset * imageTower.EmbeddingDim);
            if (scoresAll is not null)
            {
                // Glimpse towers score the windows the model chooses, as in training (or the whole grid: --glimpse-grid).
                var (cells, validCells) = imageTower.ClassificationGlimpses
                    ? Glimpses.Look(imageTower, pixelBatch, boxes, pooled, localizationMap!, localizableTags!)
                    : ClassificationPooling.Cells(spatial!, spatialMask);
                using var scores = ClassificationPooling.RawScores(pooled, cells, validCells, tagEmbeddings, classificationPooledRows!, imageTower.ClassificationPoolingTemperature, imageTower.ClassificationPoolingTopK, imageTower.ClassificationPoolingBottomWeight);
                scores.cpu().data<float>().ToArray().CopyTo(scoresAll, (long)offset * vocabularySize);
            }

            totalLoss += loss.item<float>() * chunk.Count;
            totalCount += chunk.Count;

            batchesDone++;
            onBatchComplete?.Invoke(batchesDone);
        }

        return (totalLoss / totalCount, pooledAll, scoresAll);
    }

    internal static (Tensor Pixels, IReadOnlyList<LetterboxBox> Boxes) LoadEncodedBatch(EncodedImage[] images, int inputSize)
    {
        var imageSize = 3 * inputSize * inputSize;
        var flat = new float[images.Length * imageSize];
        for (var i = 0; i < images.Length; i++)
            ImagePreprocessing.ReconstructInto(images[i], inputSize, flat.AsSpan(i * imageSize, imageSize));
        return (tensor(flat, [images.Length, 3, inputSize, inputSize]), images.Select(image => image.Content).ToArray());
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
