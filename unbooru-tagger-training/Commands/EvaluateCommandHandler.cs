using System.Diagnostics;
using Spectre.Console;
using UnbooruTagger.Core.Dataset;
using UnbooruTagger.Core.Scoring;
using UnbooruTagger.Core.Vocabulary;
using UnbooruTagger.Training.Checkpoints;
using UnbooruTagger.Training.Model;
using UnbooruTagger.Training.Training;
using static TorchSharp.torch;

namespace UnbooruTagger.Training.Commands;

/// <summary>
/// Scores a checkpoint on the same held-out split its training run used (reconstructed
/// from <see cref="TrainingProgress.ValidationSplitSeed"/>), with
/// <see cref="TagRankingEvaluator"/>'s per-tag and image-level metrics, and writes the
/// calibrated per-tag thresholds plus a full per-tag report next to it — so any checkpoint
/// (including one trained before ranking evaluation existed) can be measured and get
/// thresholds without training further.
/// </summary>
public static class EvaluateCommandHandler
{
    public static int Run(
        string checkpointDir, string cacheDir, double validationFraction, int maxImages, int batchSize,
        string? exclusiveTagsPath, string? outDir, int worstTagsToShow, string? embeddingsCachePath = null,
        string? goldPath = null, int localizationSamples = 300, float mapScoreTemperature = 0.25f, IReadOnlyList<string>? dumpTags = null,
        string? tagParentagePath = null, string? smallGoldPath = null, bool dumpOcclusion = false, OcclusionFill occlusionFill = OcclusionFill.Zero,
        string? contextCheckPath = null, string precision = "fp32", bool glimpseGrid = false, bool cropTeacher = true,
        string? localityPriorOut = null, int localityPriorImages = 20000, bool glimpseObjectOnly = false, string? twinCheckOut = null)
    {
        var contextCheck = contextCheckPath is not null ? LabelledContextCheck.Load(contextCheckPath) : null;
        var device = DeviceSelector.Best();
        using var cacheReader = new PreprocessedDatasetCacheReader(cacheDir);
        var (imageTower, config, vocabulary, embeddings, logitScaleBias) = Checkpoint.Load(checkpointDir, device);
        if (precision is "tf32" or "bf16")
            TorchSharp.torch.backends.cuda.matmul.allow_tf32 = true;
        if (precision == "bf16")
            imageTower.ComputeType = ScalarType.BFloat16;
        if (imageTower.ClassificationGlimpses)
        {
            imageTower.GlimpseGrid = glimpseGrid;
            if (glimpseObjectOnly && imageTower.GlimpseObjectDims <= 0)
                throw new ArgumentException("--glimpse-object-only needs a checkpoint trained with --glimpse-feature-split.");
            imageTower.GlimpseObjectOnly = glimpseObjectOnly;
            if (glimpseObjectOnly)
                AnsiConsole.MarkupLineInterpolated($"Glimpses score from the object half alone ({imageTower.GlimpseObjectDims} of {imageTower.EmbeddingDim} dimensions; the context half held at its training mean).");
            AnsiConsole.MarkupLineInterpolated($"Glimpses: {(glimpseGrid ? "every window of the grid" : $"the {imageTower.GlimpseCount} windows per image the model chooses from its localization map")}.");
        }

        var datasetVocabulary = TagVocabulary.Load(Path.Combine(cacheDir, "tag_vocabulary.json"));

        // Mirror training: a checkpoint that learned ratings carries the three cumulative rating
        // rows, and each image gets them per its rating; then dataset rows are matched to the
        // checkpoint's (possibly compacted) vocabulary by tag name.
        var ratingRows = RatingLabels.FindRows(vocabulary);
        var ratings = ratingRows is not null ? ImageRatings.Load(cacheDir, cacheReader.ImageCount) : null;
        IReadOnlyList<IReadOnlyList<int>> datasetRows = cacheReader.ImageTagRows;
        // Labels are read the way the run was trained: with its tag parentage (the checkpoint's copy), unless
        // --tag-parentage names other rules or 'none'. Comparing runs trained with and without them needs the same rules for both.
        var parentagePath = TagParentage.ResolvePath(tagParentagePath, checkpointDir);
        if (parentagePath is not null && tagParentagePath is not null && !File.Exists(parentagePath))
            throw new FileNotFoundException($"--tag-parentage: '{parentagePath}' doesn't exist.");
        var parentage = TagParentage.Load(parentagePath);
        if (parentage.RuleCount > 0)
            AnsiConsole.MarkupLineInterpolated($"Tag parentage: {parentage.RuleCount} rules from '{parentagePath}' (labels completed, unnamed kinds and parent-only gold boxes ignored).");
        var implications = TagImplications.Load(Path.Combine(cacheDir, TagImplications.FileName), datasetVocabulary, parentage.CompletionPairs);
        if (implications.RuleCount > 0)
            datasetRows = implications.Complete(datasetRows, out _, out _);
        if (ratingRows is not null && ratings is not null)
            datasetRows = RatingLabels.Apply(datasetRows, ratings, RatingLabels.EnsureRows(datasetVocabulary));
        var (remappedRows, unmatched) = VocabularyRemap.Apply(datasetVocabulary, datasetRows, vocabulary);
        if (unmatched > 0)
            AnsiConsole.MarkupLineInterpolated($"{unmatched} dataset tags aren't in the checkpoint's vocabulary and are ignored.");

        var seed = 0;
        if (TrainingState.HasProgress(checkpointDir))
            seed = TrainingState.LoadProgress(checkpointDir).ValidationSplitSeed;
        else
            AnsiConsole.MarkupLine("[yellow]No training_progress.json -- can't reconstruct this run's validation split; using seed 0, which may overlap its training images.[/]");

        IReadOnlyList<IReadOnlyList<int>> imageTagRows = remappedRows;
        var (trainingIndices, validationIndices) = TrainCommandHandler.SplitForValidation(cacheReader.ImageCount, validationFraction, seed);
        var heldOut = maxImages > 0 && validationIndices.Count > maxImages ? validationIndices.Take(maxImages).ToList() : validationIndices;

        // How long each phase takes, printed as it ends.
        var clock = Stopwatch.StartNew();
        void Took(string phase)
        {
            AnsiConsole.MarkupLineInterpolated($"[grey]{phase}: {clock.Elapsed.TotalSeconds:F0}s[/]");
            clock.Restart();
        }

        AnsiConsole.MarkupLine("Building per-tag statistics from the training split...");
        var tagSites = TagAgreement.LoadSiteCoverage(Path.Combine(cacheDir, TagAgreement.FileName), vocabulary);
        var imageSites = tagSites is not null ? ImageSites.Load(cacheDir, cacheReader.ImageCount) : null;
        if (imageSites is null)
            tagSites = null;
        var families = parentage.RuleCount > 0 ? parentage.Bind(vocabulary, TagImplications.LoadPairs(Path.Combine(cacheDir, TagImplications.FileName))) : null;
        var statistics = TagStatistics.Build(trainingIndices.Select(i => imageTagRows[i]).ToList(), vocabulary.Records.Count, seed: seed,
            impliedThreshold: TrainingTargetOptions.Default.ImpliedThreshold, missingThreshold: TrainingTargetOptions.Default.MissingThreshold,
            imageSites: imageSites is not null ? trainingIndices.Select(i => imageSites[i]).ToArray() : null, tagSites: tagSites,
            families: families);
        var exclusiveGroups = ExclusiveTagGroups.Load(
            exclusiveTagsPath ?? Path.Combine(checkpointDir, ExclusiveTagGroups.FileName), vocabulary, fallBackToDefaults: true);
        Took("tag statistics");

        var tagTower = TagTower.CreateFullyTrainable(
            Enumerable.Range(0, embeddings.RowCount).Select(i => embeddings.GetRow(i).ToArray()).ToArray(), config.EmbeddingDim, device);
        using var allTagIndices = tensor(Enumerable.Range(0, vocabulary.Records.Count).Select(i => (long)i).ToArray(), device: device);
        // A classifier that pools localizable tags per tag (ClassificationPooling) is scored the same way.
        var localizability = TagLocalizability.Load(Path.Combine(checkpointDir, TagLocalizability.FileName));
        using var classificationPooledRows = tensor(
            imageTower.ClassificationPoolingTemperature > 0
                ? vocabulary.Records.Where(r => localizability.IsLocalizable(r.Tag)).Select(r => (long)r.RowIndex).Order().ToArray()
                : [],
            device: device);
        if (imageTower.ClassificationPoolingTemperature > 0)
            AnsiConsole.MarkupLineInterpolated(
                $"Classification pooling: {classificationPooledRows.shape[0]:N0} localizable tags scored per tag at temperature {imageTower.ClassificationPoolingTemperature}{(imageTower.ClassificationPoolingTopK > 0 ? $", the mean of the top {imageTower.ClassificationPoolingTopK}" : "")}.");

        if (twinCheckOut is not null)
        {
            // Only the matched twins' window check, on the gold sets (TwinCheck).
            var twinGold = new List<GoldImage>();
            foreach (var path in new[] { goldPath, smallGoldPath }.OfType<string>())
                twinGold.AddRange(LocalizationEvaluator.LoadGold(path).Where(g => twinGold.All(x => x.CacheIndex != g.CacheIndex)));
            TwinCheck.Run(imageTower, Enumerable.Range(0, embeddings.RowCount).SelectMany(i => embeddings.GetRow(i).ToArray()).ToArray(), config.EmbeddingDim,
                statistics, trainingIndices, imageTagRows, twinGold, vocabulary, localizability, cacheReader, device, twinCheckOut, seed);
            Took("twin check");
            return 0;
        }

        if (localityPriorOut is not null)
        {
            // Only the locality prior: training images (shuffled by the run's seed), the tags' map peaks, pair by pair.
            var images = trainingIndices.OrderBy(i => DeterministicSeed.For(seed, i)).Take(localityPriorImages).ToList();
            var localizableRows = vocabulary.Records.OrderBy(r => r.RowIndex).Select(r => localizability.IsLocalizable(r.Tag)).ToArray();
            var allEmbeddings = Enumerable.Range(0, embeddings.RowCount).SelectMany(i => embeddings.GetRow(i).ToArray()).ToArray();
            var pairs = LocalityPrior.Measure(imageTower, allEmbeddings, config.EmbeddingDim, ids => cacheReader.ReadEncoded(ids), cacheReader.InputSize,
                imageTagRows, images, localizableRows, device, LocalityNegatives.DefaultDistance,
                done => { if (done % 2000 < 16) AnsiConsole.MarkupLineInterpolated($"locality prior: {done:N0} of {images.Count:N0} images"); });
            var written = LocalityPrior.Write(localityPriorOut, pairs, vocabulary, minSeen: 5);
            AnsiConsole.MarkupLineInterpolated($"Locality prior: {written:N0} tag pairs seen at least 5 times (of {pairs.Count:N0}) over {images.Count:N0} images, written to '{localityPriorOut}'.");
            Took("locality prior");
            return 0;
        }

        EvaluationReport? report = null;
        RatingReport? ratingReport = null;
        AnsiConsole.Progress()
            .Columns(new TaskDescriptionColumn(), new ProgressBarColumn(), new PercentageColumn(), new RemainingTimeColumn())
            .Start(ctx =>
            {
                var batches = Math.Max(1, (int)Math.Ceiling(heldOut.Count / (double)batchSize));
                var task = ctx.AddTask($"Embedding {heldOut.Count} held-out images", maxValue: batches);
                // Per-tag scores aren't cached; only the pooled embeddings are.
                var pooled = imageTower.ClassificationPoolingTemperature > 0 ? null : TryLoadEmbeddings(embeddingsCachePath, heldOut.Count * config.EmbeddingDim, checkpointDir);
                float[]? scores = null;
                if (pooled is not null)
                {
                    task.Value = batches;
                    AnsiConsole.MarkupLineInterpolated($"Reusing held-out embeddings from '{embeddingsCachePath}'.");
                }
                else
                {
                    (var loss, pooled, scores) = TrainCommandHandler.Evaluate(
                        imageTower, tagTower, logitScaleBias,
                        indices => TrainCommandHandler.LoadCacheBatch(cacheReader, indices, cacheReader.InputSize),
                        imageTagRows, heldOut, allTagIndices, vocabulary.Records.Count, device, batchSize, cacheReader.InputSize,
                        onBatchComplete: done => task.Value = done,
                        readWindow: ids => cacheReader.ReadEncoded(ids), classificationPooledRows: classificationPooledRows);
                    AnsiConsole.MarkupLineInterpolated($"mean validation loss (old metric) {loss:G4}");
                    if (embeddingsCachePath is not null)
                        SaveEmbeddings(embeddingsCachePath, pooled);
                }

                Took("held-out embeddings");

                var rankingTask = ctx.AddTask("Scoring tag rankings", maxValue: 1);
                report = TrainCommandHandler.EvaluateRankings(
                    pooled, heldOut.Select(i => imageTagRows[i]).ToList(), tagTower, allTagIndices, logitScaleBias,
                    vocabulary, statistics, exclusiveGroups, device, seed,
                    imageSites is not null ? heldOut.Select(i => imageSites[i]).ToArray() : null, scores);
                rankingTask.Value = 1;
                Took("tag rankings");

                if (dumpTags is { Count: > 0 } || contextCheck is not null)
                {
                    using var logits = TrainCommandHandler.HeldOutLogits(pooled, scores, heldOut.Count, tagTower, allTagIndices, logitScaleBias);
                    if (dumpTags is { Count: > 0 })
                        DumpScores(Path.Combine(outDir ?? checkpointDir, ScoresFileName), dumpTags, logits, vocabulary, heldOut, imageTagRows, report.Tags,
                            imageSites is not null ? heldOut.Select(i => imageSites[i]).ToArray() : null);
                    if (contextCheck is not null)
                        report = report with { LabelledContext = LabelledContextCheck.Evaluate(contextCheck, heldOut, logits, vocabulary,
                            report.Tags.ToDictionary(t => t.Tag, t => t.Threshold)) };
                }

                if (ratingRows is not null && ratings is not null)
                    ratingReport = TrainCommandHandler.EvaluateRatings(
                        pooled, heldOut.Select(i => ratings[i]).ToList(), tagTower, allTagIndices, logitScaleBias, ratingRows, config.EmbeddingDim);
                Took("scores and ratings");
            });

        var outputDirectory = outDir ?? checkpointDir;
        Directory.CreateDirectory(outputDirectory);
        TrainCommandHandler.WriteEvaluationArtifacts(outputDirectory, report!, ratingReport);

        LocalizationReport? localization = null, smallLocalization = null;
        if (localizationSamples > 0 || goldPath is not null || smallGoldPath is not null)
        {
            AnsiConsole.MarkupLine("Measuring localization...");
            var tagEmbeddingArray = Enumerable.Range(0, embeddings.RowCount).SelectMany(i => embeddings.GetRow(i).ToArray()).ToArray();
            var evaluator = new LocalizationEvaluator(
                imageTower, tagEmbeddingArray, config.EmbeddingDim, logitScaleBias.ToLogitScaling(),
                ids => cacheReader.ReadEncoded(ids), cacheReader.InputSize, device)
            {
                MapScoreTemperature = mapScoreTemperature > 0 ? mapScoreTemperature : null,
                ClassificationPooledTags = vocabulary.Records.OrderBy(r => r.RowIndex).Select(r => localizability.IsLocalizable(r.Tag)).ToArray(),
                OcclusionPairs = dumpOcclusion ? [] : null,
                GlimpsePairs = dumpOcclusion ? [] : null,
                GlimpseEmbeddingDump = dumpOcclusion ? [] : null,
                OcclusionFill = occlusionFill,
                CropTeacher = cropTeacher,
            };
            var gold = goldPath is not null ? LocalizationEvaluator.LoadGold(goldPath) : null;
            if (gold is not null && families is not null)
                gold = LocalizationEvaluator.ExpandFamilies(gold, families, vocabulary);
            localization = evaluator.Evaluate(vocabulary, imageTagRows, heldOut, gold, localizability, Math.Max(localizationSamples, 1), seed);
            WriteLocalizationReport(outputDirectory, localization);
            DumpOcclusionPairs(evaluator, Path.Combine(outputDirectory, OcclusionPairsFileName));
            Took("localization and gold");

            // A second, separately reported gold set (gold-small: small objects, held out from run 4). Only
            // its gold scoring, crop teacher and occlusion: the label-free checks are already in the main report.
            if (smallGoldPath is not null)
            {
                var smallGold = LocalizationEvaluator.LoadGold(smallGoldPath);
                if (families is not null)
                    smallGold = LocalizationEvaluator.ExpandFamilies(smallGold, families, vocabulary);
                smallLocalization = evaluator.Evaluate(vocabulary, imageTagRows, heldOut, smallGold, localizability, seed: seed, labelFreeChecks: false);
                WriteLocalizationReport(outputDirectory, smallLocalization, SmallGoldReportFileName);
                DumpOcclusionPairs(evaluator, Path.Combine(outputDirectory, SmallOcclusionPairsFileName));
                Took("gold-small");
            }
        }

        AnsiConsole.MarkupLine(TrainCommandHandler.DescribeReport(report!));
        if (ratingReport is not null)
            AnsiConsole.MarkupLine(TrainCommandHandler.DescribeRatings(ratingReport));
        else
            AnsiConsole.MarkupLine("[yellow]No rating evaluation: this checkpoint has no rating labels (trained before ratings existed) or the dataset has no ratings.bin.[/]");
        if (localization is not null)
        {
            AnsiConsole.MarkupLine(DescribeLocalization(localization));
            if (localization.GoldTags.Count > 0)
            {
                var table = new Table().Title("Gold-set localization by tag (worst pointing first)")
                    .AddColumns("tag", "pairs", "gold boxes", "pointing", "energy in box", "box precision", "box recall", "count error");
                foreach (var t in localization.GoldTags)
                    table.AddRow(Markup.Escape(t.Name), t.Pairs.ToString(), t.GoldInstances.ToString(), t.PointingAccuracy.ToString("F2"),
                        t.EnergyInBox.ToString("F2"), t.InstancePrecision.ToString("F2"), t.InstanceRecall.ToString("F2"), t.MeanCountError.ToString("F2"));
                AnsiConsole.Write(table);
            }

            foreach (var f in localization.GlimpseFunnel ?? [])
                AnsiConsole.MarkupLine(DescribeFunnel(f));
            foreach (var t in localization.Teacher ?? [])
                AnsiConsole.MarkupLine(DescribeTeacher(t));
            foreach (var o in localization.Occlusion ?? [])
                AnsiConsole.MarkupLine(DescribeOcclusion(o));
            foreach (var o in localization.MapOcclusion ?? [])
                AnsiConsole.MarkupLine(DescribeOcclusion(o with { Name = $"scored from the map, {o.Name}" }));
        }
        if (smallLocalization is not null)
        {
            foreach (var g in smallLocalization.Gold)
                AnsiConsole.MarkupLine(DescribeGold(g with { Name = $"gold-small, {g.Name}" }));
            foreach (var o in smallLocalization.Occlusion ?? [])
                AnsiConsole.MarkupLine(DescribeOcclusion(o with { Name = $"gold-small, {o.Name}" }));
            foreach (var f in smallLocalization.GlimpseFunnel ?? [])
                AnsiConsole.MarkupLine(DescribeFunnel(f with { Name = $"gold-small, {f.Name}" }));
        }
        if (report!.ContextReliance is { } context)
        {
            var table = new Table().Title("Context reliance: whole-image calls by how often the image's characters/artists/series have the tag in training")
                .AddColumns("context prior", "positives", "recall", "negatives", "false positives");
            foreach (var b in context.Bands)
                table.AddRow(Markup.Escape(b.Name), b.Positives.ToString("N0"), b.Recall.ToString("P1"), b.Negatives.ToString("N0"), b.FalsePositiveRate.ToString("P2"));
            AnsiConsole.Write(table);
        }
        if (report!.LabelledContext is { } labelled)
            AnsiConsole.MarkupLine(DescribeLabelledContext(labelled));
        if (report!.ExclusiveViolations is { Count: > 0 } violations)
        {
            var table = new Table().Title($"Exclusive violations: {report.ExclusiveViolationRate:P2} of images call two mutually exclusive tags")
                .AddColumns("pair", "images", "first true", "second true", "neither");
            foreach (var v in violations.Take(worstTagsToShow))
                table.AddRow(Markup.Escape($"{v.First} + {v.Second}"), v.Images.ToString("N0"), v.FirstTrue.ToString("N0"), v.SecondTrue.ToString("N0"),
                    (v.Images - v.FirstTrue - v.SecondTrue).ToString("N0"));
            AnsiConsole.Write(table);
        }
        PrintTags($"Worst {worstTagsToShow} tags by AP", report!.Tags.Take(worstTagsToShow));
        PrintTags($"Most-confused {worstTagsToShow} tags by hard-negative AP",
            report.Tags.Where(t => !double.IsNaN(t.HardAveragePrecision)).OrderBy(t => t.HardAveragePrecision).Take(worstTagsToShow));
        AnsiConsole.MarkupLineInterpolated(
            $"Wrote per-tag thresholds and the full report to '{outputDirectory}' ({Core.Scoring.TagThresholds.FileName}, {TrainCommandHandler.EvaluationReportFileName}).");
        return 0;
    }

    /// <summary>
    /// Held-out embeddings only depend on the image tower and the held-out image list, so a
    /// re-evaluation (different thresholds/exclusions/metrics) can skip the disk-bound
    /// embedding pass. Only reused if newer than the checkpoint's image tower and the right size.
    /// </summary>
    private static float[]? TryLoadEmbeddings(string? path, int expectedLength, string checkpointDir)
    {
        if (path is null || !File.Exists(path))
            return null;
        var tower = Path.Combine(checkpointDir, "image_tower.dat");
        if (File.Exists(tower) && File.GetLastWriteTimeUtc(tower) > File.GetLastWriteTimeUtc(path))
            return null;

        var bytes = File.ReadAllBytes(path);
        if (bytes.Length != expectedLength * sizeof(float))
            return null;
        var values = new float[expectedLength];
        Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
        return values;
    }

    internal const string ScoresFileName = "scores.jsonl";
    internal const string OcclusionPairsFileName = "occlusion_pairs.jsonl";
    internal const string SmallOcclusionPairsFileName = "occlusion_pairs_small.jsonl";

    /// <summary>Writes the occlusion pairs the evaluator collected since the last call, one JSON line each, and clears them.</summary>
    private static void DumpOcclusionPairs(LocalizationEvaluator evaluator, string path)
    {
        if (evaluator.OcclusionPairs is not { } pairs)
            return;
        File.WriteAllLines(path, pairs.Select(p => System.Text.Json.JsonSerializer.Serialize(p)));
        pairs.Clear();
        // The glimpse funnel's windows for the same gold set, beside it (glimpse_pairs.jsonl, glimpse_pairs_small.jsonl).
        if (evaluator.GlimpsePairs is { } glimpses)
        {
            File.WriteAllLines(Path.Combine(Path.GetDirectoryName(path)!, Path.GetFileName(path).Replace("occlusion_pairs", "glimpse_pairs")),
                glimpses.Select(p => System.Text.Json.JsonSerializer.Serialize(p)));
            glimpses.Clear();
        }
        if (evaluator.GlimpseEmbeddingDump is { } embeddings)
        {
            File.WriteAllLines(Path.Combine(Path.GetDirectoryName(path)!, Path.GetFileName(path).Replace("occlusion_pairs", "glimpse_embeddings")),
                embeddings.Select(e => System.Text.Json.JsonSerializer.Serialize(e)));
            embeddings.Clear();
        }
    }

    /// <summary>
    /// One JSON line per held-out image: its cache row, sites, tags, and the probability of each of
    /// <paramref name="tags"/>, with each tag's calibrated threshold in a first header line. For
    /// looking at one tag's false positives and misses directly.
    /// </summary>
    private static void DumpScores(
        string path, IReadOnlyList<string> tags, ImageTagLogits logits,
        TagVocabulary vocabulary, IReadOnlyList<int> heldOut, IReadOnlyList<IReadOnlyList<int>> imageTagRows,
        IReadOnlyList<TagEvaluation> evaluated, IReadOnlyList<BooruSites>? sites)
    {
        var rows = new List<(string Tag, int Row)>();
        foreach (var tag in tags)
            if (vocabulary.TryGet(tag, out var record))
                rows.Add((tag, record.RowIndex));
            else
                AnsiConsole.MarkupLineInterpolated($"[yellow]--dump-scores: '{tag}' isn't in the vocabulary.[/]");

        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        using var writer = new StreamWriter(path);
        var thresholds = rows.ToDictionary(r => r.Tag, r => (double?)evaluated.FirstOrDefault(e => e.Tag == r.Tag)?.Threshold);
        writer.WriteLine(System.Text.Json.JsonSerializer.Serialize(new { thresholds }));
        for (var i = 0; i < heldOut.Count; i++)
        {
            var scores = new Dictionary<string, float>();
            foreach (var (tag, row) in rows)
                scores[tag] = logits.Probability(i, row);
            writer.WriteLine(System.Text.Json.JsonSerializer.Serialize(new
            {
                cacheIndex = heldOut[i],
                sites = sites?[i].ToString(),
                tags = imageTagRows[heldOut[i]].Select(r => vocabulary.GetByRowIndex(r).Tag),
                scores,
            }));
        }
        AnsiConsole.MarkupLineInterpolated($"Wrote {heldOut.Count:N0} images' scores for {rows.Count} tag(s) to '{path}'.");
    }

    private static void SaveEmbeddings(string path, float[] values)
    {
        var bytes = new byte[values.Length * sizeof(float)];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        File.WriteAllBytes(path, bytes);
    }

    internal const string LocalizationReportFileName = "localization_report.json";

    /// <summary>The <c>--gold-small</c> set's report, beside <see cref="LocalizationReportFileName"/>.</summary>
    internal const string SmallGoldReportFileName = "localization_report_small.json";

    internal static void WriteLocalizationReport(string directory, LocalizationReport report, string fileName = LocalizationReportFileName) =>
        File.WriteAllText(
            Path.Combine(directory, fileName),
            System.Text.Json.JsonSerializer.Serialize(report, new System.Text.Json.JsonSerializerOptions
            {
                WriteIndented = true,
                // A rate with nothing to count (e.g. no exclude-crop fit an image) is NaN.
                NumberHandling = System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals
            }));

    internal static string DescribeFunnel(GlimpseFunnelResult f) =>
        Markup.Escape(
            $"glimpse funnel ({f.Name}, {f.Pairs} pairs): suspected {f.Suspected:P1} | its peak looked at {f.PeakFramed:P1} | " +
            $"object under a close window {f.Covered:P1} | best glimpse on the object {f.BestOnObject:P1} (chance {f.WindowsOnObject:P1}; " +
            $"its zoom {string.Join("/", (f.BestZoom ?? []).Select(z => z.ToString("P0")))} for 32/64/128/256px) | called {f.Called:P1} " +
            $"({f.CalledWhenCovered:P1} when looked at, {f.CalledWhenNotCovered:P1} when not)");

    internal static string DescribeTeacher(TeacherResult t) =>
        Markup.Escape(
            $"crop teacher ({t.Name}): {t.IncludePairs} include / {t.ExcludePairs} exclude crops | full-image yes {t.FullImageYesRate:P1} | " +
            $"include-crop yes {t.IncludeYesRate:P1}, exclude-crop no {t.ExcludeNoRate:P1}, balanced {t.BalancedAccuracy:P1} | " +
            $"confident (p>0.9 or <0.1) {t.ConfidentCoverage:P1} of answers, {t.ConfidentAccuracy:P1} right");

    internal static string DescribeOcclusion(OcclusionResult o) =>
        Markup.Escape(
            $"occlusion ({o.Name}, {o.Pairs} found pairs): still called with the object blanked {o.StillYesRate:P1} | " +
            $"confidence drop: object {o.ObjectDrop:F3}, same area elsewhere {o.ControlDrop:F3} | object hurts more {o.WinRate:P1} | " +
            $"object effect over all {o.ScoredPairs} pairs: {o.ObjectEffect:F2} log-odds (median {o.ObjectEffectMedian:F2})");

    internal static string DescribeLabelledContext(LabelledContextReport c) =>
        Markup.Escape(
            $"labelled context negatives: called where the object is absent {c.AbsentCalledRate:P1} of {c.Absent} (mean log-odds past the threshold {c.AbsentMeanLogit:F2}), " +
            $"where it's visible but untagged {c.PresentCalledRate:P1} of {c.Present} (mean {c.PresentMeanLogit:F2}); {c.Skipped} labelled pairs not in this held-out set");

    internal static string DescribeGold(GoldResult g) =>
        Markup.Escape(
            $"gold localization ({g.Name}, {g.Pairs} pairs): pointing {g.PointingAccuracy:P1}, energy in box {g.EnergyInBox:F3}, " +
            $"box precision {g.InstancePrecision:P2} / recall {g.InstanceRecall:P1}, count error {g.MeanCountError:F1}");

    internal static string DescribeLocalization(LocalizationReport r) =>
        Markup.Escape(
            string.Concat(r.Gold.Select(g =>
                $"gold {g.Name} ({g.Pairs} image-tag pairs): pointing {g.PointingAccuracy:P1}, energy in box {g.EnergyInBox:P1}, box precision {g.InstancePrecision:P1}, box recall {g.InstanceRecall:P1}, count error {g.MeanCountError:F2} | ")) +
            $"deletion gap {r.DeletionGap:F4} (top beats random {r.DeletionWinRate:P1}, {r.DeletionPairs} pairs) | " +
            $"two characters: heatmap correlation {r.TwoCharacterCorrelation:F3}, distinct peaks {r.TwoCharacterDistinctPeaks:P1} ({r.TwoCharacterImages} images) | " +
            $"crop consistency {r.CropConsistency:F3} ({r.CropPairs} pairs) | solo once-per-person tags with exactly one box {r.SoloSingleInstanceRate:P1} ({r.SoloPairs} pairs)");

    private static void PrintTags(string title, IEnumerable<TagEvaluation> tags)
    {
        var table = new Table().Title(Markup.Escape(title))
            .AddColumns("tag", "held-out positives", "hard negatives", "AP", "hard AP", "held-out F1", "threshold");
        foreach (var t in tags)
            table.AddRow(
                Markup.Escape(t.Tag), t.Positives.ToString(), t.HardNegatives.ToString(),
                t.AveragePrecision.ToString("F3"), double.IsNaN(t.HardAveragePrecision) ? "-" : t.HardAveragePrecision.ToString("F3"),
                double.IsNaN(t.HeldOutF1) ? "-" : t.HeldOutF1.ToString("F3"), t.Threshold.ToString("F3"));
        AnsiConsole.Write(table);
    }
}
