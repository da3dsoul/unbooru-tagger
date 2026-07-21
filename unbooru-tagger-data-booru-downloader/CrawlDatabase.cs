using Microsoft.Data.Sqlite;

namespace UnbooruTagger.Crawler;

/// <summary>Per-(tag,site,phase) pagination state — lets a re-run of <c>crawl</c> resume exactly where it left off instead of re-listing pages already consumed.</summary>
public sealed record TagProgressState(string? Cursor, int PostsFetched, bool Done);

/// <summary>
/// One newly-downloaded, not-yet-durable image, buffered by <see cref="DatasetCrawler"/>
/// until its page's checkpoint commits it — see <see cref="CrawlDatabase.CommitPendingImagesAsync"/>
/// for why this can't just be written to <c>crawl.sqlite</c> the moment it's appended.
/// </summary>
public sealed record PendingNewImage(
    string Md5,
    int CacheRowIndex,
    int Width,
    int Height,
    DateTimeOffset DownloadedAt,
    string Site,
    long PostId,
    string PostUrl,
    string Rating,
    DateTimeOffset PostDate,
    IReadOnlyCollection<string> EligibleTags,
    ulong PHash);

/// <summary>A post whose image matched an already-known one (exact md5 or perceptual near-duplicate) — just another provenance row against the canonical image's md5, buffered the same way as <see cref="PendingNewImage"/>.</summary>
public sealed record PendingAdditionalSource(
    string CanonicalMd5,
    string Site,
    long PostId,
    string PostUrl,
    string Rating,
    DateTimeOffset PostDate);

/// <summary>
/// Crawl-only bookkeeping, stored as <c>crawl.sqlite</c> inside the dataset
/// <c>--output-dir</c> alongside <c>images.bin</c>/<c>tag_rows.jsonl</c>/
/// <c>tag_vocabulary.json</c>. Never read by training/inference — this is purely this
/// project's own survey results, per-tag/site resumability, and md5 dedup index.
/// Wrapped in a single-writer lock: SQLite doesn't support true concurrent writers on
/// one file, and this project drives two site workers concurrently. Throughput here is
/// bounded by API rate limits (a handful of requests/second combined), not by
/// serializing on this lock, so a plain <see cref="SemaphoreSlim"/> is more than enough
/// rather than needing a connection pool.
/// </summary>
public sealed class CrawlDatabase : IDisposable
{
    private readonly SqliteConnection _connection;
    private readonly SemaphoreSlim _lock = new(1, 1);

    private CrawlDatabase(SqliteConnection connection)
    {
        _connection = connection;
    }

    public static async Task<CrawlDatabase> OpenOrCreateAsync(string outputDirectory, CancellationToken cancellationToken = default)
    {
        Directory.CreateDirectory(outputDirectory);
        var path = Path.Combine(outputDirectory, "crawl.sqlite");

        var connection = new SqliteConnection($"Data Source={path}");
        await connection.OpenAsync(cancellationToken).ConfigureAwait(false);

        var db = new CrawlDatabase(connection);
        await db.MigrateAsync(cancellationToken).ConfigureAwait(false);
        return db;
    }

    private async Task MigrateAsync(CancellationToken cancellationToken)
    {
        const string sql =
            """
            CREATE TABLE IF NOT EXISTS Tags (
                Name TEXT PRIMARY KEY,
                DanbooruCount INTEGER NULL,
                GelbooruCount INTEGER NULL,
                Eligible INTEGER NOT NULL,
                SurveyedAt TEXT NOT NULL,
                CombinedPositiveCount INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS TagProgress (
                TagName TEXT NOT NULL,
                Site TEXT NOT NULL,
                Phase TEXT NOT NULL,
                Cursor TEXT NULL,
                PostsFetched INTEGER NOT NULL DEFAULT 0,
                Done INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (TagName, Site, Phase)
            );

            CREATE TABLE IF NOT EXISTS Images (
                Md5 TEXT PRIMARY KEY,
                CacheRowIndex INTEGER NOT NULL,
                Width INTEGER NOT NULL,
                Height INTEGER NOT NULL,
                DownloadedAt TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS ImageSources (
                Md5 TEXT NOT NULL,
                Site TEXT NOT NULL,
                PostId INTEGER NOT NULL,
                PostUrl TEXT NOT NULL,
                Rating TEXT NOT NULL,
                PostDate TEXT NOT NULL,
                PRIMARY KEY (Site, PostId)
            );
            """;

        var command = _connection.CreateCommand();
        command.CommandText = sql;
        await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);

        // Added after the initial schema — migrate in place so an interrupted crawl
        // against an older crawl.sqlite can resume instead of needing a fresh one.
        await EnsureColumnAsync("Images", "PHash", "INTEGER NOT NULL DEFAULT 0", cancellationToken).ConfigureAwait(false);
    }

    private async Task EnsureColumnAsync(string table, string column, string definition, CancellationToken cancellationToken)
    {
        var checkCommand = _connection.CreateCommand();
        checkCommand.CommandText = $"PRAGMA table_info({table});";
        var hasColumn = false;
        await using (var reader = await checkCommand.ExecuteReaderAsync(cancellationToken).ConfigureAwait(false))
        {
            while (await reader.ReadAsync(cancellationToken).ConfigureAwait(false))
            {
                if (string.Equals(reader.GetString(1), column, StringComparison.Ordinal))
                {
                    hasColumn = true;
                    break;
                }
            }
        }

        if (!hasColumn)
        {
            var alterCommand = _connection.CreateCommand();
            alterCommand.CommandText = $"ALTER TABLE {table} ADD COLUMN {column} {definition};";
            await alterCommand.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
        }
    }

    private async Task<T> WithLockAsync<T>(Func<Task<T>> action, CancellationToken cancellationToken)
    {
        await _lock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            return await action().ConfigureAwait(false);
        }
        finally
        {
            _lock.Release();
        }
    }

    private Task WithLockAsync(Func<Task> action, CancellationToken cancellationToken) =>
        WithLockAsync(async () =>
        {
            await action().ConfigureAwait(false);
            return true;
        }, cancellationToken);

    /// <summary>
    /// Upserts every surveyed tag's per-site counts in a single transaction. A
    /// `survey-tags` run against a large vocabulary can have tens of thousands of
    /// eligible tags; one auto-committed SQLite write each (its own fsync) is the
    /// dominant cost of the whole command and gives no visible progress, whereas one
    /// transaction is a single fsync on commit — <paramref name="onRowWritten"/> lets
    /// the caller still report per-row progress.
    /// </summary>
    public Task UpsertTagSurveysAsync(
        IEnumerable<(string Name, int? DanbooruCount, int? GelbooruCount, bool Eligible)> entries,
        DateTimeOffset surveyedAt,
        Action<int>? onRowWritten,
        CancellationToken cancellationToken = default) =>
        WithLockAsync(async () =>
        {
            await using var transaction = _connection.BeginTransaction();

            var command = _connection.CreateCommand();
            command.Transaction = transaction;
            command.CommandText =
                """
                INSERT INTO Tags (Name, DanbooruCount, GelbooruCount, Eligible, SurveyedAt, CombinedPositiveCount)
                VALUES ($name, $danbooru, $gelbooru, $eligible, $surveyedAt, 0)
                ON CONFLICT(Name) DO UPDATE SET
                    DanbooruCount = $danbooru,
                    GelbooruCount = $gelbooru,
                    Eligible = $eligible,
                    SurveyedAt = $surveyedAt;
                """;
            var nameParam = command.Parameters.Add("$name", SqliteType.Text);
            var danbooruParam = command.Parameters.Add("$danbooru", SqliteType.Integer);
            var gelbooruParam = command.Parameters.Add("$gelbooru", SqliteType.Integer);
            var eligibleParam = command.Parameters.Add("$eligible", SqliteType.Integer);
            var surveyedAtParam = command.Parameters.Add("$surveyedAt", SqliteType.Text);
            surveyedAtParam.Value = surveyedAt.ToString("O");

            var written = 0;
            foreach (var entry in entries)
            {
                cancellationToken.ThrowIfCancellationRequested();

                nameParam.Value = entry.Name;
                danbooruParam.Value = (object?)entry.DanbooruCount ?? DBNull.Value;
                gelbooruParam.Value = (object?)entry.GelbooruCount ?? DBNull.Value;
                eligibleParam.Value = entry.Eligible ? 1 : 0;
                await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);

                written++;
                onRowWritten?.Invoke(written);
            }

            await transaction.CommitAsync(cancellationToken).ConfigureAwait(false);
        }, cancellationToken);

    /// <summary>All surveyed tags, eligible or not — the caller filters/orders as needed (see <see cref="TagEligibility"/>/<see cref="CrawlScheduling"/>).</summary>
    public Task<IReadOnlyList<TagSurveyResult>> GetAllSurveyedTagsAsync(CancellationToken cancellationToken = default) =>
        WithLockAsync(async () =>
        {
            var command = _connection.CreateCommand();
            command.CommandText = "SELECT Name, DanbooruCount, GelbooruCount FROM Tags;";
            var results = new List<TagSurveyResult>();
            await using var reader = await command.ExecuteReaderAsync(cancellationToken).ConfigureAwait(false);
            while (await reader.ReadAsync(cancellationToken).ConfigureAwait(false))
            {
                results.Add(new TagSurveyResult(
                    reader.GetString(0),
                    reader.IsDBNull(1) ? null : reader.GetInt32(1),
                    reader.IsDBNull(2) ? null : reader.GetInt32(2)));
            }
            return (IReadOnlyList<TagSurveyResult>)results;
        }, cancellationToken);

    public Task<TagProgressState> GetTagProgressAsync(string tagName, string site, string phase, CancellationToken cancellationToken = default) =>
        WithLockAsync(async () =>
        {
            var command = _connection.CreateCommand();
            command.CommandText = "SELECT Cursor, PostsFetched, Done FROM TagProgress WHERE TagName = $tag AND Site = $site AND Phase = $phase;";
            command.Parameters.AddWithValue("$tag", tagName);
            command.Parameters.AddWithValue("$site", site);
            command.Parameters.AddWithValue("$phase", phase);

            await using var reader = await command.ExecuteReaderAsync(cancellationToken).ConfigureAwait(false);
            if (!await reader.ReadAsync(cancellationToken).ConfigureAwait(false))
                return new TagProgressState(null, 0, false);

            return new TagProgressState(
                reader.IsDBNull(0) ? null : reader.GetString(0),
                reader.GetInt32(1),
                reader.GetInt32(2) != 0);
        }, cancellationToken);

    public Task SaveTagProgressAsync(string tagName, string site, string phase, TagProgressState state, CancellationToken cancellationToken = default) =>
        WithLockAsync(async () =>
        {
            var command = _connection.CreateCommand();
            command.CommandText =
                """
                INSERT INTO TagProgress (TagName, Site, Phase, Cursor, PostsFetched, Done)
                VALUES ($tag, $site, $phase, $cursor, $postsFetched, $done)
                ON CONFLICT(TagName, Site, Phase) DO UPDATE SET
                    Cursor = $cursor,
                    PostsFetched = $postsFetched,
                    Done = $done;
                """;
            command.Parameters.AddWithValue("$tag", tagName);
            command.Parameters.AddWithValue("$site", site);
            command.Parameters.AddWithValue("$phase", phase);
            command.Parameters.AddWithValue("$cursor", (object?)state.Cursor ?? DBNull.Value);
            command.Parameters.AddWithValue("$postsFetched", state.PostsFetched);
            command.Parameters.AddWithValue("$done", state.Done ? 1 : 0);
            await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
        }, cancellationToken);

    /// <summary>
    /// Every already-cached image's md5/cache-row-index/perceptual-hash, for seeding the
    /// in-memory dedup index (exact and near-duplicate) a crawl run checks new downloads
    /// against — see <see cref="PerceptualHash"/> and <see cref="DatasetCrawler"/>'s
    /// working state. Kept purely in memory during a run rather than re-querying per
    /// post: with a corpus that can reach millions of rows, a DB round trip per post
    /// would make dedup checks the dominant cost of the whole crawl.
    /// </summary>
    public Task<IReadOnlyList<(string Md5, int CacheRowIndex, ulong PHash)>> GetAllImagesAsync(CancellationToken cancellationToken = default) =>
        WithLockAsync(async () =>
        {
            var command = _connection.CreateCommand();
            command.CommandText = "SELECT Md5, CacheRowIndex, PHash FROM Images;";
            var results = new List<(string, int, ulong)>();
            await using var reader = await command.ExecuteReaderAsync(cancellationToken).ConfigureAwait(false);
            while (await reader.ReadAsync(cancellationToken).ConfigureAwait(false))
                results.Add((reader.GetString(0), reader.GetInt32(1), unchecked((ulong)reader.GetInt64(2))));
            return (IReadOnlyList<(string Md5, int CacheRowIndex, ulong PHash)>)results;
        }, cancellationToken);

    /// <summary>Every surveyed tag's durable <c>CombinedPositiveCount</c>, for seeding the in-memory counters a crawl run updates live and only checkpoints periodically (see <see cref="CommitPendingImagesAsync"/>).</summary>
    public Task<IReadOnlyDictionary<string, int>> GetAllCombinedPositiveCountsAsync(CancellationToken cancellationToken = default) =>
        WithLockAsync(async () =>
        {
            var command = _connection.CreateCommand();
            command.CommandText = "SELECT Name, CombinedPositiveCount FROM Tags;";
            var results = new Dictionary<string, int>(StringComparer.Ordinal);
            await using var reader = await command.ExecuteReaderAsync(cancellationToken).ConfigureAwait(false);
            while (await reader.ReadAsync(cancellationToken).ConfigureAwait(false))
                results[reader.GetString(0)] = reader.GetInt32(1);
            return (IReadOnlyDictionary<string, int>)results;
        }, cancellationToken);

    /// <summary>
    /// Durably commits every image/source buffered since the last checkpoint, in one
    /// transaction. Deliberately not one write per image (as the crawl processes posts):
    /// this is called only once the same checkpoint has already flushed the pixel/label
    /// cache and the vocabulary delta, so a crash before this point just means a
    /// still-pending image gets redownloaded and reprocessed next run (wasted work, not
    /// corruption) — writing it durably any earlier would let this dedup index end up
    /// referencing cache rows the reopened (and truncated-back-to-last-flush) cache file
    /// no longer has, which is silent, permanent data loss instead: those images would
    /// look already-cached forever and never get retried.
    /// </summary>
    public Task CommitPendingImagesAsync(
        IReadOnlyList<PendingNewImage> newImages,
        IReadOnlyList<PendingAdditionalSource> additionalSources,
        CancellationToken cancellationToken = default) =>
        WithLockAsync(async () =>
        {
            await using var transaction = _connection.BeginTransaction();

            var insertImage = _connection.CreateCommand();
            insertImage.Transaction = transaction;
            insertImage.CommandText =
                "INSERT INTO Images (Md5, CacheRowIndex, Width, Height, DownloadedAt, PHash) VALUES ($md5, $row, $w, $h, $downloadedAt, $phash);";
            var imgMd5 = insertImage.Parameters.Add("$md5", SqliteType.Text);
            var imgRow = insertImage.Parameters.Add("$row", SqliteType.Integer);
            var imgWidth = insertImage.Parameters.Add("$w", SqliteType.Integer);
            var imgHeight = insertImage.Parameters.Add("$h", SqliteType.Integer);
            var imgDownloadedAt = insertImage.Parameters.Add("$downloadedAt", SqliteType.Text);
            var imgPHash = insertImage.Parameters.Add("$phash", SqliteType.Integer);

            var insertSource = _connection.CreateCommand();
            insertSource.Transaction = transaction;
            insertSource.CommandText =
                "INSERT OR IGNORE INTO ImageSources (Md5, Site, PostId, PostUrl, Rating, PostDate) VALUES ($md5, $site, $postId, $postUrl, $rating, $postDate);";
            var srcMd5 = insertSource.Parameters.Add("$md5", SqliteType.Text);
            var srcSite = insertSource.Parameters.Add("$site", SqliteType.Text);
            var srcPostId = insertSource.Parameters.Add("$postId", SqliteType.Integer);
            var srcPostUrl = insertSource.Parameters.Add("$postUrl", SqliteType.Text);
            var srcRating = insertSource.Parameters.Add("$rating", SqliteType.Text);
            var srcPostDate = insertSource.Parameters.Add("$postDate", SqliteType.Text);

            var incrementCount = _connection.CreateCommand();
            incrementCount.Transaction = transaction;
            incrementCount.CommandText = "UPDATE Tags SET CombinedPositiveCount = CombinedPositiveCount + 1 WHERE Name = $name;";
            var incName = incrementCount.Parameters.Add("$name", SqliteType.Text);

            foreach (var image in newImages)
            {
                cancellationToken.ThrowIfCancellationRequested();

                imgMd5.Value = image.Md5;
                imgRow.Value = image.CacheRowIndex;
                imgWidth.Value = image.Width;
                imgHeight.Value = image.Height;
                imgDownloadedAt.Value = image.DownloadedAt.ToString("O");
                imgPHash.Value = unchecked((long)image.PHash);
                await insertImage.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);

                srcMd5.Value = image.Md5;
                srcSite.Value = image.Site;
                srcPostId.Value = image.PostId;
                srcPostUrl.Value = image.PostUrl;
                srcRating.Value = image.Rating;
                srcPostDate.Value = image.PostDate.ToString("O");
                await insertSource.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);

                foreach (var tag in image.EligibleTags)
                {
                    incName.Value = tag;
                    await incrementCount.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
                }
            }

            foreach (var source in additionalSources)
            {
                cancellationToken.ThrowIfCancellationRequested();

                srcMd5.Value = source.CanonicalMd5;
                srcSite.Value = source.Site;
                srcPostId.Value = source.PostId;
                srcPostUrl.Value = source.PostUrl;
                srcRating.Value = source.Rating;
                srcPostDate.Value = source.PostDate.ToString("O");
                await insertSource.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
            }

            await transaction.CommitAsync(cancellationToken).ConfigureAwait(false);
        }, cancellationToken);

    public void Dispose()
    {
        _connection.Dispose();
        _lock.Dispose();
    }
}
