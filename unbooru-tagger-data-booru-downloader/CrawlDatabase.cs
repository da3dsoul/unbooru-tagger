using Microsoft.Data.Sqlite;

namespace UnbooruTagger.Crawler;

/// <summary>Per-(tag,site,phase) pagination state — lets a re-run of <c>crawl</c> resume exactly where it left off instead of re-listing pages already consumed.</summary>
public sealed record TagProgressState(string? Cursor, int PostsFetched, bool Done);

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

    /// <summary>Inserts or refreshes one tag's per-site survey counts.</summary>
    public Task UpsertTagSurveyAsync(string name, int? danbooruCount, int? gelbooruCount, bool eligible, DateTimeOffset surveyedAt, CancellationToken cancellationToken = default) =>
        WithLockAsync(async () =>
        {
            var command = _connection.CreateCommand();
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
            command.Parameters.AddWithValue("$name", name);
            command.Parameters.AddWithValue("$danbooru", (object?)danbooruCount ?? DBNull.Value);
            command.Parameters.AddWithValue("$gelbooru", (object?)gelbooruCount ?? DBNull.Value);
            command.Parameters.AddWithValue("$eligible", eligible ? 1 : 0);
            command.Parameters.AddWithValue("$surveyedAt", surveyedAt.ToString("O"));
            await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
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

    /// <summary>The cache row index a given md5 was already appended at, or null if this image has never been seen.</summary>
    public Task<int?> FindCacheRowIndexAsync(string md5, CancellationToken cancellationToken = default) =>
        WithLockAsync(async () =>
        {
            var command = _connection.CreateCommand();
            command.CommandText = "SELECT CacheRowIndex FROM Images WHERE Md5 = $md5;";
            command.Parameters.AddWithValue("$md5", md5);
            var result = await command.ExecuteScalarAsync(cancellationToken).ConfigureAwait(false);
            return result is null ? (int?)null : Convert.ToInt32(result);
        }, cancellationToken);

    /// <summary>
    /// Records a newly-appended image: its dedup row, its provenance, and — for every
    /// eligible tag it carries — credit toward that tag's <c>CombinedPositiveCount</c>
    /// (not just whichever tag drove the search that found it).
    /// </summary>
    public Task RecordNewImageAsync(
        string md5,
        int cacheRowIndex,
        int width,
        int height,
        DateTimeOffset downloadedAt,
        string site,
        long postId,
        string postUrl,
        string rating,
        DateTimeOffset postDate,
        IReadOnlyCollection<string> eligibleTagsOnPost,
        CancellationToken cancellationToken = default) =>
        WithLockAsync(async () =>
        {
            await using var transaction = _connection.BeginTransaction();

            var insertImage = _connection.CreateCommand();
            insertImage.Transaction = transaction;
            insertImage.CommandText =
                "INSERT INTO Images (Md5, CacheRowIndex, Width, Height, DownloadedAt) VALUES ($md5, $row, $w, $h, $downloadedAt);";
            insertImage.Parameters.AddWithValue("$md5", md5);
            insertImage.Parameters.AddWithValue("$row", cacheRowIndex);
            insertImage.Parameters.AddWithValue("$w", width);
            insertImage.Parameters.AddWithValue("$h", height);
            insertImage.Parameters.AddWithValue("$downloadedAt", downloadedAt.ToString("O"));
            await insertImage.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);

            var insertSource = _connection.CreateCommand();
            insertSource.Transaction = transaction;
            insertSource.CommandText =
                "INSERT OR IGNORE INTO ImageSources (Md5, Site, PostId, PostUrl, Rating, PostDate) VALUES ($md5, $site, $postId, $postUrl, $rating, $postDate);";
            insertSource.Parameters.AddWithValue("$md5", md5);
            insertSource.Parameters.AddWithValue("$site", site);
            insertSource.Parameters.AddWithValue("$postId", postId);
            insertSource.Parameters.AddWithValue("$postUrl", postUrl);
            insertSource.Parameters.AddWithValue("$rating", rating);
            insertSource.Parameters.AddWithValue("$postDate", postDate.ToString("O"));
            await insertSource.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);

            foreach (var tag in eligibleTagsOnPost)
            {
                var increment = _connection.CreateCommand();
                increment.Transaction = transaction;
                increment.CommandText = "UPDATE Tags SET CombinedPositiveCount = CombinedPositiveCount + 1 WHERE Name = $name;";
                increment.Parameters.AddWithValue("$name", tag);
                await increment.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
            }

            await transaction.CommitAsync(cancellationToken).ConfigureAwait(false);
        }, cancellationToken);

    /// <summary>Adds a provenance row for a post whose image was already recorded under a different site/post — the same artwork cross-posted, so no new <c>Images</c> row or tag credit is needed, just the additional source.</summary>
    public Task RecordAdditionalSourceAsync(string md5, string site, long postId, string postUrl, string rating, DateTimeOffset postDate, CancellationToken cancellationToken = default) =>
        WithLockAsync(async () =>
        {
            var command = _connection.CreateCommand();
            command.CommandText =
                "INSERT OR IGNORE INTO ImageSources (Md5, Site, PostId, PostUrl, Rating, PostDate) VALUES ($md5, $site, $postId, $postUrl, $rating, $postDate);";
            command.Parameters.AddWithValue("$md5", md5);
            command.Parameters.AddWithValue("$site", site);
            command.Parameters.AddWithValue("$postId", postId);
            command.Parameters.AddWithValue("$postUrl", postUrl);
            command.Parameters.AddWithValue("$rating", rating);
            command.Parameters.AddWithValue("$postDate", postDate.ToString("O"));
            await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
        }, cancellationToken);

    public Task<int> GetCombinedPositiveCountAsync(string tagName, CancellationToken cancellationToken = default) =>
        WithLockAsync(async () =>
        {
            var command = _connection.CreateCommand();
            command.CommandText = "SELECT CombinedPositiveCount FROM Tags WHERE Name = $name;";
            command.Parameters.AddWithValue("$name", tagName);
            var result = await command.ExecuteScalarAsync(cancellationToken).ConfigureAwait(false);
            return result is null ? 0 : Convert.ToInt32(result);
        }, cancellationToken);

    public Task<int> GetTotalImageCountAsync(CancellationToken cancellationToken = default) =>
        WithLockAsync(async () =>
        {
            var command = _connection.CreateCommand();
            command.CommandText = "SELECT COUNT(*) FROM Images;";
            var result = await command.ExecuteScalarAsync(cancellationToken).ConfigureAwait(false);
            return Convert.ToInt32(result);
        }, cancellationToken);

    public void Dispose()
    {
        _connection.Dispose();
        _lock.Dispose();
    }
}
