using System.Text.Json;
using Microsoft.Data.Sqlite;

namespace UnbooruTagger.Core.Dataset;

/// <summary>
/// SQLite-backed storage for <see cref="PreprocessedDatasetCacheWriter"/>/<see cref="PreprocessedDatasetCacheReader"/>'s
/// per-image tag-row indices — one table, <c>TagRows(CacheRowIndex INTEGER PRIMARY KEY,
/// TagRows BLOB)</c>, each row's tag indices packed as contiguous little-endian
/// <c>int32</c>s rather than a JSON array of text. Replaces the original one-JSON-array-
/// per-line <c>tag_rows.jsonl</c> format: bulk-loading every row at startup (needed both
/// to seed a crawl's in-memory duplicate-tag-merge state and to feed training) used to
/// mean reading and JSON-parsing every line of a file that can run into the hundreds of
/// MB on a multi-million-image corpus, every single time. A single indexed SQLite scan
/// over packed binary rows is dramatically cheaper per row, with no string allocation or
/// JSON tokenizing at all.
///
/// Crash-safety falls out of SQLite's own transaction log for free: every <see cref="Append"/>
/// since the last <see cref="Flush"/> lives in one open, uncommitted transaction, so a
/// crash before the next <see cref="Flush"/> leaves nothing durable — SQLite rolls it back
/// automatically the moment the file is reopened. The old JSONL format had no equivalent
/// (a <see cref="StreamWriter"/> can still leave dangling trailing lines visible on disk
/// after a crash), which is why <see cref="PreprocessedDatasetCacheWriter"/> used to need
/// its own manual dangling-line probe/truncate logic — none of that is needed anymore.
/// </summary>
internal sealed class TagRowStore : IDisposable
{
    internal const string DatabaseFileName = "tag_rows.sqlite";

    /// <summary>How often (in rows) a migration/read pass reports back to an <c>onProgress</c> callback — same reasoning as the writer's own resume-progress cadence.</summary>
    private const int ProgressReportInterval = 5_000;

    private readonly SqliteConnection _connection;

    /// <summary>Non-null only for a store opened via <see cref="OpenForWriting"/> — a read-only store (<see cref="OpenForReading"/>) never buffers anything, so it never needs one.</summary>
    private SqliteTransaction? _pendingTransaction;

    private TagRowStore(SqliteConnection connection, SqliteTransaction? pendingTransaction)
    {
        _connection = connection;
        _pendingTransaction = pendingTransaction;
    }

    /// <summary>
    /// Opens (or creates) <see cref="DatabaseFileName"/> inside <paramref name="directory"/>.
    /// If the database doesn't exist yet but a legacy <c>tag_rows.jsonl</c> does, migrates
    /// it automatically (see <see cref="MigrateFromJsonl"/>) before returning — a one-time
    /// cost for a cache that predates this store, never repeated afterward. Any row at or
    /// past <paramref name="imageCount"/> (the pixel file's own authoritative count) is
    /// dropped on open: it's either a never-confirmed dangling row from a page appended
    /// after the last successful <see cref="Flush"/> and then abandoned by a crash, or
    /// (much more rarely) orphaned by <see cref="Flush"/>'s commit-then-patch-header
    /// ordering — either way it isn't part of the confirmed corpus. Begins a transaction
    /// immediately so <see cref="Append"/> has somewhere to buffer into.
    /// </summary>
    public static TagRowStore OpenForWriting(string directory, int imageCount, Action<string, int, int>? onMigrateProgress = null)
    {
        var connection = OpenConnectionWithMigration(directory, imageCount, onMigrateProgress);

        using (var trim = connection.CreateCommand())
        {
            trim.CommandText = "DELETE FROM TagRows WHERE CacheRowIndex >= $imageCount;";
            trim.Parameters.AddWithValue("$imageCount", imageCount);
            trim.ExecuteNonQuery();
        }

        return new TagRowStore(connection, connection.BeginTransaction());
    }

    /// <summary>
    /// Same open-and-migrate-if-needed behavior as <see cref="OpenForWriting"/>, for a
    /// read-only consumer (<see cref="PreprocessedDatasetCacheReader"/>, i.e. training)
    /// that only ever calls <see cref="ReadAll"/> — no transaction is opened (there's
    /// nothing to buffer for a append-free caller) and the dangling-row trim is skipped,
    /// since <see cref="ReadAll"/> already bounds its own query by <c>imageCount</c>
    /// regardless of what else the table happens to contain.
    /// </summary>
    public static TagRowStore OpenForReading(string directory, int imageCount, Action<string, int, int>? onMigrateProgress = null) =>
        new(OpenConnectionWithMigration(directory, imageCount, onMigrateProgress), pendingTransaction: null);

    /// <summary>
    /// Creates a brand new, empty store, deleting any pre-existing <see cref="DatabaseFileName"/>
    /// first — the SQLite equivalent of <see cref="PreprocessedDatasetCacheWriter"/>'s
    /// own fresh-create branch truncating <c>images.bin</c> via <c>File.Create</c>.
    /// Never migrates a legacy <c>tag_rows.jsonl</c>: a fresh cache has nothing to
    /// resume, so any such file sitting in <paramref name="directory"/> is unrelated
    /// leftover data, not this cache's history.
    /// </summary>
    public static TagRowStore CreateFresh(string directory)
    {
        var dbPath = Path.Combine(directory, DatabaseFileName);
        if (File.Exists(dbPath))
            File.Delete(dbPath);

        var connection = new SqliteConnection($"Data Source={dbPath}");
        connection.Open();
        CreateSchema(connection);
        return new TagRowStore(connection, connection.BeginTransaction());
    }

    private static SqliteConnection OpenConnectionWithMigration(string directory, int imageCount, Action<string, int, int>? onMigrateProgress)
    {
        var dbPath = Path.Combine(directory, DatabaseFileName);
        var jsonlPath = Path.Combine(directory, PreprocessedDatasetCache.LabelsFileName);
        var isNewDatabase = !File.Exists(dbPath);

        if (isNewDatabase && File.Exists(jsonlPath))
            MigrateFromJsonl(dbPath, jsonlPath, imageCount, onMigrateProgress);

        var connection = new SqliteConnection($"Data Source={dbPath}");
        connection.Open();
        CreateSchema(connection);
        return connection;
    }

    private static void CreateSchema(SqliteConnection connection)
    {
        using var create = connection.CreateCommand();
        create.CommandText = "CREATE TABLE IF NOT EXISTS TagRows (CacheRowIndex INTEGER PRIMARY KEY, TagRows BLOB NOT NULL);";
        create.ExecuteNonQuery();
    }

    /// <summary>
    /// One-time conversion of an existing <c>tag_rows.jsonl</c> to <see cref="DatabaseFileName"/>,
    /// built entirely under a temporary name and only renamed into place once every row
    /// has committed successfully — so a crash mid-migration leaves neither a half-built
    /// database (the temp file, still under its temp name, gets deleted and the whole
    /// migration retried from scratch next time) nor a lost original (the jsonl file is
    /// only renamed to <c>.migrated.bak</c> after the database swap above already
    /// succeeded). Renaming the database into place BEFORE backing up the jsonl (rather
    /// than the reverse) matters: if a crash landed between the two, the reverse order
    /// would leave neither a valid database nor a readable jsonl file behind.
    /// </summary>
    private static void MigrateFromJsonl(string dbPath, string jsonlPath, int imageCount, Action<string, int, int>? onProgress)
    {
        var tempDbPath = dbPath + ".migrating";
        if (File.Exists(tempDbPath))
            File.Delete(tempDbPath); // leftover from a prior crashed attempt -- start clean

        using (var tempConnection = new SqliteConnection($"Data Source={tempDbPath}"))
        {
            tempConnection.Open();
            CreateSchema(tempConnection);

            using var transaction = tempConnection.BeginTransaction();
            using var insert = tempConnection.CreateCommand();
            insert.Transaction = transaction;
            insert.CommandText = "INSERT INTO TagRows (CacheRowIndex, TagRows) VALUES ($idx, $blob);";
            var idxParam = insert.Parameters.Add("$idx", SqliteType.Integer);
            var blobParam = insert.Parameters.Add("$blob", SqliteType.Blob);

            var index = 0;
            foreach (var line in File.ReadLines(jsonlPath).Take(imageCount))
            {
                var tagRows = JsonSerializer.Deserialize<int[]>(line)
                              ?? throw new InvalidDataException($"'{jsonlPath}' contains an invalid tag-row label line.");
                idxParam.Value = (long)index;
                blobParam.Value = Pack(tagRows);
                insert.ExecuteNonQuery();

                index++;
                if (onProgress is not null && index % ProgressReportInterval == 0)
                    onProgress("migrating tag rows to SQLite", index, imageCount);
            }

            transaction.Commit();
            onProgress?.Invoke("migrating tag rows to SQLite", imageCount, imageCount);
        }

        // Atomic on the same volume (same directory): this only becomes the real
        // database once every row above committed. A crash before this line leaves only
        // the ".migrating" temp file (deleted and retried next time) and the original
        // jsonl untouched.
        File.Move(tempDbPath, dbPath);

        // Kept as a backup rather than deleted -- cheap insurance, and matches this
        // project's other migration (LBX1 -> LBX2 pixel format)'s own back-up-don't-
        // delete pattern.
        File.Move(jsonlPath, jsonlPath + ".migrated.bak", overwrite: true);
    }

    /// <summary>Buffers one row into the currently-open (not yet committed) transaction — durable only once <see cref="Flush"/> commits it. Only valid on a store opened via <see cref="OpenForWriting"/>.</summary>
    public void Append(int cacheRowIndex, IReadOnlyList<int> tagRows)
    {
        if (_pendingTransaction is null)
            throw new InvalidOperationException($"{nameof(Append)} requires a store opened via {nameof(OpenForWriting)}.");

        using var insert = _connection.CreateCommand();
        insert.Transaction = _pendingTransaction;
        insert.CommandText = "INSERT INTO TagRows (CacheRowIndex, TagRows) VALUES ($idx, $blob);";
        insert.Parameters.AddWithValue("$idx", (long)cacheRowIndex);
        insert.Parameters.AddWithValue("$blob", Pack(tagRows));
        insert.ExecuteNonQuery();
    }

    /// <summary>Commits every row appended since the last call and opens a new transaction for whatever comes next. Only valid on a store opened via <see cref="OpenForWriting"/>.</summary>
    public void Flush()
    {
        if (_pendingTransaction is null)
            throw new InvalidOperationException($"{nameof(Flush)} requires a store opened via {nameof(OpenForWriting)}.");

        _pendingTransaction.Commit();
        _pendingTransaction.Dispose();
        _pendingTransaction = _connection.BeginTransaction();
    }

    /// <summary>
    /// Overwrites specific already-committed rows' tag sets — the SQLite equivalent of
    /// the old format's full-label-file rewrite, except now it's exactly <c>UPDATE</c>
    /// statements scoped to the rows that actually changed, not a read-modify-write of
    /// the entire file. Commits immediately (unlike <see cref="Append"/>, which only
    /// becomes durable at the next <see cref="Flush"/>) — matching the old format's own
    /// guarantee, since the caller only calls this right after a <see cref="Flush"/>
    /// (see <see cref="PreprocessedDatasetCacheWriter.MergeTagRows"/>'s own doc comment)
    /// expecting it to be immediately safe.
    ///
    /// SQLite doesn't allow a second transaction on a connection that already has one
    /// open, so a store opened via <see cref="OpenForWriting"/> always has SOME
    /// transaction pending here (freshly begun, empty, by the <see cref="Flush"/> that
    /// must have just run) — this commits that one (a no-op, since nothing's been
    /// appended into it yet), runs the update in its own immediately-committed
    /// transaction, then reopens a fresh pending one so <see cref="Append"/> keeps
    /// working exactly as before.
    /// </summary>
    public void MergeRows(IReadOnlyDictionary<int, IReadOnlyList<int>> tagRowsByIndex)
    {
        if (tagRowsByIndex.Count == 0)
            return;

        var wasWriting = _pendingTransaction is not null;
        if (wasWriting)
        {
            _pendingTransaction!.Commit();
            _pendingTransaction.Dispose();
            _pendingTransaction = null;
        }

        using (var transaction = _connection.BeginTransaction())
        {
            using var update = _connection.CreateCommand();
            update.Transaction = transaction;
            update.CommandText = "UPDATE TagRows SET TagRows = $blob WHERE CacheRowIndex = $idx;";
            var blobParam = update.Parameters.Add("$blob", SqliteType.Blob);
            var idxParam = update.Parameters.Add("$idx", SqliteType.Integer);

            foreach (var (rowIndex, tagRows) in tagRowsByIndex)
            {
                blobParam.Value = Pack(tagRows);
                idxParam.Value = (long)rowIndex;
                update.ExecuteNonQuery();
            }

            transaction.Commit();
        }

        if (wasWriting)
            _pendingTransaction = _connection.BeginTransaction();
    }

    /// <summary>Reads every confirmed row (<c>CacheRowIndex &lt; imageCount</c>), in index order.</summary>
    public IReadOnlyList<IReadOnlyList<int>> ReadAll(int imageCount, Action<int, int>? onProgress = null)
    {
        var result = new List<int[]>(imageCount);
        for (var i = 0; i < imageCount; i++)
            result.Add([]);

        using var select = _connection.CreateCommand();
        select.CommandText = "SELECT CacheRowIndex, TagRows FROM TagRows WHERE CacheRowIndex < $imageCount ORDER BY CacheRowIndex;";
        select.Parameters.AddWithValue("$imageCount", imageCount);

        using var reader = select.ExecuteReader();
        var read = 0;
        while (reader.Read())
        {
            var index = (int)reader.GetInt64(0);
            using var blobStream = reader.GetStream(1);
            using var buffer = new MemoryStream();
            blobStream.CopyTo(buffer);
            result[index] = Unpack(buffer.ToArray());

            read++;
            if (onProgress is not null && read % ProgressReportInterval == 0)
                onProgress(read, imageCount);
        }

        onProgress?.Invoke(imageCount, imageCount);
        return result;
    }

    private static byte[] Pack(IReadOnlyList<int> tagRows)
    {
        var blob = new byte[tagRows.Count * sizeof(int)];
        for (var i = 0; i < tagRows.Count; i++)
            BitConverter.TryWriteBytes(blob.AsSpan(i * sizeof(int), sizeof(int)), tagRows[i]);
        return blob;
    }

    private static int[] Unpack(byte[] blob)
    {
        var tagRows = new int[blob.Length / sizeof(int)];
        for (var i = 0; i < tagRows.Length; i++)
            tagRows[i] = BitConverter.ToInt32(blob, i * sizeof(int));
        return tagRows;
    }

    public void Dispose()
    {
        if (_pendingTransaction is not null)
        {
            _pendingTransaction.Commit();
            _pendingTransaction.Dispose();
        }

        _connection.Dispose();
    }
}
