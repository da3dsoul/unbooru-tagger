using Microsoft.Data.Sqlite;

namespace UnbooruTagger.Crawler;

/// <summary>
/// Writes a fresh <c>tag_rows.sqlite</c> from scratch for <c>rebuild-vocabulary</c>,
/// schema- and packing-compatible with <c>UnbooruTagger.Core.Dataset.TagRowStore</c>
/// (<c>TagRows(CacheRowIndex INTEGER PRIMARY KEY, TagRows BLOB NOT NULL)</c>, each row's
/// tag indices packed as contiguous little-endian int32s) -- reimplemented here rather
/// than reused directly because <c>TagRowStore</c> is internal to
/// <c>unbooru-tagger-core</c> and its own <c>Append</c> is written for the crawler's
/// normal one-image-at-a-time, always-growing-<c>ImageCount</c> flow, not a full-corpus
/// rewrite of already-known rows. Built entirely under a temp name and only renamed into
/// place once every row has committed, so a crash mid-rebuild leaves the original
/// (already backed up by the caller anyway) untouched instead of a half-written database.
/// </summary>
internal static class TagRowsSqliteWriter
{
    public static void WriteAtomically(string directory, IReadOnlyList<(int CacheRowIndex, IReadOnlyList<int> TagRows)> rows)
    {
        var finalPath = Path.Combine(directory, "tag_rows.sqlite");
        var tempPath = finalPath + ".rebuilding";
        if (File.Exists(tempPath))
            File.Delete(tempPath);

        using (var connection = new SqliteConnection($"Data Source={tempPath}"))
        {
            connection.Open();

            using (var create = connection.CreateCommand())
            {
                create.CommandText = "CREATE TABLE TagRows (CacheRowIndex INTEGER PRIMARY KEY, TagRows BLOB NOT NULL);";
                create.ExecuteNonQuery();
            }

            using var transaction = connection.BeginTransaction();
            using var insert = connection.CreateCommand();
            insert.Transaction = transaction;
            insert.CommandText = "INSERT INTO TagRows (CacheRowIndex, TagRows) VALUES ($idx, $blob);";
            var idxParam = insert.Parameters.Add("$idx", SqliteType.Integer);
            var blobParam = insert.Parameters.Add("$blob", SqliteType.Blob);

            foreach (var (cacheRowIndex, tagRows) in rows)
            {
                idxParam.Value = (long)cacheRowIndex;
                blobParam.Value = Pack(tagRows);
                insert.ExecuteNonQuery();
            }

            transaction.Commit();
        }

        // SqliteConnection.Dispose() above already closed the file -- safe to move now.
        File.Move(tempPath, finalPath, overwrite: true);
    }

    private static byte[] Pack(IReadOnlyList<int> tagRows)
    {
        var blob = new byte[tagRows.Count * sizeof(int)];
        for (var i = 0; i < tagRows.Count; i++)
            BitConverter.TryWriteBytes(blob.AsSpan(i * sizeof(int), sizeof(int)), tagRows[i]);
        return blob;
    }
}
