namespace UnbooruTagger.Core.Embedding;

/// <summary>
/// The tag embedding table: one row per tag, indexed by
/// <see cref="Vocabulary.TagRecord.RowIndex"/>. Stored as a flat binary file
/// (row-major float32) so a single new row can be appended cheaply for the
/// "add a tag" pipeline without rewriting the whole table.
/// </summary>
public sealed class TagEmbeddingStore
{
    private readonly List<float[]> _rows;

    public int EmbeddingDim { get; }
    public int RowCount => _rows.Count;

    private TagEmbeddingStore(int embeddingDim, List<float[]> rows)
    {
        EmbeddingDim = embeddingDim;
        _rows = rows;
    }

    public static TagEmbeddingStore CreateEmpty(int embeddingDim) => new(embeddingDim, []);

    public ReadOnlySpan<float> GetRow(int rowIndex) => _rows[rowIndex];

    public void SetRow(int rowIndex, ReadOnlySpan<float> vector)
    {
        ValidateDim(vector.Length);
        vector.CopyTo(_rows[rowIndex]);
    }

    /// <summary>Appends a new row (e.g. a warm-start prior for a brand-new tag) and returns its row index.</summary>
    public int AppendRow(ReadOnlySpan<float> vector)
    {
        ValidateDim(vector.Length);
        _rows.Add(vector.ToArray());
        return _rows.Count - 1;
    }

    private void ValidateDim(int length)
    {
        if (length != EmbeddingDim)
            throw new ArgumentException($"Expected a {EmbeddingDim}-dim vector, got {length}.");
    }

    public static TagEmbeddingStore Load(string path)
    {
        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream);

        var embeddingDim = reader.ReadInt32();
        var rowCount = reader.ReadInt32();
        var rows = new List<float[]>(rowCount);

        for (var i = 0; i < rowCount; i++)
        {
            var row = new float[embeddingDim];
            for (var j = 0; j < embeddingDim; j++)
                row[j] = reader.ReadSingle();
            rows.Add(row);
        }

        return new TagEmbeddingStore(embeddingDim, rows);
    }

    public void Save(string path)
    {
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream);

        writer.Write(EmbeddingDim);
        writer.Write(RowCount);
        foreach (var row in _rows)
            foreach (var value in row)
                writer.Write(value);
    }
}
