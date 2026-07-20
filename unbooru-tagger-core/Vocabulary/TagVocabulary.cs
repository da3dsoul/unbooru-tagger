using System.Text.Json;

namespace UnbooruTagger.Core.Vocabulary;

/// <summary>
/// The tag string &lt;-&gt; embedding-row-index mapping, plus each tag's promotion
/// status. Backed by a single JSON manifest file; the embedding vectors themselves
/// live in <see cref="Embedding.TagEmbeddingStore"/>, indexed by <see cref="TagRecord.RowIndex"/>.
/// </summary>
public sealed class TagVocabulary
{
    private readonly Dictionary<string, TagRecord> _byTag;
    private readonly List<TagRecord> _byRow;

    // Every record that's new or has changed (ImageCount, Status) since the last
    // SaveDelta/Save call. The list preserves write order for the delta file; the set
    // is just there so repeat observations of the same tag within one checkpoint
    // window dedupe to a single pending entry instead of growing the list unbounded.
    private readonly List<TagRecord> _pendingDirtyRecords = [];
    private readonly HashSet<TagRecord> _pendingDirtySet = [];

    private TagVocabulary(List<TagRecord> records)
    {
        _byRow = records;
        _byTag = records.ToDictionary(r => r.Tag, StringComparer.Ordinal);
    }

    public static TagVocabulary CreateEmpty() => new([]);

    public IReadOnlyList<TagRecord> Records => _byRow;

    public bool TryGet(string tag, out TagRecord record) =>
        _byTag.TryGetValue(tag, out record!);

    public TagRecord GetByRowIndex(int rowIndex) => _byRow[rowIndex];

    /// <summary>
    /// Registers a brand-new tag at the next free row index. The caller is
    /// responsible for appending a matching warm-start row to the embedding store.
    /// </summary>
    public TagRecord AddTag(string tag)
    {
        if (_byTag.ContainsKey(tag))
            throw new InvalidOperationException($"Tag '{tag}' is already in the vocabulary.");

        var record = new TagRecord { RowIndex = _byRow.Count, Tag = tag };
        _byRow.Add(record);
        _byTag.Add(tag, record);
        MarkDirty(record);
        return record;
    }

    /// <summary>
    /// Registers one observed occurrence of <paramref name="tag"/> — creating its
    /// vocabulary row (warm-start only) first if this is the first time it's been
    /// seen — and increments its image count. Callers processing a corpus should use
    /// this instead of <see cref="TryGet"/> + <see cref="AddTag"/> + a manual
    /// <c>ImageCount++</c>, since only this path marks the record dirty for the next
    /// <see cref="SaveDelta"/> call.
    /// </summary>
    public TagRecord RecordObservation(string tag)
    {
        if (!TryGet(tag, out var record))
            record = AddTag(tag);

        record.ImageCount++;
        MarkDirty(record);
        return record;
    }

    /// <summary>
    /// Promotes a tag from <see cref="TagStatus.WarmStartOnly"/> to
    /// <see cref="TagStatus.Trained"/> once it has crossed the minimum-image
    /// threshold (CLAUDE.md: ~10-20 images) and its row has actually been fine-tuned.
    /// </summary>
    public void PromoteIfThresholdMet(string tag, int minImageThreshold)
    {
        var record = _byTag[tag];
        if (record.Status == TagStatus.WarmStartOnly && record.ImageCount >= minImageThreshold)
        {
            record.Status = TagStatus.Trained;
            MarkDirty(record);
        }
    }

    private void MarkDirty(TagRecord record)
    {
        if (_pendingDirtySet.Add(record))
            _pendingDirtyRecords.Add(record);
    }

    /// <summary>
    /// Loads the base snapshot at <paramref name="path"/> and, if <paramref name="deltaPath"/>
    /// is given and exists, replays the change log written by <see cref="SaveDelta"/>
    /// since that snapshot was last compacted by <see cref="Save"/> on top of it. A tag
    /// can appear on more than one delta line (one per checkpoint that changed it, e.g.
    /// its image count going up) — later lines win, since they carry that tag's most
    /// recent state.
    /// </summary>
    public static TagVocabulary Load(string path, string? deltaPath = null)
    {
        var json = File.ReadAllText(path);
        var records = JsonSerializer.Deserialize<List<TagRecord>>(json)
                      ?? throw new InvalidDataException($"'{path}' did not contain a valid tag vocabulary.");
        var vocabulary = new TagVocabulary(records.OrderBy(r => r.RowIndex).ToList());

        if (deltaPath is not null && File.Exists(deltaPath))
        {
            foreach (var line in File.ReadLines(deltaPath))
            {
                if (line.Length == 0)
                    continue;

                var record = JsonSerializer.Deserialize<TagRecord>(line)
                             ?? throw new InvalidDataException($"'{deltaPath}' contains an invalid tag vocabulary delta line.");

                if (vocabulary._byTag.TryGetValue(record.Tag, out var existing))
                {
                    existing.ImageCount = record.ImageCount;
                    existing.Status = record.Status;
                }
                else
                {
                    vocabulary._byRow.Add(record);
                    vocabulary._byTag.Add(record.Tag, record);
                }
            }
        }

        return vocabulary;
    }

    /// <summary>
    /// Appends every record that's new or changed (image count, status) since the
    /// vocabulary was created/loaded or since the last <see cref="SaveDelta"/> call —
    /// O(changes), not O(vocabulary size) — as a staging log meant to be periodically
    /// folded back into the base snapshot by <see cref="Save"/> rather than rewritten
    /// on every call. Lets a long build checkpoint every page's worth of new rows and
    /// image-count updates (needed so a resumed run reuses the same RowIndex for tags
    /// already baked into a cache's tag-row labels, and doesn't lose count updates to a
    /// crash) without <see cref="Save"/>'s full-file rewrite cost compounding as the
    /// vocabulary grows into the hundreds of thousands of tags (CLAUDE.md long-tail).
    /// </summary>
    public void SaveDelta(string deltaPath)
    {
        if (_pendingDirtyRecords.Count == 0)
            return;

        using (var writer = new StreamWriter(new FileStream(deltaPath, FileMode.Append, FileAccess.Write)))
        {
            foreach (var record in _pendingDirtyRecords)
                writer.WriteLine(JsonSerializer.Serialize(record));
        }

        _pendingDirtyRecords.Clear();
        _pendingDirtySet.Clear();
    }

    /// <summary>
    /// Writes the full, compacted snapshot, including every change previously
    /// appended via <see cref="SaveDelta"/>. Callers that use <see cref="SaveDelta"/>
    /// for per-page checkpointing should call this periodically (and once the run
    /// finishes) and delete the delta file, so the next <see cref="Load"/> starts from
    /// a small, fast-to-replay delta on top of a reasonably fresh base snapshot.
    /// </summary>
    public void Save(string path)
    {
        var json = JsonSerializer.Serialize(_byRow, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(path, json);

        // Everything pending is now captured in the base snapshot, so it shouldn't be
        // re-appended by a later SaveDelta call.
        _pendingDirtyRecords.Clear();
        _pendingDirtySet.Clear();
    }
}
