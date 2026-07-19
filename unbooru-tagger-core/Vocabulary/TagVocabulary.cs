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
    private readonly List<TagRecord> _pendingNewRecords = [];

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
        _pendingNewRecords.Add(record);
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
            record.Status = TagStatus.Trained;
    }

    /// <summary>
    /// Loads the base snapshot at <paramref name="path"/> and, if <paramref name="deltaPath"/>
    /// is given and exists, replays tags appended via <see cref="SaveDelta"/> since that
    /// snapshot was last compacted by <see cref="Save"/> on top of it.
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
                if (vocabulary._byTag.ContainsKey(record.Tag))
                    continue;

                vocabulary._byRow.Add(record);
                vocabulary._byTag.Add(record.Tag, record);
            }
        }

        return vocabulary;
    }

    /// <summary>
    /// Appends only the tags added since the vocabulary was created/loaded or since the
    /// last <see cref="SaveDelta"/> call — O(new tags), not O(vocabulary size). Lets a
    /// long build checkpoint new row assignments after every page (needed so a resumed
    /// run reuses the same RowIndex for tags already baked into a cache's tag-row labels)
    /// without <see cref="Save"/>'s full-file rewrite cost compounding as the vocabulary
    /// grows into the hundreds of thousands of tags (CLAUDE.md long-tail).
    /// </summary>
    public void SaveDelta(string deltaPath)
    {
        if (_pendingNewRecords.Count == 0)
            return;

        using (var writer = new StreamWriter(new FileStream(deltaPath, FileMode.Append, FileAccess.Write)))
        {
            foreach (var record in _pendingNewRecords)
                writer.WriteLine(JsonSerializer.Serialize(record));
        }

        _pendingNewRecords.Clear();
    }

    /// <summary>
    /// Writes the full, compacted snapshot, including tags previously appended via
    /// <see cref="SaveDelta"/>. Callers that use <see cref="SaveDelta"/> for per-page
    /// checkpointing should call this once the run finishes and delete the delta file, so
    /// the next <see cref="Load"/> starts from a clean, fully up-to-date base snapshot.
    /// </summary>
    public void Save(string path)
    {
        var json = JsonSerializer.Serialize(_byRow, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(path, json);

        // Everything pending is now captured in the base snapshot, so it shouldn't be
        // re-appended by a later SaveDelta call.
        _pendingNewRecords.Clear();
    }
}
