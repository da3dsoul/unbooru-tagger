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

    public static TagVocabulary Load(string path)
    {
        var json = File.ReadAllText(path);
        var records = JsonSerializer.Deserialize<List<TagRecord>>(json)
                      ?? throw new InvalidDataException($"'{path}' did not contain a valid tag vocabulary.");
        return new TagVocabulary(records.OrderBy(r => r.RowIndex).ToList());
    }

    public void Save(string path)
    {
        var json = JsonSerializer.Serialize(_byRow, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(path, json);
    }
}
