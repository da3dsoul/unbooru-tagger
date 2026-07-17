namespace UnbooruTagger.Core.Vocabulary;

/// <summary>
/// One entry in the tag vocabulary: a tag string bound to its row index in the
/// embedding table (<see cref="Embedding.TagEmbeddingStore"/>).
/// </summary>
public sealed class TagRecord
{
    public required int RowIndex { get; init; }
    public required string Tag { get; init; }
    public TagStatus Status { get; set; } = TagStatus.WarmStartOnly;
    public int ImageCount { get; set; }
}
