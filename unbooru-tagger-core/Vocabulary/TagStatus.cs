namespace UnbooruTagger.Core.Vocabulary;

/// <summary>
/// Whether a tag has enough labeled images to have its own trained embedding row,
/// or is still riding on its text warm-start prior (see CLAUDE.md long-tail handling).
/// </summary>
public enum TagStatus
{
    WarmStartOnly,
    Trained
}
