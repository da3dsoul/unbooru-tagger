using System.Text.Json;

namespace UnbooruTagger.Data;

/// <summary>
/// Tracks the DB-side cursor (last <c>ImageId</c> pulled) for a <see cref="LargeDatasetPreprocessor"/>
/// run, separately from the cache's own image count. Persisted after every page so a
/// crash — e.g. a dropped connection over a long WAN-backed run — only loses the
/// current page, and re-running the same command resumes the DB scan right here.
/// </summary>
internal sealed record LargeCacheResumeState(int? LastImageId)
{
    private const string FileName = "resume_state.json";

    public static LargeCacheResumeState Load(string directory)
    {
        var path = Path.Combine(directory, FileName);
        if (!File.Exists(path))
            return new LargeCacheResumeState((int?)null);

        var json = File.ReadAllText(path);
        return JsonSerializer.Deserialize<LargeCacheResumeState>(json)
               ?? throw new InvalidDataException($"'{path}' does not contain valid resume state.");
    }

    public void Save(string directory) =>
        File.WriteAllText(Path.Combine(directory, FileName), JsonSerializer.Serialize(this));
}
