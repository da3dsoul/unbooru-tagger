using System.Text.Json;

namespace UnbooruTagger.Core.Dataset;

/// <summary>One labeled image: its path/URI and the booru-style tags attached to it.</summary>
public sealed record DatasetImageEntry(string ImagePath, IReadOnlyList<string> Tags);

/// <summary>
/// A dataset manifest shared between training data prep and test fixtures — the
/// common input format for both the full training pass and the "add a tag" pipeline.
/// </summary>
public sealed class DatasetManifest
{
    public IReadOnlyList<DatasetImageEntry> Entries { get; }

    public DatasetManifest(IReadOnlyList<DatasetImageEntry> entries) => Entries = entries;

    public static DatasetManifest Load(string path)
    {
        var json = File.ReadAllText(path);
        var entries = JsonSerializer.Deserialize<List<DatasetImageEntry>>(json)
                     ?? throw new InvalidDataException($"'{path}' did not contain a valid dataset manifest.");
        return new DatasetManifest(entries);
    }

    public void Save(string path)
    {
        var json = JsonSerializer.Serialize(Entries, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(path, json);
    }
}
