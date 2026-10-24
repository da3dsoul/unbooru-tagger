namespace UnbooruTagger.Core.Dataset;

/// <summary>Which booru sites an image came from (bit flags).</summary>
[Flags]
public enum BooruSites : byte
{
    None = 0,
    Danbooru = 1,
    Gelbooru = 2,
}

/// <summary>
/// One <see cref="BooruSites"/> mask per cache row, stored beside images.bin as
/// <see cref="FileName"/> (the crawler's <c>export-image-sites</c>). Some tags exist on only
/// one site's tag system (Gelbooru's silver_hair, black_legwear, *_footwear; Danbooru's
/// covered_nipples), so an image from the other site alone can never carry them: its
/// absence says nothing about the image. Training uses this with per-tag site coverage
/// (TagSiteCoverage) to stop treating those absences as negatives.
/// </summary>
public static class ImageSites
{
    public const string FileName = "image_sites.bin";
    private const int Magic = 0x53495431; // "SIT1"

    public static BooruSites Parse(string site) => site switch
    {
        "danbooru" => BooruSites.Danbooru,
        "gelbooru" => BooruSites.Gelbooru,
        _ => BooruSites.None,
    };

    public static void Save(string directory, IReadOnlyList<BooruSites> sites)
    {
        var path = Path.Combine(directory, FileName);
        var temp = path + ".tmp";
        using (var writer = new BinaryWriter(File.Create(temp)))
        {
            writer.Write(Magic);
            writer.Write(sites.Count);
            foreach (var site in sites)
                writer.Write((byte)site);
        }

        File.Move(temp, path, overwrite: true);
    }

    /// <summary>Null if the dataset has no sites file.</summary>
    public static BooruSites[]? Load(string directory, int expectedCount)
    {
        var path = Path.Combine(directory, FileName);
        if (!File.Exists(path))
            return null;

        using var reader = new BinaryReader(File.OpenRead(path));
        if (reader.ReadInt32() != Magic)
            throw new InvalidDataException($"'{path}' isn't an image-sites file.");
        var count = reader.ReadInt32();
        if (count != expectedCount)
            throw new InvalidDataException(
                $"'{path}' has sites for {count} images but the cache has {expectedCount} -- re-run export-image-sites after any change to the cache.");
        return reader.ReadBytes(count).Select(b => (BooruSites)b).ToArray();
    }
}
