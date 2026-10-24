namespace UnbooruTagger.Core.Dataset;

/// <summary>Booru content rating, ordered from least to most explicit (Danbooru's g/s/q/e scale).</summary>
public enum ContentRating : byte
{
    General = 0,
    Sensitive = 1,
    Questionable = 2,
    Explicit = 3,
    Unknown = 255,
}

/// <summary>
/// One <see cref="ContentRating"/> per cache row, stored beside a dataset's images.bin as
/// <see cref="FileName"/> (exported from crawl.sqlite by the crawler's <c>export-ratings</c>).
/// Ratings are kept out of the tag rows on purpose: they're one ordinal value per image, not
/// an open-ended tag set, and training turns them into its own cumulative labels
/// (<see cref="CumulativeTags"/>).
///
/// When an image's sources disagree, the STRICTEST rating is stored: 8% of multi-source
/// images disagree, and the costly mistake is calling an explicit image safe, never the
/// reverse.
/// </summary>
public static class ImageRatings
{
    public const string FileName = "ratings.bin";
    private const int Magic = 0x52415431; // "RAT1"

    /// <summary>
    /// The three cumulative training labels: "at least sensitive", "at least questionable",
    /// "explicit". An explicit image carries all three; a general image none. Cumulative
    /// rather than four competing labels so predictions can't contradict each other, and so
    /// each cut can be calibrated for its own cost (e.g. high recall on "at least questionable").
    /// </summary>
    public static readonly string[] CumulativeTags = ["rating:at_least_sensitive", "rating:at_least_questionable", "rating:is_explicit"];

    public static ContentRating Parse(string siteRating) => siteRating.Trim().ToLowerInvariant() switch
    {
        "g" or "general" or "safe" => ContentRating.General,
        "s" or "sensitive" => ContentRating.Sensitive,
        "q" or "questionable" => ContentRating.Questionable,
        "e" or "explicit" => ContentRating.Explicit,
        _ => ContentRating.Unknown,
    };

    /// <summary>The strictest known rating among <paramref name="ratings"/>, or <see cref="ContentRating.Unknown"/> if none are known.</summary>
    public static ContentRating Strictest(IEnumerable<ContentRating> ratings)
    {
        var known = ratings.Where(r => r != ContentRating.Unknown).ToList();
        return known.Count == 0 ? ContentRating.Unknown : known.Max();
    }

    /// <summary>How many of <see cref="CumulativeTags"/> an image with <paramref name="rating"/> carries (0 for general or unknown).</summary>
    public static int CumulativeLevel(ContentRating rating) => rating == ContentRating.Unknown ? 0 : (int)rating;

    public static void Save(string directory, IReadOnlyList<ContentRating> ratings)
    {
        var path = Path.Combine(directory, FileName);
        var temp = path + ".tmp";
        using (var writer = new BinaryWriter(File.Create(temp)))
        {
            writer.Write(Magic);
            writer.Write(ratings.Count);
            foreach (var rating in ratings)
                writer.Write((byte)rating);
        }

        File.Move(temp, path, overwrite: true);
    }

    /// <summary>Null if the dataset has no ratings file.</summary>
    public static ContentRating[]? Load(string directory, int expectedCount)
    {
        var path = Path.Combine(directory, FileName);
        if (!File.Exists(path))
            return null;

        using var reader = new BinaryReader(File.OpenRead(path));
        if (reader.ReadInt32() != Magic)
            throw new InvalidDataException($"'{path}' isn't a ratings file.");
        var count = reader.ReadInt32();
        if (count != expectedCount)
            throw new InvalidDataException(
                $"'{path}' has ratings for {count} images but the cache has {expectedCount} -- re-run export-ratings after any change to the cache.");
        return reader.ReadBytes(count).Select(b => (ContentRating)b).ToArray();
    }
}
