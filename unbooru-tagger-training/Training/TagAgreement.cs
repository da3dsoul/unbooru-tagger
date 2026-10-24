using System.Text.Json;
using UnbooruTagger.Core.Dataset;
using UnbooruTagger.Core.Vocabulary;

namespace UnbooruTagger.Training.Training;

/// <summary>
/// Reads the crawler's <c>tag_agreement.json</c> (<c>export-tag-agreement</c>): over images
/// present on both Danbooru and Gelbooru, each tag's [both, Danbooru only, Gelbooru only]
/// counts. Not usable as tagging recall: Gelbooru posts are largely copied from Danbooru with
/// their tags, so agreement is 96-99% for almost every tag (median 0.967, blush 0.975). What
/// it does show reliably is which tags a site's tag system doesn't have at all.
/// </summary>
public static class TagAgreement
{
    public const string FileName = "tag_agreement.json";

    /// <summary>
    /// Per vocabulary row: which sites' tag systems use the tag. A site counts if it lists the
    /// tag on at least <paramref name="minShare"/> of the overlapping images that carry it.
    /// 1,136 of run 2's tags are effectively single-site (Gelbooru's silver_hair,
    /// black_legwear, *_footwear, female_focus; Danbooru's covered_nipples), so an image from
    /// only the other site can never carry them. Tags without enough overlap to tell are
    /// assumed to be on both. Null without an agreement file.
    /// </summary>
    public static BooruSites[]? LoadSiteCoverage(string path, TagVocabulary vocabulary, int minSupport = 200, double minShare = 0.02)
    {
        if (!File.Exists(path))
            return null;
        using var document = JsonDocument.Parse(File.ReadAllText(path));
        var sites = Enumerable.Repeat(BooruSites.Danbooru | BooruSites.Gelbooru, vocabulary.Records.Count).ToArray();
        foreach (var tag in document.RootElement.GetProperty("Tags").EnumerateObject())
        {
            if (!vocabulary.TryGet(tag.Name, out var record))
                continue;
            var c = tag.Value.EnumerateArray().Select(v => v.GetInt32()).ToArray();
            var support = c[0] + c[1] + c[2];
            if (support < minSupport)
                continue;
            var used = BooruSites.None;
            if (c[0] + c[1] >= minShare * support)
                used |= BooruSites.Danbooru;
            if (c[0] + c[2] >= minShare * support)
                used |= BooruSites.Gelbooru;
            sites[record.RowIndex] = used;
        }

        return sites;
    }
}
