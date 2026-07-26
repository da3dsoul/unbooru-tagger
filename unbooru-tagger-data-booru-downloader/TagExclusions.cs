namespace UnbooruTagger.Crawler;

/// <summary>
/// Decides whether a tag identity should be dropped from the vocabulary entirely: never
/// counted toward eligibility (so never surveyed/crawled as a target), and stripped from
/// every image's observed tag list even when another, legitimate target tag's search
/// happens to pull in a post that also carries it.
///
/// The default is a blanket rule, not a hand-maintained list: every
/// <see cref="TagCategory.Meta"/> tag (identity prefix <c>meta:</c>) is excluded, because
/// a real survey of Danbooru's meta category (~800 tags, fetched via
/// <c>tags.json?search[category]=5</c>) is overwhelmingly link/upload bookkeeping
/// (<c>bad_pixiv_id</c>, <c>md5_mismatch</c>...), per-language commentary/translation
/// (<c>french_commentary</c>, <c>translation_request</c>...), and tagging-workflow
/// placeholders (<c>character_request</c>, <c>check_copyright</c>...) — none of it
/// recoverable from the pixels alone. General/artist/copyright/character tags are never
/// touched by this rule.
///
/// Two exceptions carve back OUT of that blanket:
/// <list type="bullet">
/// <item>Any tag whose raw name ends in <c>_(medium)</c> — Danbooru's own naming
/// convention for "how this was made" (<c>pen_(medium)</c>, <c>oil_painting_(medium)</c>,
/// <c>photoshop_(medium)</c>...). A production technique/medium genuinely leaves a visual
/// signature, unlike upload bookkeeping, so these default to included automatically —
/// covers well over a hundred tags without hand-listing each one.</item>
/// <item><see cref="DefaultIncludes"/>: a short curated list of other meta tags that are
/// also about production process rather than bookkeeping but don't happen to follow the
/// <c>_(medium)</c> naming convention — <c>scan</c>, <c>ai-generated</c>,
/// <c>traditional_media</c>...</item>
/// </list>
///
/// Read fresh from <c>--output-dir</c> on every command invocation (see
/// <see cref="LoadOrCreateAsync"/>) rather than baked into <c>crawl.sqlite</c> once, the
/// same way eligibility itself is always recomputed live from <c>--min-images</c> rather
/// than stored — so hand edits to <see cref="TagExclusions.FileName"/> take effect on the
/// very next run, no re-survey needed.
/// </summary>
public sealed class TagExclusionRules(IReadOnlySet<string> excludes, IReadOnlySet<string> includes)
{
    /// <summary>
    /// Priority, highest first: an explicit <c>!identity</c> line always wins (even over
    /// an explicit exclude — it's the more specific, more deliberate override); then an
    /// explicit exclude line; then the automatic <c>_(medium)</c> carve-out; then the
    /// meta-category blanket. Anything left over (general/artist/copyright/character) is
    /// never excluded by this rule at all.
    /// </summary>
    public bool IsExcluded(string identity)
    {
        if (includes.Contains(identity))
            return false;
        if (excludes.Contains(identity))
            return true;
        if (HasMediumSuffix(identity))
            return false;

        return IsMeta(identity);
    }

    private static bool IsMeta(string identity) => identity.StartsWith("meta:", StringComparison.Ordinal);

    private static bool HasMediumSuffix(string identity) =>
        TagCategoryNaming.RawName(identity).EndsWith("_(medium)", StringComparison.Ordinal);
}

public static class TagExclusions
{
    public const string FileName = "excluded_tags.txt";

    /// <summary>
    /// Curated exceptions to the meta-category blanket exclusion that aren't already
    /// covered by the automatic <c>_(medium)</c> carve-out — every one of these is about
    /// how the image was physically or technically produced, which does leave a visual
    /// signature (a scan has visible paper grain/dust; an AI-generated image has
    /// recognizable rendering tells; a screenshot looks nothing like hand-drawn art),
    /// unlike the rest of the meta category. Seeded as <c>!</c>-prefixed lines in a fresh
    /// <see cref="FileName"/> (see <see cref="LoadOrCreateAsync"/>) so they're visible and
    /// editable, not just baked into code.
    /// </summary>
    public static readonly IReadOnlyList<string> DefaultIncludes =
    [
        "meta:scan", "meta:magazine_scan", "meta:self-scan",
        "meta:traditional_media", "meta:mixed_media", "meta:vector_art", "meta:colorized",
        "meta:cosplay_photo", "meta:anime_screenshot", "meta:game_screenshot", "meta:game_asset", "meta:game_model",
        "meta:ai-generated", "meta:ai-generated_background", "meta:ai-assisted",
        "meta:stable_diffusion", "meta:midjourney", "meta:nai_diffusion", "meta:dall-e", "meta:tensor_art",
    ];

    /// <summary>Reads <paramref name="datasetDirectory"/>'s <see cref="FileName"/>, seeding it with <see cref="DefaultIncludes"/> first if it doesn't exist yet.</summary>
    public static async Task<TagExclusionRules> LoadOrCreateAsync(string datasetDirectory, CancellationToken cancellationToken = default)
    {
        Directory.CreateDirectory(datasetDirectory);
        var path = Path.Combine(datasetDirectory, FileName);
        if (!File.Exists(path))
            await File.WriteAllLinesAsync(path, BuildDefaultFileLines(), cancellationToken).ConfigureAwait(false);

        return await LoadAsync(path, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>Reads an exclusion file directly, or empty rules if it doesn't exist — unlike <see cref="LoadOrCreateAsync"/>, never creates one.</summary>
    public static async Task<TagExclusionRules> LoadAsync(string path, CancellationToken cancellationToken = default)
    {
        var excludes = new HashSet<string>(StringComparer.Ordinal);
        var includes = new HashSet<string>(StringComparer.Ordinal);
        if (!File.Exists(path))
            return new TagExclusionRules(excludes, includes);

        var lines = await File.ReadAllLinesAsync(path, cancellationToken).ConfigureAwait(false);
        foreach (var raw in lines)
        {
            var line = raw.Trim();
            if (line.Length == 0 || line.StartsWith('#'))
                continue;

            if (line.StartsWith('!'))
                includes.Add(line[1..].Trim());
            else
                excludes.Add(line);
        }

        return new TagExclusionRules(excludes, includes);
    }

    private static IEnumerable<string> BuildDefaultFileLines()
    {
        yield return "# Every meta:* tag (Danbooru category/Gelbooru type 5 — upload bookkeeping, per-";
        yield return "# language commentary, tagging-workflow placeholders) is excluded from the vocabulary";
        yield return "# by default: never surveyed/crawled as a target, and stripped from every image's tag";
        yield return "# list even when another target tag's search happens to pull in a post carrying one.";
        yield return "# General/artist/series/character tags are never touched by this default.";
        yield return "#";
        yield return "# Two carve-outs, since some meta tags ARE about visual production technique:";
        yield return "#   1. Automatic: any tag ending in '_(medium)' (pen_(medium), oil_painting_(medium),";
        yield return "#      photoshop_(medium)...) — Danbooru's own convention for 'how this was made'.";
        yield return "#   2. This file: lines below prefixed '!' are additional included exceptions.";
        yield return "#";
        yield return "# A bare line (no '!') EXCLUDES that identity outright, regardless of category — use";
        yield return "# this for a general/artist/etc. tag you also want dropped. Lines starting with '#'";
        yield return "# and blank lines are ignored. Read fresh on every survey-tags/crawl/refresh-tags run,";
        yield return "# so edits here take effect immediately without re-running survey-tags.";
        yield return "#";
        yield return "# Identities are 'category:name' for anything non-general, e.g. character:frieren,";
        yield return "# series:sousou_no_frieren — see README.md.";
        yield return "";

        foreach (var tag in DefaultIncludes)
            yield return $"!{tag}";
    }
}
