namespace UnbooruTagger.Core.Vocabulary;

/// <summary>
/// Which tags have a region to point at. Whole-image tags — who drew it (artist:), what it's
/// from (series:), ratings, backgrounds, mediums, styles, lighting, weather, camera angle,
/// color scheme, format, people counts — have no "where", so training never gives them to the
/// per-location loss (it could only teach their heatmaps noise) and <c>detect</c> reports
/// them without boxes.
///
/// File format (<see cref="FileName"/>), one rule per line, <c>#</c> comments:
/// <list type="bullet">
/// <item><c>tag</c> — exactly that tag;</item>
/// <item><c>prefix*</c> — tags starting with prefix (e.g. <c>artist:*</c>, <c>from *</c>);</item>
/// <item><c>*suffix</c> — tags ending with suffix (e.g. <c>* background</c>, <c>*(medium)</c>);</item>
/// <item><c>!tag</c> / <c>!prefix*</c> / <c>!*suffix</c> — exception: localizable even if another rule matches.</item>
/// </list>
/// Matching ignores case and treats underscores as spaces. Rules are anchored on purpose:
/// a substring like "shadow" would also catch <c>eyeshadow</c> (makeup, localizable) and
/// <c>shadowverse</c>, and "cover" would catch <c>covered navel</c>.
/// </summary>
public sealed class TagLocalizability
{
    public const string FileName = "localizable_tags.txt";

    public const string DefaultText =
        """
        # Tags with no region to point at. See TagLocalizability for the rule syntax.
        # Categories
        artist:*
        series:*
        meta:*
        rating:*
        # Backgrounds, mediums, styles, color schemes, composition focus
        * background
        *(medium)
        *(style)
        * artstyle
        * theme
        * focus
        limited palette
        neon palette
        monochrome
        greyscale
        sepia
        spot color
        high contrast
        pastel colors
        realistic
        parody
        retro artstyle
        anime coloring
        traditional media
        watercolor
        sketch
        lineart
        pixel art
        3d
        oekaki
        # People counts and who's in frame
        solo
        1girl
        2girls
        3girls
        4girls
        5girls
        6+girls
        1boy
        2boys
        3boys
        4boys
        5boys
        6+boys
        1other
        2others
        3others
        multiple girls
        multiple boys
        multiple others
        no humans
        everyone
        # Camera, framing, angle
        from *
        pov
        portrait
        close-up
        upper body
        lower body
        full body
        cowboy shot
        wide shot
        very wide shot
        dutch angle
        straight-on
        profile
        looking at viewer
        looking away
        looking to the side
        looking back
        looking up
        looking down
        head out of frame
        feet out of frame
        out of frame
        # Lighting, weather, time of day
        backlighting
        sidelighting
        rim lighting
        dappled sunlight
        sunlight
        lens flare
        light rays
        sunbeam
        day
        night
        evening
        sunset
        sunrise
        twilight
        rain
        snowing
        fog
        outdoors
        indoors
        scenery
        # Image format and quality
        highres
        absurdres
        incredibly absurdres
        lowres
        letterboxed
        pillarboxed
        border
        white border
        black border
        rounded corners
        comic
        4koma
        manga
        multiple views
        split screen
        collage
        cover
        cover page
        official art
        promotional art
        scan
        jpeg artifacts
        blurry
        depth of field
        motion blur
        chromatic aberration
        film grain
        commentary
        commentary request
        translated
        translation request
        # Exceptions: small, real regions despite looking like metadata
        !signature
        !watermark
        !artist name
        !twitter username
        !speech bubble
        """;

    private readonly HashSet<string> _exact = [];
    private readonly List<string> _prefixes = [];
    private readonly List<string> _suffixes = [];
    private readonly HashSet<string> _exceptExact = [];
    private readonly List<string> _exceptPrefixes = [];
    private readonly List<string> _exceptSuffixes = [];

    public string SourceText { get; }

    private TagLocalizability(string sourceText) => SourceText = sourceText;

    public static TagLocalizability Default { get; } = Parse(DefaultText);

    public static TagLocalizability Parse(string text)
    {
        var result = new TagLocalizability(text);
        foreach (var rawLine in text.Split('\n'))
        {
            var line = rawLine.Trim();
            if (line.Length == 0 || line.StartsWith('#'))
                continue;

            var except = line.StartsWith('!');
            if (except)
                line = line[1..].Trim();
            line = Normalize(line);

            if (line.EndsWith('*'))
                (except ? result._exceptPrefixes : result._prefixes).Add(line[..^1]);
            else if (line.StartsWith('*'))
                (except ? result._exceptSuffixes : result._suffixes).Add(line[1..]);
            else
                (except ? result._exceptExact : result._exact).Add(line);
        }

        return result;
    }

    /// <summary>Loads <paramref name="path"/> if it exists, otherwise <see cref="Default"/>.</summary>
    public static TagLocalizability Load(string? path) =>
        path is not null && File.Exists(path) ? Parse(File.ReadAllText(path)) : Default;

    public void Save(string path) => File.WriteAllText(path, SourceText);

    public bool IsLocalizable(string tag)
    {
        var normalized = Normalize(tag);
        if (_exceptExact.Contains(normalized) || _exceptPrefixes.Any(normalized.StartsWith) || _exceptSuffixes.Any(normalized.EndsWith))
            return true;
        return !(_exact.Contains(normalized) || _prefixes.Any(normalized.StartsWith) || _suffixes.Any(normalized.EndsWith));
    }

    private static string Normalize(string tag) => tag.Replace('_', ' ').Trim().ToLowerInvariant();
}
