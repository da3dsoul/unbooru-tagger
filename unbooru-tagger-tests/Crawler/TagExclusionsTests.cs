using UnbooruTagger.Crawler;

namespace UnbooruTagger.Tests.Crawler;

public class TagExclusionsTests
{
    [Fact]
    public async Task LoadOrCreateAsync_SeedsTheFileWithDefaultIncludesOnFirstRun()
    {
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            var rules = await TagExclusions.LoadOrCreateAsync(directory);

            Assert.True(File.Exists(Path.Combine(directory, TagExclusions.FileName)));
            // Every default include is a meta tag that would otherwise be caught by the
            // blanket meta exclusion — confirms the seeded file actually rescues them.
            Assert.False(rules.IsExcluded("meta:scan"));
            Assert.False(rules.IsExcluded("meta:ai-generated"));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public async Task LoadOrCreateAsync_DoesNotOverwriteAnExistingFile()
    {
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            await File.WriteAllLinesAsync(Path.Combine(directory, TagExclusions.FileName),
                ["# custom exclusion list", "my_custom_tag"]);

            var rules = await TagExclusions.LoadOrCreateAsync(directory);

            Assert.True(rules.IsExcluded("my_custom_tag"));
            Assert.True(rules.IsExcluded("meta:scan")); // hand-edited file wins — the default include was never written
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public async Task LoadAsync_ParsesExcludesAndBangPrefixedIncludes_IgnoringBlankLinesAndComments()
    {
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            var path = Path.Combine(directory, "custom_exclusions.txt");
            await File.WriteAllLinesAsync(path, ["# a comment", "", "  general_junk_tag  ", "!meta:some_technique", "# another comment"]);

            var rules = await TagExclusions.LoadAsync(path);

            Assert.True(rules.IsExcluded("general_junk_tag"));
            Assert.False(rules.IsExcluded("meta:some_technique")); // rescued from the meta blanket by '!'
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public async Task LoadAsync_ReturnsEmptyRules_WhenFileDoesNotExist()
    {
        var rules = await TagExclusions.LoadAsync(Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString()));

        // The meta blanket is structural (baked into TagExclusionRules itself), not
        // something the file grants — it still applies with zero file-based rules.
        Assert.True(rules.IsExcluded("meta:highres"));
        Assert.False(rules.IsExcluded("1girl"));
    }
}
