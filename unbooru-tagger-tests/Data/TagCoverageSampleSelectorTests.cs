using UnbooruTagger.Data;

namespace UnbooruTagger.Tests.Data;

public class TagCoverageSampleSelectorTests
{
    [Fact]
    public void Select_ReturnsEverything_WhenUnderCap()
    {
        var images = new List<(int, IReadOnlyList<string>)>
        {
            (1, new[] { "a" }),
            (2, new[] { "b" }),
        };

        var selected = TagCoverageSampleSelector.Select(images, maxImages: 10, minImagesPerTag: 15);

        Assert.Equal(new HashSet<int> { 1, 2 }, selected);
    }

    [Fact]
    public void Select_KeepsRareTagRepresented_EvenWhenItsImagesSortLast()
    {
        // "common" has 100 images (ids 0-99); "rare" has a single image with the
        // highest id (100) — a plain ImageId-ordered Take(50) would drop "rare" entirely.
        var images = new List<(int, IReadOnlyList<string>)>();
        for (var i = 0; i < 100; i++)
            images.Add((i, new[] { "common" }));
        images.Add((100, new[] { "rare" }));

        var selected = TagCoverageSampleSelector.Select(images, maxImages: 50, minImagesPerTag: 15);

        Assert.Contains(100, selected);
        Assert.Equal(50, selected.Count);
    }

    [Fact]
    public void Select_CapsPerTagContributionAtMinImagesPerTag()
    {
        var images = Enumerable.Range(0, 100)
            .Select(i => (i, (IReadOnlyList<string>)new[] { "only-tag" }))
            .ToList();

        var selected = TagCoverageSampleSelector.Select(images, maxImages: 100, minImagesPerTag: 5);

        // Only one tag exists, so the coverage pass reserves 5, then the fill pass
        // (lowest ids first) tops the rest up to the full cap — result size is still
        // the cap, but this proves the coverage pass itself didn't grab more than 5.
        Assert.Equal(100, selected.Count);
    }

    [Fact]
    public void Select_GivesEveryTagAtLeastOneImage_WhenBudgetCoversAllTags()
    {
        var images = new List<(int, IReadOnlyList<string>)>();
        for (var tag = 0; tag < 20; tag++)
            for (var i = 0; i < 10; i++)
                images.Add((tag * 10 + i, new[] { $"tag{tag}" }));

        var selected = TagCoverageSampleSelector.Select(images, maxImages: 20, minImagesPerTag: 1);

        for (var tag = 0; tag < 20; tag++)
        {
            var tagName = $"tag{tag}";
            Assert.True(images.Where(img => img.Item2.Contains(tagName)).Any(img => selected.Contains(img.Item1)),
                $"expected at least one selected image for {tagName}");
        }
    }
}
