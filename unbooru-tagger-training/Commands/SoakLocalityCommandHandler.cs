using System.Diagnostics;
using TorchSharp;
using UnbooruTagger.Training.Model;
using UnbooruTagger.Training.Training;
using static TorchSharp.torch;

namespace UnbooruTagger.Training.Commands;

/// <summary>
/// Runs <see cref="LocalityNegatives.Step"/> and its backward over and over on random, training-shaped input (2026-11-01), and
/// prints this process's device memory as the driver counts it, to find what grew it in run 5: with the term on, VRAM rose
/// ~2 GB a minute and the trainer died after 14-47 minutes with xe "VM worker error: -12". Each image gets a new number of
/// labelled tags every step, as in training. <c>--mode</c> leaves parts out to bisect it.
/// </summary>
public static class SoakLocalityCommandHandler
{
    public static int Run(int steps, int padTo, string mode, int batch, int vocabulary, int maxPositives, int reportEvery, long maxVramMb)
    {
        var device = DeviceSelector.Best();
        Console.WriteLine($"soak-locality on {device}: {steps:N0} steps, batch {batch}, up to {maxPositives} tags per image, pad to {padTo}, mode {mode}");
        const int dim = 512, windows = 48;
        var random = new Random(0);
        using var tags = nn.Parameter(randn(vocabulary, dim, device: device) * 0.05f);
        using var scaling = new LogitScaleBias(device: device);
        using var locality = new LocalityNegatives(device) { MinObservations = 1, PerPair = true, ConfidenceGate = 0f };
        var clock = Stopwatch.StartNew();
        for (var step = 1; step <= steps; step++)
        {
            using var scope = NewDisposeScope();
            var rows = LocalityNegatives.Cap(Enumerable.Range(0, batch)
                .Select(_ => Enumerable.Range(0, random.Next(1, maxPositives + 1)).Select(_ => random.Next(vocabulary)).Distinct().ToList())
                .ToList(), random);
            var width = Math.Max(1, rows.Max(r => r.Count));
            width = (width + padTo - 1) / padTo * padTo;
            var padded = new long[batch * width];
            var valid = new bool[batch * width];
            for (var i = 0; i < batch; i++)
                for (var j = 0; j < rows[i].Count; j++)
                    (padded[i * width + j], valid[i * width + j]) = (rows[i][j], true);
            var rowTensor = mode == "no-upload" ? tensor(padded, [batch, width], device: device) : tensor(padded, [batch, width]).Upload(device);
            var validTensor = mode == "no-upload" ? tensor(valid, [batch, width], device: device) : tensor(valid, [batch, width]).Upload(device);

            var cells = (randn(batch, dim, windows, device: device) * 0.3f).requires_grad_(true);
            using var choice = new GlimpseChoice(
                randint(0, Glimpses.Zooms.Length, [batch, windows], dtype: ScalarType.Int64, device: device),
                randint(0, 480, [batch, windows], dtype: ScalarType.Int64, device: device),
                randint(0, 480, [batch, windows], dtype: ScalarType.Int64, device: device),
                ones(batch, windows, dtype: ScalarType.Bool, device: device));
            var loss = locality.Step(cells, choice, rowTensor, validTensor, tags, scaling);
            if (mode != "no-backward")
                loss.backward();
            tags.grad?.zero_();
            if (step % 5 == 0)
            {
                // As the trainer does: a read back every 5 steps, then the pinned staging copies are freed.
                _ = loss.item<float>();
                DeviceUpload.Release();
            }
            if (step % reportEvery == 0)
            {
                var vram = DeviceMemoryMb("drm-total-vram0");
                Console.WriteLine($"{clock.Elapsed:hh\\:mm\\:ss} step {step,7:N0}  vram {vram,7:N0} MB  gtt {DeviceMemoryMb("drm-total-gtt"),6:N0} MB  {clock.Elapsed.TotalMilliseconds / step:F1} ms/step");
                if (vram > maxVramMb)
                {
                    // It shares the device with a training run: stop before it can crowd that out.
                    Console.WriteLine($"stopping: over {maxVramMb:N0} MB");
                    return 2;
                }
            }
        }
        return 0;
    }

    /// <summary>This process's device memory of one kind, summed over its DRM file descriptors (as /proc/self/fdinfo reports it).</summary>
    private static long DeviceMemoryMb(string key) => DeviceMemory.Mb(key);
}
