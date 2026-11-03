using System.Diagnostics;
using TorchSharp;
using UnbooruTagger.Training.Model;
using UnbooruTagger.Training.Training;
using static TorchSharp.torch;

namespace UnbooruTagger.Training.Commands;

/// <summary>
/// Times the image tower's forward + backward on random input (the bulk of a training step: 88% in run 4's step
/// profile), per memory layout, to tune the step without a training run. TorchSharp 0.107 exposes no memory_format
/// conversion, so channels-last is made with permutes (NCHW shape, NHWC strides), which libtorch's convolutions
/// pick their layout from. "bf16" runs the whole tower (weights and activations) in bfloat16; "bf16-mixed" is training's --bf16 (fp32 weights cast
/// per call, ImageTower.ComputeType). Also reports how far
/// each layout's output is from plain NCHW's.
/// </summary>
public static class BenchmarkTowerCommandHandler
{
    public static int Run(int batchSize, int inputSize, int iterations, string[] layouts)
    {
        var device = DeviceSelector.Best();
        Tensor? reference = null;
        foreach (var layout in layouts)
        {
            manual_seed(0);
            // Run 4's tower, initialized on the CPU: manual_seed doesn't seed the XPU generator, so weights drawn there
            // would differ per layout and the output comparison would be meaningless.
            var tower = new ImageTower(512, 96, [96, 192, 384, 768], [3, 3, 9, 3], CPU, fusedStages: 3) { LocalizationFromFusedOnly = true };
            tower.to(device);
            tower.train();
            if (layout == "nhwc-all")
                foreach (var module in tower.modules().OfType<TorchSharp.Modules.Conv2d>())
                    module.weight = new TorchSharp.Modules.Parameter(ChannelsLast(module.weight!.detach()));
            var bf16 = layout == "bf16";
            if (bf16)
                tower.to(ScalarType.BFloat16);
            if (layout is "bf16-mixed" or "bf16-mixed-nhwc")
                tower.ComputeType = ScalarType.BFloat16;
            tower.ChannelsLast = layout == "bf16-mixed-nhwc";
            manual_seed(1);
            using var input = randn(batchSize, 3, inputSize, inputSize).to(device);
            using var x = layout is "nchw" or "bf16-mixed-nhwc" ? input.alias() : bf16 ? input.to_type(ScalarType.BFloat16) : ChannelsLast(input);
            var probe = tower.parameters().First();

            double Step()
            {
                using var scope = NewDisposeScope();
                var clock = Stopwatch.StartNew();
                tower.zero_grad();
                var (spatial, localization) = tower.ForwardMaps(x);
                (spatial.to_type(ScalarType.Float32).square().mean() + localization.to_type(ScalarType.Float32).square().mean()).backward();
                _ = probe.grad!.flatten()[0].to_type(ScalarType.Float32).item<float>();
                return clock.Elapsed.TotalMilliseconds;
            }

            for (var i = 0; i < 5; i++)
                Step();
            var times = Enumerable.Range(0, iterations).Select(_ => Step()).Order().ToArray();

            string difference;
            using (no_grad())
            {
                var output = tower.ForwardMaps(x).Spatial.to_type(ScalarType.Float32).contiguous().cpu();
                if (reference is null)
                {
                    reference = output.DetachFromDisposeScope();
                    difference = "reference";
                }
                else
                    difference = $"max |output - nchw| {(output - reference).abs().max().item<float>():0.###e+0} (output scale {reference.abs().mean().item<float>():0.###e+0})";
            }

            Console.WriteLine($"{layout,-12} forward+backward of {batchSize} at {inputSize}px: median {times[times.Length / 2]:0.0} ms, min {times[0]:0.0} ms; {difference}");
        }

        return 0;
    }

    /// <summary>Which ops keep a channels-last input channels-last (their output strides), in bfloat16 as training runs.</summary>
    public static int ProbeLayout()
    {
        var device = DeviceSelector.Best();
        var block = new ConvNeXtBlock(96, device);
        var norm = ConvNeXtBlock.NewGroupNorm(96, device);
        using var x = ChannelsLast(randn(2, 96, 32, 32, device: device).to_type(ScalarType.BFloat16));
        string Strides(Tensor t) => $"[{string.Join(",", t.stride())}]";
        Console.WriteLine($"input            {Strides(x)} (channels-last = [C*H*W, 1, W*C, C])");
        Console.WriteLine($"depthwise conv   {Strides(MixedPrecision.Conv(block.Depthwise, x))}");
        Console.WriteLine($"group norm       {Strides(MixedPrecision.Norm(norm, x))}");
        Console.WriteLine($"gelu             {Strides(nn.functional.gelu(x))}");
        Console.WriteLine($"x + x            {Strides(x + x)}");
        using var stemInput = ChannelsLast(randn(2, 3, 128, 128, device: device).to_type(ScalarType.BFloat16));
        var stem = nn.Conv2d(3, 96, 4, stride: 4, device: device);
        Console.WriteLine($"stem conv 4x4/4  {Strides(MixedPrecision.Conv(stem, stemInput))}");
        Console.WriteLine($"whole block      {Strides(block.forward(x))}");
        return 0;
    }

    /// <summary>Forward + backward time of each op of one ConvNeXt block per stage, at that stage's real size.</summary>
    public static int RunComponents(int batchSize, int inputSize, int iterations, bool bf16 = false)
    {
        var device = DeviceSelector.Best();
        var type = bf16 ? ScalarType.BFloat16 : ScalarType.Float32;
        (long Channels, long Size, int Blocks)[] stages = [(96, inputSize / 4, 3), (192, inputSize / 8, 3), (384, inputSize / 16, 9), (768, inputSize / 32, 3)];
        var totals = new Dictionary<string, double>();
        foreach (var (channels, size, blocks) in stages)
        {
            var block = new ConvNeXtBlock(channels, device);
            block.train();
            // As ConvNeXtBlock.forward runs them (MixedPrecision: weights cast to the input's dtype per call).
            Tensor ToRows(Tensor t) => t.permute(0, 2, 3, 1).reshape(-1, t.shape[1]);
            Tensor FromRows(Tensor rows) => rows.view(batchSize, size, size, channels).permute(0, 3, 1, 2);
            (string Name, Func<Tensor, Tensor> Op, long[] Shape)[] ops =
            [
                ("depthwise 7x7", t => MixedPrecision.Conv(block.Depthwise, t), [batchSize, channels, size, size]),
                ("groupnorm", t => MixedPrecision.Norm(block.Norm, t), [batchSize, channels, size, size]),
                ("to rows", ToRows, [batchSize, channels, size, size]),
                ("expand", t => MixedPrecision.Rows(block.PointwiseExpand, t), [batchSize * size * size, channels]),
                ("gelu", t => nn.functional.gelu(t), [batchSize * size * size, channels * 4]),
                ("gelu tanh", t => nn.functional.gelu(t, TorchSharp.Modules.GELU.Approximate.tanh), [batchSize * size * size, channels * 4]),
                ("project", t => MixedPrecision.Rows(block.PointwiseProject, t), [batchSize * size * size, channels * 4]),
                ("from rows + scale + residual", t => t + block.LayerScale.to_type(type) * FromRows(ToRows(t)), [batchSize, channels, size, size]),
                ("whole block", block.forward, [batchSize, channels, size, size]),
            ];
            var line = new List<string>();
            foreach (var (name, op, shape) in ops)
            {
                using var x = randn(shape, device: device).to_type(type).requires_grad_(true);
                double Step()
                {
                    using var scope = NewDisposeScope();
                    var clock = Stopwatch.StartNew();
                    op(x).to_type(ScalarType.Float32).square().mean().backward();
                    _ = x.grad!.flatten()[0].to_type(ScalarType.Float32).item<float>();
                    return clock.Elapsed.TotalMilliseconds;
                }
                for (var i = 0; i < 3; i++)
                    Step();
                var times = Enumerable.Range(0, iterations).Select(_ => Step()).Order().ToArray();
                var median = times[times.Length / 2];
                totals[name] = totals.GetValueOrDefault(name) + median * blocks;
                line.Add($"{name} {median:0.00}");
            }
            Console.WriteLine($"{channels}ch {size}x{size} (x{blocks} blocks): {string.Join(" | ", line)} ms");
        }
        Console.WriteLine("all 18 blocks: " + string.Join(" | ", totals.Select(t => $"{t.Key} {t.Value:0.0}")) + " ms (each op forward + backward, synchronized)");
        return 0;
    }

    /// <summary>
    /// How long a step's upload holds the CPU while the device is busy (a queue of large matrix multiplies, ~0.5s): a 31 MB
    /// float pixel batch and its 7.8 MB uint8 form, each from ordinary memory and from pinned memory without blocking.
    /// An upload that waits for the queue holds the CPU about as long as the queued work; one that doesn't returns at once.
    /// Also checks the pinned copies arrive intact.
    /// </summary>
    public static int RunUploads(int batchSize, int inputSize, int iterations)
    {
        var device = DeviceSelector.Best();
        var floats = new float[batchSize * 3 * inputSize * inputSize];
        var bytes = new byte[floats.Length];
        new Random(1).NextBytes(bytes);
        for (var i = 0; i < floats.Length; i++)
            floats[i] = bytes[i] / 255f;
        using var work = randn(4096, 4096, device: device);
        void Busy()
        {
            using var scope = NewDisposeScope();
            var x = work;
            for (var k = 0; k < 20; k++)
                x = x.matmul(work).tanh();
        }
        double Hold(Func<Tensor> upload, out Tensor uploaded)
        {
            Busy();
            var clock = Stopwatch.StartNew();
            uploaded = upload();
            var held = clock.Elapsed.TotalMilliseconds;
            _ = uploaded.flatten()[0].to_type(ScalarType.Float32).item<float>(); // drain before the next measurement
            return held;
        }
        var cases = new (string Name, Func<Tensor> Upload)[]
        {
            ("float, ordinary memory", () => tensor(floats).to(device)),
            ("uint8, ordinary memory", () => tensor(bytes).to(device)),
            ("float, pinned, non-blocking", () => tensor(floats).pin_memory().to(device, non_blocking: true)),
            ("uint8, pinned, non-blocking", () => tensor(bytes).pin_memory().to(device, non_blocking: true)),
        };
        foreach (var (name, upload) in cases)
        {
            var times = new List<double>();
            Tensor? last = null;
            for (var i = 0; i < iterations + 2; i++)
            {
                last?.Dispose();
                var held = Hold(upload, out last);
                if (i >= 2)
                    times.Add(held);
            }
            var intact = name.StartsWith("float")
                ? last!.cpu().data<float>().ToArray().AsSpan().SequenceEqual(floats)
                : last!.cpu().data<byte>().ToArray().AsSpan().SequenceEqual(bytes);
            last.Dispose();
            times.Sort();
            Console.WriteLine($"{name,-30} CPU held {times[times.Count / 2],8:0.0} ms (median of {times.Count}), {(intact ? "intact" : "CORRUPTED")}");
        }
        // The pinned source freed right after the copy is queued, and new pinned memory filled with garbage at once: the
        // host allocator must keep the source's block until the copy is done.
        var reused = true;
        for (var i = 0; i < iterations; i++)
        {
            Busy();
            Tensor arrived;
            using (var source = tensor(floats).pin_memory())
                arrived = source.to(device, non_blocking: true);
            using (var garbage = full(floats.Length, -7f).pin_memory())
                garbage.fill_(-9f);
            reused &= arrived.cpu().data<float>().ToArray().AsSpan().SequenceEqual(floats);
            arrived.Dispose();
        }
        Console.WriteLine($"pinned source freed while its copy was queued: {(reused ? "arrived intact every time" : "CORRUPTED")}");
        Busy();
        var clockBusy = Stopwatch.StartNew();
        _ = work.flatten()[0].item<float>();
        Console.WriteLine($"(the queued work alone: {clockBusy.Elapsed.TotalMilliseconds:0.0} ms)");
        return 0;
    }

    /// <summary>
    /// Forward + backward time of the glimpse path (<see cref="Glimpses"/>) on random input, part by part: the glimpses'
    /// tower pass alone, then with the per-tag log-sum-exp over them (<see cref="ClassificationPooling.RawScores"/>).
    /// With <paramref name="varyCount"/> each image drops a random few windows per iteration, as training's sampling does,
    /// so the glimpse batch's size changes from step to step; <paramref name="pad"/> pads it back to a fixed size, as training does.
    /// </summary>
    public static int RunGlimpses(int batchSize, int inputSize, int iterations, int perImage, bool varyCount, bool pad = true, int chunk = Glimpses.DefaultChunk)
    {
        var device = DeviceSelector.Best();
        manual_seed(0);
        var tower = new ImageTower(512, 96, [96, 192, 384, 768], [3, 3, 9, 3], CPU, fusedStages: 3, classificationGlimpses: true)
            { LocalizationFromFusedOnly = true, ComputeType = ScalarType.BFloat16, ClassificationPoolingTemperature = 0.5f };
        tower.to(device);
        tower.train();
        using var pixels = randn(batchSize, 3, inputSize, inputSize).to(device);
        var tags = randn(18452, 512, device: device).mul_(0.05).requires_grad_(true);
        using var pooledRows = arange(15817, dtype: ScalarType.Int64, device: device);
        var layout = Glimpses.Layout(inputSize);
        var content = Enumerable.Repeat(new UnbooruTagger.Core.Encoding.LetterboxBox(0, 0, inputSize, inputSize), batchSize).ToArray();
        var random = new Random(0);
        var probe = tower.parameters().First();

        double Step(bool withScores)
        {
            using var scope = NewDisposeScope();
            var windows = content.Select(c => Glimpses.Sample(layout, c, [], varyCount ? perImage - random.Next(4) : perImage, random)).ToArray();
            var clock = Stopwatch.StartNew();
            tower.zero_grad();
            var (cells, valid) = Glimpses.Embed(tower, pixels, content, layout, windows, chunk, padTo: pad ? batchSize * perImage : 0);
            var loss = withScores
                ? ClassificationPooling.RawScores(zeros(batchSize, 512, device: device), cells, valid, tags, pooledRows, 0.5f).square().mean()
                : cells.square().mean();
            loss.backward();
            _ = probe.grad!.flatten()[0].to_type(ScalarType.Float32).item<float>();
            return clock.Elapsed.TotalMilliseconds;
        }

        foreach (var withScores in new[] { false, true })
        {
            for (var i = 0; i < 5; i++)
                Step(withScores);
            var times = Enumerable.Range(0, iterations).Select(_ => Step(withScores)).Order().ToArray();
            Console.WriteLine($"glimpses {(withScores ? "+ scores" : "alone   ")} {batchSize} images x {perImage}{(varyCount ? " (varying)" : "")}{(pad ? " padded" : "")}: median {times[times.Length / 2]:0.0} ms, min {times[0]:0.0} ms, max {times[^1]:0.0} ms");
        }
        return 0;
    }

    private static Tensor ChannelsLast(Tensor x) => x.permute(0, 2, 3, 1).contiguous().permute(0, 3, 1, 2);
}
