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
            if (layout == "bf16-mixed")
                tower.ComputeType = ScalarType.BFloat16;
            manual_seed(1);
            using var input = randn(batchSize, 3, inputSize, inputSize).to(device);
            using var x = layout == "nchw" ? input.alias() : bf16 ? input.to_type(ScalarType.BFloat16) : ChannelsLast(input);
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

    /// <summary>Forward + backward time of each op of one ConvNeXt block per stage, at that stage's real size.</summary>
    public static int RunComponents(int batchSize, int inputSize, int iterations)
    {
        var device = DeviceSelector.Best();
        (long Channels, long Size, int Blocks)[] stages = [(96, inputSize / 4, 3), (192, inputSize / 8, 3), (384, inputSize / 16, 9), (768, inputSize / 32, 3)];
        foreach (var (channels, size, blocks) in stages)
        {
            var block = new ConvNeXtBlock(channels, device);
            block.train();
            using var input = randn(batchSize, channels, size, size, device: device).requires_grad_(true);
            var probe = block.parameters().First();
            (string Name, Func<Tensor, Tensor> Op, long InChannels)[] ops =
            [
                ("depthwise 7x7", block.Depthwise.forward, channels),
                ("groupnorm", block.Norm.forward, channels),
                ("pointwise expand", block.PointwiseExpand.forward, channels),
                ("gelu", t => nn.functional.gelu(t), channels * 4),
                ("pointwise project", block.PointwiseProject.forward, channels * 4),
                ("whole block", block.forward, channels),
            ];
            var line = new List<string>();
            foreach (var (name, op, inChannels) in ops)
            {
                using var x = randn(batchSize, inChannels, size, size, device: device).requires_grad_(true);
                double Step()
                {
                    using var scope = NewDisposeScope();
                    var clock = Stopwatch.StartNew();
                    op(x).square().mean().backward();
                    _ = x.grad!.flatten()[0].item<float>();
                    return clock.Elapsed.TotalMilliseconds;
                }
                for (var i = 0; i < 3; i++)
                    Step();
                var times = Enumerable.Range(0, iterations).Select(_ => Step()).Order().ToArray();
                line.Add($"{name} {times[times.Length / 2]:0.00}");
            }
            Console.WriteLine($"{channels}ch {size}x{size} (x{blocks} blocks): {string.Join(" | ", line)} ms");
        }
        return 0;
    }

    private static Tensor ChannelsLast(Tensor x) => x.permute(0, 2, 3, 1).contiguous().permute(0, 3, 1, 2);
}
