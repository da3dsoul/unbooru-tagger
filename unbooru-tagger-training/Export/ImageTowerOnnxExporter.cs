using Google.Protobuf;
using Onnx;
using TorchSharp.Modules;
using UnbooruTagger.Training.Model;
using static TorchSharp.torch;

namespace UnbooruTagger.Training.Export;

/// <summary>
/// Hand-builds an ONNX graph that mirrors <see cref="ImageTower"/>'s forward pass
/// node-for-node. TorchSharp has no ONNX exporter of its own — Python's torch.onnx
/// relies on tracing/scripting machinery that libtorch's C++ API (all TorchSharp binds
/// to) doesn't expose — so this walks the same structure ImageTower.forward does and
/// emits the matching ONNX ops with the trained weights as initializers. Keep this in
/// lockstep with ImageTower/ConvNeXtBlock if their forward passes change.
/// </summary>
public static class ImageTowerOnnxExporter
{
    private const string PooledOutputName = "pooled_embedding";
    private const string SpatialOutputName = "spatial_features";

    public static void Export(ImageTower tower, string outputPath, int inputSize)
    {
        var graph = new GraphProto { Name = "image_tower" };
        var names = new NameAllocator();

        graph.Input.Add(MakeValueInfo("pixel_values", 1, 3, inputSize, inputSize));

        var current = ExportConv2d(graph, tower.Stem, "pixel_values", names);
        foreach (var layer in tower.Layers)
        {
            current = layer switch
            {
                Conv2d conv => ExportConv2d(graph, conv, current, names),
                ConvNeXtBlock block => ExportConvNeXtBlock(graph, block, current, names),
                _ => throw new NotSupportedException($"No ONNX export defined for layer type {layer.GetType().Name}.")
            };
        }

        graph.Node.Add(new NodeProto
        {
            OpType = "Identity",
            Name = names.Next("Identity"),
            Input = { current },
            Output = { SpatialOutputName }
        });
        graph.Output.Add(MakeValueInfo(SpatialOutputName));

        var pooled = ExportGlobalAveragePoolAndFlatten(graph, current, names);
        ExportLinear(graph, tower.Projection, pooled, names, PooledOutputName);
        graph.Output.Add(MakeValueInfo(PooledOutputName));

        var model = new ModelProto { IrVersion = 9, Graph = graph };
        // Deliberately old/stable opset: GroupNorm and Gelu are decomposed into
        // primitive ops below rather than using the native "GroupNormalization"
        // (opset 18) and "Gelu" (opset 20) ops, since GroupNormalization's opset-18
        // definition has known scale/bias-shape ambiguity that trips up some runtimes.
        model.OpsetImport.Add(new OperatorSetIdProto { Domain = "", Version = 13 });

        using var stream = File.Create(outputPath);
        model.WriteTo(stream);
    }

    private static string ExportConv2d(GraphProto graph, Conv2d conv, string inputName, NameAllocator names)
    {
        var weightName = names.Next("conv_weight");
        graph.Initializer.Add(MakeInitializer(weightName, conv.weight));

        var inputs = new List<string> { inputName, weightName };
        if (conv.bias is not null)
        {
            var biasName = names.Next("conv_bias");
            graph.Initializer.Add(MakeInitializer(biasName, conv.bias));
            inputs.Add(biasName);
        }

        var outputName = names.Next("conv_out");
        var node = new NodeProto { OpType = "Conv", Name = names.Next("Conv"), Output = { outputName } };
        node.Input.Add(inputs);
        node.Attribute.Add(Ints("kernel_shape", conv.kernel_size));
        node.Attribute.Add(Ints("strides", conv.stride));
        node.Attribute.Add(Ints("pads", [conv.padding![0], conv.padding[1], conv.padding[0], conv.padding[1]]));
        node.Attribute.Add(Int("group", conv.groups));
        graph.Node.Add(node);

        return outputName;
    }

    private static string ExportConvNeXtBlock(GraphProto graph, ConvNeXtBlock block, string inputName, NameAllocator names)
    {
        var x = ExportConv2d(graph, block.Depthwise, inputName, names);
        x = ExportGroupNorm(graph, block.Norm, x, names);
        x = ExportConv2d(graph, block.PointwiseExpand, x, names);
        x = ExportGelu(graph, x, names);
        x = ExportConv2d(graph, block.PointwiseProject, x, names);

        var outputName = names.Next("residual_out");
        graph.Node.Add(new NodeProto
        {
            OpType = "Add",
            Name = names.Next("Add"),
            Input = { x, inputName },
            Output = { outputName }
        });
        return outputName;
    }

    /// <summary>
    /// Manually decomposes GroupNorm(num_groups: 1) — the only configuration
    /// <see cref="ConvNeXtBlock"/> uses — into ReduceMean/Sub/Mul/Div/Sqrt, rather than
    /// the native "GroupNormalization" op, whose opset-18 definition has known
    /// scale/bias-shape ambiguity across runtimes.
    /// </summary>
    private static string ExportGroupNorm(GraphProto graph, GroupNorm norm, string inputName, NameAllocator names)
    {
        if (norm.num_groups != 1)
            throw new NotSupportedException("ImageTowerOnnxExporter only supports GroupNorm(num_groups: 1); extend the decomposition below if that changes.");

        var channels = norm.weight.shape[0];

        var mean = ReduceMeanOverChannelsAndSpatial(graph, inputName, names, "gn_mean");
        var centered = ExportBinaryOp(graph, "Sub", inputName, mean, names, "gn_centered");
        var squared = ExportBinaryOp(graph, "Mul", centered, centered, names, "gn_squared");
        var variance = ReduceMeanOverChannelsAndSpatial(graph, squared, names, "gn_variance");

        var epsilonName = names.Next("gn_epsilon");
        graph.Initializer.Add(MakeScalarInitializer(epsilonName, (float)norm.eps));
        var varianceEps = ExportBinaryOp(graph, "Add", variance, epsilonName, names, "gn_variance_eps");

        var stdName = names.Next("gn_std");
        graph.Node.Add(new NodeProto { OpType = "Sqrt", Name = names.Next("Sqrt"), Input = { varianceEps }, Output = { stdName } });

        var normalized = ExportBinaryOp(graph, "Div", centered, stdName, names, "gn_normalized");

        var scaleName = names.Next("gn_scale");
        var biasName = names.Next("gn_bias");
        graph.Initializer.Add(MakeInitializer(scaleName, norm.weight, [1, channels, 1, 1]));
        graph.Initializer.Add(MakeInitializer(biasName, norm.bias!, [1, channels, 1, 1]));

        var scaled = ExportBinaryOp(graph, "Mul", normalized, scaleName, names, "gn_scaled");
        return ExportBinaryOp(graph, "Add", scaled, biasName, names, "gn_out");
    }

    private static string ReduceMeanOverChannelsAndSpatial(GraphProto graph, string inputName, NameAllocator names, string outputPrefix)
    {
        var outputName = names.Next(outputPrefix);
        var node = new NodeProto { OpType = "ReduceMean", Name = names.Next("ReduceMean"), Input = { inputName }, Output = { outputName } };
        node.Attribute.Add(Ints("axes", [1L, 2L, 3L]));
        node.Attribute.Add(Int("keepdims", 1));
        graph.Node.Add(node);
        return outputName;
    }

    private static string ExportBinaryOp(GraphProto graph, string opType, string lhs, string rhs, NameAllocator names, string outputPrefix)
    {
        var outputName = names.Next(outputPrefix);
        graph.Node.Add(new NodeProto { OpType = opType, Name = names.Next(opType), Input = { lhs, rhs }, Output = { outputName } });
        return outputName;
    }

    /// <summary>Manual erf-based decomposition (0.5 * x * (1 + erf(x / sqrt(2)))) instead of the native "Gelu" op (opset 20), again to stick to primitive, unambiguous ops.</summary>
    private static string ExportGelu(GraphProto graph, string inputName, NameAllocator names)
    {
        var invSqrt2Name = names.Next("gelu_inv_sqrt2");
        graph.Initializer.Add(MakeScalarInitializer(invSqrt2Name, 0.7071067811865476f));
        var halfName = names.Next("gelu_half");
        graph.Initializer.Add(MakeScalarInitializer(halfName, 0.5f));
        var oneName = names.Next("gelu_one");
        graph.Initializer.Add(MakeScalarInitializer(oneName, 1f));

        var scaled = ExportBinaryOp(graph, "Mul", inputName, invSqrt2Name, names, "gelu_scaled");

        var erfName = names.Next("gelu_erf");
        graph.Node.Add(new NodeProto { OpType = "Erf", Name = names.Next("Erf"), Input = { scaled }, Output = { erfName } });

        var onePlusErf = ExportBinaryOp(graph, "Add", erfName, oneName, names, "gelu_one_plus_erf");
        var halfX = ExportBinaryOp(graph, "Mul", inputName, halfName, names, "gelu_half_x");
        return ExportBinaryOp(graph, "Mul", halfX, onePlusErf, names, "gelu_out");
    }

    private static string ExportGlobalAveragePoolAndFlatten(GraphProto graph, string inputName, NameAllocator names)
    {
        var pooledName = names.Next("gap_out");
        graph.Node.Add(new NodeProto { OpType = "GlobalAveragePool", Name = names.Next("GlobalAveragePool"), Input = { inputName }, Output = { pooledName } });

        var flatName = names.Next("flatten_out");
        var flattenNode = new NodeProto { OpType = "Flatten", Name = names.Next("Flatten"), Input = { pooledName }, Output = { flatName } };
        flattenNode.Attribute.Add(Int("axis", 1));
        graph.Node.Add(flattenNode);
        return flatName;
    }

    private static void ExportLinear(GraphProto graph, Linear linear, string inputName, NameAllocator names, string outputName)
    {
        var weightName = names.Next("linear_weight");
        var biasName = names.Next("linear_bias");
        graph.Initializer.Add(MakeInitializer(weightName, linear.weight));
        graph.Initializer.Add(MakeInitializer(biasName, linear.bias!));

        var node = new NodeProto
        {
            OpType = "Gemm",
            Name = names.Next("Gemm"),
            Input = { inputName, weightName, biasName },
            Output = { outputName }
        };
        node.Attribute.Add(Int("transB", 1));
        graph.Node.Add(node);
    }

    private static TensorProto MakeInitializer(string name, Tensor tensor)
    {
        using var contiguous = tensor.detach().contiguous();
        return MakeInitializer(name, tensor, contiguous.shape);
    }

    private static TensorProto MakeInitializer(string name, Tensor tensor, long[] dims)
    {
        using var contiguous = tensor.detach().contiguous();
        var data = contiguous.data<float>().ToArray();
        var bytes = new byte[data.Length * sizeof(float)];
        Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);

        var proto = new TensorProto
        {
            Name = name,
            DataType = (int)TensorProto.Types.DataType.Float,
            RawData = ByteString.CopyFrom(bytes)
        };
        proto.Dims.Add(dims);
        return proto;
    }

    /// <summary>A rank-0 (scalar) initializer, broadcastable against any shape per ONNX's numpy-style elementwise broadcasting.</summary>
    private static TensorProto MakeScalarInitializer(string name, float value) => new()
    {
        Name = name,
        DataType = (int)TensorProto.Types.DataType.Float,
        RawData = ByteString.CopyFrom(BitConverter.GetBytes(value))
    };

    private static ValueInfoProto MakeValueInfo(string name, params long[] dims)
    {
        TensorShapeProto? shape = null;
        if (dims.Length > 0)
        {
            shape = new TensorShapeProto();
            foreach (var dim in dims)
                shape.Dim.Add(new TensorShapeProto.Types.Dimension { DimValue = dim });
        }

        return new ValueInfoProto
        {
            Name = name,
            Type = new TypeProto
            {
                TensorType = new TypeProto.Types.Tensor
                {
                    ElemType = (int)TensorProto.Types.DataType.Float,
                    Shape = shape
                }
            }
        };
    }

    private static AttributeProto Ints(string name, IEnumerable<long> values) =>
        new() { Name = name, Type = AttributeProto.Types.AttributeType.Ints, Ints = { values } };

    private static AttributeProto Int(string name, long value) =>
        new() { Name = name, Type = AttributeProto.Types.AttributeType.Int, I = value };

    private static AttributeProto Float(string name, float value) =>
        new() { Name = name, Type = AttributeProto.Types.AttributeType.Float, F = value };

    private sealed class NameAllocator
    {
        private readonly Dictionary<string, int> _counters = new();

        public string Next(string prefix)
        {
            var count = _counters.GetValueOrDefault(prefix, 0);
            _counters[prefix] = count + 1;
            return $"{prefix}_{count}";
        }
    }
}
