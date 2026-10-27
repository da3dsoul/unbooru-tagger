using System.Runtime.InteropServices;

namespace UnbooruTagger.Training.Training;

/// <summary>
/// Lets oneDNN, which runs XPU convolutions, use TF32 for fp32 math, matching what cuDNN
/// already does by default on CUDA (torch.backends.cudnn.allow_tf32). On the Arc Pro B70 it took
/// the ConvNeXt-T tower from 203 to 138 ms/step at batch 10 (relative grad error ~4e-3, the usual
/// TF32 level). TorchSharp only exposes the cuBLAS/cuDNN switches, so this calls libtorch's
/// exported at::Context members directly: on x86-64 SysV a member function is a plain function
/// whose first argument is `this`. Only the legacy flag (allowTF32OneDNN) reaches the XPU
/// convolutions in libtorch 2.10; the newer per-op setFloat32Precision reads back as set but has
/// no effect there.
/// </summary>
public static class OneDnnPrecision
{
    private const string Lib = "libtorch_cpu.so";

    [DllImport(Lib, EntryPoint = "_ZN2at13globalContextEv")]
    private static extern IntPtr GlobalContext();

    [DllImport(Lib, EntryPoint = "_ZN2at7Context18setAllowTF32OneDNNEb")]
    private static extern void SetAllowTF32OneDNN(IntPtr context, [MarshalAs(UnmanagedType.U1)] bool allow);

    public static void AllowTf32() => SetAllowTF32OneDNN(GlobalContext(), true);
}
