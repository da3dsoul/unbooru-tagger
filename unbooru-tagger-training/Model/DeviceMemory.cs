namespace UnbooruTagger.Training.Model;

/// <summary>This process's device memory as the GPU driver counts it (the DRM fdinfo keys), whatever the allocator reports.</summary>
public static class DeviceMemory
{
    /// <summary>The sum of <paramref name="key"/> over this process's DRM file descriptors, in MB; 0 without a DRM device (CUDA's driver has no such keys).</summary>
    public static long Mb(string key)
    {
        long kib = 0;
        foreach (var file in Directory.GetFiles("/proc/self/fdinfo"))
        {
            string[] lines;
            try { lines = File.ReadAllLines(file); }
            catch (IOException) { continue; }
            if (!lines.Any(l => l.StartsWith("drm-driver")))
                continue;
            foreach (var line in lines.Where(l => l.StartsWith(key + ":")))
                kib += long.Parse(line[(key.Length + 1)..].Trim().Split(' ')[0]);
        }
        return kib / 1024;
    }

    /// <summary>Resident VRAM (<c>drm-total-vram0</c>), in MB.</summary>
    public static long ProcessVramMb() => Mb("drm-total-vram0");
}
