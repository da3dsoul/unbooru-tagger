using System.Diagnostics;
using System.Runtime.InteropServices;

namespace UnbooruTagger.Training.Commands;

/// <summary>What the terminal can show beyond text: Sixel support and the character cell's size in pixels.</summary>
public static class TerminalGraphics
{
    /// <summary>
    /// Whether the terminal shows Sixel images. Inside tmux, tmux itself must support it (3.4+,
    /// built with sixel) and say so for the attached client; outside, terminals known to support it.
    /// </summary>
    public static bool SupportsSixel()
    {
        if (Console.IsOutputRedirected)
            return false;
        if (Environment.GetEnvironmentVariable("TMUX") is not null)
            return Tmux("#{client_termfeatures}")?.Split(',').Contains("sixel") == true;
        return Environment.GetEnvironmentVariable("KONSOLE_VERSION") is not null
               || Environment.GetEnvironmentVariable("WEZTERM_EXECUTABLE") is not null
               || Environment.GetEnvironmentVariable("TERM") is "foot" or "mlterm" or "xterm-sixel";
    }

    /// <summary>The character cell in pixels: from the tty, else from tmux, else a common default.</summary>
    public static (int Width, int Height) CellSize()
    {
        if (OperatingSystem.IsLinux() && Ioctl(1, 0x5413 /* TIOCGWINSZ */, out var size) == 0 && size.Columns > 0 && size.Rows > 0 && size.XPixels > 0 && size.YPixels > 0)
            return (size.XPixels / size.Columns, size.YPixels / size.Rows);
        if (Environment.GetEnvironmentVariable("TMUX") is not null
            && Tmux("#{client_cell_width} #{client_cell_height}")?.Split(' ', StringSplitOptions.RemoveEmptyEntries) is [var w, var h]
            && int.TryParse(w, out var width) && int.TryParse(h, out var height) && width > 0 && height > 0)
            return (width, height);
        return (8, 16);
    }

    private static string? Tmux(string format)
    {
        try
        {
            using var tmux = Process.Start(new ProcessStartInfo("tmux", ["display", "-p", format]) { RedirectStandardOutput = true, RedirectStandardError = true })!;
            var output = tmux.StandardOutput.ReadToEnd().Trim();
            tmux.WaitForExit();
            return tmux.ExitCode == 0 ? output : null;
        }
        catch (System.ComponentModel.Win32Exception)
        {
            return null;
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct WindowSize
    {
        public ushort Rows, Columns, XPixels, YPixels;
    }

    [DllImport("libc", EntryPoint = "ioctl", SetLastError = true)]
    private static extern int Ioctl(int fd, ulong request, out WindowSize size);
}
