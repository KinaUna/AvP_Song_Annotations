using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using System.Threading.Tasks;

namespace AvP_Song_Annotations.Extensions
{
    public static class ProcessExtensions
    {
        public static Task RunAsync(this Process process)
        {
            var tcs = new TaskCompletionSource<object>();
            process.EnableRaisingEvents = true;
            process.Exited += (s, e) => tcs.TrySetResult(null);
            // not sure on best way to handle false being returned
            if (!process.Start()) tcs.SetException(new Exception("Failed to start process."));
            return tcs.Task;
        }

        public static Task ExitedAsync(this Process p)
        {
            var tcs = new TaskCompletionSource<object>();
            p.Exited += (s, e) => tcs.TrySetResult(null);
            if (p.HasExited) tcs.TrySetResult(null);
            return tcs.Task;
        }
    }
}
