using System;
using System.Linq;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using AvP_Song_Annotations.Extensions;

namespace AvP_Song_Annotations.Services
{

    public class ProcessWrapper : Process
    {
        public enum PipeType { StdOut, StdErr }
        
        public class Output
        {

            public string Message { get; set; }
            public PipeType Pipe { get; set; }
            public override string ToString()
            {
                return $"{Pipe}: {Message}";
            }
        }

        private readonly string _command;
        private readonly string _args;
        private bool _isDisposed;

        private readonly Queue<Output> _outputQueue = new Queue<Output>();


        private readonly ManualResetEvent[] _waitHandles = new ManualResetEvent[2];
        private readonly ManualResetEvent _outputSteamWaitHandle = new ManualResetEvent(false);

        public ProcessWrapper(string startCommand, string args)
        {
            _command = startCommand;
            _args = args;
        }

        public IEnumerable<string> GetMessages()
        {

            while (!_isDisposed)
            {

                _outputSteamWaitHandle.WaitOne();
                if (_outputQueue.Any())
                    yield return _outputQueue.Dequeue().ToString();
            }
        }

        public void SendCommand(string command)
        {

            StandardInput.Write(command);
            StandardInput.Flush();
        }

        public new int Start()
        {

            ProcessStartInfo startInfo = new ProcessStartInfo
            {

                FileName = _command,
                Arguments = _args,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                RedirectStandardInput = true,
                CreateNoWindow = true
            };

            StartInfo = startInfo;

            OutputDataReceived += delegate (object sender, DataReceivedEventArgs args) {

                if (args.Data == null)
                {

                    _waitHandles[0].Set();
                }
                else if (args.Data.Length > 0)
                {

                    _outputQueue.Enqueue(new Output { Message = args.Data, Pipe = PipeType.StdOut });
                    _outputSteamWaitHandle.Set();
                }
            };

            ErrorDataReceived += delegate (object sender, DataReceivedEventArgs args) {

                if (args.Data == null)
                {

                    _waitHandles[1].Set();
                }
                else if (args.Data.Length > 0)
                {

                    _outputSteamWaitHandle.Set();
                    _outputQueue.Enqueue(new Output { Message = args.Data, Pipe = PipeType.StdErr });
                }
            };

            EnableRaisingEvents = true;

            var tcs = new TaskCompletionSource<int>();

            Exited += (s, ea) => tcs.SetResult(base.ExitCode);

            bool started = base.Start();

            if (!started)
            {
                //you may allow for the process to be re-used (started = false) 
                //but I'm not sure about the guarantees of the Exited event in such a case
                throw new InvalidOperationException("Could not start process: " + base.ProcessName);
            }

            _waitHandles[0] = new ManualResetEvent(false);
            BeginErrorReadLine();
            _waitHandles[1] = new ManualResetEvent(false);
            BeginOutputReadLine();

            return Id;
        }

        public new void Dispose()
        {

            StandardInput.Flush();
            StandardInput.Close();
            if (!WaitForExit(1000))
            {
                Kill();
            }
            if (WaitForExit(1000))
            {
                WaitHandle.WaitAll(_waitHandles);
            }
            base.Dispose();
            _isDisposed = true;
        }

    }
}