using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Forms;
using System.Windows.Media;
using AvP_Song_Annotations.Extensions;
using AvP_Song_Annotations.Models;
using AvP_Song_Annotations.Services;
using AvP_Song_Annotations.ViewModels;
using TextBox = System.Windows.Controls.TextBox;
using NetMQ;
using NetMQ.Sockets;
using System.Net.NetworkInformation;
using System.Net;

namespace AvP_Song_Annotations
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private readonly MainViewModel _viewModel; 
        private readonly string _currentDir;
        private readonly ConfigModel _config;
        private bool _showErrors = false;
        private int pythonAppId = 0;
        private string _appDataFolder;

        public MainWindow()
        {
            _appDataFolder = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData) + "\\AvP_Song_Annotations";
            if (!Directory.Exists(_appDataFolder))
            {
                Directory.CreateDirectory(_appDataFolder);
            }
            _currentDir = Directory.GetCurrentDirectory();
            _config = new ConfigModel();
            _config.RunExtractSpectogramValue = 1;
            _config.RunPredictionsValue = 1;
            _config.RunTrainingValue = 0;

            InitializeComponent();
            
            _viewModel = GetSettings();

            ContentGrid.DataContext = _viewModel;
            ExtractSpectrogramStackPanel.Visibility = Visibility.Collapsed;
            
            _viewModel.ProgressVisible = Visibility.Collapsed;
            _viewModel.Progress2Visible = Visibility.Collapsed;
            PredictRadioButton.IsChecked = true;
            // StartNetMqClient();
        }

        
        private MainViewModel GetSettings()
        {
            string settingsFile = _appDataFolder + "\\AvP_Lab_Settings.txt";
            
            MainViewModel model = new MainViewModel();
            model.MateBookFolder = _currentDir + "\\MateBook";           
            model.AudioSourceFolder = _currentDir + "\\Wav";
            model.OutputFolder = _currentDir + "\\Output";
            model.ModelFile = _currentDir + "\\song_pulse_model.h5";

            if (!Directory.Exists(model.MateBookFolder))
            {
                Directory.CreateDirectory(model.MateBookFolder);
            }
            if (!Directory.Exists(model.AudioSourceFolder))
            {
                Directory.CreateDirectory(model.AudioSourceFolder);
            }
            if (!Directory.Exists(model.OutputFolder))
            {
                Directory.CreateDirectory(model.OutputFolder);
            }

            model.Nfft = _config.ImgNFftValue.ToString(CultureInfo.InvariantCulture);
            model.BatchSize = _config.TrainBatchSizeValue.ToString(CultureInfo.InvariantCulture);
            model.ClipPadding = _config.ClipPaddingValue.ToString(CultureInfo.InvariantCulture);
            model.PulsePadding = _config.PulsePaddingValue.ToString(CultureInfo.InvariantCulture);
            model.Epochs = _config.TrainEpochsValue.ToString(CultureInfo.InvariantCulture);
            model.ImgDpi = _config.ImgDpiValue.ToString(CultureInfo.InvariantCulture);
            model.NoOverlap = _config.ImgNoOverlapValue.ToString(CultureInfo.InvariantCulture);
            model.StepLength = _config.StepLengthValue.ToString(CultureInfo.InvariantCulture);
            model.SegmentLength = _config.SegmentLengthValue.ToString(CultureInfo.InvariantCulture);
            model.PredictThreshold = _config.PredictThresholdValue.ToString(CultureInfo.InvariantCulture);
            model.PartialLength = _config.PartialLengthValue.ToString(CultureInfo.InvariantCulture);
            
            if (File.Exists(settingsFile))
            {
               List<string> settingText = System.IO.File.ReadLines(settingsFile).ToList();
               if (settingText.Count > 14)
               {
                    model.MateBookFolder = settingText[0];
                    model.AudioSourceFolder = settingText[1];
                    model.OutputFolder = settingText[2];
                    model.ModelFile = settingText[3];
                    model.SegmentLength = settingText[4];
                    model.StepLength = settingText[5];
                    model.ClipPadding = settingText[6];
                    model.PulsePadding = settingText[7];
                    model.Nfft = settingText[8];
                    model.NoOverlap = settingText[9];
                    model.ImgDpi = settingText[10];                   
                    model.Epochs = settingText[11];
                    model.BatchSize = settingText[12];
                    model.PredictThreshold = settingText[13];
                    model.PartialLength = settingText[14];
                   
               }
            }

            return model;
        }

        private async Task SaveSettings()
        {
            string settingsFile = _appDataFolder + "\\AvP_Lab_Settings.txt";

            string[] settingsText =
            {
                _viewModel.MateBookFolder,
                _viewModel.AudioSourceFolder,
                _viewModel.OutputFolder,
                _viewModel.ModelFile,
                _viewModel.SegmentLength,
                _viewModel.StepLength,
                _viewModel.ClipPadding,
                _viewModel.PulsePadding,
                _viewModel.Nfft,
                _viewModel.NoOverlap,
                _viewModel.ImgDpi,
                _viewModel.Epochs,
                _viewModel.BatchSize,
                _viewModel.PredictThreshold,
                _viewModel.PartialLength
            };

            await System.IO.File.WriteAllLinesAsync(settingsFile, settingsText);
        }

        private async Task SaveFolderHistory()
        {
            string folderHistoryFile = _appDataFolder + "\\Folder_History.txt";

            if (File.Exists(folderHistoryFile))
            {
                List<string> foldersText = System.IO.File.ReadLines(folderHistoryFile).ToList();
                foreach (string folderName in foldersText)
                {
                    if (folderName.ToUpper() == _viewModel.OutputFolder.ToUpper())
                    {
                        return;
                    }
                    foldersText.Add(_viewModel.OutputFolder);
                    await System.IO.File.WriteAllLinesAsync(folderHistoryFile, foldersText);
                    return;
                }
            }

            await System.IO.File.WriteAllTextAsync(folderHistoryFile, _viewModel.OutputFolder);
        }

        private void CalculateDataSize()
        {
            double folderSize = 0;
            string folderHistoryFile = _appDataFolder + "\\Folder_History.txt";
            List<string> newFolderHistory = new List<string>();
            if (File.Exists(folderHistoryFile))
            {
                
                List<string> foldersText = System.IO.File.ReadLines(folderHistoryFile).ToList();
                foreach (string folderName in foldersText)
                {
                    if (Directory.Exists(folderName))
                    {
                        DirectoryInfo dirInfo = new DirectoryInfo(folderName);
                        FileInfo[] files = dirInfo.GetFiles("*.npy", SearchOption.AllDirectories);
                        if(files.Length > 0)
                        {
                            foreach (FileInfo file in files)
                            {
                                folderSize += file.Length;
                            }
                        }
                        
                    }                    
                }
            }
            else
            {
                if (Directory.Exists(_viewModel.OutputFolder))
                {
                    DirectoryInfo dirInfo = new DirectoryInfo(_viewModel.OutputFolder);
                    FileInfo[] files = dirInfo.GetFiles("*.npy", SearchOption.AllDirectories);
                    if (files.Length > 0)
                    {
                        foreach (FileInfo file in files)
                        {
                            folderSize += file.Length;
                        }
                    }
                }
            }

            string[] sizeSuffixes = { "bytes", "KB", "MB", "GB", "TB" };
            int i = 0;
            while (folderSize > 1024.0 && i < 5)
            {
                folderSize = folderSize / 1024.0;
                i++;
            }
            _viewModel.DataSize = Math.Round(folderSize, 1) + sizeSuffixes[i];
        }

        private async Task DeleteDataFiles()
        {
            string folderHistoryFile = _appDataFolder + "\\Folder_History.txt";
            List<string> newFolderHistory = new List<string>();
            if (File.Exists(folderHistoryFile))
            {
                List<string> foldersText = System.IO.File.ReadLines(folderHistoryFile).ToList();
                foreach (string folderName in foldersText)
                {
                    bool deleteErrors = false;
                    DirectoryInfo dirInfo = new DirectoryInfo(folderName);
                    FileInfo[] files = dirInfo.GetFiles("*.npy", SearchOption.AllDirectories);
                    foreach (FileInfo file in files)
                    {
                        try
                        {
                            File.Delete(file.FullName);
                        }
                        catch (Exception)
                        {
                            deleteErrors = true;
                        }
                    }
                    if (deleteErrors)
                    {
                        newFolderHistory.Add(folderName);
                    }
                }
            }
            if (newFolderHistory.Any())
            {
                await System.IO.File.WriteAllLinesAsync(folderHistoryFile, newFolderHistory);
            }
            else
            {
                File.Delete(folderHistoryFile);
            }

            CalculateDataSize();
        }

        private async Task ValidateFolders()
        {
            bool sourceFolderExists = false;
            bool outputFolderExists = false;
            bool modelFileExists = false;
            bool mateBookFolderExists = false;

            if (!Directory.Exists(MateBookTextBox.Text))
            {
                MateBookTextBox.Background = new SolidColorBrush(Colors.LightPink);
            }
            else
            {
                MateBookTextBox.Background = new SolidColorBrush(Colors.White);
                mateBookFolderExists = true;
            }

            if (!Directory.Exists(AudioSourceTextBox.Text))
            {
                AudioSourceTextBox.Background = new SolidColorBrush(Colors.LightPink);
            }
            else
            {
                AudioSourceTextBox.Background = new SolidColorBrush(Colors.White);
                sourceFolderExists = true;
            }

            if (!Directory.Exists(OutputFolderTextBox.Text))
            {
                OutputFolderTextBox.Background = new SolidColorBrush(Colors.LightPink);
            }
            else
            {
                OutputFolderTextBox.Background = new SolidColorBrush(Colors.White);
                outputFolderExists = true;
            }

            if (!File.Exists(WeightsFileTextBox.Text))
            {
                WeightsFileTextBox.Background = new SolidColorBrush(Colors.LightPink);
            }
            else
            {
                WeightsFileTextBox.Background = new SolidColorBrush(Colors.White);
                modelFileExists = true;
            }

            if (sourceFolderExists && outputFolderExists && modelFileExists && mateBookFolderExists)
            {
                RunAnnotationsButton.IsEnabled = true;
                await SaveSettings();
            }
            else
            {
                RunAnnotationsButton.IsEnabled = false;
            }
        }

        private async void SelectMateBookButton_OnClick(object sender, RoutedEventArgs e)
        {
            var dialog = new FolderBrowserDialog();
            DialogResult result = dialog.ShowDialog();
            if (result == System.Windows.Forms.DialogResult.OK)
            {
                _viewModel.MateBookFolder = dialog.SelectedPath;
            }

            await ValidateFolders();
        }

        private async void SelectAudioSourceButton_OnClick(object sender, RoutedEventArgs e)
        {
            var dialog = new System.Windows.Forms.FolderBrowserDialog();
            System.Windows.Forms.DialogResult result = dialog.ShowDialog();
            if (result == System.Windows.Forms.DialogResult.OK)
            {
                _viewModel.AudioSourceFolder = dialog.SelectedPath;
            }
            await ValidateFolders();
        }

        private async void SelectOutputFolderButton_OnClick(object sender, RoutedEventArgs e)
        {
            var dialog = new FolderBrowserDialog();
            DialogResult result = dialog.ShowDialog();
            if (result == System.Windows.Forms.DialogResult.OK)
            {
                _viewModel.OutputFolder = dialog.SelectedPath;
            }
            await ValidateFolders();
        }

        private async void WeightsFileButton_OnClick(object sender, RoutedEventArgs e)
        {
            var dialog = new OpenFileDialog();
            dialog.Filter = "TensorFlow Model File|*.h5";
            DialogResult result = dialog.ShowDialog();
            if (result == System.Windows.Forms.DialogResult.OK)
            {
                _viewModel.ModelFile = dialog.FileName;
            }
            await ValidateFolders();
            string settingsFile = _viewModel.ModelFile.Replace(".h5", ".txt");
            if (File.Exists(settingsFile))
            {
                List<string> settingText = System.IO.File.ReadLines(settingsFile).ToList();
                if (settingText.Count > 6)
                {
                    _viewModel.SegmentLength = settingText[0].Split('=')[1];
                    _viewModel.StepLength = settingText[1].Split('=')[1];
                    _viewModel.ClipPadding = settingText[2].Split('=')[1];
                    _viewModel.PulsePadding = settingText[3].Split('=')[1];
                    _viewModel.Nfft = settingText[4].Split('=')[1];
                    _viewModel.NoOverlap = settingText[5].Split('=')[1];
                    _viewModel.ImgDpi = settingText[6].Split('=')[1];
                    
                    await SaveSettings();
                }
            }
        }

        private void StopAnnotationsButton_OnClick(object sender, RoutedEventArgs e)
        {
            MessageBoxResult messageBoxResult = System.Windows.MessageBox.Show("Are you sure you want to cancel?", "Confirm cancellation", System.Windows.MessageBoxButton.YesNo);
            if (messageBoxResult == MessageBoxResult.Yes)
            {
                if (pythonAppId != 0)
                {
                    var pythonProcess = Process.GetProcessById(pythonAppId);
                    pythonProcess.Kill();
                    pythonAppId = 0;
                }

                RunAnnotationsButton.Visibility = Visibility.Visible;
                StopAnnotationsButton.Visibility = Visibility.Hidden;
                RunAnnotationsButton.IsEnabled = true;
                PredictRadioButton.IsEnabled = true;
                TrainRadioButton.IsEnabled = true;
                SelectAudioSourceButton.IsEnabled = true;
                SelectMateBookButton.IsEnabled = true;
                SelectOutputFolderButton.IsEnabled = true;
                WeightsFileButton.IsEnabled = true;
                _viewModel.Progress = 0.0;
                _viewModel.Progress2 = 0.0;
                CalculateDataSize();
            }            
        }

        private async void RunAnnotationsButton_OnClick(object sender, RoutedEventArgs e)
        {
            _viewModel.ConsoleOutputText = "";

            if (pythonAppId != 0)
            {
                var pythonProcess = Process.GetProcessById(pythonAppId);
                pythonProcess.Kill();
                pythonAppId = 0;
            }

            await SaveSettings();
            await SaveFolderHistory();

            RunAnnotationsButton.Visibility = Visibility.Hidden;
            StopAnnotationsButton.Visibility = Visibility.Visible;

            _config.WavFolderValue = _viewModel.AudioSourceFolder;
            _config.OutputFolderValue = _viewModel.OutputFolder;
            _config.DataFolderValue = _viewModel.MateBookFolder;
            _config.WeightsFileValue = _viewModel.ModelFile;

            bool slParsed = int.TryParse(_viewModel.SegmentLength, NumberStyles.Any, CultureInfo.InvariantCulture, out int sl);
            if (slParsed)
            {
                _config.SegmentLengthValue = sl;
            }

            bool stplParsed = int.TryParse(_viewModel.StepLength, NumberStyles.Any, CultureInfo.InvariantCulture, out int stpl);
            if (stplParsed)
            {
                _config.StepLengthValue = stpl;
            }

            bool clpPadParsed = int.TryParse(_viewModel.ClipPadding, NumberStyles.Any, CultureInfo.InvariantCulture, out int clpPad);
            if (clpPadParsed)
            {
                _config.ClipPaddingValue = clpPad;
            }

            bool plsPadParsed = double.TryParse(_viewModel.PulsePadding, NumberStyles.Float, CultureInfo.InvariantCulture, out double plsPad);
            if (plsPadParsed)
            {
                _config.PulsePaddingValue = plsPad;
            }

            bool nfftParsed = int.TryParse(_viewModel.Nfft, NumberStyles.Any, CultureInfo.InvariantCulture, out int nfft);
            if (nfftParsed)
            {
                _config.ImgNFftValue = nfft;
            }

            bool noOvrParsed = int.TryParse(_viewModel.NoOverlap, NumberStyles.Any, CultureInfo.InvariantCulture, out int noOvr);
            if (noOvrParsed)
            {
                _config.ImgNoOverlapValue = noOvr;
            }

            bool imgDpiParsed = int.TryParse(_viewModel.ImgDpi, NumberStyles.Any, CultureInfo.InvariantCulture, out int imgDpi);
            if (imgDpiParsed)
            {
                _config.ImgDpiValue = imgDpi;
            }

            bool partialLengthParsed = int.TryParse(_viewModel.PartialLength, NumberStyles.Any, CultureInfo.InvariantCulture, out int partialLength);
            if (partialLengthParsed)
            {
                _config.PartialLengthValue = partialLength;
            }

            bool epochsParsed = int.TryParse(_viewModel.Epochs, NumberStyles.Any, CultureInfo.InvariantCulture, out int epochs);
            if (epochsParsed)
            {
                _config.TrainEpochsValue = epochs;
            }

            bool batchSizeParsed = int.TryParse(_viewModel.StepLength, NumberStyles.Any, CultureInfo.InvariantCulture, out int batchSize);
            if (batchSizeParsed)
            {
                _config.TrainBatchSizeValue = batchSize;
            }

            bool predThresParsed = double.TryParse(_viewModel.PredictThreshold, NumberStyles.Float, CultureInfo.InvariantCulture, out double predThres);
            if (predThresParsed)
            {
                _config.PredictThresholdValue = predThres;
            }

            
            _config.ConfigFileName = _appDataFolder + "\\aan_config.txt";

            int portNumber = GetOpenPort();

            string[] configText =
            {
                ConfigModel.WavFolderKey + _config.WavFolderValue,
                ConfigModel.DataFolderKey + _config.DataFolderValue,
                ConfigModel.OutputFolderKey + _config.OutputFolderValue,
                ConfigModel.SegmentLengthKey + _config.SegmentLengthValue.ToString(CultureInfo.InvariantCulture),
                ConfigModel.StepLengthKey + _config.StepLengthValue.ToString(CultureInfo.InvariantCulture),
                ConfigModel.ClipPaddingKey + _config.ClipPaddingValue.ToString(CultureInfo.InvariantCulture),
                ConfigModel.PulsePaddingKey + _config.PulsePaddingValue.ToString(CultureInfo.InvariantCulture),
                ConfigModel.ImgNFftKey + _config.ImgNFftValue.ToString(CultureInfo.InvariantCulture),
                ConfigModel.ImgNoOverlapKey + _config.ImgNoOverlapValue.ToString(CultureInfo.InvariantCulture),
                ConfigModel.ImgDpiKey + _config.ImgDpiValue.ToString(CultureInfo.InvariantCulture),
                ConfigModel.PartialLengthKey + _config.PartialLengthValue.ToString(CultureInfo.InvariantCulture),
                ConfigModel.RunTrainingKey + _config.RunTrainingValue.ToString(CultureInfo.InvariantCulture),
                ConfigModel.RunExtractSpectogramsKey + _config.RunExtractSpectogramValue.ToString(CultureInfo.InvariantCulture),
                ConfigModel.RunPredictionsKey + _config.RunPredictionsValue.ToString(CultureInfo.InvariantCulture),
                ConfigModel.TrainEpochsKey + _config.TrainEpochsValue.ToString(CultureInfo.InvariantCulture),
                ConfigModel.TrainBatchSizeKey + _config.TrainBatchSizeValue.ToString(CultureInfo.InvariantCulture),
                ConfigModel.PredictThresholdKey + _config.PredictThresholdValue.ToString(CultureInfo.InvariantCulture),
                ConfigModel.WeightsFileKey + _config.WeightsFileValue,
                ConfigModel.ZmqPortKey + portNumber
            };

            RunAnnotationsButton.IsEnabled = false;
            PredictRadioButton.IsEnabled = false;
            TrainRadioButton.IsEnabled = false;
            SelectAudioSourceButton.IsEnabled = false;
            SelectMateBookButton.IsEnabled = false;
            SelectOutputFolderButton.IsEnabled = false;
            WeightsFileButton.IsEnabled = false;

            ConsoleOutputTextBox.Visibility = Visibility.Visible;

            // Save config file
            await System.IO.File.WriteAllLinesAsync(_config.ConfigFileName, configText);
            string pythonFile = _currentDir + "\\AvpSongPulseAnnotations\\AvpSongPulseAnnotations.exe";
            _viewModel.ConsoleOutputText += "Starting processing...\n";

            StartNetMqClient(portNumber);
            await RunPythonFile(pythonFile, _config.ConfigFileName);
            CalculateDataSize();
        }

        public async Task RunPythonFile(string filename, string configFile)
        {
            var process = new ProcessWrapper(filename, "--config \"" + configFile + "\"");
            process.OutputDataReceived += (sender, eventArgs) =>
            {
                if (!string.IsNullOrWhiteSpace(eventArgs.Data))
                {
                    string dataString = eventArgs.Data.Replace("\f", "");
                    dataString = dataString.TrimStart();
                    if (dataString.StartsWith("Epoch"))
                    {
                        string[] epochText = dataString.Split(' ');
                        if(epochText.Length > 1)
                        {
                            string[] epochNumbers = epochText[1].Split('/');
                            bool epochDoneParsed = double.TryParse(epochNumbers[0], NumberStyles.Float, CultureInfo.InvariantCulture, out double epochDone);
                            bool epochTotalParsed = double.TryParse(epochNumbers[1], NumberStyles.Float, CultureInfo.InvariantCulture, out double epochTotal);
                            if(epochDoneParsed && epochTotalParsed)
                            {
                                _viewModel.Progress = 100 * epochDone / epochTotal;
                                UpdateEta();
                            }
                        }
                    }
                    _viewModel.ConsoleOutputText += dataString + "\n\n";
                }
            };
            process.ErrorDataReceived += (sender, eventArgs) =>
            {
                if (eventArgs.Data != null && _showErrors)
                {
                    _viewModel.ConsoleOutputText += eventArgs.Data + "\n\n";
                }
            };


            pythonAppId = process.Start();
            
            await process.ExitedAsync();
            pythonAppId = 0;
            
            RunAnnotationsButton.Visibility = Visibility.Visible;
            StopAnnotationsButton.Visibility = Visibility.Hidden;
            RunAnnotationsButton.IsEnabled = true;
            PredictRadioButton.IsEnabled = true;
            TrainRadioButton.IsEnabled = true;
            SelectAudioSourceButton.IsEnabled = true;
            SelectMateBookButton.IsEnabled = true;
            SelectOutputFolderButton.IsEnabled = true;
            WeightsFileButton.IsEnabled = true;
            _viewModel.Progress = 0.0;
            _viewModel.Progress2 = 0.0;
        }

        private async void FoldersGrid_OnLoaded(object sender, RoutedEventArgs e)
        {
            await ValidateFolders();
            CalculateDataSize();
        }

        private void PredictRadioButton_OnChecked(object sender, RoutedEventArgs e)
        {
            PredictThresholdStackPanel.Visibility = Visibility.Visible;
            ExtractSpectrogramStackPanel.Visibility = Visibility.Collapsed;
            PartialLengthStackPanel.Visibility = Visibility.Visible;
            _config.RunExtractSpectogramValue = 1;
            ExtractRadioButton.IsChecked = true;
            _config.RunPredictionsValue = 1;
            _config.RunTrainingValue = 0;
        }

        private void TrainRadioButton_OnChecked(object sender, RoutedEventArgs e)
        {
            PredictThresholdStackPanel.Visibility = Visibility.Collapsed;
            ExtractSpectrogramStackPanel.Visibility = Visibility.Visible;
            PartialLengthStackPanel.Visibility = Visibility.Collapsed;
            _config.RunExtractSpectogramValue = 1;
            ExtractRadioButton.IsChecked = true;
            _config.RunPredictionsValue = 0;
            _config.RunTrainingValue = 1;
        }

        private void ExtractRadioButton_OnChecked(object sender, RoutedEventArgs e)
        {
            _config.RunExtractSpectogramValue = 1;
        }

        private void DontExtractRadioButton_OnChecked(object sender, RoutedEventArgs e)
        {
            // Todo: Check if the data exists, if not disable this option.
            _config.RunExtractSpectogramValue = 0;
        }

        private void ConsoleOutputTextBox_OnTextChanged(object sender, TextChangedEventArgs e)
        {
            TextBox consoleTextBox = sender as TextBox;
            if (consoleTextBox != null)
            {
                consoleTextBox.SelectionStart = _viewModel.ConsoleOutputText.Length;
                consoleTextBox?.ScrollToEnd();
            }

            CalculateDataSize();
        }
        
        private NetMQPoller _poller;

        public void StartNetMqClient(int port)
        {
            _viewModel.StartTime = DateTime.Now;
            _viewModel.StartTime2 = DateTime.Now;
            Task.Factory.StartNew(() => {
                using (var pullSocket = new PullSocket(">tcp://127.0.0.1:" + port.ToString()))
                using (_poller = new NetMQPoller())
                {
                    pullSocket.ReceiveReady += (object sender, NetMQSocketEventArgs netMqSocketEventArgs) =>
                    {
                        string dataString = netMqSocketEventArgs.Socket.ReceiveFrameString();
                        if (!string.IsNullOrEmpty(dataString))
                        {
                            if (dataString.StartsWith('@'))
                            {
                                bool progressParsed = Double.TryParse(dataString.Remove(0, 1), NumberStyles.Float, CultureInfo.InvariantCulture, out double progress);
                                _viewModel.Progress = progress * 100;
                                if (_viewModel.FileCount > 0.001 && _viewModel.TaskCount > 0.001)
                                {
                                    double tempProgress = 100 * ((_viewModel.FileCompletedCount * _viewModel.TaskCount) + _viewModel.TaskCompletedCount + progress) / (_viewModel.FileCount * _viewModel.TaskCount);
                                    _viewModel.Progress2 = tempProgress;
                                    UpdateEta2();
                                }
                                else
                                {
                                    _viewModel.TaskCompletedCount = 0.0;
                                    _viewModel.TaskCount = 0.0;
                                    _viewModel.FileCompletedCount = 0.0;
                                    _viewModel.FileCount = 0.0;
                                    _viewModel.Progress2 = 0.0;
                                }
                                
                                UpdateEta();
                            }
                            else
                            {
                                if (dataString.StartsWith('!'))
                                {
                                    string[] progressStrings = dataString.Remove(0, 1).Split(':');
                                    if (progressStrings.Length > 1)
                                    {
                                        bool progressNumParsed = double.TryParse(progressStrings[0], NumberStyles.Float, CultureInfo.InvariantCulture, out double progressNum);
                                        bool progressCountParsed = double.TryParse(progressStrings[1], NumberStyles.Float, CultureInfo.InvariantCulture, out double progressCount);
                                        bool progressFileNumParsed = double.TryParse(progressStrings[2], NumberStyles.Float, CultureInfo.InvariantCulture, out double progressFileNum);
                                        bool progressFileCountParsed = double.TryParse(progressStrings[3], NumberStyles.Float, CultureInfo.InvariantCulture, out double progressFileCount);
                                        
                                        if (progressCount > 0.001 && progressFileCount > 0.001 && progressNumParsed && progressCountParsed && progressFileNumParsed && progressFileCountParsed)
                                        {
                                            _viewModel.TaskCompletedCount = progressNum;
                                            _viewModel.TaskCount = progressCount;
                                            _viewModel.FileCompletedCount = progressFileNum;
                                            _viewModel.FileCount = progressFileCount;
                                            _viewModel.Progress2 = 100 * ((progressFileNum * progressCount) + progressNum) / (progressFileCount * progressCount);
                                        }
                                        else
                                        {
                                            _viewModel.TaskCompletedCount = 0.0;
                                            _viewModel.TaskCount = 0.0;
                                            _viewModel.FileCompletedCount = 0.0;
                                            _viewModel.FileCount = 0.0;
                                            _viewModel.Progress2 = 0.0;
                                        }
                                        UpdateEta2();
                                    }

                                }
                                else
                                {
                                    if (dataString.StartsWith(']'))
                                    {
                                       _viewModel.StartTime = DateTime.Now;
                                       _viewModel.Progress1Text = dataString.Remove(0, 1);
                                    }
                                    else
                                    {
                                        if (dataString.StartsWith('}'))
                                        {
                                            _viewModel.StartTime2 = DateTime.Now;
                                            _viewModel.Progress2Text = dataString.Remove(0, 1);
                                        }
                                        else
                                        {
                                            _viewModel.ConsoleOutputText += dataString + "\n\n";
                                        }
                                    }
                                    
                                }
                            }
                        }
                        
                    };
                    _poller.Add(pullSocket);
                    _poller.Run();
                }
            });
        }

        public void StopNetMqClient()
        {
            _poller?.Stop();
        }


        private async void MainWindow_OnClosing(object sender, CancelEventArgs e)
        {
            StopNetMqClient();
            await SaveSettings();
            if (pythonAppId != 0)
            {
                var pythonProcess = Process.GetProcessById(pythonAppId);
                pythonProcess.Kill();
                pythonAppId = 0;
            }
        }

        private void UpdateEta()
        {
            if (_viewModel.Progress > 0.01)
            {
                TimeSpan timeSpent = DateTime.Now - _viewModel.StartTime;
                TimeSpan timeOverall = TimeSpan.FromTicks((long)(timeSpent.Ticks / (_viewModel.Progress / 100.0)));
                TimeSpan timeRemaining = timeOverall - timeSpent;
                _viewModel.ProgressEta = new TimeSpan(timeRemaining.Days, timeRemaining.Hours, timeRemaining.Minutes, timeRemaining.Seconds);
            }
            
        }

        private void UpdateEta2()
        {
            if (_viewModel.Progress2 > 0.01)
            {
                TimeSpan timeSpent2 = DateTime.Now - _viewModel.StartTime2;
                TimeSpan timeOverall2 = TimeSpan.FromTicks((long)(timeSpent2.Ticks / (_viewModel.Progress2 / 100.0)));
                TimeSpan timeRemaining2 = timeOverall2 - timeSpent2;
                _viewModel.ProgressEta2 = new TimeSpan(timeRemaining2.Days, timeRemaining2.Hours, timeRemaining2.Minutes, timeRemaining2.Seconds);
            }
            
        }

        private async void DeleteTempFilesButton_OnClick(object sender, RoutedEventArgs e)
        {
            MessageBoxResult messageBoxResult = System.Windows.MessageBox.Show("Are you sure you want to delete the data files?", "Delete Confirmation", System.Windows.MessageBoxButton.YesNo);
            if (messageBoxResult == MessageBoxResult.Yes)
            {
                await DeleteDataFiles();
            }                
        }

        public static int GetOpenPort(int startPort = 20030)
        {
            int portStartIndex = startPort;
            int count = 99;
            IPGlobalProperties properties = IPGlobalProperties.GetIPGlobalProperties();
            IPEndPoint[] tcpEndPoints = properties.GetActiveTcpListeners();

            List<int> usedPorts = tcpEndPoints.Select(p => p.Port).ToList<int>();
            int unusedPort = 0;

            unusedPort = Enumerable.Range(portStartIndex, count).Where(port => !usedPorts.Contains(port)).FirstOrDefault();
            return unusedPort;
        }
    }
}
