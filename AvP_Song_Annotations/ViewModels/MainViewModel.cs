using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;
using System.Windows;
using Microsoft.VisualBasic;

namespace AvP_Song_Annotations.ViewModels
{
    class MainViewModel: BaseViewModel
    {
        private string _mateBookFolder;
        private string _audioSourceFolder;
        private string _outputFolder;
        private string _modelFile;
        private string _consoleOutputText;
        private string _segmentLength;
        private string _stepLength;
        private string _clipPadding;
        private string _pulsePadding;
        private string _nfft;
        private string _noOverlap;
        private string _batchSize;
        private string _epochs;
        private string _imgDpi;
        private double _progress;
        private Visibility _progressVisible;
        private double _progress2;
        private Visibility _progress2Visible;
        private string _predictThreshold;
        private double _taskCount;
        private double _taskCompletedCount;
        private double _fileCount;
        private double _fileCompletedCount;
        private TimeSpan _progressEta;
        private TimeSpan _progressEta2;
        private DateTime _startTime;
        private DateTime _startTime2;
        private string _progress1Text;
        private string _progress2Text;
        private string _dataSize;
        private string _partialLength;

        public string MateBookFolder
        {
            get => _mateBookFolder;
            set => SetProperty(ref _mateBookFolder, value);
        }

        public string AudioSourceFolder
        {
            get => _audioSourceFolder;
            set => SetProperty(ref _audioSourceFolder, value);
        }

        public string OutputFolder
        {
            get => _outputFolder;
            set => SetProperty(ref _outputFolder, value);
        }

        public string ModelFile
        {
            get => _modelFile;
            set => SetProperty(ref _modelFile, value);
        }

        public string ConsoleOutputText
        {
            get => _consoleOutputText;
            set => SetProperty(ref _consoleOutputText, value);
        }

        public string SegmentLength
        {
            get => _segmentLength;
            set => SetProperty(ref _segmentLength, value);
        }

        public string StepLength
        {
            get => _stepLength;
            set => SetProperty(ref _stepLength, value);
        }

        public string ClipPadding
        {
            get => _clipPadding;
            set => SetProperty(ref _clipPadding, value);
        }

        public string PulsePadding
        {
            get => _pulsePadding;
            set => SetProperty(ref _pulsePadding, value);
        }

        public string Nfft
        {
            get => _nfft;
            set => SetProperty(ref _nfft, value);
        }

        public string NoOverlap
        {
            get => _noOverlap;
            set => SetProperty(ref _noOverlap, value);
        }

        public string ImgDpi
        {
            get => _imgDpi;
            set => SetProperty(ref _imgDpi, value);
        }

        public string PartialLength
        {
            get => _partialLength;
            set => SetProperty(ref _partialLength, value);
        }

        public string Epochs
        {
            get => _epochs;
            set => SetProperty(ref _epochs, value);
        }

        public string BatchSize
        {
            get => _batchSize;
            set => SetProperty(ref _batchSize, value);
        }

        public string PredictThreshold
        {
            get => _predictThreshold;
            set => SetProperty(ref _predictThreshold, value);
        }

        public double Progress
        {
            get => _progress;
            set
            {
                if (value > 0.01)
                {
                    ProgressVisible = Visibility.Visible;
                }
                else
                {
                    ProgressVisible = Visibility.Hidden;
                }
                SetProperty(ref _progress, value);
            }
        }

        public double Progress2
        {
            get => _progress2;
            set
            {
                if (value > 0.01)
                {
                    Progress2Visible = Visibility.Visible;
                }
                else
                {
                    Progress2Visible = Visibility.Hidden;
                }
                SetProperty(ref _progress2, value);
            }
        }

        public Visibility ProgressVisible
        {
            get => _progressVisible;
            set => SetProperty(ref _progressVisible, value);
        }

        public Visibility Progress2Visible
        {
            get => _progress2Visible;
            set => SetProperty(ref _progress2Visible, value);
        }

        public double TaskCount
        {
            get => _taskCount;
            set => SetProperty(ref _taskCount, value);
        }

        public double TaskCompletedCount
        {
            get => _taskCompletedCount;
            set => SetProperty(ref _taskCompletedCount, value);
        }

        public double FileCount
        {
            get => _fileCount;
            set => SetProperty(ref _fileCount, value);
        }

        public double FileCompletedCount
        {
            get => _fileCompletedCount;
            set => SetProperty(ref _fileCompletedCount, value);
        }

        public TimeSpan ProgressEta
        {
            get => _progressEta;
            set => SetProperty(ref _progressEta, value);
        }

        public TimeSpan ProgressEta2
        {
            get => _progressEta2;
            set => SetProperty(ref _progressEta2, value);
        }

        public DateTime StartTime
        {
            get => _startTime;
            set => SetProperty(ref _startTime, value);
        }

        public DateTime StartTime2
        {
            get => _startTime2;
            set => SetProperty(ref _startTime2, value);
        }

        public string Progress1Text
        {
            get => _progress1Text;
            set => SetProperty(ref _progress1Text, value);
        }

        public string Progress2Text
        {
            get => _progress2Text;
            set => SetProperty(ref _progress2Text, value);
        }

        public string DataSize
        {
            get => _dataSize;
            set => SetProperty(ref _dataSize, value);
        }
    }
}
