using System;
using System.Collections.Generic;
using System.Text;

namespace AvP_Song_Annotations.Models
{
    class ConfigModel
    {
        private string _wavFolderValue;
        private string _dataFolderValue;
        private string _outputFolderValue;
        private int _segmentLengthValue;
        private int _stepLengthValue;
        private int _clipPaddingValue;
        private double _pulsePaddingValue;
        private int _imgNFftValue;
        private int _imgNoOverlapValue;
        private int _imgDpiValue;
        private int _partialLengthValue;
        private int _runTrainingValue;
        private int _runExtractSpectogramValue;
        private int _runPredictionsValue;
        private int _trainEpochsValue;
        private int _trainBatchSizeValue;
        private double _predictThresholdValue;
        private string _weightsfileValue;

        public const string WavFolderKey = "wav_folder=";
        public const string DataFolderKey = "data_folder=";
        public const string OutputFolderKey = "output_folder=";
        public const string SegmentLengthKey = "segment_length=";
        public const string StepLengthKey = "step_length=";
        public const string ClipPaddingKey = "clip_padding=";
        public const string PulsePaddingKey = "pulse_padding=";
        public const string ImgNFftKey = "img_n_fft=";
        public const string ImgDpiKey = "img_dpi=";
        public const string PartialLengthKey = "partial_length=";
        public const string ImgNoOverlapKey = "img_no_overlap=";
        public const string RunTrainingKey = "run_training=";
        public const string RunExtractSpectogramsKey = "run_extract_spectograms=";
        public const string RunPredictionsKey = "run_predictions=";
        public const string TrainEpochsKey = "train_epochs=";
        public const string TrainBatchSizeKey = "train_batch_size=";
        public const string PredictThresholdKey = "predict_threshold=";
        public const string WeightsFileKey = "weights_file=";
        public const string ZmqPortKey = "zmq_port=";

        public ConfigModel()
        {
            _segmentLengthValue = 250;
            _stepLengthValue = 25;
            _clipPaddingValue = 50;
            _pulsePaddingValue = 0.25;
            _imgNFftValue = 128;
            _imgNoOverlapValue = 116;
            _imgDpiValue = 250;
            _partialLengthValue = 30;
            _runTrainingValue = 0;
            _runExtractSpectogramValue = 1;
            _runPredictionsValue = 1;
            _trainEpochsValue = 50;
            _trainBatchSizeValue = 16;
            _predictThresholdValue = 0.95;
        }

        public string WavFolderValue
        {
            get => _wavFolderValue;
            set => _wavFolderValue = value;
        }

        public string DataFolderValue
        {
            get => _dataFolderValue;
            set => _dataFolderValue = value;
        }

        public string OutputFolderValue
        {
            get => _outputFolderValue;
            set => _outputFolderValue = value;
        }

        public string WeightsFileValue
        {
            get => _weightsfileValue;
            set => _weightsfileValue = value;
        }

        public int SegmentLengthValue
        {
            get => _segmentLengthValue;
            set => _segmentLengthValue = value;
        }

        public int StepLengthValue
        {
            get => _stepLengthValue;
            set => _stepLengthValue = value;
        }

        public int ClipPaddingValue
        {
            get => _clipPaddingValue;
            set => _clipPaddingValue = value;
        }

        public double PulsePaddingValue
        {
            get => _pulsePaddingValue;
            set => _pulsePaddingValue = value;
        }

        public int ImgNFftValue
        {
            get => _imgNFftValue;
            set => _imgNFftValue = value;
        }

        public int ImgNoOverlapValue
        {
            get => _imgNoOverlapValue;
            set => _imgNoOverlapValue = value;
        }

        public int ImgDpiValue
        {
            get => _imgDpiValue;
            set => _imgDpiValue = value;
        }

        public int PartialLengthValue
        {
            get => _partialLengthValue;
            set => _partialLengthValue = value;
        }

        public int RunTrainingValue
        {
            get => _runTrainingValue;
            set => _runTrainingValue = value;
        }

        public int RunExtractSpectogramValue
        {
            get => _runExtractSpectogramValue;
            set => _runExtractSpectogramValue = value;
        }

        public int RunPredictionsValue
        {
            get => _runPredictionsValue;
            set => _runPredictionsValue = value;
        }

        public int TrainEpochsValue
        {
            get => _trainEpochsValue;
            set => _trainEpochsValue = value;
        }

        public int TrainBatchSizeValue
        {
            get => _trainBatchSizeValue;
            set => _trainBatchSizeValue = value;
        }

        public double PredictThresholdValue
        {
            get => _predictThresholdValue;
            set => _predictThresholdValue = value;
        }

        public string SplitOutputFolder => _outputFolderValue + "Seg_" + _segmentLengthValue + "_Stp_" + _stepLengthValue; // output_folder + 'Seg_' + str(segment_length) + '_Stp_' + str(step_length)

        public string SpecOutputFolder => SplitOutputFolder + @"\n" + _imgNFftValue + "o" + _imgNoOverlapValue; // split_output_folder + '\\n' + str(img_n_fft) + 'o' + str(img_no_overlap)

        public string SpecPosFolder => SpecOutputFolder + @"\pos"; // spec_output_folder + '\\pos'

        public string SpecNegFolder => SpecOutputFolder + @"\neg"; // spec_output_folder + '\\neg'

        public string SaveWeightsPath => OutputFolderValue + @"\Weights\Seg_" + SegmentLengthValue + "_Stp_" + StepLengthValue + @"\n" + ImgNFftValue + "o" + ImgNoOverlapValue; // output_folder + '\\Weights\\Seg_' + str(segment_length) + '_Stp_' + str(step_length) + '\\n' + str(img_n_fft) + 'o' + str(img_no_overlap)

        public string SaveWeightsFile => SaveWeightsPath + @"\song_model.h5"; // save_weights_path + '\\song_model.h5'

        public string ConfigFileName { get; set; }
    }
}
