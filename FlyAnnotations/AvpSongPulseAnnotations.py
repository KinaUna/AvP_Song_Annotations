from AvpAudioTools import AvpAudioTools
import argparse
import warnings
import sys
import os
from datetime import datetime

wav_folder = 'G:\\AVP\\Extract\\prm001'
data_folder = 'G:\\AVP\\Extract\\prm001p.mbp.mbd'
output_folder = 'G:\\AVP\\songdata'
weights_file = 'G:\\repos\\AvP_Song_Annotations\\AvP_Song_Annotations\\AvP_Song_Annotations\\bin\Debug\\netcoreapp3.1\\song_pulse_model.h5'
segment_length = 250
step_length = 1000
clip_padding = 50
pulse_padding = 0.33
img_n_fft = 128
img_no_overlap = 127
img_dpi = 200
partial_length = 30
run_training = 0
run_extract_spectrograms = 0
run_predictions = 1
train_epochs = 50
train_batch_size = 16
debug_mode = 0
predict_threshold = 0.95
zmq_port = 20032
parser = argparse.ArgumentParser(description='AvP Lab Song Pulse Annotations')
parser.add_argument('--config', required=False, type=str, help='The full path of the configuration file')
args = parser.parse_args()

try:
    configuration_file = open(args.config, 'r')
    configuration_lines = configuration_file.readlines()
    configuration_file.close()

    for line in configuration_lines:
        line_items = line.split('=')
        if line_items[0].startswith('wav_folder'):
            wav_folder = line_items[1].rstrip('\n')
        if line_items[0].startswith('data_folder'):
            data_folder = line_items[1].rstrip('\n')
        if line_items[0].startswith('output_folder'):
            output_folder = line_items[1].rstrip('\n')
        if line_items[0].startswith('segment_length'):
            segment_length = int(line_items[1].rstrip('\n'))
        if line_items[0].startswith('step_length'):
            step_length = int(line_items[1].rstrip('\n'))
        if line_items[0].startswith('clip_padding'):
            clip_padding = int(line_items[1].rstrip('\n'))
        if line_items[0].startswith('pulse_padding'):
            pulse_padding = float(line_items[1].rstrip('\n'))
        if line_items[0].startswith('img_n_fft'):
            img_n_fft = int(line_items[1].rstrip('\n'))
        if line_items[0].startswith('img_no_overlap'):
            img_no_overlap = int(line_items[1].rstrip('\n'))
        if line_items[0].startswith('img_dpi'):
            img_dpi = int(line_items[1].rstrip('\n'))
        if line_items[0].startswith('partial_length'):
            partial_length = int(line_items[1].rstrip('\n'))
        if line_items[0].startswith('run_training'):
            run_training = int(line_items[1].rstrip('\n'))
        if line_items[0].startswith('run_extract_spectograms'):
            run_extract_spectrograms = int(line_items[1].rstrip('\n'))
        if line_items[0].startswith('run_predictions'):
            run_predictions = int(line_items[1].rstrip('\n'))
        if line_items[0].startswith('train_epochs'):
            train_epochs = int(line_items[1].rstrip('\n'))
        if line_items[0].startswith('train_batch_size'):
            train_batch_size = int(line_items[1].rstrip('\n'))
        if line_items[0].startswith('debug_mode'):
            debug_mode = int(line_items[1].rstrip('\n'))
        if line_items[0].startswith('predict_threshold'):
            predict_threshold = float(line_items[1].rstrip('\n'))
        if line_items[0].startswith('weights_file'):
            weights_file = line_items[1].rstrip('\n')
        if line_items[0].startswith('zmq_port'):
            zmq_port = int(line_items[1].rstrip('\n'))
except:
    print('No config file loaded, running with default values.')
    sys.stdout.flush()



aat = AvpAudioTools(wav_folder, data_folder, output_folder, segment_length=segment_length, step_length=step_length,
                    clip_padding=clip_padding, pulse_padding=pulse_padding, img_n_fft=img_n_fft,
                    img_no_overlap=img_no_overlap, img_dpi=img_dpi, partial_length=partial_length, weights_file=weights_file, zmq_port=zmq_port)

if run_training == 1:
    if run_extract_spectrograms == 1:
        aat.extract_training_data()
    aat.train_song_pulses(train_epochs, train_batch_size)

if run_predictions == 1:
    aat.run_predictions(predict_threshold)
    