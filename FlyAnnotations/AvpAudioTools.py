import os
import sys
import math
import numpy as np
import librosa
import random
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn import metrics
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD
import zmq


class AvpAudioTools:
    def __init__(self, wav_folder, data_folder, output_folder, segment_length=200, step_length=25, clip_padding=50, pulse_padding=0.33, img_n_fft=128, img_no_overlap=116, img_dpi=200, partial_length=60, weights_file='song_pulse_model.h5', zmq_port=20032):
        self.wav_folder = wav_folder  # String: The folder containing sound files to extract pulses from
        self.data_folder = data_folder  # String: The folder with MateBook data.
        self.output_folder = output_folder  # String: The folder to save spectrogram and temporary data.
        self.segment_length = segment_length  # Int: The length of the sound clip for each spectrogram in number of frames.
        self.step_length = step_length  # Int: How many frames to move the segment window by.
        self.clip_padding = clip_padding  # Int: How many frames at the beginning and end of a segment to exclude when checking if training data contains a pulse.
        self.pulse_padding = pulse_padding  # Float: How big a fraction of the start and end of a pulse interval to exclude when checking if training data contains a pulse.
        self.img_n_fft = img_n_fft  # Int: Length of the Fast Fourier Transform window, in number of frames.
        self.img_no_overlap = img_no_overlap
        self.img_dpi = img_dpi  # The resolution of the spectrogram images.
        self.img_cmap = 'gist_heat'  # The colormap for the spectrogram images.
        self.partial_length = partial_length  # The length, in seconds, of each part when splitting the audio file for predictions.
        self.split_output_folder = output_folder + '\\Seg_' + str(segment_length) + '_Stp_' + str(step_length)  # Output root folder
        self.spec_output_folder = self.split_output_folder + '\\n' + str(img_n_fft) + 'o' + str(img_no_overlap)  # For temporary spectrogram files
        self.spec_pos_folder = self.spec_output_folder + '\\pos'  # Spectrogram data containing song pulses for training
        self.spec_neg_folder = self.spec_output_folder + '\\neg' # Spectrogram data not containing song pulses for training
        self.save_weights_path = output_folder + '\\Weights\\Seg_' + str(segment_length) + '_Stp_' + str(step_length) + '\\n' + str(img_n_fft) + 'o' + str(img_no_overlap)  # Folder to save the trained model parameters.
        self.save_weights_file = self.save_weights_path + '\\song_pulse_model.h5'
        self.weights_file = weights_file  # The weights file to use when predicting song pulses.
        self.zmq_port = zmq_port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.bind('tcp://127.0.0.1:' + str(zmq_port))
        self.model = Sequential()
        # sysout_file = output_folder + '\\logs\\stdout.txt'
        # sys.stdout = open(sysout_file, 'w')
        matplotlib.use('Agg')  # Matplotlib slows down over time in the windows console if this is not applied.

    @staticmethod
    def normalize_gray(array):
        return (array - array.min()) / (array.max() - array.min())

    @staticmethod
    def find_intersection(intervals, segment):
        intersections = 0
        for interval in intervals:
            if interval[1] > segment[0] and interval[1] < segment[1]:
                intersections += 1
            if interval[0] < segment[1] and interval[1] > segment[1]:
                intersections += 1

        return intersections

    @staticmethod
    def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising', kpsh=False, valley=False):
        # Source: https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb
        x = np.atleast_1d(x).astype('float64')
        if x.size < 3:
            return np.array([], dtype=int)
        if valley:
            x = -x
            if mph is not None:
                mph = -mph
        # find indices of all peaks
        dx = x[1:] - x[:-1]
        # handle NaN's
        indnan = np.where(np.isnan(x))[0]
        if indnan.size:
            x[indnan] = np.inf
            dx[np.where(np.isnan(dx))[0]] = np.inf
        ine, ire, ife = np.array([[], [], []], dtype=int)
        if not edge:
            ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
        else:
            if edge.lower() in ['rising', 'both']:
                ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
            if edge.lower() in ['falling', 'both']:
                ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
        ind = np.unique(np.hstack((ine, ire, ife)))
        # handle NaN's
        if ind.size and indnan.size:
            # NaN's and values close to NaN's cannot be peaks
            ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan - 1, indnan + 1))), invert=True)]
        # first and last values of x cannot be peaks
        if ind.size and ind[0] == 0:
            ind = ind[1:]
        if ind.size and ind[-1] == x.size - 1:
            ind = ind[:-1]
        # remove peaks < minimum peak height
        if ind.size and mph is not None:
            ind = ind[x[ind] >= mph]
        # remove peaks - neighbors < threshold
        if ind.size and threshold > 0:
            dx = np.min(np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
            ind = np.delete(ind, np.where(dx < threshold)[0])
        # detect small peaks closer than minimum peak distance
        if ind.size and mpd > 1:
            ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
            idel = np.zeros(ind.size, dtype=bool)
            for i in range(ind.size):
                if not idel[i]:
                    # keep peaks with the same height if kpsh is True
                    idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                           & (x[ind[i]] > x[ind] if kpsh else True)
                    idel[i] = 0  # Keep current peak
            # remove the small peaks and sort back the indices by their occurrence
            ind = np.sort(ind[~idel])

        return ind

    def split_original_wav(self, wav_file, silence_threshold=0.01):
        original_audio_data, sampling_rate = librosa.load(wav_file, sr=None)
        start = 0
        number_of_segments = int(len(original_audio_data) / self.step_length)
        sounds_list = []
        # print('Analyzing audio file...')
        # self.socket.send_string('Analyzing audio file...')
        i = 0
        while i < number_of_segments:
            # print(" Progress: {:2.1%}".format(float(i) / number_of_segments), end="\r")
            percent_done = float(i) / number_of_segments
            # print('@', percent_done)
            # sys.stdout.flush()
            self.socket.send_string('@' + str(percent_done))

            sound_clip = original_audio_data[start: start + self.segment_length]
            clip_len = len(sound_clip)
            peaks = self.detect_peaks(np.abs(sound_clip), mpd=self.step_length)
            if clip_len == self.segment_length and abs(sound_clip[peaks[0]]) > silence_threshold:
                sounds_list.append([sound_clip, [start, start + self.step_length]])
            start = start + self.step_length
            i += 1
        return sounds_list

    def extract_start_end_data(self, sound_annotations_file):
        sa_file = open(sound_annotations_file, 'r')
        annotations_data_lines = sa_file.readlines()
        sa_file.close()
        content = [x.strip() for x in annotations_data_lines]
        annotation_start_end_list = []
        annotation_not_sure_list = []  # to ensure that partial pulses are not included in the negative list.
        for line in content:
            frame_data = line.split()
            if frame_data[0].isdigit() and frame_data[1].isdigit():
                start_end_diff = int((int(frame_data[1]) - int(frame_data[0])) * self.pulse_padding)
                annotation_start_end_list.append(
                    [int(frame_data[0]) + start_end_diff, int(frame_data[1]) - start_end_diff])
                annotation_not_sure_list.append([int(frame_data[0]), int(frame_data[1])])

        return annotation_start_end_list, annotation_not_sure_list

    @staticmethod
    def extract_pulse_intervals(file):
        csp_file = open(file, 'r')
        clean_data_lines = csp_file.readlines()
        content = [x.strip() for x in clean_data_lines]
        song_start_end_list = []
        for line in content:
            s = line.split()
            if s[0].isdigit() and s[1].isdigit():
                song_start_end_list.append([int(s[0]), int(s[1])])

        return song_start_end_list

    def graph_spectrogram(self, data, sample_rate=10000):
        fig, ax = plt.subplots(1, dpi=self.img_dpi)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.ioff()
        ax.axis('off')
        pxx, freqs, bins, im = ax.specgram(x=data, Fs=sample_rate, noverlap=self.img_no_overlap, NFFT=self.img_n_fft, cmap=self.img_cmap)
        ax.axis('off')
        plt.rcParams['figure.figsize'] = [1.0, 0.75]
        fig.canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        matplot_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        np_image_array = np.reshape(matplot_image, (int(height), int(width), 3))
        plt.close('all')

        return np_image_array

    def save_image_data(self, audio_data, from_frame, to_frame, filename, segment_length, step_length):
        self.socket.send_string(']Current segment')
        start = 0
        number_of_segments = int(len(audio_data) / step_length)
        sound_image_list = []
        i = 0

        while i < number_of_segments:
            # print("\r Audio data extraction progress: {:2.1%}".format(float(i) / number_of_segments), end="\r")
            percent_done = float(i) / number_of_segments
            if (percent_done * 100000).is_integer():
                self.socket.send_string('@' + str(percent_done))
            #     print('@', percent_done, '\n')
            #     sys.stdout.flush()

            sound_clip = audio_data[start: start + segment_length]
            if len(sound_clip) == segment_length:
                img_data = self.graph_spectrogram(sound_clip)
                if img_data.shape[0] < self.img_dpi:
                    image_array = np.dot(img_data, [0.2989, 0.5870, 0.1140])
                    image_array = self.normalize_gray(image_array)
                    image_array = image_array.reshape(image_array.shape[0], image_array.shape[1], 1)
                    sound_image_list.append(image_array)

            start = start + step_length
            i += 1

        sound_image_list = np.array(sound_image_list)

        save_folder = self.split_output_folder + '\\' + filename
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        np_save_file = save_folder + '\\' + str(from_frame) + '_' + str(to_frame) + '.npy'
        np.save(np_save_file, sound_image_list)

        # print('@', 100.0, '\n')
        # sys.stdout.flush()
        self.socket.send_string('@' + str(0.0))

    def generate_image_data(self, audio_data, segment_length, step_length):
        self.socket.send_string(']Current segment')
        start = 0
        number_of_segments = int(len(audio_data) / step_length)
        sound_image_list = []
        i = 0

        while i < number_of_segments:
            percent_done = float(i) / number_of_segments
            if (percent_done * 100000).is_integer():
                self.socket.send_string('@' + str(percent_done))

            sound_segment = audio_data[start: start + segment_length]
            if len(sound_segment) == segment_length:
                img_data = self.graph_spectrogram(sound_segment)
                if img_data.shape[0] < self.img_dpi:
                    image_array = np.dot(img_data, [0.2989, 0.5870, 0.1140])
                    image_array = self.normalize_gray(image_array)
                    image_array = image_array.reshape(image_array.shape[0], image_array.shape[1], 1)
                    sound_image_list.append(image_array)

            start = start + step_length
            i += 1

        sound_image_list = np.array(sound_image_list)

        return sound_image_list

    @staticmethod
    def merge_time_ranges(intervals):
        intervals.sort(key=lambda x: x[0])

        merged = []
        for interval in intervals:
            # if the list of merged intervals is empty or if the current
            # interval does not overlap with the previous, simply append it.
            if not merged or merged[-1][1] < interval[0]:
                merged.append(interval)
            else:
                # otherwise, there is overlap, so we merge the current and previous
                # intervals.
                merged[-1][1] = max(merged[-1][1], interval[1])

        return merged

    def load_training_data(self):
        file_count = 0
        data_list_pos = []

        # print('Loading training data...')
        self.socket.send_string('Loading training data...')
        self.socket.send_string(']Loading training data')
        files_len = len(os.listdir(self.spec_pos_folder)) + len(os.listdir(self.spec_neg_folder))
        for entry in os.listdir(self.spec_pos_folder):
            pickle_file = os.path.join(self.spec_pos_folder, entry)
            if os.path.isfile(pickle_file) and entry.endswith('.npy'):
                try:
                    file_content = np.load(pickle_file)
                    file_count += 1
                    data_list_pos.append(file_content)
                    # print('@', file_count / files_len)
                    # sys.stdout.flush()
                    self.socket.send_string('@' + str(file_count / files_len))
                except PermissionError:
                    print('Permission error loading file ', file_count, '. File name: ', entry)
                    sys.stdout.flush()
        x_pos = np.concatenate(data_list_pos, axis=0)
        del data_list_pos

        data_list_neg = []
        for entry in os.listdir(self.spec_neg_folder):
            pickle_file = os.path.join(self.spec_neg_folder, entry)
            if os.path.isfile(pickle_file) and entry.endswith('.npy'):
                try:
                    file_content = np.load(pickle_file)
                    file_count += 1
                    data_list_neg.append(file_content)
                    # print('@', file_count / files_len)
                    # sys.stdout.flush()
                    self.socket.send_string('@' + str(file_count / files_len))
                except PermissionError:
                    print('Permission error loading file ', file_count, '. File name: ', entry)
                    sys.stdout.flush()
        x_neg = np.concatenate(data_list_neg, axis=0)
        y_pos = np.ones(len(x_pos))
        y_neg = np.zeros(len(x_neg))
        x_all = np.concatenate((x_pos, x_neg), axis=0)
        del x_pos
        del x_neg
        y_all = np.concatenate((y_pos, y_neg), axis=0)
        del y_pos
        del y_neg

        # print('Spectrogram training data loaded.')
        # sys.stdout.flush()
        self.socket.send_string('Spectrogram training data loaded.')
        # print('Number of spectrograms: ', len(y_all))
        # sys.stdout.flush()
        self.socket.send_string('Spectrogram count: ' + str(len(y_all)))

        y_cat_train = y_all.astype(int)

        x_train, x_test, y_train, y_test = train_test_split(x_all, y_cat_train, test_size=0.2, shuffle=True)
        del x_all
        del y_all
        del y_cat_train
        self.socket.send_string('@' + str(0.0))

        return x_train, x_test, y_train, y_test

    def extract_training_data(self, number_of_files=0, start_number=0):
        self.socket.send_string('}Extracting training data')
        file_count = 1
        clean_song_pulses_files = []
        for root, dirs, files in os.walk(self.data_folder):
            for file in files:
                if file == 'clean_songpulses.txt':
                    file_full_path = os.path.join(root, file)
                    clean_song_pulses_files.append(file_full_path)
        for entry in os.listdir(self.data_folder):
            if os.path.isfile(os.path.join(self.data_folder, entry)) and (
                    file_count <= number_of_files or number_of_files == 0) and file_count > start_number:
                if entry.endswith('.wav'):
                    wav_full_path = os.path.join(self.data_folder, entry)
                    for clean_file in clean_song_pulses_files:
                        if wav_full_path in clean_file:
                            # print(str(file_count) + ' - Adding data for: ' + entry)
                            # sys.stdout.flush()
                            self.socket.send_string(str(file_count) + ' - Adding data for: ' + entry)
                            self.socket.send_string('!' + str(0) + ':' + str(2) + ':' + str(file_count - 1) + ':' + str(len(clean_song_pulses_files)))
                            pulses, maybe_pulses = self.extract_start_end_data(clean_file)
                            all_clips = self.split_original_wav(wav_full_path)
                            current_audio_data_pos = []
                            current_audio_data_neg = []
                            pulse_count = 0
                            not_pulse_count = 0
                            clips_processed = 0
                            self.socket.send_string(']Analyzing training data')
                            for clip in all_clips:
                                clips_processed += 1
                                if len(clip[0]) == self.segment_length:
                                    clip_trimmed = [clip[1][0] + self.clip_padding, clip[1][1] - self.clip_padding]  # Exclude clips where start or end only contains part of the pulse
                                    is_pulse = self.find_intersection(pulses, clip_trimmed) > 0
                                    sound_file_data = [clip[0], is_pulse, entry, clip[1]]
                                    if is_pulse:
                                        current_audio_data_pos.append(sound_file_data)
                                        pulse_count += 1
                                    might_be_pulse = self.find_intersection(maybe_pulses, [clip[1][0], clip[1][1]]) > 0
                                    if not is_pulse and not might_be_pulse:
                                        current_audio_data_neg.append(sound_file_data)
                                        not_pulse_count += 1
                                self.socket.send_string('@' + str(clips_processed / len(all_clips)))

                            del all_clips
                            self.socket.send_string('!' + str(1) + ':' + str(2) + ':' + str(file_count - 1) + ':' + str(len(clean_song_pulses_files)))
                            if pulse_count > 0:
                                pos_list = []
                                neg_list = []
                                # random.seed(42)
                                random.shuffle(current_audio_data_neg)
                                current_audio_data_neg = current_audio_data_neg[:len(current_audio_data_pos) * 2]
                                current_audio_data = current_audio_data_pos + current_audio_data_neg
                                del current_audio_data_pos
                                del current_audio_data_neg
                                audio_item_count = 0
                                self.socket.send_string(']Creating spectrograms for training data')
                                for audio_item in current_audio_data:
                                    audio_item_count += 1
                                    spectrogram = self.graph_spectrogram(audio_item[0])
                                    if spectrogram.shape[0] < self.img_dpi:
                                        image_array = np.dot(spectrogram, [0.2989, 0.5870, 0.1140])
                                        image_array = self.normalize_gray(image_array)
                                        image_array = image_array.reshape(image_array.shape[0], image_array.shape[1], 1)
                                        if audio_item[1]:
                                            pos_list.append(image_array)
                                        else:
                                            neg_list.append(image_array)
                                    self.socket.send_string('@' + str(audio_item_count / len(current_audio_data)))

                                pos_all = np.array(pos_list)
                                del pos_list
                                neg_all = np.array(neg_list)
                                del neg_list
                                if not os.path.exists(self.spec_pos_folder):
                                    os.makedirs(self.spec_pos_folder)
                                if not os.path.exists(self.spec_neg_folder):
                                    os.makedirs(self.spec_neg_folder)
                                np_save_pos_file = self.spec_pos_folder + '\\' + entry.replace('.wav', '.npy')
                                np.save(np_save_pos_file, pos_all)

                                np_save_neg_file = self.spec_neg_folder + '\\' + entry.replace('.wav', '.npy')
                                np.save(np_save_neg_file, neg_all)
                                # print(file_count, ' - Pulse samples added: ', pulse_count, ' , No pulse samples added: ', not_pulse_count)
                                # sys.stdout.flush()
                                self.socket.send_string(str(file_count) + ' - Pulse samples added: ' + str(pulse_count))
                            self.socket.send_string('!' + str(2) + ':' + str(2) + ':' + str(file_count - 1) + ':' + str(len(clean_song_pulses_files)))
                            file_count = file_count + 1
        self.socket.send_string('!' + str(2) + ':' + str(2) + ':' + str(file_count) + ':' + str(len(clean_song_pulses_files)))
        return file_count

    def generate_predictions(self, wav_file, file_count=1, file_total=1, estimate_threshold=0.9, thread_id=0):
        for entryX in os.listdir(self.wav_folder):
            if os.path.isfile(os.path.join(self.wav_folder, entryX)):
                if entryX == wav_file:
                    for entryY in os.listdir(self.data_folder):
                        if os.path.isdir(os.path.join(self.data_folder, entryY)):
                            if entryY.startswith(entryX):
                                wav_full_path = os.path.join(self.wav_folder, entryX)
                                entry_time = datetime.now()
                                self.socket.send_string(str(entry_time) + ' Generating spectrogram images from file: ' + wav_full_path)
                                original_audio_data, sampling_rate = librosa.load(wav_full_path, sr=None)
                                wav_length = len(original_audio_data)
                                seg_start = 0
                                seg_num = 1
                                seg_count = math.ceil(wav_length / (self.partial_length * sampling_rate))
                                clean_songpulses = []
                                while seg_start < wav_length:
                                    self.socket.send_string('!' + str(seg_num - 1) + ':' + str(seg_count) + ':' + str(file_count - 1) + ':' + str(file_total))
                                    seg_start_time = datetime.now()
                                    seg_end = seg_start + self.partial_length * sampling_rate
                                    self.socket.send_string('File ' + str(file_count) + ' of ' + str(file_total))
                                    self.socket.send_string('Segment ' + str(seg_num) + ' of ' + str(seg_count) + ' - Analyzing frames: ' + str(seg_start / 1000) + 'k to ' + str(seg_end / 1000) + 'k of ' + str(wav_length / 1000) + 'k')
                                    seg_img_data = self.generate_image_data(original_audio_data[seg_start:seg_end], self.segment_length, self.step_length)
                                    if seg_img_data[0].shape != seg_img_data[1].shape:
                                        seg_img_data = np.delete(seg_img_data, 0)
                                    with tf.device('/cpu:0'):
                                        predictions = self.model.predict(seg_img_data)
                                    tf.keras.backend.clear_session()  # Memory leak? https://github.com/keras-team/keras/issues/13118
                                    pred_count = 1
                                    self.socket.send_string(']Filtering pulses')
                                    for pred in predictions:
                                        self.socket.send_string('@' + str(pred_count / len(predictions)))
                                        if pred[0] > estimate_threshold:
                                            framecenter = pred_count * self.step_length + int(self.segment_length / 2) + seg_start
                                            framestart = framecenter - int(self.step_length / 2 + 1)
                                            frameend = framecenter + int(self.step_length / 2 + 1)
                                            clean_songpulses.append([framestart, frameend])
                                        pred_count += 1
                                    self.socket.send_string('@' + str(0.0))
                                    seg_duration = datetime.now() - seg_start_time
                                    self.socket.send_string(
                                        'Segment ' + str(seg_num) + ' analyzed in ' + str(seg_duration))
                                    seg_start = seg_end
                                    seg_num += 1
                                    self.socket.send_string('!' + str(seg_num - 1) + ':' + str(seg_count) + ':' + str(file_count - 1) + ':' + str(file_total))

                                clean_pulses_final = self.merge_time_ranges(clean_songpulses)
                                clean_text = ''
                                for frames in clean_pulses_final:
                                    clean_text = clean_text + str(frames[0]) + ' ' + str(frames[1]) + '\n'

                                folder = self.output_folder + '\\' + entryX
                                if not os.path.exists(folder):
                                    os.makedirs(folder)
                                text_file_name = os.path.join(folder, 'song_predictions.txt')
                                text_file = open(text_file_name, "w")
                                text_file.write(clean_text)
                                text_file.close()

                                matebook_folder = self.data_folder
                                mb_data_folder = ''
                                for mb_root, mb_dirs, mb_files in os.walk(matebook_folder):
                                    for mb_dir in mb_dirs:
                                        if mb_dir.startswith(entryX):
                                            mb_data_folder = os.path.join(mb_root, mb_dir)

                                clean_songpulses_file = mb_data_folder + '\\clean_songpulses.txt'
                                songpulses_file = mb_data_folder + '\\songpulses.txt'
                                cleansongpulsecenters_file = mb_data_folder + '\\clean_songpulsecenters.txt'
                                songpulsecenters_file = mb_data_folder + '\\songpulsecenters.txt'

                                clean_songpulse_list = self.extract_pulse_intervals(text_file_name)
                                if len(clean_songpulse_list) > 0:
                                    audio_file = os.path.join(self.wav_folder, entryX)
                                    clip, sr = librosa.load(audio_file, sr=None)

                                    songpulse_centers_list = []
                                    for data_point in clean_songpulse_list:
                                        temp_clip = clip[data_point[0]:data_point[1]]
                                        clip_len = len(temp_clip)
                                        peaks = self.detect_peaks(np.abs(temp_clip), mpd=clip_len)
                                        if abs(temp_clip[peaks[0]]) > 0.02:
                                            songpulse_centers_list.append([data_point[0], data_point[0] + peaks[0]])

                                    centers_text = ''
                                    for frames in songpulse_centers_list:
                                        centers_text = centers_text + str(frames[0]) + '\t' + str(frames[1]) + '\n'

                                    centers_text_file = open(songpulsecenters_file, 'w')
                                    centers_text_file.write(centers_text)
                                    centers_text_file.close()

                                    cleancenters_text_file = open(cleansongpulsecenters_file, 'w')
                                    cleancenters_text_file.write(centers_text)
                                    cleancenters_text_file.close()

                                    songpulse_text = ''
                                    num_pulses = 0
                                    for frames in clean_songpulse_list:
                                        temp_clip = clip[frames[0]:frames[1]]
                                        clip_len = len(temp_clip)
                                        peaks = self.detect_peaks(np.abs(temp_clip), mpd=clip_len)
                                        # print(peaks.shape)
                                        if abs(temp_clip[peaks[0]]) > 0.02:
                                            songpulse_text = songpulse_text + str(frames[0]) + '\t' + str(
                                                frames[1]) + '\n'
                                            num_pulses += 1

                                    self.socket.send_string('Song pulses detected in ' + entryX + ': ' + str(num_pulses))

                                    pulse_text_file = open(songpulses_file, 'w')
                                    pulse_text_file.write(songpulse_text)
                                    pulse_text_file.close()

                                    clean_text_file = open(clean_songpulses_file, 'w')
                                    clean_text_file.write(songpulse_text)
                                    clean_text_file.close()
                                else:
                                    self.socket.send_string('No song pulses detected in ' + entryX + '. ')
                                    centers_text_file = open(songpulsecenters_file, 'w')
                                    centers_text_file.write('')
                                    centers_text_file.close()

                                    cleancenters_text_file = open(cleansongpulsecenters_file, 'w')
                                    cleancenters_text_file.write('')
                                    cleancenters_text_file.close()

                                    pulse_text_file = open(songpulses_file, 'w')
                                    pulse_text_file.write('')
                                    pulse_text_file.close()

                                    clean_text_file = open(clean_songpulses_file, 'w')
                                    clean_text_file.write('')
                                    clean_text_file.close()

    def run_predictions(self, threshold=0.9):
        self.socket.send_string('Loading machine learning model:' + self.weights_file)
        with tf.device('/cpu:0'):
            self.model = tf.keras.models.load_model(self.weights_file, compile=False)
        self.model.summary()
        sys.stdout.flush()

        filecount = 0
        file_total = 0
        file_list = []
        self.socket.send_string('}Total')
        for entryX in os.listdir(self.wav_folder):
            if os.path.isfile(os.path.join(self.wav_folder, entryX)):
                if entryX.endswith('.wav'):
                    for entryY in os.listdir(self.data_folder):
                        if os.path.isdir(os.path.join(self.data_folder, entryY)):
                            if entryY.startswith(entryX):
                                file_list.append(entryX)
                                file_total += 1

        for predfile in file_list:
            filecount += 1
            self.generate_predictions(predfile, filecount, file_total, threshold)

    def extract_spectographs(self):
        filecount = 0
        file_total = 0

        for entryX in os.listdir(self.wav_folder):
            if os.path.isfile(os.path.join(self.wav_folder, entryX)):
                if entryX.endswith('.wav'):
                    for entryY in os.listdir(self.data_folder):
                        if os.path.isdir(os.path.join(self.data_folder, entryY)):
                            if entryY.startswith(entryX):
                                file_total += 1
        self.socket.send_string('}Creating spectrograms from wav files')
        for entry in os.listdir(self.wav_folder):
            if os.path.isfile(os.path.join(self.wav_folder, entry)):
                if entry.endswith('.wav'):
                    for entry2 in os.listdir(self.data_folder):
                        if os.path.isdir(os.path.join(self.data_folder, entry2)):
                            if entry2.startswith(entry):
                                filecount += 1
                                wav_full_path = os.path.join(self.wav_folder, entry)
                                entry_time = datetime.now()
                                # print(entry_time, ' Generating spectograms from file: ', wav_full_path)
                                # sys.stdout.flush()
                                self.socket.send_string(str(entry_time) + ' Generating spectrogram images from file: ' + wav_full_path)
                                original_audio_data, sampling_rate = librosa.load(wav_full_path, sr=None)
                                wav_length = len(original_audio_data)
                                seg_start = 0
                                seg_num = 1
                                seg_count = math.ceil(wav_length / (self.partial_length * sampling_rate))
                                while seg_start < wav_length:
                                    # print('!', seg_num, ':', seg_count, ':', filecount, ':', file_total)
                                    # sys.stdout.flush()
                                    self.socket.send_string('!' + str(seg_num - 1) + ':' + str(seg_count) + ':' + str(filecount - 1) + ':' + str(file_total))
                                    seg_start_time = datetime.now()
                                    seg_end = seg_start + self.partial_length * sampling_rate
                                    # print('Segment ', seg_num, 'of ', seg_count, ' - Analyzing frames: ', seg_start, ' to ', seg_end, ' of ', wav_length)
                                    # sys.stdout.flush()
                                    self.socket.send_string('File ' + str(filecount) + ' of ' + str(file_total))
                                    self.socket.send_string('Segment ' + str(seg_num) + ' of ' + str(seg_count) + ' - Analyzing frames: ' + str(seg_start/1000) + 'k to ' + str(seg_end/1000) + 'k of ' + str(wav_length/1000) + 'k')
                                    self.save_image_data(original_audio_data[seg_start:seg_end], seg_start, seg_end, entry, self.segment_length, self.step_length)
                                    seg_duration = datetime.now() - seg_start_time
                                    # print('Segment ', seg_num, ' analyzed in ', seg_duration)
                                    # sys.stdout.flush()
                                    self.socket.send_string('Segment ' + str(seg_num) + ' analyzed in ' + str(seg_duration))
                                    seg_start = seg_end
                                    seg_num += 1
                                    # print('!', seg_num, ':', seg_count, ':', filecount, ':', file_total)
                                    # sys.stdout.flush()
                                    self.socket.send_string('!' + str(seg_num - 1) + ':' + str(seg_count) + ':' + str(filecount - 1) + ':' + str(file_total))

        self.socket.send_string('!' + str(0.0) + ':' + str(0.0) + ':' + str(0.0) + ':' + str(0.0))

    def train_song_pulses(self, train_num_epochs=25, train_batch_size=32):
        # print('Initializing machine learning model.')
        # sys.stdout.flush()
        self.socket.send_string('!' + str(0.0) + ':' + str(0.0) + ':' + str(0.0) + ':' + str(0.0))
        self.socket.send_string('Initializing machine learning model')

        if not os.path.exists(self.save_weights_path):
            os.makedirs(self.save_weights_path)

        x_train, x_test, y_train, y_test = self.load_training_data()
        in_shape = x_train[0].shape

        model = Sequential()

        model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), data_format='channels_last', input_shape=in_shape))
        model.add(LeakyReLU())
        model.add(Dropout(0.01))

        model.add(Conv2D(filters=32, kernel_size=(6, 6), strides=(2, 2)))
        model.add(LeakyReLU())
        model.add(Dropout(0.01))

        model.add(Conv2D(filters=32, kernel_size=(16, 16), strides=(8, 8)))
        model.add(LeakyReLU())
        model.add(Dropout(0.01))

        model.add(Flatten())
        model.add(Dropout(0.01))

        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.01))

        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.25))

        model.add(Dense(1, activation='sigmoid'))

        opt = SGD(lr=0.0001, decay=1e-7, momentum=0.6, nesterov=True)

        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        model.summary()
        sys.stdout.flush()
        # print('Starting model training...')
        # sys.stdout.flush()
        self.socket.send_string('Starting model training...')
        self.socket.send_string(']Training ML model')
        self.socket.send_string('}Training ML model')
        history = model.fit(x_train, y_train, epochs=train_num_epochs, batch_size=train_batch_size, validation_data=(x_test, y_test), verbose=2)
        model.save(self.save_weights_file, include_optimizer=False)  # include optimizer causing memory leak when looping over predictions? https://github.com/keras-team/keras/issues/13118
        settings_text = 'segment_length=' + str(self.segment_length)
        settings_text = settings_text + '\nstep_length=' + str(self.step_length)
        settings_text = settings_text + '\nclip_padding=' + str(self.clip_padding)
        settings_text = settings_text + '\npulse_padding=' + str(self.pulse_padding)
        settings_text = settings_text + '\nimg_n_fft=' + str(self.img_n_fft)
        settings_text = settings_text + '\nimg_no_overlap=' + str(self.img_no_overlap)
        settings_text = settings_text + '\nimg_dpi=' + str(self.img_dpi)
        save_settings_file_name = self.save_weights_file.replace('.h5', '.txt')
        text_file = open(save_settings_file_name, "w")
        text_file.write(settings_text)
        text_file.close()
        # print('Model training completed.')
        # sys.stdout.flush()
        self.socket.send_string('Model training completed.')
        # print('ML model saved to: ', self.save_weights_file)
        # sys.stdout.flush()
        self.socket.send_string('ML model saved to: ' + self.save_weights_file)

    def predict_song_pulses(self, estimate_threshold=0.99):
        # print('Loading machine learning model: ', self.save_weights_file)
        # sys.stdout.flush()
        self.socket.send_string('Loading machine learning model:' + self.weights_file)
        model = tf.keras.models.load_model(self.weights_file)
        model.summary()
        sys.stdout.flush()
        dirs_processed = 0
        self.socket.send_string('}Processing spectrogram data')
        for root, dirs, files in os.walk(self.split_output_folder):
            num_dirs = len(dirs)
            for directory in dirs:
                if directory.endswith('.wav'):
                    dirs_processed += 1
                    # print('Detecting song pulses for ', directory)
                    # sys.stdout.flush()
                    self.socket.send_string('\n\nDetecting song pulses for ' + directory)
                    clean_songpulses = []
                    folder = os.path.join(root, directory)
                    num_files = len(os.listdir(folder))
                    files_processed = 0
                    for file in os.listdir(folder):
                        files_processed += 1
                        self.socket.send_string('!' + str(files_processed - 1) + ':' + str(num_files) + ':' + str(dirs_processed - 1) + ':' + str(num_dirs))
                        if file.endswith('.npy'):
                            frame_indices = file.split('_')
                            first_frame = int(frame_indices[0])
                            # last_frame = int(frame_indices[1].replace('.npy', ''))
                            file_full_path = os.path.join(folder, file)
                            validation_arr = np.load(file_full_path)
                            if validation_arr[0].shape != validation_arr[1].shape:
                                validation_arr = np.delete(validation_arr, 0)
                            self.socket.send_string('!' + str(files_processed - 1) + ':' + str(num_files) + ':' + str(dirs_processed - 1) + ':' + str(num_dirs))
                            predictions = model.predict(validation_arr)
                            del validation_arr
                            tf.keras.backend.clear_session()  # Memory leak? https://github.com/keras-team/keras/issues/13118
                            pred_count = 1
                            self.socket.send_string(']Filtering pulses')
                            for pred in predictions:
                                self.socket.send_string('@' + str(pred_count / len(predictions)))
                                if pred[0] > estimate_threshold:
                                    framecenter = pred_count * self.step_length + int(self.segment_length / 2) + first_frame
                                    framestart = framecenter - int(self.step_length / 2 + 1)
                                    frameend = framecenter + int(self.step_length / 2 + 1)
                                    clean_songpulses.append([framestart, frameend])
                                pred_count += 1
                            self.socket.send_string('@' + str(0.0))
                            del predictions
                    clean_pulses_final = self.merge_time_ranges(clean_songpulses)
                    clean_text = ''
                    for frames in clean_pulses_final:
                        clean_text = clean_text + str(frames[0]) + ' ' + str(frames[1]) + '\n'

                    text_file_name = os.path.join(folder, 'song_predictions.txt')
                    text_file = open(text_file_name, "w")
                    text_file.write(clean_text)
                    text_file.close()

                    matebook_folder = self.data_folder
                    mb_data_folder = ''
                    for mb_root, mb_dirs, mb_files in os.walk(matebook_folder):
                        for mb_dir in mb_dirs:
                            if mb_dir.startswith(directory):
                                mb_data_folder = os.path.join(mb_root, mb_dir)

                    clean_songpulses_file = mb_data_folder + '\\clean_songpulses.txt'
                    songpulses_file = mb_data_folder + '\\songpulses.txt'
                    cleansongpulsecenters_file = mb_data_folder + '\\clean_songpulsecenters.txt'
                    songpulsecenters_file = mb_data_folder + '\\songpulsecenters.txt'

                    clean_songpulse_list = self.extract_pulse_intervals(text_file_name)
                    if len(clean_songpulse_list) > 0:
                        audio_file = os.path.join(self.wav_folder, directory)
                        clip, sr = librosa.load(audio_file, sr=None)

                        songpulse_centers_list = []
                        for data_point in clean_songpulse_list:
                            temp_clip = clip[data_point[0]:data_point[1]]
                            clip_len = len(temp_clip)
                            peaks = self.detect_peaks(np.abs(temp_clip), mpd=clip_len)
                            # print(peaks.shape)
                            if abs(temp_clip[peaks[0]]) > 0.02:
                                songpulse_centers_list.append([data_point[0], data_point[0] + peaks[0]])

                        centers_text = ''
                        for frames in songpulse_centers_list:
                            centers_text = centers_text + str(frames[0]) + '\t' + str(frames[1]) + '\n'

                        centers_text_file = open(songpulsecenters_file, 'w')
                        centers_text_file.write(centers_text)
                        centers_text_file.close()

                        cleancenters_text_file = open(cleansongpulsecenters_file, 'w')
                        cleancenters_text_file.write(centers_text)
                        cleancenters_text_file.close()

                        songpulse_text = ''
                        num_pulses = 0
                        for frames in clean_songpulse_list:
                            temp_clip = clip[frames[0]:frames[1]]
                            clip_len = len(temp_clip)
                            peaks = self.detect_peaks(np.abs(temp_clip), mpd=clip_len)
                            # print(peaks.shape)
                            if abs(temp_clip[peaks[0]]) > 0.02:
                                songpulse_text = songpulse_text + str(frames[0]) + '\t' + str(frames[1]) + '\n'
                                num_pulses += 1

                        self.socket.send_string('Song pulses detected in ' + directory + ': ' + str(num_pulses))

                        pulse_text_file = open(songpulses_file, 'w')
                        pulse_text_file.write(songpulse_text)
                        pulse_text_file.close()

                        clean_text_file = open(clean_songpulses_file, 'w')
                        clean_text_file.write(songpulse_text)
                        clean_text_file.close()
                    else:
                        self.socket.send_string('No song pulses detected in ' + directory + '. ')
                        centers_text_file = open(songpulsecenters_file, 'w')
                        centers_text_file.write('')
                        centers_text_file.close()

                        cleancenters_text_file = open(cleansongpulsecenters_file, 'w')
                        cleancenters_text_file.write('')
                        cleancenters_text_file.close()

                        pulse_text_file = open(songpulses_file, 'w')
                        pulse_text_file.write('')
                        pulse_text_file.close()

                        clean_text_file = open(clean_songpulses_file, 'w')
                        clean_text_file.write('')
                        clean_text_file.close()
        # print('Song pulse annotation completed.')
        # sys.stdout.flush()
        self.socket.send_string('!' + str(1) + ':' + str(1) + ':' + str(1) + ':' + str(1))
        self.socket.send_string('Song pulse annotation completed')
        # print('@', 100.0)
        # sys.stdout.flush()
        self.socket.send_string('@' + str(0.0))
