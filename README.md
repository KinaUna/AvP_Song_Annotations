# AvP_Song_Annotations
Convolutional Neural Network for song pulse detection

This application is using a TensorFlow 2.0 Convolutional Neural Network to detect fruit fly song pulses.

This repository consists of two applications:

A Python 3.6 application to analyse audio files and C# WPF application to manage settings and input files for the Python application.

In production the Python application is packaged with PyInstaller into a folder, and this folder is placed in the folder containing the C# application.


The training data has the following structure:
A folder containing the wav files.
- For each wav file there is a folder starting with same name as the wav file and a random number and some characters.
- Within each of these folders is a tab separated text file with the start and end frames of each song pulse, named clean_songpulses.txt.

When running predictions the wav files are placed in a folder with a sub folder for each wav file beginning with the same name. The predictions are then saved as start and end frames in a tab separated text file named clean_songpulses.txt, a long with a few other files for statistical analysis.
