"""WRITTEN BY: JIRUI FU"""
import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn

import scipy
from scipy import signal
from scipy.signal import freqz

import math

from butterworth import band_pass, low_pass, high_pass, band_stop
from utils import full_wave_rectify, plot_signal_one, plot_multiple, getEMGfeatures, toDataframe

a = np.arange(10, 5831, 10)
j = 0
for i in range(len(a)):
    j += 1
print(j)

# Setup the parameters of signal
f = 2000


path = r'/home/jerry/GitHub/EMG_regressive_model/data_process/raw_data'
# path = r'D:/GitHub/EMG_regressive_model/data_process/raw_data'
all_files = glob.glob(path+'/*.csv')
dfList = []

# Read .csv file by using panda
# for filename in all_files:
file = all_files[0]
saveName = file[-11:-4]
print('The csv file read into the program is:', file)
allData = pd.read_csv(file, skiprows = 4, header = None)


# Create the dataframe for EMG data and Joint angle
emgData = allData.iloc[:, 3:6]
angleData = allData.iloc[:, 32:33]

#Convert the dataframe to numpy array
emg = emgData.to_numpy()
time = np.array([i/f for i in range(0, len(emg), 1)]) # sampling rate 2000 Hz
mean = np.mean(emg, axis = 0)
emgAvg = emg - mean

angle = angleData.to_numpy()

emgSize = emgAvg.shape

bpEmg = np.zeros(emgSize)

for i in range(emgSize[-1]):
    input = emgAvg[:, i]
    iuput = input.T
    notch = band_stop(input, fs = f, fh = 20, fl = 60, order = 4)
    bandpass = low_pass(notch ,fs = f, low_pass = 2, order = 2)
    bpEmg[:, i] = bandpass

# erform the full wave rectification
rectEmg = full_wave_rectify(bpEmg)
print(rectEmg.shape)
emgFeatures = getEMGfeatures(rectEmg, 10, 10)
print(emgFeatures.shape)
a = ['ch '+ str(x) for x in range(1, 16)]
dfFeatures = toDataframe(emgFeatures,
        head = a,
        save = True,
        path = r'/home/jerry/GitHub/EMG_regressive_model/data_process/data/export.csv')


# a = emg.shape
# c = np.zeros(a)
# for i in range(a[-1]):
#     c[:, i] = emg[:, i]
# print(type(c),'and', type(emg))
#
#
# # In[31]:
#
#
# tit = ['EMG CH1', 'EMG CH2', 'EMG CH3']
# filename = 'emg_filter_lp_f4o10.png'
# signal = [rectEmg[:, 0], rectEmg[:, 1], rectEmg[:, 2]]
# t = [time, time, time]
#
# plot_multiple(signal, t, 3, 'Time', 'EMG', tit, filename)
