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

from utils.butterworth import band_pass, low_pass, high_pass, band_stop
from utils.utils import full_wave_rectify, plot_signal_one, plot_multiple
from utils.utils import getEMGfeatures, toDataframe, normalization



# Setup the parameters of signal
f = 2000


# path = r'/home/jerry/GitHub/EMG_regressive_model/data_process/raw_data'
# path = r'D:/GitHub/EMG_regressive_model/data_process/raw_data'
pathEmg = r'/home/jerry/GitHub/EMG_regressive_model/data_process/raw_data/type2/emg'

emg_files = glob.glob(pathEmg+'/*.csv')
dfList = []

for filename in emg_files:
    headers = [*pd.read_csv(filename, nrows  = 1)]
    df = pd.read_csv(filename, usecols=[c for c in headers if c != 'time'])
    dfList.append(df)

#Concatenate individual column dataframes into one data frame (don't forget axis)
emgData = pd.concat(dfList, axis = 1)

pathElbow = r'/home/jerry/GitHub/EMG_regressive_model/data_process/raw_data/type2/kin'
imu_files = glob.glob(pathElbow+'/*.csv')

dfList2 = []

for filename in imu_files:
    headers = [*pd.read_csv(filename, nrows  = 1)]
    df = pd.read_csv(filename, usecols=[c for c in headers if c != 'time'])
    dfList2.append(df)
#Concatenate individual column dataframes into one data frame (don't forget axis)
angleData = pd.concat(dfList2, axis = 1)
angle = angleData.to_numpy()
angle = angle[:, 0]
notch = band_stop(angle, fs = 200, fh = 9, fl = 10, order = 4)
imu_filter = low_pass(notch, fs = 200, low_pass = 2, order = 2)
imu = np.reshape(imu_filter, (-1, 1))
normImu = normalization(imu)
print(normImu.shape)


#Convert the dataframe to numpy array
emg = emgData.to_numpy()
time = np.array([i/f for i in range(0, len(emg), 1)]) # sampling rate 2000 Hz
mean = np.mean(emg, axis = 0)
emgAvg = emg - mean
emgSize = emgAvg.shape

bpEmg = np.zeros(emgSize)

for i in range(emgSize[-1]):
    input = emgAvg[:, i]
    iuput = input.T
    notch = band_stop(input, fs = f, fh = 2, fl = 20, order = 2)
    bandpass = low_pass(notch ,fs = f, low_pass = 20, order = 4)
    bpEmg[:, i] = bandpass

# erform the full wave rectification
rectEmg = full_wave_rectify(bpEmg)
print(rectEmg.shape)

#Feature extraction
emgFeatures = getEMGfeatures(rectEmg, 10, 10)
emgNorm = normalization(emgFeatures)
print(emgNorm.shape)

#Concatenate emg data and elbow data
dataset = np.hstack((emgNorm, normImu))
print(dataset.shape)

a = ['ch '+ str(x) for x in range(1, 16)]
a.append('angle')
dfFeatures = toDataframe(dataset,
        head = a,
        save = True,
        path = r'/home/jerry/GitHub/EMG_regressive_model/data_process/data/export.csv')
