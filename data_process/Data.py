#!/usr/bin/env python
# coding: utf-8

# ### Note:
# - This notebook only process three types of experiment data which are:
# 1. EMG measurement from three channels
# 2. Elbow flexion / extension

# In[1]:


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

from butterworth import band_pass, low_pass, high_pass
from utils import full_wave_rectify, plot_signal_one, plot_multiple


# In[2]:


x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
xa = np.asarray(x)
vara = (1 / (len(xa) - 1)) * np.sum([s ** 2 for s in xa])
print(vara)
a = 4
print(np.sqrt(a))


# ### Prepare feature extraction functions

# In[3]:


a = np.arange(10, 5831, 10)
j = 0
for i in range(len(a)):
    j += 1
print(j)


# In[4]:


def getEMGfeatures(emg, window = 1, step = 1):
    """
    emg: filtered rectified EMG signal
    window: size of sliding windows
    step: number of step between two windows
    """
    endPt = len(emg)
    pts = np.arange(window, endPt, step)
    j = 0
    for i in range(len(pts)):
        j += 1
        sampleEMG = emg[pts[i] - window + 1:pts[i], :]
        pass
        
    
def getfeaturesTD(emg):
    pass
    


# ### Data Preprocessing
# 1. Full-wave rectification
# 2. Remove the noise from data by using Butterworth Filter
# 3. Feature extraction
# 4. Dimensionality Reduction (*Optional)
# 5. Save the data as '.csv' file

# In[5]:


# Setup the parameters of signal
f = 2000


# In[6]:


# path = r'/home/jerry/GitHub/EMG_regressive_model/data_process/raw_data'
path = r'D:/GitHub/EMG_regressive_model/data_process/raw_data'
all_files = glob.glob(path+'/*.csv')
dfList = []


# In[7]:


# Read .csv file by using panda
# for filename in all_files:
file = all_files[0]
saveName = file[-11:-4]
print(file)
allData = pd.read_csv(file, skiprows = 4, header = None)


# In[29]:


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
    bandpass = low_pass(emgAvg[:, i],fs = f, low_pass = 600, order = 2)
    bpEmg[:, i] = bandpass


# Perform the full wave rectification
rectEmg = full_wave_rectify(bpEmg)


bpAngle = band_pass(angle, fs = f, high_band = 100, low_band = 10, order = 4, axs = -1, padB = True, pad = 0)


# In[30]:


a = emg.shape
c = np.zeros(a)
for i in range(a[-1]):
    c[:, i] = emg[:, i]
print(type(c),'and', type(emg))


# In[31]:


tit = ['EMG CH1', 'EMG CH2', 'EMG CH3']
filename = 'emgvstime_filter.png'
signal = [rectEmg[:, 0], rectEmg[:, 1], rectEmg[:, 2]]
t = [time, time, time]

plot_multiple(signal, t, 3, 'Time', 'EMG', tit, filename)


# In[32]:


# Plot unfiltered EMG
rect = full_wave_rectify(emgAvg)
tit = ['EMG CH1', 'EMG CH2', 'EMG CH3']
filename = 'emgvstime_raw.png'
signal = [rect[:, 0], rect[:, 1], rect[:, 2]]
t = [time, time, time]

plot_multiple(signal, t, 3, 'Time', 'EMG', tit, filename)


# In[ ]:





# In[11]:


row = 10
for i in range(11):
    for j in range(2):
        print(i, 'and', j)


# In[ ]:




