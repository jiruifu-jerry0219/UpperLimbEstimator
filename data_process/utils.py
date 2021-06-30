import os
import glob

import numpy as np
import math

import matplotlib.pyplot as plt
import seaborn

from getFeaturesTD import getfeaturesTD as F

import pandas as pd

def full_wave_rectify(signal):
    rectify_emg = np.absolute(signal)
    return rectify_emg

def plot_signal_one(signal, time, xlabel, ylabel, title, fname):
    fig = plt.figure()
    plt.plot(time, signal)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    fig_name =fname
    fig.set_size_inches(w=11,h=7)
    fig.savefig(fig_name)

def plot_multiple(signal, time, num_of_plot, xlabel, ylabel, title, fname):
    row_num = math.ceil(num_of_plot / 2)
    fig, a = plt.subplots(row_num, 2, figsize = (150, 150))
    assert len(signal)==len(time), "Length of signals must equal to length of time and total number of plots"
    n = 0
    for i in range(row_num):
        for j in range(2):
            if n<= num_of_plot - 1:
                a[i][j].plot(time[n], signal[n])
                a[i][j].set_title(title[n])
                n += 1
            else:
                pass
    plt.show()
    fig.savefig(fname)

def getEMGfeatures(emg, window = 1, step = 1):
    """
    emg: filterd rectified EMG signals
    window: size of sliding window
    step: number of step between two windows
    """
    featuresSet = []
    endPt = len(emg)
    pts = np.arange(window, endPt, step)
    j = 0
    for i in range(len(pts)):
        j += 1
        m1 = pts[i] - window + 1
        m2 = pts[i]
        # print('Sampling from', m1, ' to ', m2)
        sampleEMG = emg[pts[i] - window : pts[i], :]
        assert len(sampleEMG) != 0, 'please check for mistake'
        feature = F(sampleEMG, window, step)
        featuresSet.append(feature)

    featureTD = np.vstack(featuresSet)
    return featureTD

def toDataframe(data, head, save = False, path = None):
    """Convert a numpy matrix to a panda dataframe"""
    assert data.shape[-1] == len(head), 'The number of heads must equals to the columns of matrix'
    df = pd.DataFrame(data, columns = head)
    if save:
        assert path is not None, 'Nontype path is not feasible if you want to save the data'
        dataExport(path, df, idx = False, hd = True)

    return df

def dataExport(path, df, idx = False, hd = True):
    df.to_csv(path, index = idx, header = hd)
