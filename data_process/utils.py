import os
import glob

import numpy as np
import math

import matplotlib.pyplot as plt
import seaborn

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
    pts = np.arange(window - 1, endPt, step)
    print(pts)
    j = 0
    for i in range(len(pts + 1)):
        j += 1
        m1 = pts[i] - window + 1
        m2 = pts[i] - 1
        # print('Sampling from', m1, ' to ', m2)
        sampleEMG = emg[pts[i] - window + 1 : pts[i], :]
        assert len(sampleEMG) != 0, 'please check for mistake'
        feature = getfeaturesTD(sampleEMG, window, step)
        featuresSet.append(feature)

    featureTD = np.vstack(featuresSet)
    return featureTD

def getfeaturesTD(emg, windows, step):
    pool = []
    col = emg.shape
    for i in range(col[-1]):
        s = emg[:, i]
        assert len(s) != 0, "The length of input vector is zero!"
        # print('The length of current vector is:', len(s))

        MAV = (1 / len(s)) * np.sum([abs(x) for x in s])


        SSI = np.sum([x ** 2 for x in s])

        VAR = (1 / (len(s) - 1)) * np.sum([x ** 2 for x in s])

        RMS = np.sqrt((1 / len(s)) * np.sum([x ** 2 for x in s]))

        LOG = math.exp((1 / len(s)) * sum([abs(x) for x in s]))

        cln = np.vstack((MAV, SSI, VAR, RMS, LOG))


        pool.append(cln)

    featureSet = np.vstack(pool)
    # print('The shape of feature set in this iteration is', featureSet.T.shape)
    return featureSet.T

def toDataframe(data, head, save = False, path = None):
    """Convert a numpy matrix to a panda dataframe"""
    assert data.shape[-1] == len(head), 'The number of heads must equals to the columns of matrix'
    df = pd.DataFrame(data, columns = head)
    if save:
        assert path is not None, 'Nontype path is not feasible if you want to save the data'
        dataExport(path, df, idx = False, hd = True)

    return df

def normalization(data):
    """Data must be trial * channels"""
    size = data.shape
    pool = []
    for i in range(size[-1]):
        process = data[:, i]
        minmax = (process - process.min()) / (process.max() - process.min())
        norm = np.reshape(minmax, (-1, 1))

        print('The size of', i,'th array is', norm.shape)
        pool.append(norm)


    result = np.hstack(pool)
    return result



def dataExport(path, df, idx = False, hd = True):
    df.to_csv(path, index = idx, header = hd)
