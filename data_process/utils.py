import os
import glob

import numpy as np
import math

import matplotlib.pyplot as plt
import seaborn

def full_wave_rectify(signal):
    rectify_emg = np.absolute(signal)
    return rectify_emg

def plot_signal_one(signal, time, xlabel, ylabel, title, fname):
    fir = plt.figure()
    plt.plot(time, signal)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    fig_name =fname
    fig.set_size_inches(w=11,h=7)
    fig.savefig(fig_name)

def plot_multiple(signal, time, num_of_plot, xlabel, ylabel, title, fname):
    row_num = math.ceil(num_of_plot / 2)
    fig, a = plt.subplots(row_num, 2, figsize = (15, 15))
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
