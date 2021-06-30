import scipy
from scipy import signal
from scipy.signal import freqz
import numpy as np
import os

"""
The program for Butterworth Filter:
1. Highpass Filter
2. Lowpass Filter
3. Bandpass Filter
"""
def band_stop(x, fs, fh, fl, order):
    """
    Apply a bandstop filter
    """
    fny = 0.5 * fs
    assert fh < fl ,'High cutoff frequency must be less than low cutoff frequency'
    cutoffHigh = fh / fny
    cutoffLow = fl / fny

    [b, a] = signal.butter(order, [cutoffHigh, cutoffLow], btype = 'stop')
    emg_notch = signal.filtfilt(b, a, x)

    return emg_notch

def band_pass(signal, fs = 2000, high_band = 1000, low_band = 10, order = 4):
    """
    signal: rectified EMG data
    high_band: high pass filter cut off frequency
    low_band: low pass filter cut off frequency
    fs: sampling frequency
    order: order of filter
    """
    # normalize cut-off frequency by sampling frequency
    high_band = high_band / (fs / 2)
    low_band = low_band / (fs / 2)
    # create bandpass filter gain
    b, a = scipy.signal.butter(order, [low_band, high_band], btype='bandpass')

    # Filter EMG
    emg_bandpass = scipy.signal.filtfilt(b, a, signal)
    return emg_bandpass

def low_pass(signal, fs = 2000, low_pass = 10, order = 4):
    """
    signal: rectified EMG data
    low_pass: low pass filter cut off frequency
    fs: sampling frequency
    order: rder of filter
    """

    # normalize cut-off frequency
    low_pass = low_pass / (fs / 2)
    #create lowpass filter
    b, a = scipy.signal.butter(order, low_pass, btype = 'lowpass')

    emg_envelop = scipy.signal.filtfilt(b, a, signal)

    return emg_envelop

def high_pass(signal, fs = 2000, high_pass = 10, order = 4):
    """
    signal: rectified EMG data
    high_pass: high pass filter cut off frequency
    fs: sampling frequency
    order: order of filter
    """

    high_pass = high_pass / (fs / 2)
    b, a = scipy.signal.butter(order, high_pass, btype = 'highpass')
    emg_envelop = scipy.signal.filtfilt(b, a, signal)

    return emg_envelop
