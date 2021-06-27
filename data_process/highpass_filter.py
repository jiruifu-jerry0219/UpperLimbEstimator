from butterworth_filter import butter_highpass_filter as bhpf
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import math

def hpf(data, fs, fc, order):
    """
    Apply the Butterworth high pass Filter
    data: raw signal for processing
    fs: sampling rate
    fc: cutoff Frequency
    order: order of Filter
    """
    b, a = bhpf(fs, fc, order)
    y = signal.filtfilt(b, a, data)
    return y
