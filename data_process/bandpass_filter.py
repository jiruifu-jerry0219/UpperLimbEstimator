from butterworth_filter import butter_bandpass_filter as bbpf
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import math

def bpf(data, fs, lowcut, highcut, order):
    """
    Apply the Butterworth high pass Filter
    data: raw signal for processing
    fs: sampling rate
    lowcut: lowcut Frequency
    highcut: highcut Frequency
    order: order of Filter
    """
    b, a = bbpf(fs,
        lowcut,
        highcut,
        order)
    y = signal.filtfilt(b,
        a,
        data)
    return y
