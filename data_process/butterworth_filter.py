import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import math

# def butter_highpass_filter(f_s, f_pass, f_stop, fs = 0.5, td = 1, g_pass = 1, g_stop = 50, wc = None):
#     """
#     Return explanation:
#     N: number of orders
#     b: numerator of the Filter
#     a: denominator of the filter
#     """
#     wp = f_pass / (f_s / 2)
#     ws = f_stop / (f_s / 2)
#     omega_p = (2 / td) * np.tan(wp / 2)
#     omega_s = (2 / td) * np.tan(ws / 2)
#     # Find the order and natural frequency of the highpass filter
#     N, wn = signal.buttord(omega_p, omega_s, g_pass, g_stop, analog = True)
#     # Find the Gain of the highpass filter
#     if wc:
#         b, a = signal.butter(N, wc, btype = 'high', analog = True)
#         wn = wc
#     else:
#         b, a = signal.butter(N, wn, btype = 'high', analog = True)
#     return N, b, a, wn

def butter_highpass_filter(fs, fc, order = 5):
    """
    Function explanation:
    fs: sampling rate
    fc: cutoff frequency
    order: Number of orders
    Return: denominator and numerator of the filter's transfer function
    """
    nyq = 0.5 * fs
    normal_fc = fc / nyq
    b, a = signal.butter(order, normal_fc, btype = 'high', analog = False)
    return b, a

def butter_bandpass_filter(fs, lowcut, highcut, order = 5):
    """
    Function explanation:
    fs: sampling rate
    lowcut: lowcut Frequency
    highcut:highcut Frequency
    order: Number of orders
    Return: denominator and numerator of the filter's transfer function
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype = 'band', analog = False)
    return b, a
