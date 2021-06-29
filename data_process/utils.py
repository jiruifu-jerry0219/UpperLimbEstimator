
import peakutils  # peak detection
import numpy as np  # to handle datas
import math  # to handle mathematical stuff (example power of 2)
from scipy.signal import butter, lfilter, welch, square  # for signal filtering


class getFeaturesTD:
    def __init__(self, signal, windowSize, step):
        self.emg = signal
        self.window = windowSize
        self.step = step

    def getMAV(self):
        """
        Mean absolute value
        """
        MAV = (1 / len(self.emg)) * np.sum([abs(x) for x in self.emg])
        return MAV

    def getSSI(self):
        """
        Simple square integral
        """
        SSI = np.sum([x ** 2 for x in self.emg])
        return SSI

    def getVAR(self):
        """
        Variance of EMG
        """
        VAR = (1 / (len(self.emg) - 1)) * np.sum([x ** 2 for x in self.emg])
        return VAR

    def getRMS(self):
        """
        Root mean square
        """
        RMS = np.sqrt((1 / len(self.emg)) * np.sum([x ** 2 for x in self.emg]))
        return RMS
