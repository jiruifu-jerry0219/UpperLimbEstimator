import numpy as np  # to handle datas
import math  # to handle mathematical stuff (example power of 2)
from scipy.signal import butter, lfilter, welch, square  # for signal filtering


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
