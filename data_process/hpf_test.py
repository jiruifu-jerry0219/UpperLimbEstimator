from butterworth_filter import butter_highpass_filter as bhpf
from highpass_filter import hpf
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import math

def main(data = None, fs = None, fc = None, order = None):
    from scipy.signal import freqz
    fs = 1000
    fc = 10

    # Plot the frequency response for a few different orders.
    plt.figure(1)
    plt.clf()
    for order in [3, 6, 9]:
        b, a = bhpf(fs, fc, order=order)
        w, h = freqz(b, a, 512)
        plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)

    plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
             '--', label='sqrt(0.5)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.legend(loc='best')
    plt.show()


    # Impulse respnse
    imp = signal.unit_impulse(40)

    response = hpf(imp, 3500, 600, order = 6)

    # Illustrating impulse response
    plt.stem(np.arange(0, 40), imp, markerfmt='D', use_line_collection=True)
    plt.stem(np.arange(0, 40), response, use_line_collection=True)
    plt.margins(0, 0.1)

    plt.xlabel('Time [samples]')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

    # # Filter a noisy signal.
    # T = 0.05
    # nsamples = int(T * fs)
    # t = np.linspace(0, T, nsamples, endpoint=False)
    # a = 0.02
    # f0 = 10.0
    # x = 0.1 * np.sin(2 * np.pi * 1.2 * np.sqrt(t))
    # x += 0.01 * np.cos(2 * np.pi * 312 * t + 0.1)
    # x += a * np.cos(2 * np.pi * f0 * t + .11)
    # x += 0.03 * np.cos(2 * np.pi * 2000 * t)
    # plt.figure(2)
    # plt.clf()
    # plt.plot(t, x, label='Noisy signal')
    #
    # y = hpf(x, fs, fc, order=6)
    # plt.plot(t, y, label='Filtered signal (%g Hz)' % f0)
    # plt.xlabel('time (seconds)')
    # plt.hlines([-a, a], 0, T, linestyles='--')
    # plt.grid(True)
    # plt.axis('tight')
    # plt.legend(loc='upper left')

    # plt.show()

if __name__ == '__main__':
    main()
