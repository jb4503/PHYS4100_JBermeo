import numpy as np


# Fourier transform function
def dft(y, per=0.15):
    """
    Performs the Fast Fourier Transform
    Zeros is the percentage of elements that are converted to zero
    """
    # use FFT to get array with .5N + 1 elements
    c = np.fft.rfft(y)
    # set arbitrary % of array to zero
    d = int(len(c) * per)
    e = len(c) - d
    c = (c[:d])
    c = np.pad(c, (0, e), 'constant')
    z = np.fft.irfft(c)
    return z


def square_wave(x):
    """
    Creates a square wave
    Takes an array argument x and returns the square wave function
    The amplitude of the wave is 1
    """
    sqwv = np.arange(0, 1000, 1)
    for i in range((len(x))):
        if x[1] < 0.5000:
            sqwv[i] = 0
        if x[i] > 0.5000:
            sqwv[i] = 1
    return sqwv


def sawtooth_wave(x):
    """
    Creates a sawtooth wave
    Takes an argument x
    The amplitude of the wave is 1
    """
    sawt = x
    return sawt


def modulated_sine_wave(x):
    """
    Takes an argument x, as an array
    Returns the modulated sine wave with amplitude 1
    """
    N = (len(x))
    modw = np.arange(0, 1000, 1)/1000
    for i in range(N):
        modw[i] = np.sin(np.pi * i / N) * np.sin(20 * np.pi * i / N)
    return modw
