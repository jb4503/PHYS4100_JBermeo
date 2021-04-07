import numpy as np
import matplotlib.pyplot as plt
from functions_lab7 import dft, square_wave, sawtooth_wave, modulated_sine_wave


def main():
    # Number of points
    n_1 = np.linspace(0, 1, 1000)

    sq_wave = square_wave(n_1)
    sawtooth = sawtooth_wave(n_1)
    mod_sine = modulated_sine_wave(n_1)

    square_wave_coefficients = dft(sq_wave)
    sawtooth_wave_coefficients = dft(sawtooth_wave)
    modulated_sine_coefficients = dft(mod_sine)

    fig, axs = plt.subplots(3, 2)
    axs[0, 0].plot(sq_wave)
    axs[0, 0].set_title("Square wave")
    axs[0, 1].plot(abs(square_wave_coefficients))
    axs[0, 1].set_title("DFT of the Square wave")
    axs[1, 0].sharex(axs[0, 0])
    axs[1, 0].plot(sawtooth)
    axs[1, 0].set_title("Sawtooth Wave")
    axs[1, 1].plot(abs(sawtooth_wave_coefficients[100:200]))
    axs[1, 1].set_title("DFT of the Sawtooth wave")
    axs[1, 0].sharex(axs[0, 0])
    axs[2, 0].plot(mod_sine)
    axs[2, 0].set_title("Modulated Sine Wave")
    axs[2, 1].plot(abs(modulated_sine_coefficients))
    axs[2, 1].set_title("DFT of the Modulated Sine wave")
    fig.tight_layout()
    fig.show()


if __name__ == '__main__':
    main()
