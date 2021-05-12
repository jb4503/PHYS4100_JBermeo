import time
import cProfile
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import scipy.fftpack as fft
from Lab_10_Functions import psi_function

# Constants
ELECTRON_MASS = 9.109e-31
L = 1e-8  # m
TIME_STEP = 1e-18  # seconds
SIGMA = 1e-10  # m
KAPPA = 5e10  # 1/m
H_BAR = 6.626e-34
N_SLICES = 1000  # number of spatial slices
A_DIST = L / N_SLICES


def main():
    # For timing purposes
    start = time.time()
    pr = cProfile.Profile()
    pr.enable()

    def inv_discrete_sine(t):
        """
        Inverse discrete sine at a specific time of a wave function
        """
        sin_cos = t * (((np.pi ** 2) * H_BAR * (position_k ** 2)) / (2 * ELECTRON_MASS * (L ** 2)))
        a_term = np.cos(sin_cos) * c_real
        b_term = np.sin(sin_cos) * c_imag
        z = a_term - b_term
        inverse_z = fft.idst(z) / N_SLICES
        return inverse_z

    # Initial parameters

    start_time = TIME_STEP * np.arange(N_SLICES)
    end_time = N_SLICES
    position_k = np.delete((np.linspace(0, N_SLICES, num=N_SLICES+1, endpoint=False)), 0)
    x_points = np.delete((A_DIST * np.linspace(0, N_SLICES, num=N_SLICES+1, endpoint=False)), 0)

    # Animation portion
    def init():
        line.set_data(x_points, psi_function(x_points))
        return line,

    def animate(i):
        x = x_points
        y = inv_discrete_sine(time[i])
        line.set_data(x, y)
        return line,

    psi_wave = psi_function(x_points)
    coeffs = fft.dst(psi_wave)
    # Split the coefficients
    c_real = np.real(coeffs)
    c_imag = np.imag(coeffs)

    plt.plot(x_points, np.real(coeffs) ** 2)
    #plt.plot(x_points, np.real(coeffs[49]) ** 2)
    #plt.plot(x_points, np.real(coeffs[250]) ** 2)
    plt.xlabel("x (m)")
    plt.ylabel("$\psi(x)$")
    plt.show()

    # Animation
    fig = plt.figure()
    axis = plt.axes(xlim=(-1e-9, 11e-9), ylim=(-2, 2))
    axis.set_xlabel('x')
    axis.set_ylabel('$\psi(x)$')
    axis.set_title('Spectral Wave')
    line, = axis.plot([], [], lw=.5)

    animate_wave = animation.FuncAnimation(
        fig, animate, init_func=init, frames=end_time, interval=10, blit=True,
        save_count=50)
    my_writer = animation.PillowWriter(fps=30, codec='libx264', bitrate=2)
    plt.show()

    pr.disable()
    # pr.print_stats(sort='time')
    end = time.time()
    print(f"Program took :{end - start} seconds to run")


if __name__ == '__main__':
    main()
