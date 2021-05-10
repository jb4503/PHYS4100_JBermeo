import time
import cProfile
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy import abs, exp, linspace, array, zeros, empty, real, imag, dot, delete
from numpy.linalg import solve, pinv
from scipy.linalg import solve_banded


# Constants
ELECTRON_MASS = 9.109e-31
L = 1e-8  # m
TIME_STEP = 1e-18  # seconds
SIGMA = 1e-10  # m
KAPPA = 5e10  # 1/m
H_BAR = 6.626e-34
N_SLICES = 1000  # number of spatial slices
A_DIST = L / N_SLICES


def psi_function(x=1, length=1e-8, sigma=1e-10, kappa=1e10):
    """
    Returns the first value of psi at t=0 for all x values
    :param x:
    :param length: length of the box, in meters
    :param sigma:
    :param kappa:
    :return:
    """
    x_0 = length / 2
    num = (x - x_0) ** 2
    den = 2 * (sigma ** 2)
    psi_0 = exp(-(num / den)) * exp(1j * kappa * x)
    return psi_0


def TriDiag_Matrix(A, a1, a2,N=100):
    """
    Creates a Tridiag Matrix A
    given diagonal element a1
    and banded elements a2
    """
    for i in range(N):
        A[i, i] = a1
        if i == 0:
            A[i, i+1] = a2
        elif i == N-1:
            A[i, i-1] = a2
        else:
            A[i, i+1] = a2
            A[i, i-1] = a2
    return A


def main():
    # For timing purposes
    start = time.time()
    pr = cProfile.Profile()
    pr.enable()

    # Initial parameters
    # Constants

    a1 = 1 + TIME_STEP*((1j * H_BAR) /(2 * ELECTRON_MASS * (A_DIST**2)))
    a2 = - TIME_STEP * H_BAR * 1j / (4 * ELECTRON_MASS * A_DIST ** 2)
    b1 = 1 - 1j * TIME_STEP * H_BAR / (2 * ELECTRON_MASS * A_DIST ** 2)
    b2 = - a2

    # initial conditions
    x_points = linspace(0, L, N_SLICES + 1, endpoint=False)
    psi = array(list(map(psi_function, x_points)), complex)
    psi[0] = psi[N_SLICES - 1] = 0

    # Create the matrix A
    A2 = empty([3, N_SLICES + 1], complex)
    A2[0, 0] = 0
    A2[0, 1:] = a2
    A2[1, :] = a1
    A2[2, 0: N_SLICES] = a2
    A2[2, N_SLICES] = 0
    # store the wave function at each time step in a list
    solution = [psi]
    for i in range(500):
        psi[1: N_SLICES] = b1 * psi[1: N_SLICES] + b2 * (psi[2:] + psi[0: N_SLICES - 1])
        psi = solve_banded((1, 1), A2, psi)
        solution.append(psi)

    plt.plot(x_points, real(solution[25]))
    plt.plot(x_points, real(solution[49]) ** 2)
    plt.plot(x_points, real(solution[250]) ** 2)
    plt.xlabel("x (m)")
    plt.ylabel("$\psi(x)$")
    plt.show()

    # Animation

    def init():
        line.set_data(x_points, psi_function(xpoints, L, SIGMA, KAPPA))
        return line,

    A_2 = zeros([N_SLICES, N_SLICES], dtype=complex)
    A_2 = TriDiag_Matrix(A_2, a1, a2)
    B = zeros([N_SLICES, N_SLICES], dtype=complex)
    B = TriDiag_Matrix(B, b1, b2, N_SLICES)

    xpoints = (L / N_SLICES) * linspace(0, N_SLICES, num=N_SLICES + 1, endpoint=False)
    xpoints = delete(xpoints, 0)

    def animate(i):
        v = dot(B, psi[i])
        psi_2.append(solve(A_2, v))
        x = xpoints
        y = psi[i]
        i += 1
        line.set_data(x, y)
        return line,

        # calling the animation function
        # creating a blank window
        # for the animation

    psi_2 = [psi_function(xpoints, L, SIGMA, KAPPA)]
    fig = plt.figure()
    axis = plt.axes(xlim=(-1e-9, 11e-9), ylim=(-1, 1))
    axis.set_xlabel('x')
    axis.set_ylabel('$\psi(x)$')
    axis.set_title('Crank Nicolson Wave Function')
    line, = axis.plot([], [], lw=.5)

    animate_wave = animation.FuncAnimation(
        fig, animate, init_func=init, frames=100000, interval=500, blit=True,
        save_count=50)
    my_writer = animation.PillowWriter(fps=30, codec='libx264', bitrate=2)
    plt.show()


    pr.disable()
    # pr.print_stats(sort='time')
    end = time.time()
    print(f"Program took :{end - start} seconds to run")


if __name__ == '__main__':
    main()
