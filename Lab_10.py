import time
import cProfile
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy import abs, exp, linspace, array, zeros, empty, real, imag
from scipy.linalg import solve_banded


# Constants
ELECTRON_MASS = 9.109e-31
L = 1e-8  # m
TIME_STEP = 1e-18  # seconds
SIGMA = 1e-10  # m
K = 5e10  # 1/m
H_BAR = 6.626e-34
N_SLICES = 1000  # number of spatial slices
A_DIST = L / N_SLICES


def main():
    # For timing purposes
    start = time.time()
    pr = cProfile.Profile()
    pr.enable()

    # Initial parameters
    # Constants

    x0 = L / 2

    a1 = 1 + 1j * TIME_STEP * H_BAR / (2 * ELECTRON_MASS * A_DIST ** 2)
    a2 = - TIME_STEP * H_BAR * 1j / (4 * ELECTRON_MASS * A_DIST ** 2)
    b1 = 1 - 1j * TIME_STEP * H_BAR / (2 * ELECTRON_MASS * A_DIST ** 2)
    b2 = - a2

    def psi_0(x):
        return exp(-(x - x0) ** 2 / (2 * SIGMA ** 2)) * exp(1j * K * x)

    # initial conditions
    x_points = linspace(0, L, N_SLICES + 1)
    psi = array(list(map(psi_0, x_points)), complex)
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

    plt.plot(x_points, real(solution[4]))
    #plt.plot(x_points, real(solution[49]) ** 2)
    #plt.plot(x_points, real(solution[250]) ** 2)
    plt.xlabel("x (m)")
    plt.ylabel("$\psi(x)$")
    plt.show()

    # creating a blank window
    # for the animation
    psi_2 = []
    psi_2.append(psi_0(x_points))
    fig = plt.figure()
    axis = plt.axes(xlim=(-1e-9, 11e-9), ylim=(-1, 1))
    axis.set_xlabel('x')
    axis.set_ylabel('$\psi(x)$')
    axis.set_title('Crank Nicolson Wave Function')
    line, = axis.plot([], [], lw=.5)

    # what will our line dataset
    # contain?
    def init():

        line.set_data(x_points, psi_0(x_points))
        return line,

    # animation function
    solution = [psi_2]
    def animate(i):
        solution = [psi_2]
        for i in range(500):
            psi_2[1: N_SLICES] = b1 * psi_2[1: N_SLICES] + b2 * (psi_2[2:] + psi[0: N_SLICES - 1])
            psi_2 = solve_banded((1, 1), A2, psi)
            solution.append(psi_2)
        # x, y values to be plotted
        x = x_points
        y = solution[i]
        # appending values to the previously
        i += 2500
        line.set_data(x, y)

        return line,

        # calling the animation function

    anim = animation.FuncAnimation(fig, animate,
                                   init_func=init,
                                   frames=500,
                                   interval=20,
                                   blit=True)

    plt.show()
    anim.save('Lab10GIF.gif', writer=animation.PillowWriter(fps=30))

    pr.disable()
    # pr.print_stats(sort='time')
    end = time.time()
    print(f"Program took :{end - start} seconds to run")


if __name__ == '__main__':
    main()
