from pylab import imshow, gray, show, colorbar, streamplot
import matplotlib.pyplot as plt
import scipy.constants as cons
import numpy as np
import time
import cProfile


def main():
    # For timing purposes
    start = time.time()
    pr = cProfile.Profile()
    pr.enable()
    # Ex 5.21

    q1 = [0.05, 0.0]
    q2 = [-0.05, 0.0]
    a = -0.5
    b = 0.5
    num_squares = 100

    x = np.linspace(a, b, num_squares)
    xcoor, ycoor = np.meshgrid(x, x)

    phi = np.zeros((num_squares, num_squares))
    E = np.zeros((num_squares, num_squares))
    for i in range(num_squares):
        for j in range(num_squares):
            r1 = np.sqrt((xcoor[i][j] - q1[0])**2 + (ycoor[i][j] - q1[1])**2)
            r2 = np.sqrt((xcoor[i][j] - q2[0])**2 + (ycoor[i][j] - q2[1]) ** 2)
            phi[i][j] = (1/ 4 * np.pi*cons.epsilon_0) * (1/r1 - 1/r2)

    plt.imshow(phi)
    plt.title("electric potential of two particles")
    plt.colorbar()
    plt.show()

    Ex, Ey = np.gradient(phi)

    # Distance formula
    def R(x,y):
        return np.sqrt(x ** 2 + y ** 2)

    # Potential for every point
    q1x, q1y = xcoor - q1[0], ycoor - q1[1]
    q2x, q2y = xcoor - q2[0], ycoor - q2[1]

    # Electric field via partial derivative
    h = 1e-5

    def V(q, x, y):
        return q*(1 / 4 * np.pi*cons.epsilon_0)*(1/R(x, y))

    vx1 = (V(1, (q1x + h / 2), q1y) - V(1, (q1x - h / 2), q1y)) / h
    vy1 = (V(1, q1x, (q1y + h / 2)) - V(1, q1x, (q1y - h / 2))) / h
    vx2 = (V(-1, (q2x + h / 2), q2y) - V(-1, (q2x - h / 2), q2y)) / h
    vy2 = (V(-1, q2x, (q2y + h / 2)) - V(-1, q2x, (q2y - h / 2))) / h
    Vx = vx1 + vx2
    Vy = vy1 + vy2

    streamplot(xcoor, ycoor, Vx, Vy, density=2.5, linewidth=0.5, arrowsize=0.8)
    plt.title("Electric Field of a Dipole")
    plt.show()

    pr.disable()
    # pr.print_stats(sort='time')
    end = time.time()
    print(f"Program took :{end - start} seconds to run")


if __name__ == '__main__':
    main()
