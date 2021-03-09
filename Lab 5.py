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
    a = -.5
    b = .5
    num_squares = 100

    x = np.linspace(a, b, num_squares)
    xcoor, ycoor = np.meshgrid(x, x)

    phi = np.zeros((num_squares, num_squares))
    for i in range(num_squares):
        for j in range(num_squares):
            r1 = np.sqrt((xcoor[i][j] - q1[0])**2 + (ycoor[i][j] - q1[1])**2)
            r2 = np.sqrt((xcoor[i][j] - q2[0])**2 + (ycoor[i][j] - q2[1]) ** 2)
            phi[i][j] = (1/ 4 * np.pi*cons.epsilon_0) * (1/r1 - 1/r2)

    field_x, field_y = np.gradient(phi)
    streamplot(xcoor, ycoor, field_x, field_y, density=1, color=field_x, cmap='gray')
    plt.show()

    plt.matshow(phi)
    plt.colorbar()
    plt.show()

    Ex, Ey = np.gradient(phi)
    E = np.sqrt(Ex ** 2 + Ey ** 2)
    plt.quiver(-E[Ey][::4, ::4], E[Ex][::4, ::4])
    plt.show()
    pr.disable()
    # pr.print_stats(sort='time')
    end = time.time()
    print(f"Program took :{end - start} seconds to run")


if __name__ == '__main__':
    main()
