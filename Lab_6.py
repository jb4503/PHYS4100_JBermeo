import matplotlib.pyplot as plt
import math
import numpy as np
import time
import timeit
import cProfile
import scipy.constants as cons
from astropy import units as u


def main():

    # For timing purposes
    start = time.time()
    pr = cProfile.Profile()
    pr.enable()

    # Exercise 6.14

    # Preliminary values

    w = 1e-9  # in nanometers
    V = 20  # in electronVolts
    m = cons.m_e  # in kilograms
    c_e = cons.elementary_charge  # 1 eV = 1 Joule
    hbar = cons.hbar  # in J*s

    # multiplying w^2*m by c_e allows to get a tangent graph and not funky
    # lines like before
    c = c_e*((w ** 2) * m) / (2 * hbar ** 2)

    # First function
    def y1(e):
        return np.tan(np.sqrt(c * e))

    # Second function
    def y2(e):
        return np.sqrt((V - e) / e)

    # Third function
    def y3(e):
        return -np.sqrt(e / (V - e))

    # Finds the roots to a desired accuracy
    def roots(f, x1, x2, acc):
        def midpoint(x, y):
            return (x + y) / 2
        def have_same_sign(x, y):
            if x < 0 and y < 0 or x > 0 and y > 0:
                return True
            else:
                return False
        while abs(x1 - x2) > acc:
            x = midpoint(x1, x2)
            if have_same_sign(f(x1), f(x)):
                x1 = x
            elif have_same_sign(f(x), f(x2)):
                x2 = x
            elif abs(x) < acc:
                return x

        return midpoint(x1, x2)

    def f1(x):
        return y1(x) - y2(x)

    def f2(x):
        return y1(x) - y3(x)

    # For the different values of E
    E = np.linspace(.05, 19.95, 100)

    # Initially I had used the names y1,y2 and y3
    # This was a fatal mistake and caused call issues down below
    # Now I have learned never to reuse variable and functions names
    Y1 = y1(E)
    Y2 = y2(E)
    Y3 = y3(E)
    # Plot the values of E
    plt.plot(E,Y1, '-', label='tangent')
    plt.plot(E, Y2, '-', label='positive values')
    plt.plot(E, Y3, '-', label='negative values')
    plt.legend()
    plt.xlabel('E in electronVolts')
    plt.show()


    # Finding the first 6 values for energy levels
    accuracy = 1e-5  # in Ev
    print('E0 = ', roots(f1, 0.02, 0.5, accuracy))
    print('E1 = ', roots(f2, 0.75, 1.5, accuracy))
    print('E2 = ', roots(f1, 2.5, 3, accuracy))
    print('E3 = ', roots(f2, 5, 8, accuracy))
    print('E4 = ', roots(f1, 7, 9, accuracy))
    print('E5 = ', roots(f2, 10, 12.5, accuracy))

    pr.disable()
    # pr.print_stats(sort='time')
    end = time.time()
    print(f"Program took :{end - start} seconds to run")


if __name__ == '__main__':
    main()
