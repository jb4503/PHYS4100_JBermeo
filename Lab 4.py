import scipy.constants as cons
import numpy as np
import time
import cProfile


def main():
    # For timing purposes
    start = time.time()
    pr = cProfile.Profile()
    pr.enable()

    # Ex 5.12

    # Term outside of the integral

    def outer_term(t):
        # all the constants can be multiplied and divided together to make this simpler
        multiplier = (cons.k**4)/((4*cons.pi**2)*(cons.c**2)*(cons.hbar**3))
        return t ** 4 * multiplier
    # Define the integral

    def f(x):
        a = (x**3)
        b = (np.exp(x))-1
        return a/b

    # Simpsons rule:

    def simpson_integration(a, b, n):
        h = (b-a)/n
        odds = 0
        for k in range(1, n, 2):
            odds += f(a + k*h)
        even = 0
        for k in range(2, n, 2):
            even += f(a + k * h)

        return (1/3) * h * (f(a) + f(b) + 4*odds + 2*even)
    # Stefan-Boltzmann Equation

    integral_lower = 1e-10
    integral_top = float(100)
    num_slices = int(10000)

    s_b = outer_term(1) * simpson_integration(integral_lower, integral_top, num_slices)

    # Error finding

    error = abs(cons.Stefan_Boltzmann - s_b)

    print('The calculated value of the constant is: {}'.format(s_b))
    print('The theoretical value of the constant is: {}'.format(cons.Stefan_Boltzmann))
    print('The difference between found and theoretical values is: {}'.format(error))

    pr.disable()
    # pr.print_stats(sort='time')
    end = time.time()
    print(f"Program took :{end - start} seconds to run")


if __name__ == '__main__':
    main()
