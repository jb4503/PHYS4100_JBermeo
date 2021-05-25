import matplotlib.pyplot as plt
import numpy as np
import time
import cProfile


def main():
    # For timing purposes
    start = time.time()
    pr = cProfile.Profile()
    pr.enable()

    # Prove the denominator for p(x) is 2
    def den(x):
        return x ** -.5

    N = 10000000  # Number of samples
    rand_vals = np.random.random(N)
    denominator = 1 / (N) * np.sum(den(rand_vals))
    denominator = round(denominator, 2)
    print('p(x): 1/(' + str(denominator) + 'x ** .5)')

    # Evaluating the integral
    def f(x):
        return (x ** -.5) / (1 + np.exp(x))

    def p(x):
        return 1 / (2 * np.sqrt(x))

    def g(x):
        return 1 / (1 + np.exp(x))

    # Solving using f(x) and p(x)
    integral_A = np.sum(f(rand_vals)) / np.sum(p(rand_vals))
    print('Using f(x)/p(x), the integral is:' + str(integral_A))

    # Solving using g(x)
    integral_B = 1 / N * denominator * np.sum(g(rand_vals ** 2))
    print('Using g(x), the integral is:' + str(integral_B))

    plt.title('Probability Density')
    plt.hist(rand_vals ** 2)
    plt.show()

    pr.disable()
    # pr.print_stats(sort='time')
    end = time.time()
    print(f"Program took :{end - start} seconds to run")


if __name__ == '__main__':
    main()
