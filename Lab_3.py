import matplotlib.pyplot as plt
import numpy as np
import time
import cProfile

def main ():
    # For timing purposes
    start = time.time()
    pr = cProfile.Profile()
    pr.enable()

    # We define the function first
    def f(x):
        return x * (x - 1)

    x = 1
    y = f(x)
    delta = 10 ** -2
    # Derivative
    df_dx = ((f(x + delta)) - y) / delta
    print(df_dx)

    #Now let's use more values of delta
    d_arr = [-2, -4,-6, -8, -10, -12, -14,-16, -18, -20, -22, -30]
    derivative_matrix = []
    exp_arr = []

    for i in range (len(d_arr)):
        exponent = d_arr[i]
        new_delta = 10**(exponent)
        exp_arr.append(new_delta)
        df_dx2 = ((f(x + new_delta)) - y) / new_delta
        print(df_dx2)
        derivative_matrix.append(df_dx2)

    # I had these printed so I can see what values I was getting
    #print(exp_arr)
    #print(derivative_matrix)
    error_arr = [abs(i-1) for i in derivative_matrix]
    # Plotting time
    plt.loglog(exp_arr, error_arr)
    plt.ylabel('Derivatives')
    plt.xlabel(r'$\delta$ Values')
    plt.show()

    pr.disable()
    #pr.print_stats(sort='time')
    end = time.time()
    print(f"Program for EX 4.3 took :{end - start} seconds to run")


    # 4.4
    # For timing purposes
    start = time.time()
    pr = cProfile.Profile()
    pr.enable()

    #def f_2(x):
        #return np.sqrt(1 - x**2)

    a = -1; b = 1
    N = 1000000 # If this number is too large it will be slow
    h = 2/N
    x_2 = np.linspace(a, b, N)
    #y_2 = f_2(x_2)
    y_2 = np.sqrt(1 - x_2**2)

    Integral = h * sum(y_2)
    d = Integral - (np.pi/2)

    print ('The numerical value of the integral is: I = {}'.format(Integral))
    print('The delta value is  = {}'.format(d))
    pr.disable()
    #pr.print_stats(sort='time')
    end = time.time()
    print(f"Program for EX 4.4 took :{end - start} seconds to run")
if __name__ == '__main__':
    main()
