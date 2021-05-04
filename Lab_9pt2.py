import time
import cProfile
import matplotlib.pyplot as plt
import numpy as np
from vpython import cylinder, vector, sphere, rate

# Constants
LITTLE_G = 9.81  # in m/s**2
L = 0.1
THETA_0 = 179 * np.pi / 180
OMEGA_0 = 0.0
T_0 = 0.0
T_MAX = 10.0
N = 5000
H = (T_MAX - T_0) / N


def f(r, t):
    theta = r[0]
    omega = r[1]
    f_theta = omega
    f_omega = - (LITTLE_G / L) * np.sin(theta)
    return np.array([f_theta, f_omega], float)


def runge_kuta(minim, maxim, step, r):
    t_points = np.arange(minim, maxim, step)
    new_points = []
    for t in t_points:
        new_points.append(r[0])
        k1 = H * f(r, t)
        k2 = H * f(r + 0.5 * k1, t + 0.5 * H)
        k3 = H * f(r + 0.5 * k2, t + 0.5 * H)
        k4 = H * f(r + k3, t + H)
        r += (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return new_points


def main():
    # For timing purposes
    start = time.time()
    pr = cProfile.Profile()
    pr.enable()

    # Using fourth-order Runge-Kutta
    t_points = np.arange(T_0, T_MAX, H)
    theta_points = []
    r = np.array([THETA_0, OMEGA_0], float)
    for t in t_points:
        theta_points.append(r[0])
        k1 = H * f(r, t)
        k2 = H * f(r + 0.5 * k1, t + 0.5 * H)
        k3 = H * f(r + 0.5 * k2, t + 0.5 * H)
        k4 = H * f(r + k3, t + H)
        r += (k1 + 2 * k2 + 2 * k3 + k4) / 6

    # Plot theta vs t
    plt.plot(t_points, (np.array(theta_points, float) * 180 / np.pi))
    plt.xlabel('t (s)')
    plt.ylabel('theta (degrees)')
    plt.show()

    # make animation
    rod = cylinder(pos=vector(0, 0, 0), axis=vector(L * np.cos(THETA_0 - np.pi / 2), L * np.sin(THETA_0 - np.pi / 2), 0)
                   , radius=L / 40)
    bob = sphere(pos=vector(L * np.cos(THETA_0 - np.pi / 2), L * np.sin(THETA_0 - np.pi / 2), 0), radius=L / 10)
    for theta in theta_points:
        rate(N // 10)
        rod.axis = vector(L * np.cos(theta - np.pi / 2), L * np.sin(theta - np.pi / 2), 0)
        bob.pos = vector(L * np.cos(theta - np.pi / 2), L * np.sin(theta - np.pi / 2), 0)

    # pr.print_stats(sort='time')
    end = time.time()
    print(f"Program took :{end - start} seconds to run")


if __name__ == '__main__':
    main()
