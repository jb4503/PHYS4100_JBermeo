import time
import cProfile
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Constants
AIR_DENSITY = 1.22  # kg/m3
DRAG_COEFFICIENT = 0.47
LITTLE_G = 9.81  # m/s2


def multiplier(mass, radius=1):
    """
    :param mass: Takes the mass of an object and multiplies it by all the other constants
    :param radius: specify the radius of the object, set to 1m as default
    :return: pi*R^2*rho*C/2m  needed for both x and y
    """
    c = np.pi * radius ** 2 * AIR_DENSITY * DRAG_COEFFICIENT / 2
    return c / mass


def F(r, t, mass=1, radius=1):
    """
    :param r: position, in array form 
    :param t: range of points in time
    :param mass: mass of the object, set to 1kg by default
    :param radius: radius of the object, set to 1m by default
    :return: 
    """
    v_x = r[1]
    v_y = r[3]
    v = np.sqrt(v_x**2 + v_y**2)
    return np.array([v_x, -multiplier(mass, radius) * v_x * v,
                     v_y, - LITTLE_G - multiplier(mass, radius) * v_y * v], float)


def trajectory(v_0=1, mass=1, theta=np.pi/2, t_0=0, t_f=2, n=100):
    """
    :param v_0: initial velocity, in meters per second
    :param mass: Mass of the object in kg, set to 1kg as default
    :param theta: Angle of the trajectory's shot, set to pi/2 as default
    :param t_0: initial time, in seconds, default as 0
    :param t_f: final time, in seconds, default as 2
    :param n: Number of points
    :return: array with x and y coordinates of the trajectory
    """
    h = (t_f - t_0)/n
    # t_points = np.arange(t_0, t_f, h)
    t = t_0
    x_points = []
    y_points = []
    r = np.array([0, v_0 * np.cos(theta), 0, v_0 * np.sin(theta)], float)

    while r[2] >= 0 and t < t_f:
        x_points.append(r[0])
        y_points.append(r[2])
        k1 = h * F(r, t, mass)
        k2 = h * F(r + 0.5 * k1, t + 0.5 * h, mass)
        k3 = h * F(r + 0.5 * k2, t + 0.5 * h, mass)
        k4 = h * F(r + k3, t + h, mass)
        r += (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t = t + h
    return np.array(x_points, float), np.array(y_points, float)


def main():
    # For timing purposes
    start = time.time()
    pr = cProfile.Profile()
    pr.enable()

    # Initial parameters
    cannonball_radius = 0.08  # meters
    cannonball_mass = 1  # kg
    theta_0 = 30 * np.pi / 180  # in radians
    cannonball_v0 = 100
    t_0 = 0
    t_f = 2
    n = 100

    trajectory_x, trajectory_y = trajectory(cannonball_v0, cannonball_mass, theta_0, t_0, t_f, n)
    dummy_var = len(trajectory_x)
    size_array = list(range(0, dummy_var))
    dict_array = list(range(0, dummy_var*2))
    new_cannonball_mass = 2
    new_cannonball_mass_2 = 3
    trajectory_x2, trajectory_y2 = trajectory(cannonball_v0, new_cannonball_mass, theta_0, t_0, t_f, n)
    trajectory_x3, trajectory_y3 = trajectory(cannonball_v0, new_cannonball_mass_2, theta_0, t_0, t_f, n)
    figure = px.scatter(x=trajectory_x, y=trajectory_y, size=size_array)
    figure.show()

    multiple_figure = go.Figure()
    multiple_figure.add_trace(go.Scatter(x=trajectory_x, y=trajectory_y, mode='markers'
                                         , marker=dict(size=dict_array)))
    multiple_figure.add_trace(go.Scatter(x=trajectory_x2, y=trajectory_y2, mode='markers'
                                         , marker=dict(size=dict_array)))
    multiple_figure.add_trace(go.Scatter(x=trajectory_x3, y=trajectory_y3, mode='markers'
                                         , marker=dict(size=dict_array)))
    multiple_figure.show()

    pr.disable()
    # pr.print_stats(sort='time')
    end = time.time()
    print(f"Program took :{end - start} seconds to run")


if __name__ == '__main__':
    main()
