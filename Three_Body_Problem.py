import cProfile
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
import pylab as py
import time

# Constants
EARTH = 6e24  # Mass of Earth in kg
SUN = 2e30  # Mass of Sun in kg
THIRD_PLANET = 1.9e27  # Mass of third planet
BIG_G = 6.673e-11  # Gravitational Constant
N_DISTANCE = 1.496e11  # Normalizing distance in m (= 1 AU)
NORM_FACTOR = 6e24  # Normalization factor for masses ( so that Earth's mass = 1)
N_TIME = 365 * 24 * 60 * 60.0  # Normalizing time (1 year)
FORCE_UNIT = (BIG_G * NORM_FACTOR ** 2) / N_DISTANCE ** 2  # Unit force
ENERGY_UNIT = FORCE_UNIT * N_DISTANCE  # Energy in Joules
N_BIG_G = (NORM_FACTOR * BIG_G * N_TIME ** 2) / (N_DISTANCE ** 3)
N_EARTH = EARTH / NORM_FACTOR  # Normalized mass of Earth
N_SUN = SUN / NORM_FACTOR  # Normalized mass of Sun
N_THIRD_PLANET = 500 * THIRD_PLANET / NORM_FACTOR  # Normalized mass of third planet/Super size of the third planet
t_i = 0  # initial time = 0
t_f = 120  # final time in years
N = 100 * t_f  # points per year
t = np.linspace(t_i, t_f, N)  # time array from ti to tf with N points
h = t[2] - t[1]  # time step


def main():
    # For timing purposes
    start = time.time()
    pr = cProfile.Profile()
    pr.enable()

    # Animation Method
    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        line3.set_data([], [])
        ttl.set_text('')

        return line1, line2, line3, ttl

    def F_Earth_on_Sun(r):
        """
        Calculates the force the Earth exerts on the Sun wih only a distance parameter
        :param r: Distance between the earth and the sun
        :return: The x and y components of the Gravitational force
        """
        F = np.zeros(2)
        Fmag = N_BIG_G * N_EARTH * N_SUN / (np.linalg.norm(r) + 1e-20) ** 2
        theta = math.atan(np.abs(r[1]) / (np.abs(r[0]) + 1e-20))
        F[0] = Fmag * np.cos(theta)
        F[1] = Fmag * np.sin(theta)
        if r[0] > 0:
            F[0] = -F[0]
        if r[1] > 0:
            F[1] = -F[1]

        return F

    def F_Planet_on_Sun(r):
        """
        Calculates the force a given planetary mass exerts on the Sun wih only a distance parameter
        :param r: Distance between the planet and the sun
        :return: The x and y components of the Gravitational force
        """
        F = np.zeros(2)
        Fmag = N_BIG_G * N_THIRD_PLANET * N_SUN / (np.linalg.norm(r) + 1e-20) ** 2
        theta = math.atan(np.abs(r[1]) / (np.abs(r[0]) + 1e-20))
        F[0] = Fmag * np.cos(theta)
        F[1] = Fmag * np.sin(theta)
        if r[0] > 0:
            F[0] = -F[0]
        if r[1] > 0:
            F[1] = -F[1]

        return F

    def Force_Earth_on_Planet(re, rj):
        """
        Calculates the force Earth exerts on a given planetary mass wih only a distance parameter
        :param r: Distance between the earth and the planet
        :return: The x and y components of the Gravitational force
        """
        r = np.zeros(2)
        F = np.zeros(2)
        r[0] = re[0] - rj[0]
        r[1] = re[1] - rj[1]
        Fmag = N_BIG_G * N_EARTH * N_THIRD_PLANET / (np.linalg.norm(r) + 1e-20) ** 2
        theta = math.atan(np.abs(r[1]) / (np.abs(r[0]) + 1e-20))
        F[0] = Fmag * np.cos(theta)
        F[1] = Fmag * np.sin(theta)
        if r[0] > 0:
            F[0] = -F[0]
        if r[1] > 0:
            F[1] = -F[1]

        return F

    def force(r, planet, ro, vo):
        if planet == 'earth':
            return F_Earth_on_Sun(r) + Force_Earth_on_Planet(r, ro)
        if planet == 'Planet':
            return F_Planet_on_Sun(r) - Force_Earth_on_Planet(r, ro)

    def dr_dt(t, r, v, planet, ro, vo):
        return v

    def dv_dt(t, r, v, planet, ro, vo):
        F = force(r, planet, ro, vo)
        if planet == 'earth':
            y = F / N_EARTH
        if planet == 'Planet':
            y = F / N_THIRD_PLANET
        return y

    def Runge_Kutta(t, r, v, h, planet, ro, vo):
        """
        Runge Kutta 4th order solver
        :param t: Time
        :param r: Position
        :param v: Velocity
        :param h: Time step
        :param planet: Desired planet for the calculation
        :param ro:
        :param vo:
        :return:
        """
        k11 = dr_dt(t, r, v, planet, ro, vo)
        k21 = dv_dt(t, r, v, planet, ro, vo)
        k12 = dr_dt(t + 0.5 * h, r + 0.5 * h * k11, v + 0.5 * h * k21, planet, ro, vo)
        k22 = dv_dt(t + 0.5 * h, r + 0.5 * h * k11, v + 0.5 * h * k21, planet, ro, vo)
        k13 = dr_dt(t + 0.5 * h, r + 0.5 * h * k12, v + 0.5 * h * k22, planet, ro, vo)
        k23 = dv_dt(t + 0.5 * h, r + 0.5 * h * k12, v + 0.5 * h * k22, planet, ro, vo)
        k14 = dr_dt(t + h, r + h * k13, v + h * k23, planet, ro, vo)
        k24 = dv_dt(t + h, r + h * k13, v + h * k23, planet, ro, vo)
        y0 = r + h * (k11 + 2. * k12 + 2. * k13 + k14) / 6.
        y1 = v + h * (k21 + 2. * k22 + 2. * k23 + k24) / 6.
        z = np.zeros([2, 2])
        z = [y0, y1]
        return z

    def KineticEnergy(v):
        """
        Calculates the kinetic energy of Earth throughout the problem
        :param v: Earth's velocity
        :return: Kinetic enery over time
        """
        vn = np.linalg.norm(v)
        return 0.5 * N_EARTH * vn ** 2

    def PotentialEnergy(r):
        fmag = np.linalg.norm(F_Earth_on_Sun(r))
        rmag = np.linalg.norm(r)
        return -fmag * rmag

    def AngMomentum(r, v):
        rn = np.linalg.norm(r)
        vn = np.linalg.norm(v)
        r = r / rn
        v = v / vn
        rdotv = r[0] * v[0] + r[1] * v[1]
        theta = math.acos(rdotv)
        return N_EARTH * rn * vn * np.sin(theta)

    def AreaCalc(r1, r2):
        r1n = np.linalg.norm(r1)
        r2n = np.linalg.norm(r2)
        r1 = r1 + 1e-20
        r2 = r2 + 1e-20
        theta1 = math.atan(abs(r1[1] / r1[0]))
        theta2 = math.atan(abs(r2[1] / r2[0]))
        rn = 0.5 * (r1n + r2n)
        del_theta = np.abs(theta1 - theta2)
        return 0.5 * del_theta * rn ** 2

    # Initialization

    kinetic = np.zeros(N)  # Kinetic energy
    potential = np.zeros(N)  # Potential energy
    angular = np.zeros(N)  # Angular momentum
    area_value = np.zeros(N) # Area swept

    # Positional and velocity parameters for all three bodies

    earth_position = np.zeros([N, 2])  # position vector of Earth
    earth_velocity = np.zeros([N, 2])  # velocity vector of Earth
    planet_position = np.zeros([N, 2])  # position vector of third planet
    planet_velocity = np.zeros([N, 2])  # velocity vector of third planet
    sun_position = np.zeros([N, 2])  # position vector of third planet
    sun_velocity = np.zeros([N, 2])  # velocity vector of third planet

    e_position_i = [1496e8 / N_DISTANCE, 0]  # initial position of earth
    p_position_i = [5.2, 0]  # initial position of third planet
    sun_position_i = [2, 2]  # initial position of sun

    e_vel_mag = np.sqrt(N_SUN * N_BIG_G / e_position_i[0])  # Magnitude of Earth's initial velocity
    sun_vel_mag = np.sqrt(N_SUN * N_BIG_G / sun_position_i[0])
    p_vel_j = 13.06e3 * N_TIME / N_DISTANCE  # Magnitude of third planet's initial velocity

    e_vel_i = [0, e_vel_mag * 1.0]  # Initial velocity vector for Earth along y direction
    p_vel_i = [0, p_vel_j * 1.0]  # Initial velocity vector for third planet
    s_vel_i = [0, sun_vel_mag * 1.0]  # Initial velocity vector for third planet

    # Initializing the arrays with initial values.
    t[0] = t_i
    earth_position[0, :] = e_position_i
    earth_velocity[0, :] = e_vel_i
    planet_position[0, :] = p_position_i
    planet_velocity[0, :] = p_vel_i
    sun_position[0, :] = sun_position_i
    sun_velocity[0, :] = s_vel_i

    kinetic[0] = KineticEnergy(earth_velocity[0, :])
    potential[0] = PotentialEnergy(earth_position[0, :])
    angular[0] = AngMomentum(earth_position[0, :], earth_velocity[0, :])
    area_value[0] = 0

    for i in range(0, N - 1):
        [earth_position[i + 1, :], earth_velocity[i + 1, :]] = Runge_Kutta(t[i], earth_position[i, :],
                                                                           earth_velocity[i, :], h, 'earth',
                                                                           planet_position[i, :], planet_velocity[i, :])
        [planet_position[i + 1, :], planet_velocity[i + 1, :]] = Runge_Kutta(t[i], planet_position[i, :],
                                                                             planet_velocity[i, :], h, 'Planet',
                                                                             earth_position[i, :], earth_velocity[i, :])
        [sun_position[i + 1, :], sun_velocity[i + 1, :]] = Runge_Kutta(t[i], sun_position[i, :],
                                                                       sun_velocity[i, :], h, 'Planet',
                                                                       earth_position[i, :], earth_velocity[i, :])
        kinetic[i + 1] = KineticEnergy(earth_velocity[i + 1, :])
        potential[i + 1] = PotentialEnergy(earth_position[i + 1, :])
        angular[i + 1] = AngMomentum(earth_position[i + 1, :], earth_velocity[i + 1, :])
        area_value[i + 1] = area_value[i] + AreaCalc(earth_position[i, :], earth_position[i + 1, :])

    # The mini plots below give out information such as angular momentum, kinetic and potential energy, area swept

    def mini_plot(fig_num, x, y, xl, yl, clr, lbl):
        py.figure(fig_num)
        py.xlabel(xl)
        py.ylabel(yl)
        return py.plot(x, y, clr, linewidth=1.0, label=lbl)

    label_ = 'orbit'
    py.plot(0, 0, 'ro', linewidth=7)
    mini_plot(1, earth_position[:, 0], earth_position[:, 1], r'$x$ position (AU)', r'$y$ position (AU)', 'blue',
              'Earth')
    mini_plot(1, planet_position[:, 0], planet_position[:, 1], r'$x$ position (AU)', r'$y$ position (AU)', 'green',
              'Third Planet')
    py.ylim([-9, 9])

    py.axis('equal')
    mini_plot(2, t, kinetic, r'Time, $t$ (years)',
              r'Kinetic Energy, $KE$ ($\times$' + str("%.*e" % (2, ENERGY_UNIT)) + ' Joule)',
              'blue',
              'KE')
    mini_plot(2, t, potential, r'Time, $t$ (years)',
              r'Potential Energy, $KE$ ($\times$' + str("%.*e" % (2, ENERGY_UNIT)) + ' Joule)', 'red', 'PE')
    mini_plot(2, t, kinetic + potential, r'Time, $t$ (years)', r'Total Energy, $KE$ ($\times$' +
              str("%.*e" % (2, ENERGY_UNIT)) + ' Joule)', 'black', 'Total Energy')
    q = py.legend(loc=0)
    q.draw_frame(False)
    py.ylim([-180, 180])

    mini_plot(3, t, angular, r'Time, $t$ (years)', r'Angular Momentum', 'black', label_)
    py.ylim([4, 8])

    mini_plot(4, t, area_value, r'Time, $t$ (years)', r'Swept Area ($AU^2$)', 'black', label_)

    # Animation function
    def animate(i):
        # Changing these orbits make the graph act in very particular ways
        earth_orbit = 50
        planet_orbit = 200
        sun_orbit = 600
        tm_yr = 'Elapsed time = ' + str(round(t[i], 1)) + ' years'
        ttl.set_text('Elapsed time = ' + str(round(t[i], 1)) + ' years')
        line1.set_data(earth_position[i:max(1, i - earth_orbit):-1, 0],
                       earth_position[i:max(1, i - earth_orbit):-1, 1])
        line2.set_data(planet_position[i:max(1, i - planet_orbit):-1, 0],
                       planet_position[i:max(1, i - planet_orbit):-1, 1])
        line3.set_data(sun_position[i:max(1, i - sun_orbit):-1, 0],
                       sun_position[i:max(1, i - sun_orbit):-1, 1])
        return line1, line2, line3, ttl

    # Animation

    fig, ax = py.subplots()
    ax.axis('square')
    ax.set_xlim((-8.2, 8.2))
    ax.set_ylim((-8.2, 8.2))
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.patch.set_facecolor('black')
    fig.patch.set_alpha(0.)
    fig.tight_layout()

    line1, = ax.plot([], [], 'o-', color='#d2eeff', markersize=8, markeredgecolor="#d2eeff", markerfacecolor='#0077BE',
                     lw=2, markevery=100,)
    line2, = ax.plot([], [], 'o-', color='#e3dccb', markersize=8, markerfacecolor='#f66338', markeredgecolor="#e3dccb",
                     lw=2, markevery=1000)
    line3, = ax.plot([], [], 'o-', color="#FDB813", markersize=8, markerfacecolor="#FDB813", markeredgecolor="#FD7813",
                     lw=2, markevery=1000,)

    ax.plot([1, 2], [-7.2, -7.2], 'r-')
    ax.text(2.1, -7.4, r'1 AU = $1.496 \times 10^8$ km', color='white')

    ax.plot(-6.2, -7.2, 'o', color='#d2eeff', markerfacecolor='#0077BE')
    ax.text(-5.85, -7.4, 'First', color='white')

    ax.plot(-4.3, -7.2, 'o', color='#e3dccb', markersize=8, markerfacecolor='#f66338')
    ax.text(-3.9, -7.4, 'Second', color='white')

    ax.plot(-1.6, -7.2, 'o', markersize=9, markerfacecolor="#FDB813", markeredgecolor="#FD7813")
    ax.text(-1.1, -7.4, 'Third', color='white')
    ttl = ax.text(1.5, 7.3, '', fontsize=9, color='white')

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=4000, interval=5, blit=True)
    plt.show()
    # anim.save('animation.gif', writer="imagemagick", savefig_kwargs=dict(facecolor='#black'))

    pr.disable()
    # pr.print_stats(sort='time')
    end = time.time()
    print(f"Program took :{end - start} seconds to run")


if __name__ == '__main__':
    main()
