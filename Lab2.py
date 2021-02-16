import matplotlib.pyplot as plt
import numpy as np


def main():

    # Creating an array of zeros for both x and y
    x = np.empty(0)
    y = np.empty(0)
    # The actual deltoid curve loop
    pi = np.pi
    Theta = 0
    while Theta < 2 * pi:
        x = np.append(x, 2 * (np.cos(Theta)) + (np.cos(2 * Theta)))
        y = np.append(y, 2 * (np.sin(Theta)) - (np.sin(2 * Theta)))
        Theta += 0.005

    # We need another theta for this graph
    theta = np.radians(np.linspace(0, 360 * 5, 1000))
    r = theta ** 2

    # Standard equations
    x_2 = r * np.cos(theta)
    y_2 = r * np.sin(theta)

    # This is Fey's function
    def fey(angle):
        return np.exp(np.cos(angle)) - 2 * np.cos(4 * angle) + (np.sin(angle / 12)) ** 5

    # Let's break into x and y components

    def x_comp(fey, angle):
        return fey * np.cos(angle)

    def y_comp(fey, angle):
        return fey * np.sin(angle)

    angles = np.radians(np.linspace(0, 4320, 2000))
    xC = x_comp(fey(angles), angles)
    yC = y_comp(fey(angles), angles)

    fig, axs = plt.subplots(3, figsize=[12, 24])
    axs[0].plot(x, y)
    axs[0].set_title('Deltoid Curve')
    axs[1].plot(x_2, y_2)
    axs[1].set_title('Galilean Curve')
    axs[2].plot(xC, yC)
    axs[2].set_title("Fey's Function")
    plt.show()


if __name__ == '__main__':
    main()
