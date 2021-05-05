import time
import cProfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants
L = 100
i = 50
j = 50


def main():

    # For timing purposes
    start = time.time()
    pr = cProfile.Profile()
    pr.enable()

    position_x = [i]
    position_y = [j]
    fig = plt.figure(figsize=(20, 20))
    fig.set_dpi(100)
    fig.set_size_inches(7, 6.5)
    ax = plt.axes(xlim=(-10, 110), ylim=(-10, 110))
    particle, = ax.plot(0, 0, color='blue')
    bounds = plt.Rectangle((0, 0), 100, 100, color='0.8')
    ax.add_patch(bounds)
    plt.title('Brownian Motion')

    def init(i):
        i += 1
        np.random.seed()
        a = np.random.randint(1, 4)
        # Move Right
        if a == 1:
            if position_x[-1] == L:
                position_x.append(position_x[-1] - 1)
                position_y.append(position_y[-1])
            else:
                position_x.append(position_x[-1] + 1)
                position_y.append(position_y[-1])
        # Move Up
        elif a == 2:
            if position_y[-1] == L:
                position_y.append(position_y[-1] - 1)
                position_x.append(position_x[-1])
            else:
                position_y.append(position_y[-1] + 1)
                position_x.append(position_x[-1])
        # Move Left
        elif a == 3:
            if position_x[-1] == 0:
                position_x.append(position_x[-1] + 1)
                position_y.append(position_y[-1])
            else:
                position_x.append(position_x[-1] - 1)
                position_y.append(position_y[-1])
        # Move Down
        elif a == 4:
            if position_y[-1] == 0:
                position_y.append(position_y[-1] + 1)
                position_x.append(position_x[-1])
            else:
                position_y.append(position_y[-1] - 1)
                position_x.append(position_x[-1])

        # Save 5 time steps
        x = position_x[-1::-1]
        y = position_y[-1::-1]

        if len(x) >= 5:
            x = x[11:1:-1]
            x.append(position_x[-1])
            y = y[11:1:-1]
            y.append(position_y[-1])

        particle.set_xdata(x)
        particle.set_ydata(y)
        return particle,

    animated = animation.FuncAnimation(fig, init, frames=360, interval=20)
    plt.show()

    pr.disable()
    # pr.print_stats(sort='time')
    end = time.time()
    print(f"Program took :{end - start} seconds to run")


if __name__ == '__main__':
    main()