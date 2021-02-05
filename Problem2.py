import math
import argparse
import sys

# A spaceship travels from Earth in a straight line at relativistic speed v to another planet x light years away.
# Write a program to ask the user for the value of x and the speed v as a fraction of the speed of light c,
# then print out the time in years that the spaceship takes to reach its destination (a) in the rest frame of
# an observer on Earth and (b) as perceived by a passenger on board the ship.
# Use your program to calculate the answers for a planet 10 light years away with v = 0.99c.

# Let's ask for velocity first


def main():
    parser1 = argparse.ArgumentParser(description= "Parses the values for this problem")
    parser1.add_argument('x', type=float, help="Distance traveled in light years: ")
    parser1.add_argument('v', type=float, help="Speed of the spaceship as a fraction of the speed of light, c: ")
    args = parser1.parse_args()

    v = args.v
    while v >= 1 or v <= 0:
        print("That's not a valid value for the velocity!")
        v = float(input("Enter the velocity as a fraction of the speed of light, c: "))
    x = args.x
    while x <= 0:
        print("That's not a valid value for the distance!")
        x = float(input("Enter the distance in light years: "))
    # Equation for dilation
    gamma = 1 / (math.sqrt(1 - v ** 2))
    # Distance
    x_c = x / gamma
    # Time
    time_earth = round((x / v), 2)
    time_spaceship = round((x_c / v), 2)
    print("The time in Earth's frame is {} years".format(str(time_earth)))
    print("The time in spaceship's frame is {} years".format(str(time_spaceship)))


if __name__ == '__main__':
    main()


