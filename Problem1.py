#!/usr/bin/env python3
import math
import sys
# A ball is dropped from a tower of height h with initial velocity zero.
# Write a program that asks the user to enter the height in meters of the tower
# and then calculates and prints the time the ball takes until it hits the ground,
# ignoring air resistance. Use your program to calculate the time for a ball dropped from a 100 m high tower.


def main():
    # Initial Velocity v_0 = 0
    # Ask user for tower height:
    if len(sys.argv) < 2:
        print("Please provide a height value")
    else:
        def is_number(string):
            try:
                float(string)
                return True
            except ValueError:
                return False

        for arg in sys.argv[1]:
            if not is_number(arg):
                sys.exit("You must enter a positive integer value. Try again.")

        height = sys.argv[1]
        h = float(height)
        delta_y = - h
        # Don't forget about little g
        g = 9.8
        # acceleration
        a = -g

        # Now let's calculate the time it takes to hit the ground
        # We can use deltaY = V_0*t + 1/2 a t^2
        # Since V_0 is 0, deltaY becomes 1/2 a*t^2

        t = str(round(math.sqrt((2 * delta_y) / a), 2))
        print('It would take the ball ' + t + ' seconds to hit the ground')


if __name__ == '__main__':
    main()
