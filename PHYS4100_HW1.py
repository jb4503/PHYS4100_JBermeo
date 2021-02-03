import math
import argparse
# A ball is dropped from a tower of height h with initial velocity zero.
# Write a program that asks the user to enter the height in meters of the tower
# and then calculates and prints the time the ball takes until it hits the ground,
# ignoring air resistance. Use your program to calculate the time for a ball dropped from a 100 m high tower.


# Initial Velocity
v_0 = 0;
# Ask user for tower height:
h = int(input("Enter the tower's height in meters : "))
while h <= 0:
    print("Height can't be negative!")
    h = int(input("Enter the tower's height in meters : "))
deltaY= - h
# Don't fprget about little g
g = 9.8;
# acceleration
a = -g

# Now let's calculate the time it takes to hit the ground
# We can use deltaY = V_0*t + 1/2 a t^2
# Since V_0 is 0, deltaY becomes 1/2 a*t^2

t = str(round(math.sqrt( (2*deltaY)/a), 2))

print ('It would take the ball '+t+' seconds to hit the ground')


