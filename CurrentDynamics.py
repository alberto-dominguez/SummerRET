import math
import numpy as np

# constants
PI = math.pi
WORLD_HEIGHT = 20
WORLD_WIDTH = 40
SPACE_SCALE_FACTOR = 10

# field parameters
E = 0.25
W = PI / 20
A = 0.1

# numerical differentiation constants
delta = 0.0001
dt = 0.1


def a(t):
    return E * math.sin(W * t)


def b(t):
    return 1 - 2 * E * math.sin(W * t)


def f(x, t):
    return a(t) * x**2 + b(t) * x


def velocity(x, y, t):
    vx = -PI * A * math.sin(PI*f(x, t)) * math.cos(PI * y)
    vy = -PI * A * math.cos(PI*f(x, t)) * math.sin(PI * y) * (2 * a(t) * x + b(t))
    return vx, vy


# create a double gyre velocity field
# This function is intended to reproduce double gyre stream flow described on the LCS website
# https://shaddenlab.berkeley.edu/uploads/LCS-tutorial/examples.html#Sec7.1
def double_gyre(t):
    CURR_X = np.zeros((WORLD_HEIGHT, WORLD_WIDTH), dtype=float)
    CURR_Y = np.zeros((WORLD_HEIGHT, WORLD_WIDTH), dtype=float)
    for i in range(0, WORLD_HEIGHT):
        x = i / SPACE_SCALE_FACTOR
        for j in range(0, WORLD_WIDTH):
            y = j / SPACE_SCALE_FACTOR
            CURR_X[i][j], CURR_Y[i][j] = velocity(x, y, t)
    return CURR_X, CURR_Y
