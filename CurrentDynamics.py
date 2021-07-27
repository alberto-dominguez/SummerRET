import math
import numpy as np
# import matplotlib.pyplot as plt

# constants
PI = math.pi
E = 0.1
W = PI / 5
A = 0.1
WORLD_HEIGHT = 10
WORLD_WIDTH = 20
SPACE_SCALE_FACTOR = 10


def a(t):
    return E * math.sin(W * t)


def b(t):
    return 1 - 2 * E * math.sin(W * t)


def f(xx, t):
    return a(t) * xx**2 + b(t) * xx


def u(xx, yy, t):
    return -PI * A * math.sin(PI*f(xx, t)) * math.cos(PI * yy)


def v(xx, yy, t):
    return PI * A * math.cos(PI*f(xx, t)) * math.sin(PI * yy) * (2 * a(t) * xx + b(t))


def double_gyre(t):
    # create double gyre velocity field
    # This code is intended to reproduce double gyre stream flow described on the LCS website
    # https://shaddenlab.berkeley.edu/uploads/LCS-tutorial/examples.html#Sec7.1
    CURR_X = np.zeros((WORLD_HEIGHT, WORLD_WIDTH), dtype=float)
    CURR_Y = np.zeros((WORLD_HEIGHT, WORLD_WIDTH), dtype=float)
    for ii in range(0, WORLD_HEIGHT):
        x = ii / SPACE_SCALE_FACTOR
        for jj in range(0, WORLD_WIDTH):
            y = jj / SPACE_SCALE_FACTOR
            CURR_X[ii][jj] = v(x, y, t)
            CURR_Y[ii][jj] = u(x, y, t)
#    xv, yv = np.meshgrid(np.linspace(0, 2, WORLD_WIDTH), np.linspace(0, 1, WORLD_HEIGHT))
#    plt.quiver(xv, yv, CURR_X, CURR_Y)
#    plt.show()
    return CURR_X, CURR_Y
