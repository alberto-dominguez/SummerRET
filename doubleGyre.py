import pylab
import numpy as np
import matplotlib.animation as animation

# numerical constants
PI = np.pi
SPACE_SCALE_FACTOR = 20

# parameters for double gyre field flow
A = 0.1
eps = 0.25
w = PI / 20

# parameters for numerical work
delta = 0.0001  # for finite difference numerical differentiation algorithm
dt = 0.01       # step size for RK4 algorithm

# number of particles
N = 2

# defining particle colors (red, yellow)
col = ['r', 'y']


def f(x, t):
    return eps * np.sin(w * t) * x ** 2 + (1 - 2 * eps * np.sin(w * t)) * x


# field equation that defines the characteristics of double gyre field
def phi(x, y, t):
    return A * np.sin(PI * f(x, t)) * np.sin(PI * y)


# function to calculate the derivatives of x and y
def velocity(x, y, t):
    vx = (phi(x, y + delta, t) - phi(x, y - delta, t)) / (2 * delta)
    vy = (phi(x - delta, y, t) - phi(x + delta, y, t)) / (2 * delta)
    return -1 * vx, -1 * vy


# function that computes velocity of particle at each point
def update(r, t):
    x = r[0]
    y = r[1]
    vx = (phi(x, y + delta, t) - phi(x, y - delta, t)) / (2 * delta)
    vy = (phi(x - delta, y, t) - phi(x + delta, y, t)) / (2 * delta)
    return np.array([-1 * vx, -1 * vy], float)


# animation for particle moving along the vector field
def animate(t, Q, X, Y, C, R, N):
    Vx, Vy = velocity(X, Y, t)
    Q.set_UVC(Vx, Vy)
    # update particles' positions using Runge–Kutta method (4th order) to solve ODEs
    # Ref: DeVries, A First Course in Computational Physics, p. 215
    for i in range(0, N):
        for j in range(0, 10):
            r = R[i][:]
            k1 = dt * update(r, t)
            k2 = dt * update(r + 0.5 * k1, t + 0.5 * dt)
            k3 = dt * update(r + 0.5 * k2, t + 0.5 * dt)
            k4 = dt * update(r + k3, t + dt)
            R[i][:] += (k1 + 2 * k2 + 2 * k3 + k4) / 6
        C[i].center = (R[i][0], R[i][1])
    return Q, C


if __name__ == "__main__":

    # create a 2D mesh grid with its corresponding vector field
    fig, ax = pylab.subplots(1, 1, figsize=(10, 5))
    X, Y = pylab.meshgrid(np.arange(0, 2, 1 / SPACE_SCALE_FACTOR), np.arange(0, 1, 1 / SPACE_SCALE_FACTOR))
    Vx, Vy = velocity(X, Y, 0.1)
    Q = ax.quiver(X, Y, Vx, Vy, scale=10)

    # initialize array of particles
    C = np.empty([N], pylab.Circle)
    for i in range(0, N):
        C[i] = pylab.Circle((-1, -1), radius=0.02, fc=col[i])

    # define particles' initial positions
    R = np.empty([N, 2], float)
    R[0][0] = 1
    R[0][1] = 0.5
    C[0].center = (R[0][0], R[0][1])
    ax.add_patch(C[0])
    R[1][0] = 1
    R[1][1] = 0.52
    C[1].center = (R[1][0], R[1][1])
    ax.add_patch(C[1])

    # run animation
    ani = animation.FuncAnimation(fig, animate, frames=350, fargs=(Q, X, Y, C, R, N), interval=100, blit=False)
    pylab.title("Chaotic behavior: Particles with slightly different starting points end up in different regions")
    pylab.ylabel('y')
    pylab.xlabel('x')
    ani.save("animation.gif", writer=animation.PillowWriter(fps=30))
    pylab.show()
