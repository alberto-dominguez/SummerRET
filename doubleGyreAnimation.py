import pylab as plt
import numpy as num
import matplotlib.animation as animation
import CurrentDynamics as cd

# numerical differentiation parameters
delta = 0.0001
dt = 0.1

# number of particles
N = 2

# defining particle colors (red, yellow)
col = ['r', 'y']


# wave function that defines the characteristics of double gyre
def phi(x, y, t):
    temp = cd.A * num.sin(cd.PI * cd.f(x, t)) * num.sin(cd.PI * y)
    return temp


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
    return num.array([-1 * vx, -1 * vy], float)


# make a 2D mesh grid of size 40*20
X, Y = plt.meshgrid(num.arange(0, 2, 1 / cd.WORLD_HEIGHT), num.arange(0, 1, 1 / cd.WORLD_HEIGHT))
Vx, Vy = velocity(X, Y, 0.1)

# vector arrows
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
Q = ax.quiver(X, Y, Vx, Vy, scale=10)

# initialize array of particles
C = num.empty([N], plt.Circle)
for i in range(0, N):
    C[i] = plt.Circle((-1, -1), radius=0.02, fc=col[i])

# Defining particle initial position
R = num.empty([N, 2], float)
R[0][0] = 1
R[0][1] = 0.5
C[0].center = (R[0][0], R[0][1])
ax.add_patch(C[0])
R[1][0] = 1
R[1][1] = 0.52
C[1].center = (R[1][0], R[1][1])
ax.add_patch(C[1])


# animation for particle moving along the vector field
def animate(num, Q, X, Y, C, R, N):
    t = num / 1
    dt = 1 / 100
    Vx, Vy = velocity(X, Y, t)
    Q.set_UVC(Vx, Vy)
    # update particles' positions using Runge–Kutta method (4th order) to solve ODEs
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


if __name__ == '__main__':
    ani = animation.FuncAnimation(fig, animate, fargs=(Q, X, Y, C, R, N), interval=100, blit=False)
    plt.title("Chaotic behavior: Particles with slightly different starting points end up in different regions")
    plt.ylabel('y')
    plt.xlabel('x')
    plt.show()
