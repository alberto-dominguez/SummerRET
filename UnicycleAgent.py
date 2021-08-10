import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pylab
from scipy.io import loadmat

# import ROMS data (MATLAB .MAT format)
# Contains UU and VV structures, each with dimensions Time x Depth x Latitude x Longitude
roms = loadmat('UUVV711b.mat')
vx = roms['UU']
vy = roms['VV']

# dimensions
WORLD_HEIGHT = 21
WORLD_WIDTH = 28
TIME_RANGE = 384
N = 1

# possible actions
ACTION_SPACE_SIZE = 6
IDLE = 0
MOVE_FWD = 1
ROT_CW = 2
ROT_CCW = 3
ASCEND = 4
DESCEND = 5
ACTIONS = [IDLE, MOVE_FWD, ROT_CW, ROT_CCW, ASCEND, DESCEND]

# possible bearings
NORTH = 0  #   0
NE = 1     #  45
EAST = 2   #  90
SE = 3     # 135
SOUTH = 4  # 180
SW = 5     # 225
WEST = 6   # 270
NW = 7     # 315

# possible depths
# index 0 1  2  3  4  5  6  7  8  9  10  11
# depth 0 5 10 15 20 30 40 50 60 75 100 125
DEPTH_COUNT = 12

# In nautical navigation the bearing is the cw angle from N; we will start with a N bearing
# Start and goal positions of the agent
# A state is defined as (x, y, depth, bearing)
START_BEARING = NORTH
START = [0, 0, 0, START_BEARING]
GOAL = [9, 9, 0, 0]  # The bearing at the goal is irrelevant


# This function defines how the agent moves on the grid.
# gremlin represents the probability that action is random
# noise/uncertainty to account for unexpected disturbances, modeling error, and/or unknown dynamics
def step(state, action, gremlin, time):

    x, y, depth, bearing = state
    t = time % TIME_RANGE
    CURR_X = vx[t, 0, x, y]
    CURR_Y = vy[t, 0, x, y]
    dx = int(CURR_X)
    dy = int(CURR_Y)

    if np.random.binomial(1, gremlin) == 1:
        action = np.random.choice(ACTIONS)

    if action == DESCEND:
        # descending (going to larger depth) means incrementing the array index
        depth = max(min(depth + 1, DEPTH_COUNT - 1), 0)
        return [max(min(x + dx, WORLD_HEIGHT - 1), 0), max(min(y + dy, WORLD_WIDTH - 1), 0), depth, bearing]
    elif action == ASCEND:
        # ascending (going to smaller depth) means decrementing the array index
        depth = max(min(depth - 1, DEPTH_COUNT - 1), 0)
        return [max(min(x + dx, WORLD_HEIGHT - 1), 0), max(min(y + dy, WORLD_WIDTH - 1), 0), depth, bearing]
    elif action == ROT_CW:
        bearing  = (bearing + 1) % 8
        return [max(min(x + dx, WORLD_HEIGHT - 1), 0), max(min(y + dy, WORLD_WIDTH - 1), 0), depth, bearing]
    elif action == ROT_CCW:
        bearing  = (bearing - 1) % 8
        return [max(min(x + dx, WORLD_HEIGHT - 1), 0), max(min(y + dy, WORLD_WIDTH - 1), 0), depth, bearing]
    elif action == IDLE:
        return [max(min(x + dx, WORLD_HEIGHT - 1), 0), max(min(y + dy, WORLD_WIDTH - 1), 0), depth, bearing]
    else:  # action == MOVE_FWD
        if bearing == NORTH:
            return [max(min(x - 1 + dx, WORLD_HEIGHT - 1), 0), max(min(y     + dy, WORLD_WIDTH - 1), 0), depth, bearing]
        elif bearing == NE:
            return [max(min(x - 1 + dx, WORLD_HEIGHT - 1), 0), max(min(y + 1 + dy, WORLD_WIDTH - 1), 0), depth, bearing]
        elif bearing == EAST:
            return [max(min(x     + dx, WORLD_HEIGHT - 1), 0), max(min(y + 1 + dy, WORLD_WIDTH - 1), 0), depth, bearing]
        elif bearing == SE:
            return [max(min(x + 1 + dx, WORLD_HEIGHT - 1), 0), max(min(y + 1 + dy, WORLD_WIDTH - 1), 0), depth, bearing]
        elif bearing == SOUTH:
            return [max(min(x + 1 + dx, WORLD_HEIGHT - 1), 0), max(min(y     + dy, WORLD_WIDTH - 1), 0), depth, bearing]
        elif bearing == SW:
            return [max(min(x + 1 + dx, WORLD_HEIGHT - 1), 0), max(min(y - 1 + dy, WORLD_WIDTH - 1), 0), depth, bearing]
        elif bearing == WEST:
            return [max(min(x     + dx, WORLD_HEIGHT - 1), 0), max(min(y - 1 + dy, WORLD_WIDTH - 1), 0), depth, bearing]
        elif bearing == NW:
            return [max(min(x - 1 + dx, WORLD_HEIGHT - 1), 0), max(min(y - 1 + dy, WORLD_WIDTH - 1), 0), depth, bearing]


# play for an episode, return amount of time spent in the episode
def episode(q_value, eps, gremlin, alpha):

    # initialize the counter that will track the total time steps in this episode
    time = 0

    # initialize state
    state = START

    # choose an initial action based on epsilon-greedy algorithm
    if np.random.binomial(1, eps) == 1:
        # eps% of the time, we select an action randomly
        action = np.random.choice(ACTIONS)
    else:
        # (1-eps)% of the time, we select an action greedily
        # Select the action associated with the maximum q_value found among all q_values stored in values_
        # Algorithmically:
        # for each action_ and value_ stored in values_
        #   if the variable value_ is maximum in the list values_
        #     select the action_ associated with that q_value.
        values_ = q_value[state[0], state[1], :]
        action = np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

    # keep going until get to the goal state
    while state[0] != GOAL[0] or state[1] != GOAL[1]:  # bearing doesn't matter if we arrived at the goal cell
        # determine the next state
        next_state = step(state, action, gremlin, time)
        # choose the next action based on epsilon-greedy algorithm
        if np.random.binomial(1, eps) == 1:
            next_action = np.random.choice(ACTIONS)
        else:
            values_ = q_value[next_state[0], next_state[1], :]
            next_action = np.random.choice(
                [action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])
        # SARSA update - For more info about SARSA algorithm, please refer to p. 129 of the S&B textbook.
        # TODO - make reward function dependent on action taken to approximate energy expended
        reward = -1
        q_value[state[0], state[1], action] = q_value[state[0], state[1], action] + alpha * (
                    reward + q_value[next_state[0], next_state[1], next_action] -
                    q_value[state[0], state[1], action])
        state = next_state
        action = next_action
        time += 1

    # return the total time steps in this episode
    return time


def figure_6_3(eps, gremlin, alpha):

    episode_limit = 2000

    q_value = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, ACTION_SPACE_SIZE))
    steps = []
    ep = 0
    while ep < episode_limit:
        steps.append(episode(q_value, eps, gremlin, alpha))
        ep += 1

    steps = np.add.accumulate(steps)
    plt.figure(1)
    plt.xlabel('Episodes')
    plt.ylabel('Aggregate Time Steps')
    plt.plot(np.arange(1, len(steps) + 1), steps)

    divisor = np.add.accumulate(np.ones(episode_limit))
    average_steps = steps/divisor
    print(average_steps[episode_limit - 1])
    plt.figure(2)
    plt.xlabel('Episodes')
    plt.ylabel('Average Time Steps')
    plt.plot(np.arange(1, len(average_steps) + 1), average_steps)

    # display the optimal policy
#    optimal_policy = []
#    for i in range(0, WORLD_HEIGHT):
#        optimal_policy.append([])
#        for j in range(0, WORLD_WIDTH):
#            if i == GOAL[0] and j == GOAL[1]:
#                optimal_policy[-1].append('G ')
#                continue
#            bestAction = np.argmax(q_value[i, j, :])
#            if bestAction == MOVE_FWD:
#                optimal_policy[-1].append('F ')
#            elif bestAction == ROT_CW:
#                optimal_policy[-1].append('+ ')
#            elif bestAction == ROT_CCW:
#                optimal_policy[-1].append('- ')
#            else:  # bestAction == IDLE
#                optimal_policy[-1].append('I ')
#    print('Optimal policy when epsilon equals', eps, 'and the random dynamics parameter equals', gremlin, 'is:')
#    for row in optimal_policy:
#        print(row)


# animation of the time-dependent vector field
def animate(t, Q, X, Y, C, R, N):
    VX = vx[t, 0, :, :]
    VY = vy[t, 0, :, :]
    Q.set_UVC(VX, VY)
    return Q


def create_gif(ind, depth):

    # create the 2D mesh grid with its corresponding vector field
    fig, ax = pylab.subplots(1, 1, figsize=(10, 5))
    X, Y = pylab.meshgrid(np.arange(0, WORLD_HEIGHT, 1), np.arange(0, WORLD_WIDTH, 1))
    VX = vx[0, ind, :, :]
    VY = vy[0, ind, :, :]
    Q = ax.quiver(VX, VY)

    # initialize particle (particle won't move, at least initially)
    C = np.empty([1], pylab.Circle)
    C[0] = pylab.Circle((-1, -1), radius=0.02)
    R = np.empty([1, 2], float)
    R[0][0] = 0
    R[0][1] = 0
    C[0].center = (R[0][0], R[0][1])
    ax.add_patch(C[0])

    # run animation
    ani = animation.FuncAnimation(fig, animate, frames=350, fargs=(Q, X, Y, C, R, N), interval=100, blit=False)
    plt.title("ROMS current field at depth = " + str(depth) + " meters as a function of time")
    filename = "animation_roms_" + str(depth) + ".gif"
    ani.save(filename, writer=animation.PillowWriter(fps=30))  # ignore compiler warning; PillowWriter works correctly
    plt.show()


if __name__ == '__main__':

    # Create GIF of current field at various depths
#    depths = [0, 5, 10, 15, 20, 30, 40, 50, 60, 75, 100, 125]
#    for i in range(0, len(depths)):
#        create_gif(i, depths[i])

    # Experiment with various values of epsilon in the epsilon-greedy algorithm
    figure_6_3(0.2,  0.1, 0.5)
    figure_6_3(0.1,  0.1, 0.5)
    figure_6_3(0.05, 0.1, 0.5)
    figure_6_3(0,    0.1, 0.5)
    leg = ["eps = 0.2", "eps = 0.1", "eps = 0.05", "eps = 0"]
    plt.figure(1)
    plt.legend(leg)
    plt.figure(2)
    plt.legend(leg)
    plt.show()

    # reset plot
    plt.figure(1).clear()
    plt.figure(2).clear()

    # Experiment with various values of the noise/uncertainty parameter
    figure_6_3(0.1, 0.2,  0.5)
    figure_6_3(0.1, 0.1,  0.5)
    figure_6_3(0.1, 0.05, 0.5)
    figure_6_3(0.1, 0,    0.5)
    leg = ["noise = 0.2", "noise = 0.1", "noise = 0.05", "noise = 0"]
    plt.figure(1)
    plt.legend(leg)
    plt.figure(2)
    plt.legend(leg)
    plt.show()

    # reset plot
    plt.figure(1).clear()
    plt.figure(2).clear()

    # Experiment with various values of the alpha parameter
    figure_6_3(0.1, 0.1, 0.7)
    figure_6_3(0.1, 0.1, 0.6)
    figure_6_3(0.1, 0.1, 0.5)
    figure_6_3(0.1, 0.1, 0.4)
    figure_6_3(0.1, 0.1, 0.3)
    leg = ["alpha = 0.7", "alpha = 0.6", "alpha = 0.5", "alpha = 0.4", "alpha = 0.3"]
    plt.figure(1)
    plt.legend(leg)
    plt.figure(2)
    plt.legend(leg)
    plt.show()
