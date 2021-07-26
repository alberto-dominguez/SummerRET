#####################################################################################################
# Double Gyre Reinforcement Learning                                                                #
#                                                                                                   #
# Commit history:                                                                                   #
# ALD 12-JUL-2021 First version, assuming heterogeneous wind conditions across each row and column  #
# ALD 13-JUL-2021 Created wind matrix initialization and added stochastic gusts                     #
# ALD 16-JUL-2021 Moved gust logic from individual episodes to initial matrix setup (per Paulo)     #
#                 Changed epsilon from a hard-coded constant to a parameter for experimentation     #
# ALD 17-JUL-2021 Added idle action; made cosmetic changes (UDRL->NSEW and wind->current)           #
# ALD 17-JUL-2021 Separated parameters for epsilon-greedy algorithm and chance of random action     #
# ALD 20-JUL-2021 Added average time step graph, switched axes on graphs for readability            #
# ALD 21-JUL-2021 Corrected code so avg graph and agg graph are generated from same data series     #
# ALD 22-JUL-2021 Added NE, NW, SE, SW actions                                                      #
# ALD 23-JUL-2021 Implemented double gyre velocity field                                            #
#####################################################################################################

import math
import numpy as np
import matplotlib.pyplot as plt

# constants
PI = math.pi
E = 0.1
W = PI / 5
A = 0.1

# The SARSA learning rate alpha determines to what extent newly acquired information overrides old information.
# A factor of 0 makes the agent not learn anything; a factor of 1 makes the agent consider only the most recent information.
ALPHA = 0.667

# world height and width (depth will be ignored)
WORLD_HEIGHT = 10
WORLD_WIDTH = 20
SCALE_FACTOR = 10

# possible actions
IDLE = 0
MOVE_NORTH = 1
MOVE_SOUTH = 2
MOVE_EAST = 3
MOVE_WEST = 4
MOVE_NE = 5
MOVE_NW = 6
MOVE_SE = 7
MOVE_SW = 8
ACTIONS = [IDLE, MOVE_NORTH, MOVE_SOUTH, MOVE_EAST, MOVE_WEST, MOVE_NE, MOVE_NW, MOVE_SE, MOVE_SW]

# Start and goal positions of the agent
START = [0, 0]
GOAL = [9, 9]


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


# create double gyre velocity field
# This code is intended to reproduce double gyre stream flow described on the LCS website
# https://shaddenlab.berkeley.edu/uploads/LCS-tutorial/examples.html#Sec7.1
CURR_X = np.zeros((WORLD_HEIGHT, WORLD_WIDTH), dtype=float)
CURR_Y = np.zeros((WORLD_HEIGHT, WORLD_WIDTH), dtype=float)
for ii in range(0, WORLD_HEIGHT):
    x = ii / SCALE_FACTOR
    for jj in range(0, WORLD_WIDTH):
        y = jj / SCALE_FACTOR
        CURR_X[ii][jj] = v(x, y, 0)
        CURR_Y[ii][jj] = u(x, y, 0)
xv, yv = np.meshgrid(np.linspace(0, 2, WORLD_WIDTH), np.linspace(0, 1, WORLD_HEIGHT))
plt.quiver(xv, yv, CURR_X, CURR_Y)
plt.show()


# This function defines how the agent moves on the grid.
# gremlin represents the probability that action is random
# noise/uncertainty to account for unexpected disturbances, modeling error, and/or unknown dynamics
def step(state, action, gremlin):
    i, j = state
    dx = int(CURR_X[i][j] * SCALE_FACTOR)
    dy = int(CURR_Y[i][j] * SCALE_FACTOR)
    if np.random.binomial(1, gremlin) == 1:
        action = np.random.choice(ACTIONS)
    if action == MOVE_NORTH:
        return [max(min(i - 1 + dx, WORLD_HEIGHT - 1), 0), max(min(j + dy, WORLD_WIDTH - 1), 0)]
    elif action == MOVE_SOUTH:
        return [max(min(i + 1 + dx, WORLD_HEIGHT - 1), 0), max(min(j + dy, WORLD_WIDTH - 1), 0)]
    elif action == MOVE_WEST:
        return [max(min(i + dx, WORLD_HEIGHT - 1), 0),     max(min(j - 1 + dy, WORLD_WIDTH - 1), 0)]
    elif action == MOVE_EAST:
        return [max(min(i + dx, WORLD_HEIGHT - 1), 0),     max(min(j + 1 + dy, WORLD_WIDTH - 1), 0)]
    elif action == MOVE_NE:
        return [max(min(i - 1 + dx, WORLD_HEIGHT - 1), 0), max(min(j + 1 + dy, WORLD_WIDTH - 1), 0)]
    elif action == MOVE_NW:
        return [max(min(i - 1 + dx, WORLD_HEIGHT - 1), 0), max(min(j - 1 + dy, WORLD_WIDTH - 1), 0)]
    elif action == MOVE_SE:
        return [max(min(i + 1 + dx, WORLD_HEIGHT - 1), 0), max(min(j + 1 + dy, WORLD_WIDTH - 1), 0)]
    elif action == MOVE_SW:
        return [max(min(i + 1 + dx, WORLD_HEIGHT - 1), 0), max(min(j - 1 + dy, WORLD_WIDTH - 1), 0)]
    else:  # action == IDLE
        return [max(min(i + dx, WORLD_HEIGHT - 1), 0),     max(min(j + dy, WORLD_WIDTH - 1), 0)]


# play for an episode, return amount of time spent in the episode
def episode(q_value, eps, gremlin):

    # track the total time steps in this episode
    time = 0

    # initialize state
    state = START

    # choose an action based on epsilon-greedy algorithm.
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
    while state != GOAL:
        next_state = step(state, action, gremlin)
        if np.random.binomial(1, eps) == 1:
            next_action = np.random.choice(ACTIONS)
        else:
            values_ = q_value[next_state[0], next_state[1], :]
            next_action = np.random.choice(
                [action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])
        # Sarsa update - For more info about Sarsa Algorithm, please refer to p. 129 of the textbook.
        if action == MOVE_NORTH or action == MOVE_SOUTH or action == MOVE_EAST or action == MOVE_WEST:
            reward = -1
        elif action == MOVE_NE or action == MOVE_NW or action == MOVE_SE or action == MOVE_SW:
            reward = -1
        else:  # action == IDLE:
            reward = -1
        q_value[state[0], state[1], action] = q_value[state[0], state[1], action] + ALPHA * (
                    reward + q_value[next_state[0], next_state[1], next_action] -
                    q_value[state[0], state[1], action])
        state = next_state
        action = next_action
        time += 1
    return time


def figure_6_3(eps, gremlin):

    episode_limit = 2000

    q_value = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 9))
    steps = []
    ep = 0
    while ep < episode_limit:
        steps.append(episode(q_value, eps, gremlin))
        ep += 1
    steps = np.add.accumulate(steps)

    plt.figure(1)
    plt.xlabel('Episodes')
    plt.ylabel('Aggregate Time Steps')
    plt.plot(np.arange(1, len(steps) + 1), steps)

    plt.figure(2)
    divisor = np.add.accumulate(np.ones(episode_limit))
    average_steps = steps/divisor
    plt.xlabel('Episodes')
    plt.ylabel('Average Time Steps')
    plt.plot(np.arange(1, len(average_steps) + 1), average_steps)

    # display the optimal policy
    optimal_policy = []
    for i in range(0, WORLD_HEIGHT):
        optimal_policy.append([])
        for j in range(0, WORLD_WIDTH):
            if [i, j] == GOAL:
                optimal_policy[-1].append('G ')
                continue
            bestAction = np.argmax(q_value[i, j, :])
            if bestAction == MOVE_NORTH:
                optimal_policy[-1].append('N ')
            elif bestAction == MOVE_SOUTH:
                optimal_policy[-1].append('S ')
            elif bestAction == MOVE_WEST:
                optimal_policy[-1].append('W ')
            elif bestAction == MOVE_EAST:
                optimal_policy[-1].append('E ')
            elif bestAction == MOVE_NE:
                optimal_policy[-1].append('NE')
            elif bestAction == MOVE_NW:
                optimal_policy[-1].append('NW')
            elif bestAction == MOVE_SE:
                optimal_policy[-1].append('SE')
            elif bestAction == MOVE_SW:
                optimal_policy[-1].append('SW')
            elif bestAction == IDLE:
                optimal_policy[-1].append('I ')
    print('Optimal policy when epsilon equals', eps, 'and the random dynamics parameter equals', gremlin, 'is:')
    for row in optimal_policy:
        print(row)


if __name__ == '__main__':

    # Experiment with various values of epsilon in the epsilon-greedy algorithm
    figure_6_3(0.2,  0.1)
    figure_6_3(0.1,  0.1)
    figure_6_3(0.05, 0.1)
    figure_6_3(0,    0.1)
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
    figure_6_3(0.1, 0.2)
    figure_6_3(0.1, 0.1)
    figure_6_3(0.1, 0.05)
    figure_6_3(0.1, 0)
    leg = ["noise = 0.2", "noise = 0.1", "noise = 0.05", "noise = 0"]
    plt.figure(1)
    plt.legend(leg)
    plt.figure(2)
    plt.legend(leg)
    plt.show()
