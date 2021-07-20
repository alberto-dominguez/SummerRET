#####################################################################################################
# Extension of Windy Grid World                                                                     #
#                                                                                                   #
# Revision history:                                                                                 #
# ALD 11-JUL-2021 First version, assuming heterogeneous wind conditions across each row and column  #
# ALD 13-JUL-2021 Created wind matrix initialization and added stochastic gusts                     #
# ALD 16-JUL-2021 Moved gust logic from individual episodes to initial matrix setup (per Paulo)     #
#                 Changed epsilon from a hard-coded constant to a parameter for experimentation     #
# ALD 17-JUL-2021 Added idle action; made cosmetic changes (UDRL->NSEW and wind->current)           #
#                 Separated parameters for epsilon-greedy algorithm and chance of random action     #
# ALD 20-JUL-2021 Added average time step graph, switched axes on graphs for readability            #
#####################################################################################################

import numpy as np
import matplotlib.pyplot as plt

# world height and width (depth will be ignored)
WORLD_HEIGHT = 10
WORLD_WIDTH = 10

# initialize current strength matrices to 0
CURR_X = np.zeros((WORLD_HEIGHT, WORLD_WIDTH), dtype=int)
CURR_Y = np.zeros((WORLD_HEIGHT, WORLD_WIDTH), dtype=int)
# Create an arbitrary current field
# left edge of the world has current to the east and top edge of the world has current to the south
for k in range(0, WORLD_HEIGHT):
    CURR_X[k][0] = 1
for k in range(1, WORLD_HEIGHT):  # start at 1 to avoid placing both CURR_X and CURR_Y at (0,0)
    CURR_Y[0][k] = 1
# all other cells are set randomly to either the cell to the immediate west or the cell to the immediate north
for ii in range(1, WORLD_HEIGHT):
    for jj in range(1, WORLD_WIDTH):
        if np.random.binomial(1, 0.5) == 1:
            CURR_X[ii][jj] = CURR_X[ii][jj-1]
            CURR_Y[ii][jj] = CURR_Y[ii][jj-1]
        else:
            CURR_X[ii][jj] = CURR_X[ii-1][jj]
            CURR_Y[ii][jj] = CURR_Y[ii-1][jj]
# introduce stochastic gusts
for ii in range(0, WORLD_HEIGHT):
    for jj in range(0, WORLD_WIDTH):
        gust = 1
        if np.random.binomial(1, 0.1) == 1:  # 10% probability of a gust
            gust *= 2
        if np.random.binomial(1, 0.1) == 1:  # 10% probability of current changing direction
            gust *= -1
        CURR_X[ii][jj] = CURR_X[ii][jj] * gust
        CURR_Y[ii][jj] = CURR_Y[ii][jj] * gust

# print current grid
print('Current grid:')
for ii in range(0, WORLD_HEIGHT):
    for jj in range(0, WORLD_WIDTH):
        curr = abs(CURR_X[ii][jj] + CURR_Y[ii][jj])  # Works because x and y are mutually exclusive
        print(curr, end='')
        direction = ""
        if CURR_X[ii][jj] > 0:
            direction = "E"
        elif CURR_X[ii][jj] < 0:
            direction = "W"
        elif CURR_Y[ii][jj] > 0:
            direction = "S"
        elif CURR_Y[ii][jj] < 0:
            direction = "N"
        print(direction, end='')
        print(" ", end='')
    print()

# possible actions
IDLE = 0
MOVE_NORTH = 1
MOVE_SOUTH = 2
MOVE_WEST = 3
MOVE_EAST = 4

# Sarsa step size.  Sarsa algorithm is explained on p. 129 of the textbook.
ALPHA = 0.5

# reward for each step (what this means is we want to minimize the number of steps)
# if we introduce diagonal steps, we'll need a separate reward = -sqrt(2) for them - ALD
REWARD = -1.0

# Start and goal positions of the agent
START = [0, 0]
GOAL = [5, 9]
ACTIONS = [IDLE, MOVE_NORTH, MOVE_SOUTH, MOVE_WEST, MOVE_EAST]


# This function defines how the agent moves on the grid.
def step(state, action, gremlin):

    i, j = state

    # gremlin represents the probability that action is random
    # noise/uncertainty to account for unexpected disturbances, modeling error, and/or unknown dynamics
    if np.random.binomial(1, gremlin) == 1:
        action = np.random.choice(ACTIONS)
    if action == IDLE:
        return [max(min(i + CURR_X[i][j], WORLD_HEIGHT - 1), 0), max(min(j + CURR_Y[i][j], WORLD_WIDTH - 1), 0)]
    if action == MOVE_NORTH:
        return [max(min(i - 1 + CURR_X[i][j], WORLD_HEIGHT - 1), 0), max(min(j + CURR_Y[i][j], WORLD_WIDTH - 1), 0)]
    elif action == MOVE_SOUTH:
        return [max(min(i + 1 + CURR_X[i][j], WORLD_HEIGHT - 1), 0), max(min(j + CURR_Y[i][j], WORLD_WIDTH - 1), 0)]
    elif action == MOVE_WEST:
        return [max(min(i + CURR_X[i][j], WORLD_HEIGHT - 1), 0), max(min(j - 1 + CURR_Y[i][j], WORLD_WIDTH - 1), 0)]
    elif action == MOVE_EAST:
        return [max(min(i + CURR_X[i][j], WORLD_HEIGHT - 1), 0), max(min(j + 1 + CURR_Y[i][j], WORLD_WIDTH - 1), 0)]
    # Per discussion with Paulo, we are ignoring diagonal moves, at least for the moment, possibly always
    else:
        assert False  # This should never happen since all potential actions are accounted for


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
        # Sarsa update
        # For more info about Sarsa Algorithm, please refer to p. 129 of the textbook.
        q_value[state[0], state[1], action] = q_value[state[0], state[1], action] + ALPHA * (
                    REWARD + q_value[next_state[0], next_state[1], next_action] -
                    q_value[state[0], state[1], action])
        state = next_state
        action = next_action
        time += 1
    return time


def figure_6_3(eps, gremlin):

    episode_limit = 1000

    q_value = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 5))
    steps = []
    ep = 0
    while ep < episode_limit:
        steps.append(episode(q_value, eps, gremlin))
        ep += 1
    steps = np.add.accumulate(steps)
    divisor = np.add.accumulate(np.ones(episode_limit))
    average_steps = steps/divisor

    plt.plot(np.arange(1, len(average_steps) + 1), average_steps)
    title = 'epsilon = ' + str(eps) + ' and random dynamics parameter = ' + str(gremlin)
    plt.title(title)
    plt.xlabel('Episodes')
    plt.ylabel('Average Time Steps')
    plt.show()

    plt.plot(np.arange(1, len(steps) + 1), steps)
    title = 'epsilon = ' + str(eps) + ' and random dynamics parameter = ' + str(gremlin)
    plt.title(title)
    plt.xlabel('Episodes')
    plt.ylabel('Aggregate Time Steps')
    plt.show()

    # display the optimal policy
    optimal_policy = []
    for i in range(0, WORLD_HEIGHT):
        optimal_policy.append([])
        for j in range(0, WORLD_WIDTH):
            if [i, j] == GOAL:
                optimal_policy[-1].append('G')
                continue
            bestAction = np.argmax(q_value[i, j, :])
            if bestAction == MOVE_NORTH:
                optimal_policy[-1].append('N')
            elif bestAction == MOVE_SOUTH:
                optimal_policy[-1].append('S')
            elif bestAction == MOVE_WEST:
                optimal_policy[-1].append('W')
            elif bestAction == MOVE_EAST:
                optimal_policy[-1].append('E')
            elif bestAction == IDLE:
                optimal_policy[-1].append('I')
    print('Optimal policy when epsilon equals', eps, 'and the random dynamics parameter equals', gremlin, 'is:')
    for row in optimal_policy:
        print(row)


if __name__ == '__main__':
    # original base scenario
    figure_6_3(0.1, 0.1)
    # Experiment with various values of epsilon in the epsilon-greedy algorithm
    figure_6_3(0.2, 0.1)
    figure_6_3(0.05, 0.1)
    figure_6_3(0, 0.1)
    # Experiment with various values of the noise/uncertainty parameter
    figure_6_3(0.1, 0.2)
    figure_6_3(0.1, 0.05)
    figure_6_3(0.1, 0)
