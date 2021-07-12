#####################################################################################################
# Extension of Windy Grid World                                                                     #
#                                                                                                   #
# Revision history:                                                                                 #
# ALD 11-JUL-2021 First version, assuming heterogeneous wind conditions across each row and column  #
#####################################################################################################

import numpy as np
import matplotlib.pyplot as plt

# world height
WORLD_HEIGHT = 10

# world width
WORLD_WIDTH = 10

# wind strength
# TODO - Convert to a full grid
WIND_X = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
WIND_Y = [1, 1, 1, 0, 0, 0, 0, 0, 0, 1]

# possible actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

# Instead of selecting the action with the highest value (greedy selection), we let the agent explore the set of
# actions and select one of them randomly. The probability of exploration is given by epsilon.
# The complete motivation for epsilon-greedy methods is explained on p. 27-30 of the textbook.
EPSILON = 0.1

# Sarsa step size.  Sarsa algorithm is explained on p. 129 of the textbook.
ALPHA = 0.5

# reward for each step (what this means is we want to minimize the number of steps)
REWARD = -1.0

# Start and goal positions of the agent as shown on p. 130 of the textbook.
START = [3, 0]
GOAL = [3, 7]
ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]


# This function defines how the agent moves on the grid.
def step(state, action):
    i, j = state
    if action == ACTION_UP:
        return [max(i - 1 - WIND_X[j], 0), max(j - WIND_Y[i], 0)]
    elif action == ACTION_DOWN:
        return [max(min(i + 1 - WIND_X[j], WORLD_HEIGHT - 1), 0), max(j - WIND_Y[i], 0)]
    elif action == ACTION_LEFT:
        return [max(i - WIND_X[j], 0), max(j - 1 - WIND_Y[i], 0)]
    elif action == ACTION_RIGHT:
        return [max(i - WIND_X[j], 0), max(min(i + 1 - WIND_Y[i], WORLD_WIDTH - 1), 0)]
    # TODO - Add diagonal actions (need to think about the dynamics/geometry)
    else:
        assert False  # This should never happen


# play for an episode
def episode(q_value):

    # track the total time steps in this episode
    time = 0

    # initialize state
    state = START

    # choose an action based on epsilon-greedy algorithm.
    # Because we chose EPSILON = 0.1, there's a 10% chance that we select an action randomly.
    # This is represented by computing the binomial distribution with n = 1 experiment and probability
    # of occurrence epsilon = 0.1.
    if np.random.binomial(1, EPSILON) == 1:  # ignore PyCharm compiler warning
        action = np.random.choice(ACTIONS)
    else:
        # Most of the times (90%), we select an action greedily.
        # That means we select the action associated with the maximum q_value found among all q_values stored in values_
        # Algorithmically:
        # for each action_ and value_ stored in values_
        #   if the variable value_ is maximum in the list values_
        #       select the action_ associated with that q_value.
        values_ = q_value[state[0], state[1], :]
        action = np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

    # keep going until get to the goal state
    while state != GOAL:
        next_state = step(state, action)
        if np.random.binomial(1, EPSILON) == 1:  # ignore PyCharm compiler warning
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


def figure_6_3():
    q_value = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 4))
    episode_limit = 500

    steps = []
    ep = 0
    while ep < episode_limit:
        steps.append(episode(q_value))
        ep += 1
    steps = np.add.accumulate(steps)

    plt.plot(steps, np.arange(1, len(steps) + 1))
    plt.xlabel('Time steps')
    plt.ylabel('Episodes')
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
            if bestAction == ACTION_UP:
                optimal_policy[-1].append('U')
            elif bestAction == ACTION_DOWN:
                optimal_policy[-1].append('D')
            elif bestAction == ACTION_LEFT:
                optimal_policy[-1].append('L')
            elif bestAction == ACTION_RIGHT:
                optimal_policy[-1].append('R')
    print('Optimal policy is:')
    for row in optimal_policy:
        print(row)


if __name__ == '__main__':
    figure_6_3()
