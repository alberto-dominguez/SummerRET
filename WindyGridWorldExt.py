#####################################################################################################
# Extension of Windy Grid World                                                                     #
#                                                                                                   #
# Revision history:                                                                                 #
# ALD 11-JUL-2021 First version, assuming heterogeneous wind conditions across each row and column  #
# ALD 13-JUL-2021 Created wind matrix initialization and added stochastic gusts                     #
# ALD 16-JUL-2021 Moved gust logic from individual episodes to initial matrix setup (per Paulo)     #
#                 Changed epsilon from a hard-coded constant to a parameter for experimentation     #
#####################################################################################################

import numpy as np
import matplotlib.pyplot as plt

# world height
WORLD_HEIGHT = 10

# world width
WORLD_WIDTH = 10

# initialize wind strength matrices to 0
# TODO - Change dynamics wording from wind to current, matching our subject domain
WIND_X = np.zeros((WORLD_HEIGHT, WORLD_WIDTH), dtype=int)
WIND_Y = np.zeros((WORLD_HEIGHT, WORLD_WIDTH), dtype=int)
# Created a completely arbitrary wind system.
# left edge has wind to the right (+1) and top edge has wind down (+1)
for k in range(0, WORLD_HEIGHT):
    WIND_X[k][0] = 1
for k in range(1, WORLD_HEIGHT):  # start at 1 to avoid placing both WIND_X and WIND_Y in (0,0)
    WIND_Y[0][k] = 1
# other cells are set randomly to either the cell to the left or the cell above
for ii in range(1, WORLD_HEIGHT):
    for jj in range(1, WORLD_WIDTH):
        if np.random.binomial(1, 0.5) == 1:
            WIND_X[ii][jj] = WIND_X[ii][jj-1]
            WIND_Y[ii][jj] = WIND_Y[ii][jj-1]
        else:
            WIND_X[ii][jj] = WIND_X[ii-1][jj]
            WIND_Y[ii][jj] = WIND_Y[ii-1][jj]
# gusts
for ii in range(0, WORLD_HEIGHT):
    for jj in range(0, WORLD_WIDTH):
        gust = 1
        if np.random.binomial(1, 0.1) == 1:  # 10% probability of a wind gust
            gust = 2
        if np.random.binomial(1, 0.1) == 1:  # 10% probability of wind changing direction
            gust = gust * -1
        WIND_X[ii][jj] = WIND_X[ii][jj] * gust
        WIND_Y[ii][jj] = WIND_Y[ii][jj] * gust

# print wind grid
print('Wind grid:')
for ii in range(0, WORLD_HEIGHT):
    for jj in range(0, WORLD_WIDTH):
        wind = abs(WIND_X[ii][jj] + WIND_Y[ii][jj])  # Works because x and y are mutually exclusive
        print(wind, end='')
        direction = ""
        if WIND_X[ii][jj] > 0:
            direction = "R"
        elif WIND_X[ii][jj] < 0:
            direction = "L"
        elif WIND_Y[ii][jj] > 0:
            direction = "D"
        elif WIND_Y[ii][jj] < 0:
            direction = "U"
        print(direction, end='')
        print(" ", end='')
    print()

# possible actions
# TODO - Rename N,S,E,W
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

# Sarsa step size.  Sarsa algorithm is explained on p. 129 of the textbook.
ALPHA = 0.5

# reward for each step (what this means is we want to minimize the number of steps)
REWARD = -1.0

# Start and goal positions of the agent
START = [0, 0]
GOAL = [5, 9]
ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]


# This function defines how the agent moves on the grid.
# Note that for improved readability I changed the sign convention from the original code. - ALD
def step(state, action, eps):

    i, j = state

    if np.random.binomial(1, eps) == 1:
        action = np.random.choice(ACTIONS)
    if action == ACTION_UP:
        return [max(min(i - 1 + WIND_X[i][j], WORLD_HEIGHT - 1), 0), max(min(j + WIND_Y[i][j], WORLD_WIDTH - 1), 0)]
    elif action == ACTION_DOWN:
        return [max(min(i + 1 + WIND_X[i][j], WORLD_HEIGHT - 1), 0), max(min(j + WIND_Y[i][j], WORLD_WIDTH - 1), 0)]
    elif action == ACTION_LEFT:
        return [max(min(i + WIND_X[i][j], WORLD_HEIGHT - 1), 0), max(min(j - 1 + WIND_Y[i][j], WORLD_WIDTH - 1), 0)]
    elif action == ACTION_RIGHT:
        return [max(min(i + WIND_X[i][j], WORLD_HEIGHT - 1), 0), max(min(j + 1 + WIND_Y[i][j], WORLD_WIDTH - 1), 0)]
    # Per discussion with Paulo, we are ignoring diagonal moves, at least for the moment, possibly always
    # TODO - add IDLE as an action
    else:
        assert False  # This should never happen since all potential actions are accounted for


# play for an episode
def episode(q_value, eps):

    # track the total time steps in this episode
    time = 0

    # initialize state
    state = START

    # choose an action based on epsilon-greedy algorithm.
    if np.random.binomial(1, eps) == 1:
        action = np.random.choice(ACTIONS)
    else:
        # Most of the time (90%), we select an action greedily.
        # We select the action associated with the maximum q_value found among all q_values stored in values_
        # 10% of time, we select an action randomly.
        # Algorithmically:
        # for each action_ and value_ stored in values_
        #   if the variable value_ is maximum in the list values_
        #       select the action_ associated with that q_value.
        values_ = q_value[state[0], state[1], :]
        action = np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

    # keep going until get to the goal state
    while state != GOAL:
        next_state = step(state, action, eps)
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


def figure_6_3(eps):

    episode_limit = 1000

    q_value = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 4))
    steps = []
    ep = 0
    while ep < episode_limit:
        steps.append(episode(q_value, eps))
        ep += 1
    steps = np.add.accumulate(steps)

    plt.plot(steps, np.arange(1, len(steps) + 1))
    plt.title(str(eps))
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
    figure_6_3(0.2)
    figure_6_3(0.1)
    figure_6_3(0.05)
    figure_6_3(0.025)
    figure_6_3(0.012)
    figure_6_3(0)
