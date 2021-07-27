#####################################################################################################
# Double Gyre Reinforcement Learning                                                                #
#                                                                                                   #
# Version history:                                                                                  #
# ALD 12-JUL-2021 First version, assuming heterogeneous wind conditions across each row and column  #
# ALD 13-JUL-2021 Created wind matrix initialization and added stochastic gusts                     #
# ALD 16-JUL-2021 Moved gust logic from individual episodes to initial matrix setup (per Paulo)     #
#                 Changed epsilon from a hard-coded constant to a parameter for experimentation     #
# ALD 17-JUL-2021 Added idle action; made cosmetic changes (UDRL->NSEW and wind->current)           #
#                 Separated parameters for epsilon-greedy algorithm and chance of random action     #
# ALD 20-JUL-2021 Added average time step graph, switched axes on graphs for readability            #
# ALD 21-JUL-2021 Corrected code so avg graph and agg graph are generated from same data series     #
# ALD 22-JUL-2021 Added NE, NW, SE, SW actions                                                      #
# ALD 23-JUL-2021 Implemented double gyre velocity field                                            #
# ALD 27-JUL-2021 Separate current dynamics into a separate python file                             #
#                 Introduced time dependence into the current grid                                  #
#####################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import CurrentDynamics as cd

# The SARSA learning rate alpha determines to what extent newly acquired information overrides old information.
# A factor of 0 makes the agent not learn anything; a factor of 1 makes the agent consider only the most recent info.
ALPHA = 0.4

# dimensions
WORLD_HEIGHT = 10
WORLD_WIDTH = 20
SPACE_SCALE_FACTOR = 10
TIME_SCALE_FACTOR = 20

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


# This function defines how the agent moves on the grid.
# gremlin represents the probability that action is random
# noise/uncertainty to account for unexpected disturbances, modeling error, and/or unknown dynamics
def step(state, action, gremlin, time):
    i, j = state
    CURR_X, CURR_Y = cd.double_gyre(time)
    dx = int(CURR_X[i][j] * SPACE_SCALE_FACTOR)
    dy = int(CURR_Y[i][j] * SPACE_SCALE_FACTOR)
    if np.random.binomial(1, gremlin) == 1:
        action = np.random.choice(ACTIONS)
    if action == MOVE_NORTH:
        return [max(min(i - 1 + dx, WORLD_HEIGHT - 1), 0), max(min(j     + dy, WORLD_WIDTH - 1), 0)]
    elif action == MOVE_SOUTH:
        return [max(min(i + 1 + dx, WORLD_HEIGHT - 1), 0), max(min(j     + dy, WORLD_WIDTH - 1), 0)]
    elif action == MOVE_WEST:
        return [max(min(i     + dx, WORLD_HEIGHT - 1), 0), max(min(j - 1 + dy, WORLD_WIDTH - 1), 0)]
    elif action == MOVE_EAST:
        return [max(min(i     + dx, WORLD_HEIGHT - 1), 0), max(min(j + 1 + dy, WORLD_WIDTH - 1), 0)]
    elif action == MOVE_NE:
        return [max(min(i - 1 + dx, WORLD_HEIGHT - 1), 0), max(min(j + 1 + dy, WORLD_WIDTH - 1), 0)]
    elif action == MOVE_NW:
        return [max(min(i - 1 + dx, WORLD_HEIGHT - 1), 0), max(min(j - 1 + dy, WORLD_WIDTH - 1), 0)]
    elif action == MOVE_SE:
        return [max(min(i + 1 + dx, WORLD_HEIGHT - 1), 0), max(min(j + 1 + dy, WORLD_WIDTH - 1), 0)]
    elif action == MOVE_SW:
        return [max(min(i + 1 + dx, WORLD_HEIGHT - 1), 0), max(min(j - 1 + dy, WORLD_WIDTH - 1), 0)]
    else:  # action == IDLE
        return [max(min(i     + dx, WORLD_HEIGHT - 1), 0), max(min(j     + dy, WORLD_WIDTH - 1), 0)]


# play for an episode, return amount of time spent in the episode
def episode(q_value, eps, gremlin):

    # initialize the counter that will track the total time steps in this episode
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
        t = time / TIME_SCALE_FACTOR
        next_state = step(state, action, gremlin, t)
        if np.random.binomial(1, eps) == 1:
            next_action = np.random.choice(ACTIONS)
        else:
            values_ = q_value[next_state[0], next_state[1], :]
            next_action = np.random.choice(
                [action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])
        # Sarsa update - For more info about Sarsa Algorithm, please refer to p. 129 of the textbook.
        # TODO - differentiate reward for Idle (0) vs N, S, E, W (1) vs NE, NW, SE, SW (sqrt 2)
        reward = -1
        q_value[state[0], state[1], action] = q_value[state[0], state[1], action] + ALPHA * (
                    reward + q_value[next_state[0], next_state[1], next_action] -
                    q_value[state[0], state[1], action])
        state = next_state
        action = next_action
        time += 1
    return time


def figure_6_3(eps, gremlin):

    episode_limit = 200

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
            else:  # bestAction == IDLE
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
