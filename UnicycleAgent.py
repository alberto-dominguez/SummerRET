#####################################################################################################
# Reinforcement Learning Using Unicycle Agent                                                       #
#                                                                                                   #
# Version history:                                                                                  #
# ALD 31-JUL-2021 First version                                                                     #
#####################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import CurrentDynamics as cd

# The SARSA learning rate alpha determines to what extent newly acquired information overrides old information.
# A factor of 0 makes the agent not learn anything; a factor of 1 makes the agent consider only the most recent info.
ALPHA = 0.5

# dimensions
WORLD_HEIGHT = 10
WORLD_WIDTH = 20
SPACE_SCALE_FACTOR = 10
TIME_SCALE_FACTOR = 20

# possible actions
ACTION_SPACE_SIZE = 4
IDLE = 0
MOVE_FWD = 1
ROT_CW = 2
ROT_CCW = 3
ACTIONS = [IDLE, MOVE_FWD, ROT_CW, ROT_CCW]

# possible bearings
NORTH = 0
NE = 1
EAST = 2
SE = 3
SOUTH = 4
SW = 5
WEST = 6
NW = 7

# In nautical navigation the bearing is the cw angle from N; we will start with a N bearing
# Start and goal positions of the agent
START_BEARING = NORTH
START = [0, 0, START_BEARING]
GOAL = [9, 9, START_BEARING]  # TODO - we don't really know what the goal bearing should be


# This function defines how the agent moves on the grid.
# gremlin represents the probability that action is random
# noise/uncertainty to account for unexpected disturbances, modeling error, and/or unknown dynamics
def step(state, action, gremlin, time):
    i, j, bearing = state
    CURR_X, CURR_Y = cd.double_gyre(time)
    dx = int(CURR_X[i][j] * SPACE_SCALE_FACTOR)
    dy = int(CURR_Y[i][j] * SPACE_SCALE_FACTOR)
    if np.random.binomial(1, gremlin) == 1:
        action = np.random.choice(ACTIONS)
    if action == ROT_CW:
        bearing  = (bearing + 1) % 8
        return [max(min(i + dx, WORLD_HEIGHT - 1), 0), max(min(j + dy, WORLD_WIDTH - 1), 0), bearing]
    elif action == ROT_CCW:
        bearing  = (bearing - 1) % 8
        return [max(min(i + dx, WORLD_HEIGHT - 1), 0), max(min(j + dy, WORLD_WIDTH - 1), 0), bearing]
    elif action == IDLE:
        return [max(min(i + dx, WORLD_HEIGHT - 1), 0), max(min(j + dy, WORLD_WIDTH - 1), 0), bearing]
    else:  # action == MOVE_FWD
        if bearing == NORTH:
            return [max(min(i - 1 + dx, WORLD_HEIGHT - 1), 0), max(min(j     + dy, WORLD_WIDTH - 1), 0), bearing]
        elif bearing == NE:
            return [max(min(i - 1 + dx, WORLD_HEIGHT - 1), 0), max(min(j + 1 + dy, WORLD_WIDTH - 1), 0), bearing]
        elif bearing == EAST:
            return [max(min(i + dx, WORLD_HEIGHT - 1), 0),     max(min(j + 1 + dy, WORLD_WIDTH - 1), 0), bearing]
        elif bearing == SE:
            return [max(min(i + 1 + dx, WORLD_HEIGHT - 1), 0), max(min(j + 1 + dy, WORLD_WIDTH - 1), 0), bearing]
        elif bearing == SOUTH:
            return [max(min(i + 1 + dx, WORLD_HEIGHT - 1), 0), max(min(j     + dy, WORLD_WIDTH - 1), 0), bearing]
        elif bearing == SW:
            return [max(min(i + 1 + dx, WORLD_HEIGHT - 1), 0), max(min(j - 1 + dy, WORLD_WIDTH - 1), 0), bearing]
        elif bearing == WEST:
            return [max(min(i     + dx, WORLD_HEIGHT - 1), 0), max(min(j - 1 + dy, WORLD_WIDTH - 1), 0), bearing]
        elif bearing == NW:
            return [max(min(i - 1 + dx, WORLD_HEIGHT - 1), 0), max(min(j - 1 + dy, WORLD_WIDTH - 1), 0), bearing]


# play for an episode, return amount of time spent in the episode
def episode(q_value, eps, gremlin):

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
    while state != GOAL:
        # determine the next state
        t = time / TIME_SCALE_FACTOR
        next_state = step(state, action, gremlin, t)
        # choose the next action based on epsilon-greedy algorithm
        if np.random.binomial(1, eps) == 1:
            next_action = np.random.choice(ACTIONS)
        else:
            values_ = q_value[next_state[0], next_state[1], :]
            next_action = np.random.choice(
                [action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])
        # Sarsa update - For more info about Sarsa algorithm, please refer to p. 129 of the textbook.
        reward = -1
        q_value[state[0], state[1], action] = q_value[state[0], state[1], action] + ALPHA * (
                    reward + q_value[next_state[0], next_state[1], next_action] -
                    q_value[state[0], state[1], action])
        state = next_state
        action = next_action
        time += 1

    # return the total time steps in this episode
    return time


def figure_6_3(eps, gremlin):

    episode_limit = 20

    q_value = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, ACTION_SPACE_SIZE))
    steps = []
    ep = 0
    while ep < episode_limit:
        steps.append(episode(q_value, eps, gremlin))
        ep += 1
#        if ep % 1000 == 0:
        print(".", end="")

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
    optimal_policy = []
    for i in range(0, WORLD_HEIGHT):
        optimal_policy.append([])
        for j in range(0, WORLD_WIDTH):
            if [i, j] == GOAL:
                optimal_policy[-1].append('G ')
                continue
            bestAction = np.argmax(q_value[i, j, :])
            if bestAction == MOVE_FWD:
                optimal_policy[-1].append('F ')
            elif bestAction == ROT_CW:
                optimal_policy[-1].append('+ ')
            elif bestAction == ROT_CCW:
                optimal_policy[-1].append('- ')
            else:  # bestAction == IDLE
                optimal_policy[-1].append('I ')
#    print('Optimal policy when epsilon equals', eps, 'and the random dynamics parameter equals', gremlin, 'is:')
#    for row in optimal_policy:
#        print(row)


if __name__ == '__main__':

    # Experiment with various values of epsilon in the epsilon-greedy algorithm
#    figure_6_3(0.2,  0.1)
#    figure_6_3(0.1,  0.1)
#    figure_6_3(0.05, 0.1)
#    figure_6_3(0,    0.1)
#    leg = ["eps = 0.2", "eps = 0.1", "eps = 0.05", "eps = 0"]
#    plt.figure(1)
#    plt.legend(leg)
#    plt.figure(2)
#    plt.legend(leg)
#    plt.show()

    # reset plot
#    plt.figure(1).clear()
#    plt.figure(2).clear()

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
