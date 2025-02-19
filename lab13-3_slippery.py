import gym
from gym.envs.registration import register
import colorama as cr
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


cr.init(autoreset=True)

env = gym.make('FrozenLake-v0')
env.reset()
env.render()

Q = np.zeros([env.observation_space.n, env.action_space.n])

learning_rate = .85
dis = .99
num_episodes = 2000

rList = []
for i in range(num_episodes) :
    state = env.reset()
    rAll = 0
    done = False

    # the Q-Table learning algorithm
    while not done :
        # Choose an action by greedity (with noise) picking from Q-table
        action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) / (i + 1))

        # Get new state and reward from environment
        new_state, reward, done, _ = env.step(action)

        # Update Q-Table with new knowledge using learning ratge
        Q[state, action] = (1 - learning_rate) * Q[state, action] \
            + learning_rate * (reward + dis * np.max(Q[new_state, :]))
        
        rAll += reward
        state = new_state

    rList.append(rAll)


print("Score over time: " + str(sum(rList)/num_episodes))
print("Final Q-State Values")
print(Q)
plt.bar(range(len(rList)), rList, color="blue")
plt.savefig('./Q-Function_Rewards.png')
