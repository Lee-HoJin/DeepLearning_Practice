import gym
from gym.envs.registration import register
import colorama as cr
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

cr.init(autoreset=True)

register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4',
            'is_slippery': False}
)

env = gym.make('FrozenLake-v3')

Q = np.zeros([env.observation_space.n, env.action_space.n])

dis = .99
num_episodes = 2000

rList = []
for i in range(num_episodes) :
    state = env.reset()
    rAll = 0
    done = False

    e = 1. / ((i // 100) + 1)
    if i < 30 :
        print(e)

    # the Q-Table learning algorithm
    while not done :

        ## Random Noise
        # # Choose an action by greedity (with noise) picking from Q-table
        # action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) / (i + 1))

        ## E-Greedy
        if np.random.rand(1) < e:
            action = env.action_space.sample()
        else :
            action = np.argmax(Q[state, :])

        # Get new state and reward from environment
        new_state, reward, done, _ = env.step(action)

        # Update Q-Table with new knowledge using learning ratge
        Q[state, action] = reward + dis * np.max(Q[new_state, :])
    
        rAll += reward
        state = new_state

    rList.append(rAll)


print("Success Rate: " + str(sum(rList)/num_episodes))
print("Final Q-State Values")
print(Q)
plt.bar(range(len(rList)), rList, color="blue")
plt.savefig('./Q-Function_Rewards.png')
