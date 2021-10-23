import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from stable_baselines3 import PPO
import gym
import acrobot2
from gym.wrappers import TimeLimit

env = gym.make("Acrobot2-v0")
env = TimeLimit(env, 500)

model = PPO.load("ppo")

print('Starting loops')

for env_seed in [1, 2, 3]:
    env.seed(env_seed)

    obs_list = []

    obs = env.reset()
    obs_list.append(obs)
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        obs_list.append(obs)
        if done:
            break

    for i in [0, 1, 2, 3]:

        xs = []
        ys = []
        zs = []

        for obs in obs_list:
            xs.append(obs[0-i])
            ys.append(obs[1-i])
            zs.append(obs[2-i])

        ax = plt.gca(projection="3d")
        #ax.scatter(xs, ys, zs, c='r')
        ax.plot(xs, ys, zs, color='r')

        print('Loop saving')

        plt.savefig('./plots/' + str(env_seed) + '_' + str(i) + '.png')
