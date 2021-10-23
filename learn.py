import gym
import acrobot2

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
env = make_vec_env("Acrobot2-v0", n_envs=4)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=125000)
model.save("ppo")

del model  # remove to demonstrate saving and loading

model = PPO.load("ppo")

model.set_random_seed(1)
env.seed(1)

obs_list = []

obs = env.reset()
obs_list.append(obs)
while True:
    action, _states = model.predict(obs, determinstic=True)
    obs, rewards, dones, info = env.step(action)
    obs_list.append(obs)


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

xs = []
ys = []
zs = []


for obs in obs_list:
    xs.append(obs[0])
    ys.append(obs[0])
    zs.append(obs[0])

ax = plt.gca(projection="3d")
ax.scatter(xs, ys, zs, c='r', s=100)
ax.plot(xs, ys, zs, color='r')

plt.savefig('sweet_potatoes.png')
