import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from stable_baselines3 import PPO
import gym
import acrobot2

print('a')

env = gym.make("Acrobot2-v0")

model = PPO.load("ppo")

model.set_random_seed(1)
env.seed(1)

obs_list = []

print('b')

obs = env.reset()
obs_list.append(obs)
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    obs_list.append(obs)
    if done:
        break

xs = []
ys = []
zs = []

print('c')

for obs in obs_list:
    xs.append(obs[0])
    ys.append(obs[0])
    zs.append(obs[0])

print('d')

ax = plt.gca(projection="3d")
ax.scatter(xs, ys, zs, c='r', s=100)
ax.plot(xs, ys, zs, color='r')

print('e')

plt.savefig('sweet_potatoes.png')

print('f')
