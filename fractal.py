import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from stable_baselines3 import PPO
import gym
import acrobot2
import numpy as np

env = gym.make("Acrobot2-v0")
env.seed(42)

model = PPO.load("ppo")

print('Starting loops')

array = np.zeros((20, 20))

for x in range(-.2, .2, 1000):
    for y in range(-.2, .2, 1000):
 
        obs_list = []

        obs = env.reset(x, y)
        i = 0
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            i += 1
            if done:
                break

        array[x, y] = i

im = plt.imshow(model, cmap='plasma')

plt.savefig('hopefully_fractal.png')
