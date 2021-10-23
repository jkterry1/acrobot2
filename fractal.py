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

res = 20

array = np.zeros((res, res)).astype('float32')

steps = np.linspace(-.1, .1, num=res)

for i in range(res):
    for j in range(res):
 
        obs_list = []

        obs = env.reset(steps[i], steps[j])
        k = 0
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            k += 1
            if done:
                break

        array[i, j] = float(k)

im = plt.imshow(model, cmap='plasma')

plt.savefig('hopefully_fractal.png')
