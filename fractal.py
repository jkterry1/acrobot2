import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from stable_baselines3 import PPO
import gym
import acrobot2
import numpy as np
import time
from tqdm import tqdm

env = gym.make("Acrobot2-v0")
env.seed(42)

model = PPO.load("ppo")

res = 50

array = np.zeros((res, res)).astype('float32')

steps = np.linspace(theta1=-.1, theta2=.1, num=res)

start = time.time()

for i in tqdm(range(res)):
    for j in range(res):
 
        obs_list = []

        obs = env.reset(theta1=steps[i], theta2=steps[j])
        k = 0
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            k += 1
            if done:
                break

        array[i, j] = float(k)

end = time.time()

print(end-start)

im = plt.imshow(array, cmap='plasma')

plt.savefig('hopefully_fractal.png')
