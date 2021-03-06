import gym
import acrobot2
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
env = make_vec_env("Acrobot2-v0", n_envs=4)


model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=200000)
model.save("ppo")
