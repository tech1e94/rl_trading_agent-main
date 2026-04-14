from stable_baselines3 import PPO
import numpy as np

model = PPO.load("models/saved/baseline_agent.zip")

# FIXED shape
state = np.zeros((1, 25))

action, _ = model.predict(state)

print("Action:", action)