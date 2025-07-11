import gym
import numpy as np
from gym.wrappers import RecordVideo

# Create and wrap the environment to record video
env = gym.make("CartPole-v1", render_mode="rgb_array")
env = RecordVideo(env, video_folder="./cartpole_videos", episode_trigger=lambda e: True)

# Run one episode and save the video
for episode in range(1):
    state, _ = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

env.close()
print("🎥 Video saved in ./cartpole_videos")
