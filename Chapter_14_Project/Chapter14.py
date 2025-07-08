import gym
import numpy as np
import random

# Create the environment
env = gym.make("FrozenLake-v1", is_slippery=True)

# Parameters
num_states = env.observation_space.n
num_actions = env.action_space.n
q_table = np.zeros((num_states, num_actions))

# Hyperparameters
num_episodes = 10000
max_steps_per_episode = 100
learning_rate = 0.1
discount_factor = 0.99
exploration_rate = 1.0
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

# Training loop
for episode in range(num_episodes):
    state = env.reset()[0]  # reset() returns (observation, info)
    done = False

    for step in range(max_steps_per_episode):
        # Exploration-exploitation tradeoff
        if random.uniform(0, 1) < exploration_rate:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[state, :])  # Exploit

        new_state, reward, done, truncated, info = env.step(action)

        # Bellman equation update
        old_value = q_table[state, action]
        next_max = np.max(q_table[new_state, :])
        new_value = old_value + learning_rate * (reward + discount_factor * next_max - old_value)
        q_table[state, action] = new_value

        state = new_state

        if done:
            break

    # Decay exploration rate
    exploration_rate = min_exploration_rate + \
        (1.0 - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)

# Display final Q-table
print("\nTrained Q-table:")
print(np.round(q_table, 3))

