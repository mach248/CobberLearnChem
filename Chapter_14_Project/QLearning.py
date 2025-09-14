# Import necessary libraries
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import clear_output
import time

# Create the FrozenLake environment
env = gym.make('FrozenLake-v1', render_mode=None)

# Print observation and action space information
print("Observation Space:", env.observation_space)
print("Action Space:", env.action_space)
print("Number of states:", env.observation_space.n)
print("Number of actions:", env.action_space.n)

# Create the Q-table
num_states = env.observation_space.n
num_actions = env.action_space.n
q_table = np.zeros((num_states, num_actions))
print("Q-table shape:", q_table.shape)

# Define hyperparameters
learning_rate = 0.1
discount_factor = 0.99
exploration_rate = 1.0
max_exploration_rate = 1.0
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

# Training parameters
num_episodes = 10000
max_steps_per_episode = 100

# Lists to track progress
rewards_all_episodes = []
success_count = 0

# Training the Q-learning agent
for episode in range(num_episodes):
    # Reset the environment
    state, _ = env.reset()
    done = False
    rewards_current_episode = 0

    for step in range(max_steps_per_episode):
        # Exploration-exploitation trade-off
        exploration_threshold = np.random.uniform(0, 1)

        if exploration_threshold > exploration_rate:
            # Exploit: choose the action with highest Q-value
            action = np.argmax(q_table[state, :])
        else:
            # Explore: choose a random action
            action = env.action_space.sample()

        # Take action and observe the result
        new_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Update Q-table using the Bellman equation
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
                                 learning_rate * (reward + discount_factor * np.max(q_table[new_state, :]))

        # Update state and tracking variables
        state = new_state
        rewards_current_episode += reward

        # Break if the episode is done
        if done:
            if reward == 1:  # Successfully reached the goal
                success_count += 1
            break

    # Decay exploration rate
    exploration_rate = min_exploration_rate + \
                       (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)

    # Add current episode reward to list
    rewards_all_episodes.append(rewards_current_episode)

    # Print progress periodically
    if episode % 1000 == 0:
        average_reward = np.mean(rewards_all_episodes[-1000:])
        success_rate = success_count / 1000 * 100 if episode >= 1000 else success_count / (episode + 1) * 100
        print(
            f"Episode: {episode}, Average Reward: {average_reward:.4f}, Success Rate: {success_rate:.2f}%, Exploration Rate: {exploration_rate:.4f}")
        success_count = 0

# Print the final Q-table
print("\nFinal Q-table:")
print(q_table)


# Function to interpret the Q-table as a policy grid
def print_policy(q_table):
    actions = ['←', '↓', '→', '↑']
    policy = np.array([actions[np.argmax(q_table[i])] for i in range(16)])
    policy_grid = policy.reshape(4, 4)

    print("Policy Grid:")
    for row in policy_grid:
        print(' '.join(row))

    return policy_grid


# Display the policy
policy_grid = print_policy(q_table)


# Plot rewards over time
def plot_rewards():
    plt.figure(figsize=(12, 5))

    # Plot episode rewards
    plt.subplot(1, 2, 1)
    plt.plot(rewards_all_episodes)
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    # Plot moving average of rewards
    plt.subplot(1, 2, 2)
    moving_avg = np.convolve(rewards_all_episodes, np.ones(100) / 100, mode='valid')
    plt.plot(moving_avg)
    plt.title('Moving Average of Rewards (100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')

    plt.tight_layout()
    plt.savefig('frozen_lake_learning_progress.png')
    plt.show()


# Plot the learning progress
plot_rewards()


# Test the trained agent
def test_agent(num_test_episodes=100):
    success = 0

    for episode in range(num_test_episodes):
        state, _ = env.reset()
        done = False

        while not done:
            # Choose the best action according to the learned policy
            action = np.argmax(q_table[state, :])

            # Take the action
            new_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = new_state

            if reward == 1:  # Successfully reached the goal
                success += 1
                break

    success_rate = success / num_test_episodes * 100
    print(f"\nTesting Results: {success} successful episodes out of {num_test_episodes}")
    print(f"Success Rate: {success_rate:.2f}%")

    return success_rate


# Test the trained agent
test_success_rate = test_agent()


# Visualize a few episodes with the trained agent
def visualize_episode():
    env_vis = gym.make('FrozenLake-v1', render_mode='human')
    state, _ = env_vis.reset()
    done = False

    while not done:
        # Choose the best action according to the learned policy
        action = np.argmax(q_table[state, :])

        # Take the action
        new_state, reward, terminated, truncated, _ = env_vis.step(action)
        done = terminated or truncated
        state = new_state

        # Slow down visualization
        time.sleep(0.5)

    env_vis.close()

# Uncomment to visualize an episode (if in an environment that supports rendering)
# visualize_episode()