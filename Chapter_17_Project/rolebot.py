import gym
import time

# Create the CartPole environment
env = gym.make('CartPole-v1', render_mode='human')  # 'human' for visualization

# Run for a few episodes
num_episodes = 5
for episode in range(num_episodes):
    # Reset the environment at the start of each episode
    state = env.reset()

    # Initialize episode variables
    total_reward = 0
    step = 0
    done = False

    # Run until the episode is done
    while not done:
        # Choose a random action (0 = move left, 1 = move right)
        action = env.action_space.sample()

        # Apply the action to the environment
        next_state, reward, done, info, _ = env.step(action)  # Note: newer gym versions return 5 values

        # Print step information
        step += 1
        total_reward += reward
        print(f"Episode: {episode + 1}, Step: {step}, Action: {action}, Reward: {reward}")

        # Update the state
        state = next_state

        # Small delay to make visualization viewable
        time.sleep(0.05)

    print(f"\nEpisode {episode + 1} finished after {step} steps with total reward: {total_reward}\n")

# Close the environment
env.close()