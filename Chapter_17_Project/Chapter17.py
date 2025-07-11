import gym

# Create the environment
env = gym.make("CartPole-v1", render_mode='human')  # Set render_mode='human' if you want to see the animation

# Number of episodes to run
num_episodes = 5

for episode in range(num_episodes):
    print(f"\nEpisode {episode + 1}")

    # Reset environment and unpack the observation (Gym 0.26+ returns a tuple)
    state, _ = env.reset()
    done = False
    step = 0

    while not done:
        # Randomly choose an action
        action = env.action_space.sample()

        # Apply the action to the environment
        next_state, reward, done, truncated, info = env.step(action)

        # Print step information
        print(f"  Step {step}: Reward = {reward}")

        # Update state
        state = next_state
        step += 1

# Close the environment when done
env.close()
