# 📦 Import packages
import numpy as np
import gym
import random
import time

# ✅ Create environment with visual rendering for testing later
env = gym.make("FrozenLake-v1", is_slippery=True)

# 🧠 Initialize Q-table (16 states × 4 actions)
state_space = env.observation_space.n
action_space = env.action_space.n
q_table = np.zeros((state_space, action_space))

# ⚙️ Hyperparameters
alpha = 0.8           # Learning rate
gamma = 0.95          # Discount factor
epsilon = 1.0         # Initial exploration rate
epsilon_decay = 0.995 # Decay after each episode
min_epsilon = 0.01
episodes = 200000
max_steps = 100

# 📊 Track rewards for stats
rewards = []

# 🔁 Training loop
for episode in range(episodes):
    state = env.reset()[0]
    total_rewards = 0

    for step in range(max_steps):
        # 🎲 Epsilon-greedy strategy
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        # 🧮 Take action → observe result
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # 🔄 Bellman update
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        q_table[state, action] = old_value + alpha * (reward + gamma * next_max - old_value)

        # Move to next state
        state = next_state
        total_rewards += reward

        if done:
            break

    # 📉 Decay epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    rewards.append(total_rewards)

    # Print every 100 episodes
    if (episode + 1) % 100 == 0:
        avg = np.mean(rewards[-100:])
        print(f"Episode {episode + 1}, Avg Reward (last 100): {avg:.2f}, Epsilon: {epsilon:.3f}")

# ✅ Training complete
print("\n✅ Training completed!\n")

import pickle

# Save Q-table
with open("q_table.pkl", "wb") as f:
    pickle.dump(q_table, f)

print("✅ Q-table saved to q_table.pkl")
