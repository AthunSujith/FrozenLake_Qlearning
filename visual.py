# ⚠️ Recreate env with rendering (requires pygame)
import numpy as np
import gym
import time
import pickle


# Load Q-table
with open("q_table.pkl", "rb") as f:
    q_table = pickle.load(f)

    
env = gym.make("FrozenLake-v1", render_mode="human", is_slippery=True)

test_episodes = 10
test_max_steps = 100
successes = 0

print("🎮 Testing Trained Agent...\n")

for episode in range(test_episodes):
    state = env.reset()[0]
    done = False
    print(f"Episode {episode + 1}:")

    for step in range(test_max_steps):
        time.sleep(0.5)  # To visually see the agent move
        action = np.argmax(q_table[state])  # Choose best action
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        state = next_state

        if done:
            if reward == 1:
                print("✅ Reached the goal!")
                successes += 1
            else:
                print("❌ Fell into a hole.")
            time.sleep(1.5)
            break

print(f"\n🏁 Agent succeeded in {successes}/{test_episodes} episodes.")
