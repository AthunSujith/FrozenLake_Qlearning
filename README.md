# ❄️ FrozenLake Q-Learning Solver

Successfully navigate the treacherous frozen lake using Reinforcement Learning! This project implements a **Q-Learning** agent to solve the OpenAI Gym `FrozenLake-v1` environment, even with the "slippery" setting enabled.

![FrozenLake](https://gymnasium.farama.org/_images/frozen_lake.gif)

## 🚀 Overview

The goal of Frozen Lake is to cross a 4x4 grid from the Start (S) to the Goal (G) without falling into any Holes (H). The ice is slippery, so the agent doesn't always move in the intended direction, making it a perfect challenge for Reinforcement Learning.

This project uses the **Bellman Equation** to iteratively update a Q-Table, which the agent uses to make optimal decisions.

## ✨ Key Features

- **Q-Learning Implementation**: A robust training loop with epsilon-greedy exploration.
- **Dynamic Learning**: Hyperparameters tuned for the 4x4 slippery environment.
- **Visual Demonstration**: A script to watch the trained agent navigate the lake in real-time.
- **Model Persistence**: Saves the trained Q-Table for later use.

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/frozenlake-q-learning.git
   cd frozenlake-q-learning
   ```

2. **Install dependencies**:
   Ensure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 How to Use

### 1. Train the Agent
Run the training script to generate the Q-Table. This will run for 200,000 episodes (configurable) to ensure a high success rate.
```bash
python learn.py
```

### 2. Visualize the Results
Watch the trained agent in action! This script loads the saved `q_table.pkl` and renders the environment.
```bash
python visual.py
```

## 🧠 Reinforcement Learning Details

The agent learns using the Q-Learning update rule:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

- **$\alpha$ (Alpha)**: Learning rate (0.8)
- **$\gamma$ (Gamma)**: Discount factor (0.95)
- **$\epsilon$ (Epsilon)**: Exploration rate (starts at 1.0, decays to 0.01)

## 📁 Project Structure

- `learn.py`: Core logic for training the Q-Learning agent.
- `visual.py`: Script for testing and rendering the trained model.
- `Q_learn.ipynb`: Interactive notebook version of the solver.
- `q_table.pkl`: The serialized Q-Table (generated after training).
- `requirements.txt`: List of Python dependencies.

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---
*Developed with ❤️ by Antigravity*
