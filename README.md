# Autonomous Snake AI: Ray-Casting & Deep Q-Learning

An advanced Reinforcement Learning implementation where an agent learns to play Snake from scratch using **Deep Q-Networks (DQN)**. By Generation 2000, this model has achieved a record score of **66**, demonstrating a sophisticated understanding of spatial navigation and self-preservation.

## 🧠 The AI Architecture

The agent makes decisions based on a **3-layer Deep Neural Network** with **512 hidden neurons**. It doesn't "see" the board pixels; instead, it interprets the world through high-dimensional sensory data.

### 📡 Sensory Input (28-Dimensional State)

The agent utilizes **8-directional Ray-Casting** (N, NE, E, SE, S, SW, W, NW). For each direction, the ray returns:

1. **Distance to Wall:** 1 / distance to the board edge.
2. **Distance to Body:** 1 / distance to its own segments (to avoid self-collision).
3. **Distance to Food:** 1 / distance to the target.

By using the reciprocal ($1/d$), the agent perceives objects that are closer as "stronger" signals, making the learning process more stable.

### ⚖️ Reward Engineering

The agent's behavior is shaped through a custom reward function designed to balance hunger and safety:

- **Positive Reinforcement (+20):** Successfully eating food.
- **Negative Reinforcement (-20):** Dying (Wall or Self-collision) or Starvation (Looping for too long).
- **Distance Shaping:** Small rewards (+0.2) for moving closer to food; penalties (-0.3) for moving away.
- **Adjacency Penalty (-0.15):** A "personal space" sensor that penalizes the head for being directly next to its own body segments, encouraging the snake to keep its path open.

## 🚀 Performance Milestones

- **Gen 0-100:** Random exploration. The snake frequently hits walls while populating its memory.
- **Gen 100-500:** The "Hunter" phase. The snake aggressively pursues food but often traps itself in "U" turns.
- **Gen 1000+:** Refined coiling. The snake learns to utilize the adjacency penalty to avoid tight loops.
- **Gen 2000+:** **Record Score: 66.** The agent can navigate complex body mazes to fill a significant portion of the board.

## 🛠️ Tech Stack

- **Python:** Core logic and UI.
- **PyTorch:** Neural network construction and backpropagation.
- **Tkinter:** Real-time game visualization.
- **NumPy:** Mathematical state processing.

## 📂 Project Structure

- `agent.py`: The brain. Manages the Epsilon-Greedy strategy, memory buffer, and state extraction.
- `model.py`: The neural network architecture (`LinearQNet`) and the `QTrainer` logic.
- `snake_game_ai.py`: The custom environment including physics, collision detection, and reward logic.

## ⚙️ Hyperparameters

```python
BATCH_SIZE = 2000
LR = 0.0001
GAMMA = 0.9
MAX_MEMORY = 100_000
EPSILON_DECAY = max(150 - n_games, 5)
```
