# Autonomous Snake AI: Ray-Casting & Deep Q-Learning

An advanced Reinforcement Learning implementation where an agent learns to play Snake from scratch using **Deep Q-Networks (DQN)**. After training for over 2,700 generations, this model achieved a record score of **66**.

## 🧠 The AI Architecture

The agent makes decisions based on a **3-layer Deep Neural Network** with **512 hidden neurons**. It interprets the world through high-dimensional sensory data rather than raw pixels.

### 📡 Sensory Input (28-Dimensional State)

The agent utilizes **8-directional Ray-Casting** (N, NE, E, SE, S, SW, W, NW). For each direction, the ray returns three specific values:

1. **Distance to Wall:** 1 / distance to the board edge.
2. **Distance to Body:** 1 / distance to its own segments (to avoid self-collision).
3. **Distance to Food:** 1 / distance to the target.

By using the reciprocal ($1/d$), the agent perceives objects that are closer as "stronger" signals, allowing for more precise maneuvering in high-density environments.

### ⚖️ Reward Engineering

The agent's behavior is shaped through a custom reward function designed to balance hunger and safety:

- **Positive Reinforcement (+20):** Successfully eating food.
- **Negative Reinforcement (-20):** Dying (Wall or Self-collision) or Starvation (Looping for too long).
- **Distance Shaping:** Small rewards (+0.2) for moving closer to food; penalties (-0.3) for moving away.
- **Adjacency Penalty (-0.15):** A "personal space" sensor that penalizes the head for being directly next to its own body segments, encouraging the snake to keep its path open and avoid trapping itself.

## 🚀 Performance Milestones

- **The "Zero" Phase (Gens 0-10):** The agent consistently scored 0 as it explored the boundaries and physics of the grid.
- **First Discovery (Gen 11):** The agent successfully ate its first piece of food, setting the initial record to **1**.
- **Rapid Growth (Gens 400-500):** The AI moved from a record of 4 to **25**, showing a clear understanding of the "scent" of the food.

### 📈 The Middle Era (Gens 500–2000)

- **The "Greedy Hunter" Phase (Gens 500–800):** The agent mastered the 8-directional rays to pinpoint food instantly. The record climbed steadily from **25 to 40**, though the agent was still prone to reckless "U-turns."
- **The "Spatial Awareness" Breakthrough (Gens 800–1200):** The model began to prioritize the **Distance to Body** rays. The snake started "looping"—intentionally moving in circles to wait for its tail to clear, a behavior that emerged naturally to avoid adjacency penalties.
- **Refining the Path (Gens 1200–1800):** The record moved from **48 to 58**. As **Epsilon (Exploration)** decayed to its floor, the agent began "zoning," staying toward the center of the map to reduce accidental wall deaths.

### 🏆 Advanced Mastery

- **The High-Density Era (Gens 1800–2100):** As the board became increasingly crowded, the agent reached a **Record of 65 at Game 2090**.
- **Ultimate Record (Gen 2174):** The model achieved its peak performance with a **record score of 66**. At this level, the AI effectively manages its own body as a moving obstacle.

## 🛠️ Tech Stack

- **Python:** Core logic and UI (Tkinter).
- **PyTorch:** Neural network construction and backpropagation.
- **NumPy:** Mathematical state processing.

## 📂 Project Structure

- `agent.py`: Manages the Epsilon-Greedy strategy, memory buffer, and state extraction.
- `model.py`: Contains the `LinearQNet` architecture and the `QTrainer` logic.
- `snake_game_ai.py`: The custom game environment, collision detection, and reward logic.

## ⚙️ Hyperparameters

```python
BATCH_SIZE = 2000
LEARNING_RATE = 0.0001
GAMMA (Discount Rate) = 0.9
MAX_MEMORY = 100,000
EPSILON_DECAY = max(150 - n_games, 5) # 2.5% minimum randomness
```
