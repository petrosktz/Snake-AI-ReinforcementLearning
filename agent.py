import torch
import random
import numpy as np
from collections import deque
from snake_game_ai import SnakeGameAI, SPACE_SIZE, GAME_WIDTH, GAME_HEIGHT
from model import LinearQNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 2000
LR = 0.0001


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()

        self.model = LinearQNet(28, 512, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake_coordinates[0]
        point_l = [head[0] - SPACE_SIZE, head[1]]
        point_r = [head[0] + SPACE_SIZE, head[1]]
        point_u = [head[0], head[1] - SPACE_SIZE]
        point_d = [head[0], head[1] + SPACE_SIZE]

        # Compass directions: N, NE, E, SE, S, SW, W, NW
        # (dx, dy) changes for each direction
        directions = [
            (0, -SPACE_SIZE),  # N
            (SPACE_SIZE, -SPACE_SIZE),  # NE
            (SPACE_SIZE, 0),  # E
            (SPACE_SIZE, SPACE_SIZE),  # SE
            (0, SPACE_SIZE),  # S
            (-SPACE_SIZE, SPACE_SIZE),  # SW
            (-SPACE_SIZE, 0),  # W
            (-SPACE_SIZE, -SPACE_SIZE),  # NW
        ]

        state = []

        for dx, dy in directions:
            distance_wall = 0
            distance_body = 0
            distance_food = 0

            # Use ray to scan in this direction
            x, y = head[0], head[1]
            found_body = False
            found_food = False

            # Keep moving the ray until we hit a wall
            count = 0
            while True:
                x += dx
                y += dy
                count += 1

                # Check for Wall
                if x < 0 or x >= GAME_WIDTH or y < 0 or y >= GAME_HEIGHT:
                    distance_wall = count
                    break

                # Check for Body (first time only)
                if not found_body and [x, y] in game.snake_coordinates[1:]:
                    distance_body = count
                    found_body = True

                # Check for Food (first time only)
                if not found_food and [x, y] == game.food_coordinates:
                    distance_food = count
                    found_food = True

            # Wall Input
            state.append(1.0 / distance_wall)

            # Body Input
            if found_body:
                state.append(1.0 / distance_body)
            else:
                state.append(0)  # No body in this direction

            # Food Input
            if found_food:
                state.append(1.0 / distance_food)
            else:
                state.append(0)  # No food in this direction

        dir_l = game.direction == "left"
        dir_r = game.direction == "right"
        dir_u = game.direction == "up"
        dir_d = game.direction == "down"
        state.extend([dir_l, dir_r, dir_u, dir_d])

        return np.array(state, dtype=float)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Epsilon-greedy
        self.epsilon = max(150 - self.n_games, 5)
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move


# def train():
#     record = 0  # Track the highest score
#     agent = Agent()
#     game = SnakeGameAI()

#     while True:
#         state_old = agent.get_state(game)
#         final_move = agent.get_action(state_old)
#         reward, done, score = game.step(final_move)
#         state_new = agent.get_state(game)

#         agent.train_short_memory(state_old, final_move, reward, state_new, done)
#         agent.remember(state_old, final_move, reward, state_new, done)
#         if done:
#             game.reset()
#             agent.n_games += 1
#             agent.train_long_memory()

#             if score > record:
#                 record = score
#                 agent.model.save()

#             print(f"Game: {agent.n_games}, Score: {score}, Record: {record}")
