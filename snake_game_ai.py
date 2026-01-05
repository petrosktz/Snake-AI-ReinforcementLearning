from tkinter import *
import random
import numpy as np

# Constants
GAME_WIDTH = 400
GAME_HEIGHT = 400
SPEED = 0  # milliseconds
SPACE_SIZE = 25
BODY_PARTS = 3
SNAKE_COLOR = "#00FF00"
FOOD_COLOR = "#FF0000"
BACKGROUND_COLOR = "#000000"
GENERATION = 1


class SnakeGameAI:
    def __init__(self):
        self.window = Tk()
        self.window.title("Snake Game AI")
        self.window.resizable(False, False)

        self.generation = 0

        self.label = Label(self.window, text="Score: 0", font=("consolas", 40))
        self.label.pack()
        self.generation_label = Label(
            self.window, text="Generation: 0", font=("consolas", 20)
        )
        self.generation_label.pack()

        self.canvas = Canvas(
            self.window, bg=BACKGROUND_COLOR, height=GAME_HEIGHT, width=GAME_WIDTH
        )
        self.canvas.pack()

        self.window.update()
        self.reset()

    def reset(self):
        self.generation += 1
        self.generation_label.config(text="Generation: {}".format(self.generation))

        self.direction = "down"
        self.score = 0
        self.frame_iteration = 0

        self.canvas.delete(ALL)

        self.snake_coordinates = [[0, 0] for _ in range(BODY_PARTS)]

        self.snake_squares = []
        for x, y in self.snake_coordinates:
            square = self.canvas.create_rectangle(
                x, y, x + SPACE_SIZE, y + SPACE_SIZE, fill=SNAKE_COLOR, tag="snake"
            )
            self.snake_squares.append(square)

        self._place_food()
        self.label.config(text="Score: {}".format(self.score))

    def _place_food(self):
        while True:
            x = random.randint(0, (GAME_WIDTH // SPACE_SIZE) - 1) * SPACE_SIZE
            y = random.randint(0, (GAME_HEIGHT // SPACE_SIZE) - 1) * SPACE_SIZE
            self.food_coordinates = [x, y]

            if self.food_coordinates not in self.snake_coordinates:
                break
        self.canvas.create_oval(
            x, y, x + SPACE_SIZE, y + SPACE_SIZE, fill=FOOD_COLOR, tag="food"
        )

    def step(self, action, visual=False):
        self.frame_iteration += 1

        # Capture old distance to food (for the distance reward)
        head_x, head_y = self.snake_coordinates[0]
        food_x, food_y = self.food_coordinates
        old_dist = abs(head_x - food_x) + abs(head_y - food_y)

        # Movement
        # Actions: [1,0,0] -> Straight, [0,1,0] -> Right, [0,0,1] -> Left
        clock_wise = ["up", "right", "down", "left"]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [0, 1, 0]):
            new_idx = (idx + 1) % 4
            self.direction = clock_wise[new_idx]  # Turn Right
        elif np.array_equal(action, [0, 0, 1]):
            new_idx = (idx - 1) % 4
            self.direction = clock_wise[new_idx]  # Turn Left

        # Calculate new head position
        x, y = self.snake_coordinates[0]
        if self.direction == "up":
            y -= SPACE_SIZE
        elif self.direction == "down":
            y += SPACE_SIZE
        elif self.direction == "left":
            x -= SPACE_SIZE
        elif self.direction == "right":
            x += SPACE_SIZE

        # Insert new head
        self.snake_coordinates.insert(0, [x, y])
        square = self.canvas.create_rectangle(
            x, y, x + SPACE_SIZE, y + SPACE_SIZE, fill=SNAKE_COLOR
        )
        self.snake_squares.insert(0, square)

        # Initialize Reward and Game State
        reward = 0
        game_over = False

        # Collision Check (Wall or Body)
        if self._is_collision():
            game_over = True
            reward = -20  # Significant penalty for death
        elif self.frame_iteration > 100 * len(self.snake_coordinates):
            # Prevents the snake from looping indefinitely
            game_over = True
            reward = -20
            return reward, game_over, self.score

        # Small penalty for every move to encourage speed
        # reward -= 0.01

        # Distance Reward: Reward moving toward food, penalize moving away
        new_dist = abs(x - food_x) + abs(y - food_y)
        if new_dist < old_dist:
            reward += 0.2  # Getting closer
        else:
            reward -= 0.3  # Moving away (slightly higher penalty to break loops)

        # Adjacency Penalty: Penalize the head for being next to its own body
        neighbors = [
            [x + SPACE_SIZE, y],
            [x - SPACE_SIZE, y],
            [x, y + SPACE_SIZE],
            [x, y - SPACE_SIZE],
        ]
        for neighbor in neighbors:
            # We ignore snake_coordinates[1] (the neck)
            if neighbor in self.snake_coordinates[2:]:
                reward -= 0.15

        # 6. Food Check
        if self.snake_coordinates[0] == self.food_coordinates:
            self.score += 1
            reward = 20  # High reward for success
            self.label.config(text="Score: {}".format(self.score))
            self.canvas.delete("food")
            self._place_food()
        else:
            del self.snake_coordinates[-1]
            self.canvas.delete(self.snake_squares[-1])
            del self.snake_squares[-1]

        # Update the UI
        if visual:
            try:
                self.window.update()
            except TclError:
                pass

        return reward, game_over, self.score

    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.snake_coordinates[0]
        x, y = pt
        if x < 0 or x >= GAME_WIDTH or y < 0 or y >= GAME_HEIGHT:
            return True
        if pt in self.snake_coordinates[1:]:
            return True
        return False


if __name__ == "__main__":
    from agent import Agent

    MAX_LENGTH = (GAME_WIDTH * GAME_HEIGHT) // (SPACE_SIZE * SPACE_SIZE)

    agent = Agent()
    game = SnakeGameAI()
    record = 0

    try:
        while True:
            state_old = agent.get_state(game)
            final_move = agent.get_action(state_old)

            reward, done, score = game.step(final_move, visual=True)
            state_new = agent.get_state(game)

            agent.train_short_memory(state_old, final_move, reward, state_new, done)
            agent.remember(state_old, final_move, reward, state_new, done)

            if done:
                # Check for Ultimate Victory
                if score >= MAX_LENGTH:
                    print(f"Win - Max Length {MAX_LENGTH} reached.")
                    break

                agent.train_long_memory()
                if score > record:
                    record = score
                    agent.model.save()

                print(f"Game: {agent.n_games}, Score: {score}, Record: {record}")

                game.reset()
                agent.n_games += 1
    except TclError:
        print("Game window closed. Training stopped.")
