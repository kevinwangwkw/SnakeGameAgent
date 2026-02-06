import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class SnakeGameEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, grid_size=10, max_steps_per_food=100):
        super().__init__()
        self.grid_size = grid_size
        self.render_mode = render_mode
        self.max_steps_per_food = max_steps_per_food

        # Actions: 0=up, 1=right, 2=down, 3=left
        self.action_space = spaces.Discrete(4)

        # Observation: 3 channels of grid_size x grid_size
        # Channel 0: snake body, Channel 1: food, Channel 2: snake head
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(3, grid_size, grid_size), dtype=np.float32
        )

        # Direction vectors: up, right, down, left
        self._direction_vectors = [
            np.array([-1, 0]),  # up
            np.array([0, 1]),   # right
            np.array([1, 0]),   # down
            np.array([0, -1]), # left
        ]

        # Opposite directions for preventing 180-degree turns
        self._opposite = {0: 2, 1: 3, 2: 0, 3: 1}

        # Pygame rendering
        self._cell_size = 40
        self._screen_size = self.grid_size * self._cell_size
        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Snake starts in the middle, moving right, length 3
        mid = self.grid_size // 2
        self.snake = [
            np.array([mid, mid]),      # head
            np.array([mid, mid - 1]),  # body
            np.array([mid, mid - 2]), # tail
        ]
        self.direction = 1  # moving right

        # Step counter for truncation (prevents infinite loops)
        self.steps_since_food = 0

        # Place food
        self._place_food()

        observation = self._get_obs()
        return observation, {}

    def _place_food(self):
        """Place food on a random empty square. Returns False if grid is full (win condition)."""
        snake_positions = {tuple(s) for s in self.snake}
        empty = [
            (r, c)
            for r in range(self.grid_size)
            for c in range(self.grid_size)
            if (r, c) not in snake_positions
        ]
        if len(empty) == 0:
            # Snake fills the entire grid - win condition!
            return False
        idx = self.np_random.integers(len(empty))
        self.food = np.array(empty[idx])
        return True

    def _get_obs(self):
        obs = np.zeros((3, self.grid_size, self.grid_size), dtype=np.float32)
        # Channel 0: snake body (including head)
        for segment in self.snake:
            obs[0, segment[0], segment[1]] = 1.0
        # Channel 1: food
        obs[1, self.food[0], self.food[1]] = 1.0
        # Channel 2: head only
        head = self.snake[0]
        obs[2, head[0], head[1]] = 1.0
        return obs

    def step(self, action):
        # Prevent 180-degree turns
        if action == self._opposite[self.direction]:
            action = self.direction
        self.direction = action

        self.steps_since_food += 1

        # Calculate new head position
        head = self.snake[0].copy()
        new_head = head + self._direction_vectors[self.direction]

        # Check wall collision
        if (
            new_head[0] < 0
            or new_head[0] >= self.grid_size
            or new_head[1] < 0
            or new_head[1] >= self.grid_size
        ):
            observation = self._get_obs()
            return observation, -10.0, True, False, {}

        # Check self collision (exclude tail since it will move)
        for segment in self.snake[:-1]:
            if np.array_equal(new_head, segment):
                observation = self._get_obs()
                return observation, -10.0, True, False, {}

        # Check food
        ate_food = np.array_equal(new_head, self.food)

        # Move snake
        self.snake.insert(0, new_head)
        if ate_food:
            reward = 1.0
            self.steps_since_food = 0
            # Try to place new food - if grid is full, the agent won!
            if not self._place_food():
                # Win condition: snake fills the entire grid
                # Give a big bonus reward for winning
                observation = self._get_obs()
                return observation, reward + 100.0, True, False, {"win": True}
        else:
            reward = 0.0
            self.snake.pop()

        # Truncate if too many steps without eating (prevents loops)
        truncated = self.steps_since_food >= self.max_steps_per_food

        observation = self._get_obs()
        return observation, reward, False, truncated, {}

    def render(self):
        if self.render_mode is None:
            return None

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.set_caption("Snake Game")
            self.window = pygame.display.set_mode(
                (self._screen_size, self._screen_size)
            )
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self._screen_size, self._screen_size))
        canvas.fill((0, 0, 0))

        # Draw grid lines (subtle)
        for i in range(self.grid_size + 1):
            pygame.draw.line(
                canvas, (40, 40, 40),
                (i * self._cell_size, 0),
                (i * self._cell_size, self._screen_size),
            )
            pygame.draw.line(
                canvas, (40, 40, 40),
                (0, i * self._cell_size),
                (self._screen_size, i * self._cell_size),
            )

        # Draw snake body
        for i, segment in enumerate(self.snake):
            rect = pygame.Rect(
                segment[1] * self._cell_size,
                segment[0] * self._cell_size,
                self._cell_size,
                self._cell_size,
            )
            if i == 0:
                pygame.draw.rect(canvas, (0, 200, 0), rect)  # head: bright green
            else:
                pygame.draw.rect(canvas, (0, 150, 0), rect)  # body: darker green

        # Draw food
        food_rect = pygame.Rect(
            self.food[1] * self._cell_size,
            self.food[0] * self._cell_size,
            self._cell_size,
            self._cell_size,
        )
        pygame.draw.rect(canvas, (200, 0, 0), food_rect)

        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None