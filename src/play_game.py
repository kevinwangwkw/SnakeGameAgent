import pygame
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from snake_env import SnakeGameEnv

def main():
    env = SnakeGameEnv(render_mode="human")
    obs, info = env.reset()
    env.render()

    # Map pygame keys to action indices: 0=up, 1=right, 2=down, 3=left
    key_to_action = {
        pygame.K_UP: 0,
        pygame.K_RIGHT: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_w: 0,
        pygame.K_d: 1,
        pygame.K_s: 2,
        pygame.K_a: 3,
    }

    score = 0
    game_over = False
    action = 1  # default: moving right

    print("=== Snake Game ===")
    print("Arrow keys or WASD to move")
    print("R to restart, ESC/Q to quit")
    print(f"Score: {score}")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r and game_over:
                    obs, info = env.reset()
                    env.render()
                    score = 0
                    game_over = False
                    action = 1
                    print(f"\n=== Restarted! Score: {score} ===")
                elif event.key in key_to_action and not game_over:
                    action = key_to_action[event.key]

        if not game_over and running:
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()

            if reward > 0:
                score += int(reward)
                print(f"Score: {score}")

            if terminated:
                game_over = True
                print(f"\nGame Over! Final score: {score}")
                print("Press R to restart or ESC/Q to quit")

    env.close()

if __name__ == "__main__":
    main()