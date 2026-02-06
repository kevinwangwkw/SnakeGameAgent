import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

# Add parent directory to path to import snake_env
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from snake_env import SnakeGameEnv

def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """Initialize a layer with orthogonal weights and constant bias."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class CNNAgent(nn.Module):
    """
    CNN-based PPO Agent for the Snake Game.
    Must match the architecture from train_ppo.py exactly.
    """

    def __init__(self, obs_shape=(3, 10, 10), n_actions=4):
        super().__init__()

        # Shared CNN feature extractor
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(obs_shape[0], 32, kernel_size=3, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=3, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, padding=1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate flattened size after convolutions
        with torch.no_grad():
            sample = torch.zeros(1, *obs_shape)
            flat_size = self.network(sample).shape[1]

        # Actor head (policy)
        self.actor = nn.Sequential(
            layer_init(nn.Linear(flat_size, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, n_actions), std=0.01),
        )

        # Critic head (value function)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(flat_size, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )

    def get_action(self, x: torch.Tensor, deterministic: bool = False) -> int:
        """Get action from observation."""
        features = self.network(x)
        logits = self.actor(features)

        if deterministic:
            # Choose the action with highest probability
            action = logits.argmax(dim=-1)
        else:
            # Sample from the distribution
            probs = torch.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).squeeze(-1)

        return action.item()

def main():
    parser = argparse.ArgumentParser(description="Watch a trained agent play Snake")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model (.pt file)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to play (default: 5)",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=10,
        help="Size of the snake game grid (default: 10)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic actions (always pick best action)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for rendering (default: 10)",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record videos to videos/enjoy/ folder",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Don't show the game window (useful with --record)",
    )
    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        print("\nAvailable models:")
        if os.path.exists("models"):
            for f in os.listdir("models"):
                if f.endswith(".pt"):
                    print(f"  models/{f}")
        else:
            print("  No models directory found. Train a model first!")
        sys.exit(1)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create environment
    render_mode = None if args.no_render else "human"
    env = SnakeGameEnv(render_mode=render_mode, grid_size=args.grid_size)

    # Optionally wrap with video recorder
    if args.record:
        from gymnasium.wrappers import RecordVideo
        os.makedirs("videos/enjoy", exist_ok=True)
        env = RecordVideo(
            env,
            video_folder="videos/enjoy",
            episode_trigger=lambda ep: True,  # Record all episodes
            name_prefix="snake-enjoy",
        )
        print("Recording videos to videos/enjoy/")

    # Load model
    print(f"Loading model from {args.model_path}")
    agent = CNNAgent(
        obs_shape=(3, args.grid_size, args.grid_size),
        n_actions=4,
    ).to(device)

    # Load state dict
    state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
    agent.load_state_dict(state_dict)
    agent.eval()
    print("Model loaded successfully!")

    # Update FPS
    env.metadata["render_fps"] = args.fps

    # Play episodes
    print(f"\nPlaying {args.episodes} episodes...")
    print("Press Ctrl+C to stop early\n")
    print("-" * 40)

    total_rewards = []
    total_lengths = []
    total_food = []

    try:
        for ep in range(args.episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            food_eaten = 0

            while not done:
                # Get action from model
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                with torch.no_grad():
                    action = agent.get_action(obs_tensor, deterministic=args.deterministic)

                # Take action
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1

                if reward > 0:
                    food_eaten += int(reward)

                # Render
                if render_mode == "human":
                    env.render()

            total_rewards.append(episode_reward)
            total_lengths.append(episode_length)
            total_food.append(food_eaten)

            print(f"Episode {ep + 1}: Reward = {episode_reward:.1f}, "
                  f"Length = {episode_length}, Food = {food_eaten}")

    except KeyboardInterrupt:
        print("\nStopped by user")

    # Print summary
    print("-" * 40)
    print(f"\nSummary ({len(total_rewards)} episodes):")
    print(f"  Average reward: {np.mean(total_rewards):.2f}")
    print(f"  Average length: {np.mean(total_lengths):.1f}")
    print(f"  Average food:   {np.mean(total_food):.1f}")
    print(f"  Best reward:    {max(total_rewards):.1f}")
    print(f"  Most food:      {max(total_food)}")

    # Cleanup
    env.close()

if __name__ == "__main__":
    main()
