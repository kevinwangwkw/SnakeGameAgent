### Based on CleanRL's SAC Atari implementation

import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from gymnasium.wrappers import RecordVideo
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path to import snake_env
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from snake_env import SnakeGameEnv

# Import ReplayBuffer from CleanRL
cleanrl_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cleanrl")
sys.path.insert(0, cleanrl_path)
from cleanrl_utils.buffers import ReplayBuffer

@dataclass
class Args:
    """Training configuration arguments."""

    # Experiment settings
    exp_name: str = "sac_snake"
    """Name of this experiment"""
    seed: int = 1
    """Random seed for reproducibility"""
    torch_deterministic: bool = True
    """If True, sets torch.backends.cudnn.deterministic=True"""
    cuda: bool = True
    """If True, enables CUDA if available"""

    # Wandb settings
    track: bool = True
    """If True, tracks experiment with Weights & Biases"""
    wandb_project_name: str = "snake-sac"
    """Wandb project name"""
    wandb_entity: Optional[str] = None
    """Wandb entity (team) name"""

    # Video recording settings
    capture_video: bool = True
    """If True, captures videos of agent performance"""
    video_freq: int = 50
    """Record video every N episodes (only for env 0)"""

    # Environment settings
    grid_size: int = 10
    """Size of the snake game grid"""
    max_steps_per_food: int = 100
    """Maximum steps without eating before truncation"""

    # SAC hyperparameters
    total_timesteps: int = 5_000_000
    """Total timesteps for training"""
    buffer_size: int = 200_000
    """Replay buffer size"""
    gamma: float = 0.99
    """Discount factor"""
    tau: float = 0.005
    """Target network soft update coefficient"""
    batch_size: int = 2048
    """Batch size for training"""
    learning_starts: int = 5_000
    """Number of steps before learning starts"""
    policy_lr: float = 3e-4
    """Learning rate for policy network"""
    q_lr: float = 3e-4
    """Learning rate for Q networks"""
    update_frequency: int = 4 #1
    """Frequency of training updates (every N steps)"""
    target_network_frequency: int = 8 #1
    """Frequency of target network updates"""
    alpha: float = 0.2
    """Entropy regularization coefficient"""
    autotune: bool = True
    """If True, automatically tunes entropy coefficient"""
    target_entropy_scale: float = 0.3
    """Scale factor for target entropy"""

def make_env(args: Args, idx: int, capture_video: bool, run_name: str, video_folder: str):
    # Factory function to create a snake game environment.
    def thunk():
        if capture_video and idx == 0:
            env = SnakeGameEnv(
                render_mode="rgb_array",
                grid_size=args.grid_size,
                max_steps_per_food=args.max_steps_per_food,
            )
            env = RecordVideo(
                env,
                video_folder=video_folder,
                episode_trigger=lambda ep: ep % args.video_freq == 0,
                name_prefix=f"snake-sac-{run_name}",
            )
        else:
            env = SnakeGameEnv(
                render_mode=None,
                grid_size=args.grid_size,
                max_steps_per_food=args.max_steps_per_food,
            )

        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk

def layer_init(layer: nn.Module, bias_const: float = 0.0) -> nn.Module:
    """Initialize layer with Kaiming normal initialization."""
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class SoftQNetwork(nn.Module):
    # Soft Q-Network for discrete actions.
    def __init__(self, obs_shape: tuple, n_actions: int):
        super().__init__()

        # CNN encoder for grid observations
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(obs_shape[0], 32, kernel_size=3, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=3, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, padding=1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate flattened size
        with torch.inference_mode():
            output_dim = self.conv(torch.zeros(1, *obs_shape)).shape[1]

        # Fully connected layers
        self.fc1 = layer_init(nn.Linear(output_dim, 256))
        self.fc_q = layer_init(nn.Linear(256, n_actions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return Q-values for all actions."""
        x = F.relu(self.conv(x))
        x = F.relu(self.fc1(x))
        q_vals = self.fc_q(x)
        return q_vals

class Actor(nn.Module):
    # Actor network for discrete SAC.
    def __init__(self, obs_shape: tuple, n_actions: int):
        super().__init__()

        # CNN encoder (separate from Q-network to avoid gradient interference)
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(obs_shape[0], 32, kernel_size=3, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=3, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, padding=1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate flattened size
        with torch.inference_mode():
            output_dim = self.conv(torch.zeros(1, *obs_shape)).shape[1]

        # Fully connected layers
        self.fc1 = layer_init(nn.Linear(output_dim, 256))
        self.fc_logits = layer_init(nn.Linear(256, n_actions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return action logits."""
        x = F.relu(self.conv(x))
        x = F.relu(self.fc1(x))
        logits = self.fc_logits(x)
        return logits

    def get_action(self, x: torch.Tensor):
        # Sample action from the policy.
        logits = self(x)
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        # Action probabilities for calculating the adapted soft-Q loss
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=1)
        return action, log_prob, action_probs

def train():
    """Main training loop for Discretized SAC."""

    # Parse command-line arguments
    args = tyro.cli(Args)

    # Create run name for logging
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"

    # Setup directories
    video_folder = f"videos/{run_name}"
    os.makedirs(video_folder, exist_ok=True)
    os.makedirs(f"runs/{run_name}", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Initialize Weights & Biases
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=False,  # Disabled to avoid conflict with gymnasium's RecordVideo
            save_code=True,
        )

    # Initialize TensorBoard writer
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]),
    )

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    # Create vectorized environment (single env for SAC)
    envs = gym.vector.SyncVectorEnv(
        [make_env(args, 0, args.capture_video, run_name, video_folder)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "Only discrete action space is supported"

    obs_shape = envs.single_observation_space.shape
    n_actions = envs.single_action_space.n

    print(f"Observation space: {envs.single_observation_space}")
    print(f"Action space: {envs.single_action_space}")

    # Initialize networks
    actor = Actor(obs_shape, n_actions).to(device)
    qf1 = SoftQNetwork(obs_shape, n_actions).to(device)
    qf2 = SoftQNetwork(obs_shape, n_actions).to(device)
    qf1_target = SoftQNetwork(obs_shape, n_actions).to(device)
    qf2_target = SoftQNetwork(obs_shape, n_actions).to(device)

    # Initialize target networks with same weights
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    # Print model summary
    total_params = (
        sum(p.numel() for p in actor.parameters()) +
        sum(p.numel() for p in qf1.parameters()) +
        sum(p.numel() for p in qf2.parameters())
    )
    print(f"Total parameters: {total_params:,}")

    # Optimizers
    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()),
        lr=args.q_lr,
        eps=1e-4
    )
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.policy_lr, eps=1e-4)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -args.target_entropy_scale * torch.log(1 / torch.tensor(n_actions))
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        alpha_optimizer = optim.Adam([log_alpha], lr=args.q_lr, eps=1e-4)
    else:
        alpha = args.alpha

    # Initialize replay buffer
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )

    # Track episode statistics
    episode_count = 0
    episode_rewards = []
    episode_lengths = []
    best_reward = float('-inf')

    print(f"\nStarting training for {args.total_timesteps:,} timesteps")
    print(f"Learning starts at step {args.learning_starts:,}")
    print("-" * 60)

    # Initialize environment
    start_time = time.time()
    obs, _ = envs.reset(seed=args.seed)

    # Main training loop
    for global_step in range(args.total_timesteps):
        # Action selection
        if global_step < args.learning_starts:
            # Random exploration before learning starts
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            # Sample from policy
            with torch.no_grad():
                actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
                actions = actions.cpu().numpy()

        # Execute action
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # Log episode statistics
        if "_episode" in infos:
            for i, done_flag in enumerate(infos["_episode"]):
                if done_flag:
                    ep_return = float(infos["episode"]["r"][i])
                    ep_length = int(infos["episode"]["l"][i])
                    episode_count += 1
                    episode_rewards.append(ep_return)
                    episode_lengths.append(ep_length)

                    writer.add_scalar("charts/episodic_return", ep_return, global_step)
                    writer.add_scalar("charts/episodic_length", ep_length, global_step)
                    writer.add_scalar("charts/episode_count", episode_count, global_step)

                    if ep_return > best_reward:
                        best_reward = ep_return
                        # Save best model
                        torch.save({
                            'actor': actor.state_dict(),
                            'qf1': qf1.state_dict(),
                            'qf2': qf2.state_dict(),
                        }, f"models/{run_name}_best.pt")

                    if episode_count % 100 == 0:
                        avg_reward = np.mean(episode_rewards[-100:])
                        avg_length = np.mean(episode_lengths[-100:])
                        print(f"Episode {episode_count} | Step {global_step:,} | "
                              f"Avg Reward (100 ep): {avg_reward:.2f} | "
                              f"Avg Length: {avg_length:.1f} | "
                              f"Best: {best_reward:.2f}")

        # Handle truncation (use final observation for proper bootstrapping)
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc and "final_observation" in infos:
                real_next_obs[idx] = infos["final_observation"][idx]

        # Store transition in replay buffer
        # Note: infos needs to be a list of dicts for the buffer
        infos_list = [{} for _ in range(envs.num_envs)]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos_list)

        # Update observation
        obs = next_obs

        # Training
        if global_step > args.learning_starts:
            if global_step % args.update_frequency == 0:
                # Sample batch from replay buffer
                data = rb.sample(args.batch_size)

                # ----- Critic Training -----
                with torch.no_grad():
                    # Get next actions and their log probabilities
                    _, next_state_log_pi, next_state_action_probs = actor.get_action(data.next_observations)

                    # Compute target Q-values
                    qf1_next_target = qf1_target(data.next_observations)
                    qf2_next_target = qf2_target(data.next_observations)

                    # Use action probabilities instead of MC sampling for expectation
                    min_qf_next_target = next_state_action_probs * (
                        torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                    )
                    # Sum over actions for discrete Q-target
                    min_qf_next_target = min_qf_next_target.sum(dim=1)

                    # Compute TD target
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * min_qf_next_target

                # Current Q-values for taken actions
                qf1_values = qf1(data.observations)
                qf2_values = qf2(data.observations)
                qf1_a_values = qf1_values.gather(1, data.actions.long()).view(-1)
                qf2_a_values = qf2_values.gather(1, data.actions.long()).view(-1)

                # Critic loss
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                # Update critics
                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                # ----- Actor Training -----
                _, log_pi, action_probs = actor.get_action(data.observations)

                with torch.no_grad():
                    qf1_values = qf1(data.observations)
                    qf2_values = qf2(data.observations)
                    min_qf_values = torch.min(qf1_values, qf2_values)

                # Actor loss: expectation over actions (no reparameterization needed for discrete)
                actor_loss = (action_probs * ((alpha * log_pi) - min_qf_values)).mean()

                # Update actor
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # ----- Entropy Tuning -----
                if args.autotune:
                    alpha_loss = (action_probs.detach() * (
                        -log_alpha.exp() * (log_pi + target_entropy).detach()
                    )).mean()

                    alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    alpha_optimizer.step()
                    alpha = log_alpha.exp().item()

            # Update target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            # Logging
            if global_step % 1000 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)

                sps = int(global_step / (time.time() - start_time))
                writer.add_scalar("charts/SPS", sps, global_step)

                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

        # Periodic progress report
        if global_step > 0 and global_step % 10000 == 0:
            sps = int(global_step / (time.time() - start_time))
            print(f"Step {global_step:,}/{args.total_timesteps:,} | SPS: {sps}")

    # Training complete
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Total episodes: {episode_count}")
    print(f"Best reward: {best_reward:.2f}")
    if len(episode_rewards) > 0:
        print(f"Final average reward (100 ep): {np.mean(episode_rewards[-100:]):.2f}")
    print("=" * 60)

    # Save final model
    torch.save({
        'actor': actor.state_dict(),
        'qf1': qf1.state_dict(),
        'qf2': qf2.state_dict(),
    }, f"models/{run_name}_final.pt")
    print(f"Models saved to models/{run_name}_*.pt")

    # Cleanup
    try:
        envs.close()
    except AttributeError:
        pass
    writer.close()

    if args.track:
        wandb.finish()

if __name__ == "__main__":
    train()
