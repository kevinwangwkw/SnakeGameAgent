### Based on CleanRL's PPO implementation

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
import torch.optim as optim
import tyro
from gymnasium.wrappers import RecordVideo
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path to import snake_env
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from snake_env import SnakeGameEnv

@dataclass
class Args:
    """Training configuration arguments."""

    # Experiment settings
    exp_name: str = "ppo_snake"
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
    wandb_project_name: str = "snake-ppo"
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

    # PPO hyperparameters
    total_timesteps: int = 10_000_000
    """Total timesteps for training"""
    learning_rate: float = 3e-4
    """Learning rate for optimizer"""
    num_envs: int = 8
    """Number of parallel environments"""
    num_steps: int = 256
    """Number of steps per environment per rollout"""
    anneal_lr: bool = True
    """If True, anneals learning rate over training"""
    gamma: float = 0.99
    """Discount factor"""
    gae_lambda: float = 0.95
    """Lambda for Generalized Advantage Estimation"""
    num_minibatches: int = 4
    """Number of minibatches per update"""
    update_epochs: int = 4
    """Number of epochs per policy update"""
    norm_adv: bool = True
    """If True, normalizes advantages"""
    clip_coef: float = 0.2
    """PPO clipping coefficient"""
    clip_vloss: bool = True
    """If True, clips value function loss"""
    ent_coef: float = 0.01
    """Entropy coefficient for exploration"""
    vf_coef: float = 0.5
    """Value function coefficient"""
    max_grad_norm: float = 0.5
    """Maximum gradient norm for clipping"""
    target_kl: Optional[float] = None
    """Target KL divergence for early stopping (None = disabled)"""

    # Computed at runtime
    batch_size: int = 0
    """Batch size (computed: num_envs * num_steps)"""
    minibatch_size: int = 0
    """Minibatch size (computed: batch_size // num_minibatches)"""
    num_iterations: int = 0
    """Number of iterations (computed: total_timesteps // batch_size)"""

def make_env(args: Args, idx: int, capture_video: bool, run_name: str, video_folder: str):
    # Factory function to create a snake game environment.
    def thunk():
        # Create snake environment with rgb_array render mode for video
        if capture_video and idx == 0:
            env = SnakeGameEnv(
                render_mode="rgb_array",
                grid_size=args.grid_size,
                max_steps_per_food=args.max_steps_per_food,
            )
            # Record video every video_freq episodes
            env = RecordVideo(
                env,
                video_folder=video_folder,
                episode_trigger=lambda ep: ep % args.video_freq == 0,
                name_prefix=f"snake-{run_name}",
            )
        else:
            env = SnakeGameEnv(
                render_mode=None,
                grid_size=args.grid_size,
                max_steps_per_food=args.max_steps_per_food,
            )

        # Wrap with episode statistics for logging
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk

def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    # Initialize a layer with orthogonal weights and constant bias.
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class CNNAgent(nn.Module):
    # CNN-based PPO Agent for the Snake Game.
    def __init__(self, envs):
        super().__init__()

        obs_shape = envs.single_observation_space.shape  # (3, H, W)
        n_actions = envs.single_action_space.n  # 4 actions

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
        # For a 10x10 grid with padding=1, spatial dims are preserved
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

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        features = self.network(x)
        return self.critic(features)

    def get_action_and_value(self, x: torch.Tensor, action: Optional[torch.Tensor] = None):
        # Compute action, log probability, entropy, and state value.
        features = self.network(x)
        logits = self.actor(features)
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), self.critic(features)

def train():
    """Main training loop."""

    # Parse command-line arguments
    args = tyro.cli(Args)

    # Compute derived values
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

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

    # Create vectorized environments
    envs = gym.vector.SyncVectorEnv(
        [make_env(args, i, args.capture_video, run_name, video_folder) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "Only discrete action space is supported"

    print(f"Observation space: {envs.single_observation_space}")
    print(f"Action space: {envs.single_action_space}")

    # Initialize agent and optimizer
    agent = CNNAgent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Print model summary
    total_params = sum(p.numel() for p in agent.parameters())
    print(f"Total parameters: {total_params:,}")

    # Storage for rollout data
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Initialize environment
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # Track episode statistics
    episode_count = 0
    episode_rewards = []
    episode_lengths = []
    best_reward = float('-inf')

    print(f"\nStarting training for {args.total_timesteps:,} timesteps ({args.num_iterations} iterations)")
    print(f"Batch size: {args.batch_size}, Minibatch size: {args.minibatch_size}")
    print("-" * 60)

    # Main training loop
    for iteration in range(1, args.num_iterations + 1):
        # Learning rate annealing
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # Collect rollout
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # Get action from policy
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Execute action in environment
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(next_done).to(device)

            # Log episode statistics (gymnasium uses '_episode' mask and 'episode' data)
            if "_episode" in infos:
                # infos['_episode'] is a boolean array indicating which envs finished
                # infos['episode']['r'] and infos['episode']['l'] contain the data
                for i, done_flag in enumerate(infos["_episode"]):
                    if done_flag:
                        ep_return = float(infos["episode"]["r"][i])
                        ep_length = int(infos["episode"]["l"][i])
                        episode_count += 1
                        episode_rewards.append(ep_return)
                        episode_lengths.append(ep_length)

                        # Log to tensorboard and wandb
                        writer.add_scalar("charts/episodic_return", ep_return, global_step)
                        writer.add_scalar("charts/episodic_length", ep_length, global_step)
                        writer.add_scalar("charts/episode_count", episode_count, global_step)

                        # Track best reward
                        if ep_return > best_reward:
                            best_reward = ep_return
                            # Save best model
                            torch.save(agent.state_dict(), f"models/{run_name}_best.pt")

                        # Print progress
                        if episode_count % 100 == 0:
                            avg_reward = np.mean(episode_rewards[-100:])
                            avg_length = np.mean(episode_lengths[-100:])
                            print(f"Episode {episode_count} | Step {global_step:,} | "
                                  f"Avg Reward (100 ep): {avg_reward:.2f} | "
                                  f"Avg Length: {avg_length:.1f} | "
                                  f"Best: {best_reward:.2f}")

        # Compute advantages using GAE
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # Flatten batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Policy and value network updates
        b_inds = np.arange(args.batch_size)
        clipfracs = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                # Gradient update
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            # Early stopping based on KL divergence
            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # Compute explained variance
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Log training metrics
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/SPS", sps, global_step)

        # Periodic logging
        if iteration % 10 == 0:
            print(f"Iteration {iteration}/{args.num_iterations} | SPS: {sps}")

    # Training complete
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Total episodes: {episode_count}")
    print(f"Best reward: {best_reward:.2f}")
    if len(episode_rewards) > 0:
        print(f"Final average reward (100 ep): {np.mean(episode_rewards[-100:]):.2f}")
    print("=" * 60)

    # Save final model
    torch.save(agent.state_dict(), f"models/{run_name}_final.pt")
    print(f"Models saved to models/{run_name}_*.pt")

    # Cleanup
    try:
        envs.close()
    except AttributeError:
        # Workaround for wandb/gymnasium compatibility issue
        pass
    writer.close()

    if args.track:
        wandb.finish()

if __name__ == "__main__":
    train()
