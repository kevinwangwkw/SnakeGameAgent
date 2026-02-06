# SnakeGame Reinforcement Learning Agent

## Setup

### 1. Create Virtual Environment
```bash
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2. Install Project Dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Enable NVIDIA GPU for PyTorch
```bash
pip uninstall -y torch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

## Project Structure
```
SnakeGameAgent/
├── src/
│   ├── snake_env.py      # Gymnasium environment of snake game
│   ├── train_ppo.py      # PPO training script
│   ├── train_sac.py      # Discretized-SAC training script
│   ├── observe_ppo.py    # Watch PPO agent play
│   ├── observe_sac.py    # Watch SAC agent play
│   ├── play_game.py      # Manual gameplay
│   └── ...
├── cleanrl/              # CleanRL library (reference)
├── models/               # Saved model checkpoints
├── videos/               # Recorded training videos
├── runs/                 # TensorBoard logs
├── requirements.txt      # Python dependencies
└── README.md
```

## Test Snake Game Manually
```bash
python src/play_game.py
```

**Controls:** Arrow keys or WASD to move, R to restart, ESC/Q to quit.

## Training with PPO

### Quick Start
Train the PPO agent with default settings (wandb tracking and video recording enabled):
```bash
python src/train_ppo.py
```

### Training with Custom Settings
```bash
# Train without wandb (offline mode)
python src/train_ppo.py --no-track

# Train without video recording (faster)
python src/train_ppo.py --no-capture-video

# Custom hyperparameters
python src/train_ppo.py --total-timesteps 2000000 --learning-rate 1e-4 --num-envs 16
```
### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--exp-name` | `ppo_snake` | Experiment name for logging |
| `--seed` | `1` | Random seed |
| `--cuda/--no-cuda` | `True` | Use GPU if available |
| `--track/--no-track` | `True` | Enable Weights & Biases logging |
| `--wandb-project-name` | `snake-ppo` | W&B project name |
| `--capture-video/--no-capture-video` | `True` | Record training videos |
| `--video-freq` | `50` | Record video every N episodes |
| `--grid-size` | `10` | Snake game grid size |
| `--total-timesteps` | `10000000` | Total training steps |
| `--learning-rate` | `3e-4` | Learning rate |
| `--num-envs` | `8` | Parallel environments |
| `--num-steps` | `256` | Steps per rollout |
| `--gamma` | `0.99` | Discount factor |
| `--gae-lambda` | `0.95` | GAE lambda |
| `--num-minibatches` | `4` | Minibatches per update |
| `--update-epochs` | `4` | Epochs per update |
| `--clip-coef` | `0.2` | PPO clip coefficient |
| `--ent-coef` | `0.01` | Entropy coefficient |
| `--vf-coef` | `0.5` | Value function coefficient |

## Training with Discretized SAC

### Quick Start
Train the SAC agent with default settings:
```bash
python src/train_sac.py
```

### Training with Custom Settings
```bash
# Train without wandb (offline mode)
python src/train_sac.py --no-track

# Train without video recording (faster)
python src/train_sac.py --no-capture-video

# Custom hyperparameters
python src/train_sac.py --total-timesteps 500000 --learning-starts 10000
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--exp-name` | `sac_snake` | Experiment name for logging |
| `--seed` | `1` | Random seed |
| `--cuda/--no-cuda` | `True` | Use GPU if available |
| `--track/--no-track` | `True` | Enable Weights & Biases logging |
| `--wandb-project-name` | `snake-sac` | W&B project name |
| `--capture-video/--no-capture-video` | `True` | Record training videos |
| `--video-freq` | `50` | Record video every N episodes |
| `--total-timesteps` | `5000000` | Total training steps |
| `--buffer-size` | `200000` | Replay buffer size |
| `--batch-size` | `2048` | Training batch size |
| `--learning-starts` | `5000` | Steps before learning begins |
| `--policy-lr` | `3e-4` | Policy network learning rate |
| `--q-lr` | `3e-4` | Q-network learning rate |
| `--gamma` | `0.99` | Discount factor |
| `--tau` | `0.005` | Target network soft update coefficient |
| `--alpha` | `0.2` | Entropy coefficient (if not autotuning) |
| `--autotune/--no-autotune` | `True` | Automatic entropy tuning |


## Visualize Performance

### Watch Trained PPO Agent Play
```bash
    python src/observe_ppo.py --model-path models/ppo_snake__1__1770324317_best.pt
    python src/observe_ppo.py --model-path models/ppo_snake__1__1770324317_best.pt --episodes 10
    python src/observe_ppo.py --model-path models/ppo_snake__1__1770324317_best.pt --record
```

### Watch Trained SAC Agent Play
```bash
    python src/observe_sac.py --model-path models/sac_snake__1__1770343783_best.pt
    python src/observe_sac.py --model-path models/sac_snake__1__1770343783_best.pt --episodes 10
    python src/observe_sac.py --model-path models/sac_snake__1__1770343783_best.pt --record
```