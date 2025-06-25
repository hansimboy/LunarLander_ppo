# PPO for LunarLanderContinuous-v3

This repository implements the Proximal Policy Optimization (PPO) algorithm and applies it to the OpenAI Gym/Gymnasium LunarLanderContinuous-v3 environment.  
Based on the original paper: Schulman et al., “Proximal Policy Optimization Algorithms” (2017) ([arXiv:1707.06347](https://arxiv.org/abs/1707.06347)).

---

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Features](#features)  
3. [Installation & Setup](#installation--setup)  
4. [Usage](#usage)  
   - [Training](#training)  
   - [Recording Video](#recording-video)  
5. [Hyperparameters](#hyperparameters)  
6. [Example Results](#example-results)  
7. [References](#references)  
8. [Contributing & License](#contributing--license)  

---

## Project Overview
This project provides:
- A PyTorch-based implementation of PPO from scratch.
- Application to the LunarLanderContinuous-v3 environment so that an agent learns to land successfully.
- Clipped surrogate objective for policy updates.
- Generalized Advantage Estimation (GAE) for advantage computation.
- Actor-Critic network architecture.
- Automatic saving of the best model during training.
- Scripts for training, evaluation, and recording agent behavior as video.

---

## Features
- Pure PyTorch implementation of PPO (no high-level RL libraries).
- Configurable hyperparameters via `argparse`
- Scripts:
  - `train.py` for training.
  - `record_video.py` for recording an episode as an MP4.

---

## Installation & Setup
- pip install --upgrade pip
- pip install -r requirements.txt

## Usage
Training:

```bash
python src/train.py
```
Recoding Video

```bash
python src/record_video.py
```
## Hyperparameters
The default hyperparameters used for PPO training (as defined in `train.py`):

| Parameter             | Description                                            | Default                          |
|-----------------------|--------------------------------------------------------|----------------------------------|
| `--env-name`          | Gym environment name                                   | `LunarLanderContinuous-v3`       |
| `--iteration`         | Number of training iterations (outer loop)             | `100`                            |
| `--actors`            | Number of parallel rollouts per iteration              | `10`                             |
| `--rollout-length`    | Maximum steps per rollout before reset                 | `1000`                           |
| `--epochs`            | Number of update epochs per iteration                  | `10`                             |
| `--batch-size`        | Minibatch size for policy/value updates                | `64`                             |
| `--clip-eps`          | PPO clipping ε                                         | `0.2`                            |
| `--lr`                | Learning rate                                          | `3e-4`                           |
| `--gamma`             | Discount factor                                        | `0.99`                           |
| `--lamb`              | GAE λ                                                  | `0.95`                           |
| `--vf-coef`           | Value function loss coefficient                        | `0.05`                           |
| `--entropy-bonus-coef`| Entropy bonus coefficient                              | `0.005`                          |
| `--device`            | Torch device to use (`cpu` or `cuda`)                  | `"cuda"` if available, else `"cpu"` |
| `--save-dir`          | Directory to save model checkpoints                    | `models`                         |
| `--seed`              | Random seed for reproducibility (optional)             | `None`                           |

running `train.py`, for example:

```bash
python src/train.py \
  --iteration 200 \
  --actors 8 \
  --rollout-length 500 \
  --epochs 10 \
  --batch-size 64 \
  --clip-eps 0.2 \
  --lr 3e-4 \
  --gamma 0.99 \
  --lamb 0.95 \
  --vf-coef 0.05 \
  --entropy-bonus-coef 0.005 \
  --device cpu \
  --save-dir models \
  --seed 42
```

## References
- Schulman et al. (2017). Proximal Policy Optimization Algorithms. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
- Gymnasium LunarLanderContinuous-v3 documentation
- PyTorch documentation

## Contributing & License
Contributions welcome.  
Licensed under MIT License. See [LICENSE](LICENSE.txt) for details.
