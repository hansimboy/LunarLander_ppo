import gymnasium as gym
import numpy as np
import torch

def make_env(env_name: str, seed: int = None):
    # Gymnasium 환경 생성. 필요 시 wrapper 추가 (예: 시드 고정, 모니터링 등)
    if seed is not None:
        def init():
            env = gym.make(env_name)
            env.reset(seed=seed)
            return env
        return init  # 벡터 환경 사용 시 callable 반환
    else:
        return lambda: gym.make(env_name)

def preprocess_obs(obs_np: np.ndarray, device: torch.device):
    # numpy -> torch tensor
    return torch.tensor(obs_np, dtype=torch.float32).to(device)