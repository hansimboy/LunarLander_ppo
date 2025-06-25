import gymnasium as gym
import argparse
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from src.model import ActorCritic
from src.memory import Memory, calculate_gae
from src.env_utils import make_env, preprocess_obs

def parse_args():
    parser = argparse.ArgumentParser(description="PPO for LunarLanderContinuous-v3")
    parser.add_argument("--env-name", type=str, default="LunarLanderContinuous-v3")
    parser.add_argument("--iteration", type=int, default=100)
    parser.add_argument("--actors", type=int, default=10)
    parser.add_argument("--rollout-length", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lamb", type=float, default=0.95)
    parser.add_argument("--vf-coef", type=float, default=0.05)
    parser.add_argument("--entropy-bonus-coef", type=float, default=0.005)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", type=str, default="models")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)
    # 환경 생성
    env = make_env(args.env_name, args.seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    model = ActorCritic(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    memory = Memory()
    best_return = -float("inf")

    os.makedirs(args.save_dir, exist_ok=True)

    for iter in range(args.iteration):
        memory.clear()

        for actor in range(args.actors):
            obs_np, _ = env.reset() # numpy (8,)
            episode_return = 0

            for _ in range(args.rollout_length):
                obs = preprocess_obs(obs_np, device)

                with torch.no_grad():
                    action, log_prob, value = model.get_action(obs)

                action_np = action.cpu().numpy()
                value_np = value.cpu().item()
                log_prob_np = log_prob.cpu().item()
                next_obs_np, reward, done, truncated, _ = env.step(action_np)
                done_flag = done or truncated

                memory.store(obs_np, action_np, log_prob_np, reward, done_flag, value_np)
                obs_np = next_obs_np
                episode_return += reward

                if done_flag:
                    if episode_return > best_return:
                        best_return = episode_return
                        save_path = os.path.join(args.save_dir, "best_model.pth")
                        torch.save(model.state_dict(), save_path)
                        print(f"[Iter {iter}] New best return: {episode_return:.2f}, saved to {save_path}")

                    obs_np, _ = env.reset()
                    episode_return = 0
                    break

        advantages_np, returns_np = calculate_gae(memory.rewards, memory.values, memory.dones, gamma=args.gamma, lamb=args.lamb)

        states = torch.from_numpy(np.array(memory.states)).float().to(device)
        actions = torch.from_numpy(np.array(memory.actions)).float().to(device)
        old_log_probs = torch.tensor(memory.log_probs, dtype=torch.float32).to(device)
        advantages = torch.tensor(advantages_np, dtype=torch.float32).to(device)
        returns = torch.tensor(returns_np, dtype=torch.float32).to(device)

        dataset_size = len(advantages)

        for _ in range(args.epochs):
            perm = torch.randperm(dataset_size)
            for i in range(0, dataset_size, args.batch_size):
                idx = perm[i:i+args.batch_size]
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]

                new_log_probs, entropy, values = model.evaluate_actions(batch_states, batch_actions)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values.squeeze(-1), batch_returns)

                loss = policy_loss + args.vf_coef * value_loss - args.entropy_bonus_coef * entropy.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    env.close()

if __name__ == "__main__":
    main()