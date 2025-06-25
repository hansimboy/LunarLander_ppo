import argparse
import os
import imageio
from tqdm import tqdm
import torch
import gymnasium as gym
from src.model import ActorCritic

def parse_args():
    parser = argparse.ArgumentParser(description="Record video for a trained PPO model")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--env-name", type=str, default="LunarLanderContinuous-v3")
    parser.add_argument("--output", type=str, default="ppo_lunarlander.mp4")
    parser.add_argument("--episode-length", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()

def record_video(model_path: str, env_name: str, output_filename: str, episode_length: int, device: str):
    env = gym.make(env_name, render_mode="rgb_array")
    obs, _ = env.reset()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    model = ActorCritic(obs_dim, act_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    episode_reward = 0
    frames = []
    for _ in tqdm(range(episode_length), desc="Recording"):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
        with torch.no_grad():
            action, _, _ = model.get_action(obs_tensor)
        obs, reward, done, truncated, _ = env.step(action.cpu().numpy())
        episode_reward += reward
        frame = env.render()
        frames.append(frame)
        if done or truncated:
            break
    env.close()
    os.makedirs(os.path.dirname(output_filename) or ".", exist_ok=True)
    imageio.mimsave(output_filename, frames, fps=30)
    print(f"Video saved to {output_filename}")

def main():
    args = parse_args()
    record_video(args.model_path, args.env_name, args.output, args.episode_length, args.device)

if __name__ == "__main__":
    main()
