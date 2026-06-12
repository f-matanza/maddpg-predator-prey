import os
import torch
import numpy as np
import pandas as pd
from pettingzoo.mpe import simple_tag_v3

from src import config
from src.independent_ddpg import IndependentDDPG
from src.replay_buffer import ReplayBuffer

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # init env
    env = simple_tag_v3.parallel_env(
        num_adversaries    = config.NUM_ADVERSARIES,
        num_good           = config.NUM_GOOD_AGENTS,
        num_obstacles      = config.NUM_OBSTACLES,
        max_cycles         = config.MAX_CYCLES,
        continuous_actions = config.CONTINUOUS_ACTIONS
    )
    
    env.reset()
    agent_names = env.possible_agents
    
    # init controller & buffer
    controller = IndependentDDPG(env, device=device)
    buffer     = ReplayBuffer(config.BUFFER_SIZE, agent_ids=agent_names)
    
    # tracking
    rewards_history = []
    os.makedirs("results",           exist_ok=True)
    os.makedirs("checkpoints/iddpg", exist_ok=True)
    
    print("Starting Training (Independent DDPG)...")
    for ep in range(1, config.NUM_EPS + 1):
        obs, _ = env.reset()
        
        # calc exploration noise decay
        noise_std = config.NOISE_STD_START - (config.NOISE_STD_START - config.NOISE_STD_END) * (ep / config.NOISE_DECAY_EPS)
        noise_std = max(config.NOISE_STD_END, noise_std)
        
        ep_reward = {a: 0.0 for a in agent_names}
        
        while env.agents:
            actions = controller.select_actions(obs, noise_std=noise_std)
            next_obs, rewards, terminations, truncations, _ = env.step(actions)
            
            # pad next_obs and done for dead agents to keep dicts consistent
            padded_next_obs = {a: next_obs.get(a, np.zeros_like(obs[a])) for a in agent_names}
            padded_dones    = {a: terminations.get(a, True) or truncations.get(a, True) for a in agent_names}
            
            buffer.push(obs, actions, rewards, padded_next_obs, padded_dones)
            
            for a in env.agents:
                ep_reward[a] += rewards[a]
                
            obs = next_obs
            
            # update networks
            if len(buffer) >= config.BATCH_SIZE:
                batch = buffer.sample(config.BATCH_SIZE)
                controller.update(batch)
                
        # logging
        adv_rewards = [ep_reward[a] for a in agent_names if "adversary" in a]
        adv_mean    = np.mean(adv_rewards) if adv_rewards else 0.0
        
        log_entry = {
            "episode": ep,
            "adversary_reward_mean": adv_mean,
            **ep_reward
        }
        rewards_history.append(log_entry)
        
        if ep % config.LOG_FREQ == 0:
            avg_rew = {a: np.mean([rh[a] for rh in rewards_history[-config.LOG_FREQ:]]) for a in agent_names}
            # simplified print for readability
            avg_str = ", ".join([f"{a}: {rew:.2f}" for a, rew in avg_rew.items()])
            print(f"Ep {ep:05d}/{config.NUM_EPS} | Noise: {noise_std:.2f} | Avg Rewards: [{avg_str}]")
            
        if ep % config.CKPT_FREQ == 0:
            # save weights for each agent
            for a_name, agent in controller.agents.items():
                torch.save(agent.actor.state_dict(), f"checkpoints/iddpg/{a_name}_actor_ep{ep}.pt")
                
    df = pd.DataFrame(rewards_history)
    df.to_csv("results/iddpg_rewards.csv", index=False)
    print("Training complete! Logs saved to results/iddpg_rewards.csv")

if __name__ == "__main__":
    train()
