import copy
import torch
import numpy as np
from torch.optim import Adam

from src.networks import Actor, Critic

class DDPGAgent:
    def __init__(self, obs_dim, act_dim, critic_input_dim, lr_actor, lr_critic, device="cpu"):
        self.device = device
        
        # init primary networks
        self.actor  = Actor(obs_dim, act_dim).to(device)
        self.critic = Critic(critic_input_dim).to(device)
        
        # init target networks
        self.target_actor  = copy.deepcopy(self.actor).to(device)
        self.target_critic = copy.deepcopy(self.critic).to(device)
        
        # optimizers
        self.actor_optimizer  = Adam(self.actor.parameters(),  lr=lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic)
    
    def select_action(self, obs, noise_std=0.0):
        """selects an action given an obs, w/ opt exploration noise"""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(obs_tensor).cpu().numpy()[0]
        self.actor.train()
        
        # add exploration noise
        if noise_std > 0:
            action += np.random.normal(0, noise_std, size=action.shape)
            
        # MPE expects continuous actions in [0, 1]
        return np.clip(action, 0, 1)
        
    def soft_update(self, tau):
        """soft-updates the target networks: θ_target = τ*θ + (1 - τ) * θ_target"""
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
            
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
