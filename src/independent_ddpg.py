import torch
import torch.nn.functional as F
from src.ddpg_agent import DDPGAgent
from src import config

class IndependentDDPG:
    def __init__(self, env, device="cpu"):
        self.device      = device
        self.agents      = {}
        self.agent_names = env.possible_agents
        
        for agent_name in self.agent_names:
            obs_dim = env.observation_space(agent_name).shape[0]
            act_dim = env.action_space(agent_name).shape[0]
            
            # for Indep. DDPG, critic only sees agent's OWN obs & action
            critic_input_dim = obs_dim + act_dim
            
            self.agents[agent_name] = DDPGAgent(
                obs_dim          = obs_dim,
                act_dim          = act_dim,
                critic_input_dim = critic_input_dim,
                lr_actor         = config.LR_ACTOR,
                lr_critic        = config.LR_CRITIC,
                device           = device
            )
            
    def select_actions(self, obs_dict, noise_std=0.0):
        actions = {}
        for agent_name, agent_obs in obs_dict.items():
            actions[agent_name] = self.agents[agent_name].select_action(agent_obs, noise_std)
        return actions

    def update(self, batch_dict):
        """
        updates each agent independently using the batch from the replay buffer
        """
        agent_ids      = batch_dict["agent_ids"]
        obs_batch      = batch_dict["obs"]
        act_batch      = batch_dict["actions"]
        rew_batch      = batch_dict["rewards"]
        next_obs_batch = batch_dict["next_obs"]
        don_batch      = batch_dict["dones"]
        
        for idx, agent_name in enumerate(agent_ids):
            if agent_name not in self.agents:
                continue
                
            agent = self.agents[agent_name]
            
            # extract the data for this specific agent
            o_i  = torch.FloatTensor(obs_batch[idx]).to(self.device)
            a_i  = torch.FloatTensor(act_batch[idx]).to(self.device)
            # rewards/dones are shape (batch_size, num_agents)
            r_i  = torch.FloatTensor(rew_batch[:, idx]).unsqueeze(1).to(self.device)
            no_i = torch.FloatTensor(next_obs_batch[idx]).to(self.device)
            d_i  = torch.FloatTensor(don_batch[:, idx]).unsqueeze(1).to(self.device)
            
            # --- critic update ---
            with torch.no_grad():
                next_a_i       = agent.target_actor(no_i)
                target_q_input = torch.cat([no_i, next_a_i], dim=1)
                target_q       = agent.target_critic(target_q_input)
                # y = r + gamma * (1-d) * Q_target
                y = r_i + config.GAMMA * (1.0 - d_i) * target_q
                
            current_q_input = torch.cat([o_i, a_i], dim=1)
            current_q       = agent.critic(current_q_input)
            
            critic_loss = F.mse_loss(current_q, y)
            
            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            agent.critic_optimizer.step()
            
            # --- actor update ---
            # maximize Q(s, mu(s)) -> minimize -Q(s, mu(s))
            pred_a_i      = agent.actor(o_i)
            actor_q_input = torch.cat([o_i, pred_a_i], dim=1)
            actor_loss    = -agent.critic(actor_q_input).mean()
            
            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            agent.actor_optimizer.step()
            
            # --- soft update target networks ---
            agent.soft_update(config.TAU)
