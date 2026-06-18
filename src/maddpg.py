import torch
import torch.nn.functional as F

from src.ddpg_agent import DDPGAgent
from src import config


class MADDPG:
    def __init__(self, env, device="cpu"):
        self.device      = device
        self.agents      = {}
        self.agent_names = env.possible_agents

        total_obs_dim = 0
        total_act_dim = 0
        for agent_name in self.agent_names:
            total_obs_dim += env.observation_space(agent_name).shape[0]
            total_act_dim += env.action_space(agent_name).shape[0]

        critic_input_dim = total_obs_dim + total_act_dim

        for agent_name in self.agent_names:
            obs_dim = env.observation_space(agent_name).shape[0]
            act_dim = env.action_space(agent_name).shape[0]

            self.agents[agent_name] = DDPGAgent(
                obs_dim          = obs_dim,
                act_dim          = act_dim,
                critic_input_dim = critic_input_dim,
                lr_actor         = config.LR_ACTOR,
                lr_critic        = config.LR_CRITIC,
                device           = device,
            )

    def select_actions(self, obs_dict, noise_std=0.0):
        actions = {}
        for agent_name, agent_obs in obs_dict.items():
            actions[agent_name] = self.agents[agent_name].select_action(agent_obs, noise_std)
        return actions

    @staticmethod
    def _joint_input(obs_tensors, act_tensors):
        return torch.cat(obs_tensors + act_tensors, dim=1)

    def update(self, batch_dict):
        agent_ids      = batch_dict["agent_ids"]
        obs_batch      = batch_dict["obs"]
        act_batch      = batch_dict["actions"]
        rew_batch      = batch_dict["rewards"]
        next_obs_batch = batch_dict["next_obs"]
        don_batch      = batch_dict["dones"]

        obs_tensors = [
            torch.FloatTensor(obs_batch[idx]).to(self.device)
            for idx in range(len(agent_ids))
        ]
        act_tensors = [
            torch.FloatTensor(act_batch[idx]).to(self.device)
            for idx in range(len(agent_ids))
        ]
        next_obs_tensors = [
            torch.FloatTensor(next_obs_batch[idx]).to(self.device)
            for idx in range(len(agent_ids))
        ]

        for idx, agent_name in enumerate(agent_ids):
            if agent_name not in self.agents:
                continue

            agent = self.agents[agent_name]
            r_i   = torch.FloatTensor(rew_batch[:, idx]).unsqueeze(1).to(self.device)
            d_i   = torch.FloatTensor(don_batch[:, idx]).unsqueeze(1).to(self.device)

            # --- critic update ---
            with torch.no_grad():
                next_actions = [
                    self.agents[name].target_actor(next_obs_tensors[j])
                    for j, name in enumerate(agent_ids)
                ]
                target_q_input = self._joint_input(next_obs_tensors, next_actions)
                target_q       = agent.target_critic(target_q_input)
                y              = r_i + config.GAMMA * (1.0 - d_i) * target_q

            current_q_input = self._joint_input(obs_tensors, act_tensors)
            current_q       = agent.critic(current_q_input)
            critic_loss     = F.mse_loss(current_q, y)

            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            agent.critic_optimizer.step()

            # --- actor update ---
            policy_actions = []
            for j, name in enumerate(agent_ids):
                if j == idx:
                    policy_actions.append(self.agents[name].actor(obs_tensors[j]))
                else:
                    policy_actions.append(act_tensors[j].detach())

            actor_q_input = self._joint_input(obs_tensors, policy_actions)
            actor_loss    = -agent.critic(actor_q_input).mean()

            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            agent.actor_optimizer.step()

            agent.soft_update(config.TAU)
