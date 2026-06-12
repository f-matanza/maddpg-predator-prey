import random
from collections import deque
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, obs, act, rew, next_obs, done):
        """
        expects tuples/lists where each element belongs to one agent
        e.g., obs = (obs_agent_1, obs_agent_2, ...)
        """
        self.buffer.append((obs, act, rew, next_obs, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, act, rew, next_obs, don = zip(*batch)
        
        # transpose so that we get a list of batches per agent
        num_agents = len(obs[0])
        
        obs_batch = [np.array([o[i] for o in obs]) for i in range(num_agents)]
        act_batch = [np.array([a[i] for a in act]) for i in range(num_agents)]
        rew_batch = [np.array([r[i] for r in rew]) for i in range(num_agents)]
        next_obs_batch = [np.array([no[i] for no in next_obs]) for i in range(num_agents)]
        don_batch = [np.array([d[i] for d in don]) for i in range(num_agents)]
        
        return obs_batch, act_batch, rew_batch, next_obs_batch, don_batch
    
    def __len__(self):
        return len(self.buffer)
