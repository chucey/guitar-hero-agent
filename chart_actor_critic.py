import torch
import torch.nn as nn
import torch.nn.functional as F

class ChartActorCritic(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Actor head
        self.actor = nn.Linear(hidden_dim, action_dim)
    
        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_logits = self.actor(x)
        state_value = self.critic(x)
        return action_logits, state_value
        

    def get_action(self, x):
            action_logits, _ = self.forward(x)
            distribution = torch.distributions.Bernoulli (logits=action_logits)
            action = distribution.sample()
            log_prob = distribution.log_prob(action).sum(dim=-1)
            return action, log_prob