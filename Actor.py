import torch
import torch.nn as nn

class Actor(nn.Module):
    """
    Policy Network
    Output Logits
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        return self.net(state)

    def get_action(self, state, action=None):
        logits = self.forward(state)
        dist = torch.distributions.Bernoulli(logits=logits)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob