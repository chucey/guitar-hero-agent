import torch.nn as nn

class Critic(nn.Module):
    """
    State-value function V(s).
    """
    
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        return self.net(state).squeeze(-1)