'''
Defines the ActorCritic model for the Guitar Hero agent. The model takes in a sequence of frames and outputs both the action probabilities (actor) and the value estimate (critic).'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class VisionActorCritic(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(VisionActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Linear(conv_out_size, 512)

        # Actor head
        self.actor = nn.Linear(512, num_actions)

        # Critic head
        self.critic = nn.Linear(512, 1)

    def _get_conv_out(self, shape):
        o = torch.zeros(1, *shape)
        o = self.conv1(o)
        o = self.conv2(o)
        o = self.conv3(o)
        return int(torch.prod(torch.tensor(o.size())))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))

        action_logits = self.actor(x)
        state_value = self.critic(x)

        return action_logits, state_value

    def get_action(self, x):
        action_logits, _ = self.forward(x)
        distribution = torch.distributions.Bernoulli (logits=action_logits)
        action = distribution.sample()
        log_prob = distribution.log_prob(action).sum(dim=-1)
        return action, log_prob

if __name__ == "__main__":
    # Test the model with a dummy input
    model = VisionActorCritic(input_shape=(12, 192, 128), num_actions=6)
    dummy_input = torch.randn(1, 12, 192, 128)  # Batch size of 1
    action_logits, state_value = model(dummy_input)
    print("Action logits:", action_logits)
    print("State value:", state_value)
    print(f'actions: {model.get_action(dummy_input)}')