'''
class defintion of the vision actor, which is a neural network that takes in a stack of 4 images as the current state of the game and outputs an action to take.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class VisionActor(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(VisionActor, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(self._get_conv_output(input_shape), 512)
        self.fc2 = nn.Linear(512, num_actions)

    def _get_conv_output(self, shape):
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
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Use sigmoid to output values between 0 and 1
        return x

if __name__ == "__main__":
    # Example usage
    input_shape = (4, 96, 64)  # 4 stacked frames of size 84x84
    num_actions = 6  # Number of possible actions
    model = VisionActor(input_shape, num_actions)
    print(model)
    logits = model(torch.randn(1, *input_shape))  # Batch size of 1
    print("Logits:", logits)

       

    