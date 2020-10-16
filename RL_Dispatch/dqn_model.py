import torch 
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, in_channels, num_actions, num_actor):
        """
        Initialize a deep Q-learning network as described in
        https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
        Arguments:
            in_channels: number of channel of input.
                i.e The number of most recent frames stacked together as describe in the paper
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super(DQN, self).__init__()
        self.num_actor = num_actor
        self.fc1 = nn.Linear(in_channels, 128)
        self.fc2 = nn.Linear(128, 64)
        self.l_fc3 = [nn.Linear(64, 5) for _ in range(self.num_actor)]

    def forward(self, x):
        # 用于unsqueeze的dim，不这样会在shape上出问题
        dim = len(x.shape) - 1
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        l_y = [torch.unsqueeze(F.relu(fc3(x)), dim) for fc3 in self.l_fc3]
        return torch.cat(l_y, dim)
