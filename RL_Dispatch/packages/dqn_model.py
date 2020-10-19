import torch 
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, in_channels, num_actions, num_actor, num_layer, layer_size):
        """
        Initialize a deep Q-learning network as described in
        https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
        Arguments:
            in_channels: number of channel of input.
                i.e The number of most recent frames stacked together as describe in the paper
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        assert num_layer == len(layer_size), "num_layer not equals to length of layer size"
        super(DQN, self).__init__()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, layer_size[0]))
        for i in range(num_layer - 1):
            self.fcs.append(nn.Linear(layer_size[i], layer_size[i+1]))
        self.fcs_end = [nn.Linear(layer_size[-1], num_actions) for _ in range(num_actor)]

    def forward(self, x):
        # 用于unsqueeze的dim，不这样会在shape上出问题
        dim = len(x.shape) - 1
        for fc in self.fcs:
            x = F.relu(fc(x))
        l_y = [torch.unsqueeze(F.relu(fc_end(x)), dim) for fc_end in self.fcs_end]
        return torch.cat(l_y, dim)
