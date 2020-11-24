import torch 
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, in_channels, num_action, num_actor, num_layer, layer_size, use_batchnorm, dropout=0.1):
        assert num_layer == len(layer_size), "num_layer not equals to length of layer size"
        super(DQN, self).__init__()

        self.num_actor = num_actor
        self.num_action = num_action

        # 首先过batch norm层
        self.num_layer = num_layer
        self.use_batchnorm = use_batchnorm
        if self.use_batchnorm:
            self.bn = nn.BatchNorm1d(in_channels, track_running_stats=True)
        self.dropout = nn.Dropout()
        # 新建一个list存在之后动态建层结果
        self.fcs = nn.ModuleList()
        self.dps = nn.ModuleList()
        # 起始层
        self.fcs.append(nn.Linear(in_channels, layer_size[0]))
        self.dps.append(nn.Dropout(p=dropout))
        # 中间层
        for i in range(num_layer - 1):
            self.fcs.append(nn.Linear(layer_size[i], layer_size[i+1]))
            self.dps.append(nn.Dropout(p=dropout))
        # 结果层，输出q值
        self.m = nn.Softmax(dim=2)
        self.fc_end = nn.Linear(layer_size[-1], num_action * num_actor)


    def forward(self, x):
        # 用于unsqueeze的dim，不这样会在shape上出问题
        dim = len(x.shape) - 1
        if self.use_batchnorm:
            x = self.bn(x)
        for i in range(self.num_layer):
            x = self.dps[i](F.relu(self.fcs[i](x)))
        x = self.fc_end(x).view(-1, self.num_actor, self.num_action)
        res = self.m(x)
        return res