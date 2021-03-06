'''
Unittest 主函数，用来做各个模块的单元测试
'''
'''
单元测试主程序
'''
import unittest
import sys
import os
import yaml
from unittest import TestCase

import torch
import numpy as np
import pandapower as pp
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from packages.dqn_model import DQN
from packages.lib.utils.env import Env
from packages.lib.utils.pp_wrapper import Wrapper
from packages.lib.utils.replay_buffer import ReplayBuffer
from packages.lib.generator.case_generator import case_generate, fluc_value

# test case generator
class Test(TestCase):

    def test_fluc(self):
        net = pp.from_json(base_grid)
        obj, attr, area = obj2change[0], attr2change[0], fluc_area[0]
        new_net = fluc_value(net, obj, attr, area)
        base_value = net[obj][attr].values
        bottom_value = base_value * area[0]
        top_value = base_value * area[1]
        new_value = new_net[obj][attr].values
        self.assertTrue((top_value > new_value).all())
        self.assertTrue((bottom_value < new_value).all())

    def test_generate(self):
        case_generate(d, output=False)

    # test wrapper
    # test env
    # test replaybuffer

    # test model
    def test_forward(self):
        model = DQN(env.num_observation, len(action_enum), num_actor, num_layer, layer_size)
        x = torch.from_numpy(np.ones([env.num_observation])).float()
        y = model(x)
        self.assertEqual(y.shape().numpy(), [num_actor, len(action_enum)])


if __name__ == '__main__':
    with open('config.yaml') as file:
        d = yaml.load(file)
    # 把dict加载到locals变量中
    for key, value in d.items():
        locals()[key] = value
    env = Env(d)

    unittest.main()
