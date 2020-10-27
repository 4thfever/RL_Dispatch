'''
Unittest 主函数，用来做各个模块的单元测试
'''
import unittest
import sys
import os
import yaml
from unittest import TestCase

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import torch
import pandapower as pp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from packages.dqn_model import DQN
from packages.lib.utils.env import Env
from packages.lib.utils.pp_wrapper import Wrapper
from packages.lib.utils.replay_buffer import ReplayBuffer
from packages.lib.generator.case_generator import case_generate, fluc_value

# test case generator
class TestGenerator(TestCase):

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
# class TestModel(TestCase):
#     def test_forward(self):
#         model = DQN(env.num_observation, len(action_enum), num_actor, num_layer, layer_size)
#         y = model(input_ex)
#         self.assertEqual(y.size(), output_ex.size())
#         self.assertAlmostEqual(y, output_ex, places=3)


if __name__ == '__main__':
    a = np.array([1.0])
    a = torch.from_numpy(a)
    print(a.dtype)
    a = a.double()
    print(a.dtype)
