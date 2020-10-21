import unittest
import sys
import os
import yaml
from unittest import TestCase

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from packages.dqn_model import DQN
from packages.lib.utils.env import Env
from packages.lib.utils.pp_wrapper import Wrapper
from packages.lib.utils.replay_buffer import ReplayBuffer
from packages.lib.generator.case_generator import case_generate

# test model
# class TestModel(TestCase):
#     def test_forward(self):
#       model = DQN()
#         y = model(input_ex)
#         self.assertEqual(y.size(), output_ex.size())
#         self.assertAlmostEqual(y, output_ex, places=3)

# test env
# test wrapper
# test replaybuffer
# test case generator


if __name__ == '__main__':
    with open('./config.yaml') as file:
        d = yaml.load(file)
    l = locals()
    for key, value in d:
        l[key] = value
    print(l)

    unittest.main()