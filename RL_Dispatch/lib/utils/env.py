"""
把pandapower打包成环境env
和pandapower的network交互
相应的，dqn_learn是不和network直接交互的
"""
import random

import torch
import numpy as np
import yaml

from .pp_wrapper import Wrapper

class Env():
    def __init__(self, total_network_num=None):
        """
        和强化学习程序交互的接口
        """
        with open('config.yaml') as file:
            d = yaml.load(file)
            self.total_step = d["total_step"]
            self.total_step = 10000
            self.folder = d["folder"]

        self.network_num = 0
        self.num_step = 0
        self.total_network_num = total_network_num
        self.stop_expr = False
        # pandapower wrapper
        self.wrapper = Wrapper(self.folder)
        self.wrapper.load_network(self.network_num)
        if not total_network_num:
            self.total_network_num = self.wrapper.count_network_num()

    def step(self, action):
        """
        执行下一步调度过程，并输出各项信息
        """
        action = self.wrapper.trans_action(action)
        obs = self.wrapper.extract_obs()
        self.wrapper.input_action(action)
        self.wrapper.run_network(obs)
        obs = self.wrapper.extract_obs()
        reward = self.wrapper.calcu_reward(obs)
        obs = self.wrapper.extra_feature(obs)
        done = self.wrapper.is_done(obs)
        return obs, reward, done

    def reset(self):
        """
        在某个网络的调度过程结束（稳定或者解列）的前提下，
        进入下一个网络。
        """
        self.network_num += 1
        self.wrapper.load_network(self.network_num)
        obs = self.wrapper.extract_obs()
        return obs

    def stopping_criterion(self):
        """
        整个实验停止的指标
        """
        if self.num_step >= self.total_step:
            self.stop_expr = True
        return self.stop_expr


    @staticmethod
    def onehot_encode(value, act_per_gen=5):
        res = np.zeros((len(value), act_per_gen))
        res[np.arange(len(value)), value] = 1
        return res
        
    def rand_action(self, num_gen, act_per_gen):
        ran_a = [random.randint(0, act_per_gen-1) for _ in range(num_gen)]
        return self.onehot_encode(ran_a,act_per_gen)

    @staticmethod
    def onehot_decode(x):
        return torch.argmax(x, dim=-1)