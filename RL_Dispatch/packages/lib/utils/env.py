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
    def __init__(self, d, num_total_network=None):
        """
        和强化学习程序交互的接口
        """
        
        self.total_step = d["total_step"]
        self.data_folder = d["data_folder"]

        self.idx_network = 0
        self.num_step = 0
        self.num_episode = 0
        self.num_total_network = num_total_network
        self.stop_expr = False
        # pandapower wrapper
        self.wrapper = Wrapper(self.data_folder, d)
        self.wrapper.load_network(self.idx_network)
        if not num_total_network:
            self.num_total_network = self.wrapper.count_network_num()

        self.num_observation = self.wrapper.num_observation

    def step(self, action):
        """
        执行下一步调度过程，并输出各项信息
        """
        obs = self.wrapper.extract('obs')
        action = self.wrapper.trans_action(action)
        self.wrapper.input_action(action)
        self.wrapper.run_network()
        tar = self.wrapper.extract('tar')
        obs = self.wrapper.extract('obs')
        self.wrapper.check_diverge(tar)
        reward = self.wrapper.calcu_reward(tar)
        obs = self.wrapper.extra_feature(obs)
        done = self.wrapper.is_done(obs)
        self.wrapper.step += 1
        return obs, reward, done

    def initial_run(self):
        self.wrapper.load_network(self.idx_network)
        obs = self.wrapper.extract('obs')
        return obs

    def reset(self):
        """
        在某个网络的调度过程结束（稳定或者解列）的前提下，
        进入下一个网络。
        """
        self.idx_network += 1
        self.num_episode += 1
        if self.idx_network >= self.num_total_network:
            self.idx_network = 0
        return self.initial_run()

    def stopping_criterion(self):
        """
        整个实验停止的指标
        """
        if self.num_step >= self.total_step:
            self.stop_expr = True
        return self.stop_expr


    @staticmethod
    def onehot_encode(value, act_per_gen):
        res = np.zeros((len(value), act_per_gen))
        res[np.arange(len(value)), value] = 1
        return res
        
    def rand_action(self, num_gen, act_per_gen):
        ran_a = [random.randint(0, act_per_gen-1) for _ in range(num_gen)]
        return self.onehot_encode(ran_a,act_per_gen)

    @staticmethod
    def onehot_decode(x):
        return torch.argmax(x, dim=-1)

    def cal_epi_reward(self, df, _num_episode):
        df = df[df['Episode'].isin(list(range(_num_episode-100, _num_episode)))]
        _mean_steps_per_episode = df.shape[0]/100
        gb = df.groupby('Episode').apply(lambda x:x.iloc[-1])
        _mean_episode_reward = gb.mean()['Reward']
        return _mean_steps_per_episode, _mean_episode_reward