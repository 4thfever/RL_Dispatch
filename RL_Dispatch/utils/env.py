"""
把pandapower打包成环境env
和pandapower的network交互
相应的，dqn_learn是不和network直接交互的
"""
from pp_wrapper import Wrapper
import numpy as np
class Env():
    def __init__(self, total_network_num=None):
        """
        和强化学习程序交互的接口
        """
        with open('../expr.yaml') as file:
            d = yaml.load(file)
            self.total_step = d["TOTAL_STEP"]

        self.network_num = 0
        self.step = 0
        self.total_network_num = total_network_num
        self.stop_expr = False
        # pandapower wrapper
        self.wrapper = Wrapper()
        self.wrapper.load_network(self.network_num)
        if not total_network_num:
            self.total_network_num = self.wrapper.count_network_num()

    def step(self, action):
        """
        执行下一步调度过程，并输出各项信息
        """
        action = self.wrapper.trans_action(action)
        self.wrapper.run_network()
        obs = self.wrapper.extract_obs()
        reward = self.wrapper.calcu_reward(obs)
        obs = extra_feature(obs)
        done = self.wrapper.is_done()
        return obs, reward, done

    def reset(self):
        """
        在某个网络的调度过程结束（稳定或者解列）的前提下，
        进入下一个网络。
        """
        self.network_num += 1
        self.wrapper.load_network(self.network_num)
        obs = extract_obs(net)
        return obs

    def stopping_criterion(self):
        """
        整个实验停止的指标
        """
        if step >= total_step:
            self.stop_expr = True