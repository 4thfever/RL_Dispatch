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
        记录现有的网络编号
        """
        self.network_num = 0
        self.wrapper = Wrapper()
        wrapper.load_network(self.network_num)
        self.total_network_num = total_network_num
        if not total_network_num:
            self.total_network_num = count_network_num()

    def _step(self, action):
        """
        执行下一步调度过程，并输出各项信息
        """
        action = trans_action(action)

        return obs, reward, done

    def _reset(self):
        """
        在某个网络的调度过程结束（稳定或者解列）的前提下，
        进入下一个网络。
        """
        self.network_num += 1
        wrapper.load_network(self.network_num)
        obs = extract_obs(net)
        return obs
