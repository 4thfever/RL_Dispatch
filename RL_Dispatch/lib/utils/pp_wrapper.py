'''
和pandapower有关的接口
'''
import os

import numpy as np
import pandapower as pp
import yaml


class Wrapper():
    def __init__(self, folder):
        with open('config.yaml') as file:
            d = yaml.load(file)
            self.rb = d["reward_border"]
            self.db = d["diverge_border"]
            self.actor = d["actor"]
            self.action_enum = d["action_enum"]
            self.action_attribute = d["action_attribute"]
            self.observer = d["observer"]
            self.observe_attribute = d["observe_attribute"]

        self.folder = folder
        self.net = None
        self.is_diverged = False

        
    def count_network_num(self):
        total_network_num = len(os.listdir(self.folder))
        return total_network_num

    # 从文件中根据编号读入网络
    def load_network(self, num):
        self.net = pp.from_json(self.folder + '/' + os.listdir(self.folder)[num])
        self.is_diverged = False
        obs = self.extract_obs()
        self.run_network(obs)

    def calcu_reward(self, obs):
        '''
        根据观察量计算reward
        对于多个bus的情况，是向下取整
        若有一个bus在最坏区间，那么reward就是最差
        输入的应当是标幺值
        '''
        reward = 100
        for ele in obs:
            if ele < self.rb[0] or ele > self.rb[-1]:
                reward = -100
                break
            if (((ele > self.rb[0]) and (ele < self.rb[1])) or 
                ((ele < self.rb[-1]) and (ele > self.rb[-2]))):
                reward = -50
        return reward

    def extract_obs(self):
        # 从网络中提取观测值（如必要，压缩）
        obs_raw = self.net[self.observer][self.observe_attribute]
        return obs_raw

    def trans_action(self, action_raw):
        # 把NN输出的action变换成pandapower能理解的形式
        
        action = [self.action_enum[np.argmax(sublist)] for sublist in action_raw]
        # print(action_raw, action)
        return action

    def input_action(self, action):
        # 给network输入action
        self.net[self.actor][self.action_attribute] = action

    def is_done(self, obs):
        '''
        判断这个net是否已经结束
        已经最优化或者崩溃了
        '''
        done_mask = (self.calcu_reward(obs) == 100) or self.is_diverged
        return done_mask

    def check_diverge(self, obs):
        # 输入的应当是标幺值
        for ele in obs:
            if ele < self.db[0] or ele > self.db[1]:
                return True
        return False

    def run_network(self, obs):
        pp.runpp(self.net)
        if self.check_diverge(obs):
            self.is_diverged = True

    @staticmethod
    def extra_feature(obs_raw):
        # 把观测结果降维
        # 具体怎么降维留着之后再研究
        obs = obs_raw
        return obs