'''
和pandapower有关的接口
'''
import os

import numpy as np
import pandapower as pp


class Wrapper():
    def __init__(self, folder, d):
        self.rb = d["reward_border"]
        self.db = d["diverge_border"]
        self.actor = d["actor"]
        self.action_enum = d["action_enum"]
        self.action_attribute = d["action_attribute"]
        self.observer = d["observer"]
        self.observe_attribute = d["observe_attribute"]
        self.max_step = d["max_step"]
        self.reward_value = d["reward_value"]

        self.folder = folder
        self.net = None
        self.is_diverged = False
        self.step = 0

        
    def count_network_num(self):
        total_network_num = len(os.listdir(self.folder))
        return total_network_num

    # 从文件中根据编号读入网络
    def load_network(self, num):
        self.net = pp.from_json(self.folder + '/' + os.listdir(self.folder)[num])
        self.is_diverged = False
        self.step = 0
        self.run_network()

    def calcu_reward(self, obs):
        '''
        根据观察量计算reward
        对于多个bus的情况，是向下取整
        若有一个bus在最坏区间，那么reward就是最差
        输入的应当是标幺值
        '''
        rew_bad, rew_normal, rew_best = self.reward_value
        reward = rew_best
        for ele in obs:
            # print(ele)
            if ele < self.rb[0] or ele > self.rb[-1]:
                reward = rew_bad
                break
            if (((ele > self.rb[0]) and (ele < self.rb[1])) or 
                ((ele < self.rb[-1]) and (ele > self.rb[-2]))):
                reward = rew_normal
        return reward

    def extract_obs(self):
        # 从网络中提取观测值（如必要，压缩）
        obs_raw = self.net[self.observer][self.observe_attribute].values
        return obs_raw

    def trans_action(self, action_raw):
        # 把NN输出的action变换成pandapower能理解的形式
        action = [self.action_enum[np.argmax(sublist)] for sublist in action_raw]
        return action

    def input_action(self, action):
        # 给network输入action
        self.net[self.actor][self.action_attribute] = action

    def is_done(self, obs):
        '''
        判断这个net是否已经结束
        已经最优化或者崩溃了
        '''
        # print(obs, self.is_diverged, self.step)
        done_mask = ((self.calcu_reward(obs) == 100) or 
                    self.is_diverged or 
                    self.exceed_max_step())
        return done_mask

    # 每个网络最多的调度步数
    def exceed_max_step(self):
        return self.step >= self.max_step

    def check_diverge(self, obs):
        # 输入的应当是标幺值
        for ele in obs:
            if ele < self.db[0] or ele > self.db[1]:
                self.is_diverged = True

    def run_network(self):
        pp.runpp(self.net)

    @staticmethod
    def extra_feature(obs_raw):
        # 把观测结果降维
        # 具体怎么降维留着之后再研究
        obs = obs_raw
        return obs