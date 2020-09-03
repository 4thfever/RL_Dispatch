'''
和pandapower有关的接口
'''
import os
import pandapower as pp
import yaml

with open('../expr.yaml') as file:
    d = yaml.load(file)
    # reward_border = [0.8, 0.95, 1.05, 1.25]
    reward_border = d["REWARD_BORDER"]
    diverge_border = d["DIVERGE_BORDER"]

class Wrapper(object):
    def __init__(self, folder):
        self.folder = folder
        self.net = None
        self.rb = reward_border
        self.db = diverge_border
        self.is_diverged = False
        
    def count_network_num(self):
        total_network_num = len(os.listdir(self.folder))
        return total_network_num

    # 从文件中根据编号读入网络
    def load_network(self, num):
        self.net = pp.from_json(os.listdir(self.folder)[num])
        self.is_diverged = False
        self.run_network()


    @staticmethod
    def calcu_reward(obs):
        '''
        根据观察量计算reward
        对于多个bus的情况，是向下取整
        若有一个bus在最坏区间，那么reward就是最差
        输入的应当是标幺值
        '''
        reward = 100
        for ele in obs:
            if obs < self.rb[0] or obs > self.rb[-1]:
                reward = -100
                break
            if ((obs > self.rb[0]) and (obs < self.rb[1])) or
                ((obs < self.rb[-1]) and (obs > self.rb[-2]))
                reward = -50
        return reward

    def extract_obs(self):
        # 从网络中提取观测值（如必要，压缩）
        obs_raw = self.net['res_bus']['vm_pu']
        return obs_raw

    @staticmethod
    def extra_feature(obs_raw):
        # 把观测结果降维
        pass
        return obs

    def trans_action(self, action_out):
        # 把NN输出的action变换成pandapower能理解的形式
        pass
        return action

    def input_action(self, action):
        # 给network输入action
        self.net["gen"]["p_mw"] = action


    def is_done(self, obs):
        '''
        判断这个net是否已经结束
        已经最优化或者崩溃了
        '''
        done_mask = (calcu_reward(obs) == 100) or self.is_diverged
        return done_mask

    @staticmethod
    def check_diverge(obs):
        # 输入的应当是标幺值
        for ele in obs:
            if obs < self.db[0] or obs > self.db[1]:
                return True
        return False

    def run_network(self, obs):
        self.net.runpp()
        if check_diverge(obs):
            self.is_diverged = True