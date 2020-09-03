'''
和pandapower有关的接口
'''
import os
import pandapower as pp
import yaml

with open('expr.yaml') as file:
    d = yaml.load(file)
    # reward_border = [0.8, 0.95, 1.05, 1.25]
    reward_border = d["REWARD_BORDER"]

class Wrapper(object):
    def __init__(self, folder):
        self.folder = folder
        self.net = None
        self.is_diverged = False
        
    def count_network_num(self):
        total_network_num = len(os.listdir(self.folder))
        return total_network_num

    # 从文件中根据编号读入网络
    def load_network(self, num):
        self.net = pp.from_json(os.listdir(self.folder)[num])
        self.is_diverged = False

    # 根据观察量计算reward
    # 对于多个bus的情况，是向下取整
    # 若有一个bus在最坏区间，那么reward就是最差
    @static
    def calcu_reward(obs):
        pass
        return reward

    # 从网络中提取观测值（如必要，压缩）
    def extract_obs(self):
        obs = self.net['res_bus']['vm_pu']
        obs = extra_feature(obs)
        return obs

    # 把NN输出的action变换成pandapower能理解的形式
    def trans_action(self,action_out):
        pass
        return action

    # 给network输入action
    def input_action(self,action):
        self.net["gen"]["p_mw"] = action

    # 判断这个net是否已经结束
    # 已经最优化或者崩溃了
    def is_done(self, obs):
        done_mask = (calcu_reward(obs) == 100) or self.is_diverged
        return done_mask

    def run_network(self):
        info = self.net.runpp()
        if blabla(info):
            self.is_diverged = True