"""
把pandapower打包成环境env
和pandapower的network交互
相应的，dqn_learn是不和network直接交互的
"""
import random

import torch
import pandas as pd
import numpy as np
import yaml
from torch import optim
from torch.optim.lr_scheduler import StepLR

from .pp_wrapper import Wrapper
from .schedule import LinearSchedule
from .replay_buffer import ReplayBuffer

class Env():
    def __init__(self, d):
        """
        和强化学习程序交互的接口
        """
        
        # self.total_step = d["total_step"]
        # self.data_folder = d["data_folder"]
        # self.num_actor = d["num_actor"]
        # self.action_enum = d["action_enum"]
        # self.log_every_n_steps = d["log_every_n_steps"]
        self.d = d

        self.count_network = 0
        self.num_step = 0
        self.num_episode = 0
        self.stop_expr = False

        # pandapower wrapper
        self.wrapper = Wrapper(d)
        self.num_observation = self.wrapper.num_observation

        # network
        self.num_total_network = self.wrapper.count_network_num()
        self.idxs_network = np.random.permutation(self.num_total_network)
        self.wrapper.load_network(self.idxs_network[self.count_network])

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
        done = self.wrapper.is_done(tar)
        self.wrapper.step += 1
        return obs, reward, done

    def initial_run(self):
        self.wrapper.load_network(self.idxs_network[self.count_network])
        obs = self.wrapper.extract('obs')
        return obs

    def reset(self):
        """
        在某个网络的调度过程结束（稳定或者解列）的前提下，
        进入下一个网络。
        """
        self.count_network += 1
        self.num_episode += 1
        if self.count_network >= self.num_total_network:
            self.count_network = 0
            self.idxs_network = np.random.permutation(self.num_total_network)
        return self.initial_run()

    def stopping_criterion(self):
        """
        整个实验停止的指标
        """
        if self.num_step >= self.d["total_step"]:
            self.stop_expr = True
        return self.stop_expr


    @staticmethod
    def onehot_encode(value, act_per_gen):
        res = np.zeros((len(value), act_per_gen))
        res[np.arange(len(value)), value] = 1
        return res

    @staticmethod
    def onehot_decode(x):
        return torch.argmax(x, dim=-1)

    def rand_action(self, num_gen, act_per_gen):
        ran_a = [random.randint(0, act_per_gen-1) for _ in range(num_gen)]
        return self.onehot_encode(ran_a,act_per_gen)

    # 计算按照episode整理的reward
    def cal_epi_reward(self, df, _num_episode):
        start = _num_episode-self.d["log_every_n_steps"]
        df = df[df['Episode'].isin(range(start, _num_episode))]
        gb = df.groupby('Episode').apply(lambda x:x.iloc[-1])
        mean_steps_episode = df.shape[0]/self.d["log_every_n_steps"]
        mean_reward_episode = gb.mean()['Reward']
        return mean_steps_episode, mean_reward_episode

    # Construct an epilson greedy policy with given exploration schedule
    def select_epilson_greedy_action(self, model, obs, eps_threshold, device):
        sample = random.random()
        if sample > eps_threshold:
            obs = torch.from_numpy(obs).unsqueeze(0).float().to(device)
            with torch.no_grad():
                # 这里不记录梯度信息
                # 这是因为之后会在batch的阶段重新计算
                model.eval()
                output = model(obs).data
                output = output.squeeze().max(1)[1].cpu()
                model.train()
            action_buffer = self.onehot_encode(output, len(self.d["action_enum"]))
        else:
            action_buffer = self.rand_action(self.d["num_actor"], len(self.d["action_enum"]))
        return action_buffer

    def create_explor_schedule(self):
        if self.d["schedule_type"] == "linear":
            schedule_timesteps = self.d["schedule_timesteps"]
            final_p = self.d["final_p"]
            exploration = LinearSchedule(schedule_timesteps, final_p)
        return exploration

    def create_optim(self, params):
        if self.d["optim_type"] == "RMSprop":
            optimizer = optim.RMSprop(params,
                              lr=self.d["learning_rate"], 
                              alpha=self.d["alpha"], 
                              eps=self.d["eps"],
                              )
        elif self.d["optim_type"] == "Adam":
            ptimizer = optim.Adam(params,
                                  lr=self.d["learning_rate"], 
                                  eps=self.d["eps"],
                                 )
            # optimizer = 
        return optimizer

    # 负责learning rate的变化
    def create_optim_scheduler(self, optimizer):
        if self.d["step_optimizer"] == True:
            optim_scheduler = StepLR(optimizer, 
                                     step_size=self.d["step_size"], 
                                     gamma=self.d["gamma_lr"],
                                     )
            return optim_scheduler
        else:
            return None

    @staticmethod
    def create_df_res():
        cols = ["Timestep", "Episode", "Reward", "Exploration"]
        return pd.DataFrame(columns=cols)

    def create_replay_buffer(self):
        # Construct the replay buffer
        return ReplayBuffer(
                            self.d["replay_buffer_size"], 
                            self.d["num_actor"],
                            len(self.d["action_enum"]),
                            self.num_observation,
                            )