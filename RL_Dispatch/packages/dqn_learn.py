"""
用来设定dqn的学习超参，
执行学习训练和更新过程
"""
import sys
import time
import random
import pickle
from collections import namedtuple
from itertools import count

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import matplotlib.pyplot as plt
from torch import optim
from torch.optim.lr_scheduler import StepLR

from .lib.utils.replay_buffer import ReplayBuffer

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dqn_learing(
    env,
    q_func,
    exploration,
    d,
    ):

    ###############
    # BUILD MODEL #
    ###############
    # 读入参数，不用locals()是因为我希望所有的param都有据可依
    batch_size = d["batch_size"]
    gamma = d["gamma"]
    replay_buffer_size = d["replay_buffer_size"]
    learning_starts = d["learning_starts"]
    learning_freq = d["learning_freq"]
    target_update_freq = d["target_update_freq"]
    num_actor = d["num_actor"]
    action_enum = d["action_enum"]
    observe_attribute = d["observe_attribute"]
    num_layer = d["num_layer"]
    layer_size = d["layer_size"]
    log_every_n_steps = d["log_every_n_steps"]
    learning_rate = d["learning_rate"]
    alpha = d["alpha"]
    eps = d["eps"]

    # Initialize target q function and q function
    Q = q_func(env.num_observation, len(action_enum), num_actor, num_layer, layer_size).to(device)
    target_Q = q_func(env.num_observation, len(action_enum), num_actor, num_layer, layer_size).to(device)

    optimizer = optim.RMSprop(Q.parameters(),
                              lr=learning_rate, 
                              alpha=alpha, 
                              eps=eps,
                              )
    # 负责learning rate的变化
    if d["step_optimizer"] == True:
        step_size = d["step_size"]
        gamma_lr = d["gamma_lr"]
        optim_scheduler = StepLR(optimizer, 
                                 step_size=step_size, 
                                 gamma=gamma_lr,
                                 )

    # 初始化log info
    best_mean_reward = -1
    cols = ["Timestep", "Episode", "Reward", "Exploration"]
    df_res = pd.DataFrame(columns=cols)

    # Construct the replay buffer
    replay_buffer = ReplayBuffer(
                                replay_buffer_size, 
                                num_actor,
                                len(action_enum),
                                env.num_observation,
                                )

    ###############
    # RUN ENV     #
    ###############
    explor_value = 'None'
    mean_episode_reward = -float('nan')
    best_mean_episode_reward = -float('inf')
    last_obs = env.initial_run()

    time_point = time.time()
    for t in count():
        if d["step_optimizer"] == True:
            optim_scheduler.step()

        env.num_step = t
        if env.stopping_criterion():
            break

        ### Step the env and store the transition
        # Store lastest observation in replay memory and last_idx can be used to store action, reward, done
        last_idx = replay_buffer.store_obs(last_obs)

        # Choose random action if not yet start learning
        if t > learning_starts:
            explor_value = exploration.value(t)
            Q.eval()
            action = env.select_epilson_greedy_action(Q, last_obs, explor_value, device)
            Q.train()
        else:
            action = env.rand_action(num_actor, len(action_enum))

        # Advance one step
        last_obs, reward, done = env.step(action)
        replay_buffer.store_result(last_idx, action.squeeze(), reward, done)
        if done:
            last_obs = env.reset()

        # 初始化batch norm
        # 这个是因为batch norm刚开始是不知道mean和var的
        # 需要先给一个sample来训练上
        if t == learning_starts:
            input_ = torch.FloatTensor(replay_buffer.obs[:learning_starts+1]).to(device)
            input_ = Q.bn(input_)


        ### Perform experience replay and train the network.
        if (t > learning_starts and
            t % learning_freq == 0 and
            replay_buffer.can_sample(batch_size)):
            # Use the replay buffer to sample a batch of transitions
            # Note: done_mask[i] is 1 if the next state corresponds to the end of an episode,
            # in which case there is no Q-value at the next state; at the end of an
            # episode, only the current state reward contributes to the target
            obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = replay_buffer.sample(batch_size)
           
            # 看是不是gpu版本 
            obs_batch = torch.from_numpy(obs_batch).to(device)
            act_batch = torch.from_numpy(act_batch).to(device)
            rew_batch = torch.from_numpy(rew_batch).to(device)
            next_obs_batch = torch.from_numpy(next_obs_batch).to(device)
            not_done_mask = torch.from_numpy(1 - done_mask).to(device)

            # Compute current Q value, q_func takes only state and output value for every state-action pair
            # 把Q value中对应实际动作的那部分挑出来，并整理shape
            current_Q_values = Q(obs_batch).gather(-1, env.onehot_decode(act_batch).unsqueeze(-1))
            # 挑出q值中最大的那个，映射到没有结束的调度序列中
            # Detach, 因为next Q的梯度是不希望传播的
            next_max_q = target_Q(next_obs_batch).detach().max(2)[0]
            next_Q_values = not_done_mask.unsqueeze(1) * next_max_q

            # Compute the target of the current Q values
            rew_batch = rew_batch.unsqueeze(1)
            # 拼接reward让其可以对应二维输出
            rew_batch_resize = torch.cat((rew_batch, rew_batch, rew_batch, rew_batch), 1)
            target_Q_values = rew_batch_resize + (gamma * next_Q_values)
            # Compute Bellman error
            bellman_error = target_Q_values - current_Q_values.squeeze()
            # clip the bellman error between [-1 , 1]
            clipped_bellman_error = bellman_error.clamp(-1, 1)
            # Note: clipped_bellman_delta * -1 will be right gradient
            d_error = clipped_bellman_error * -1.0
            # Clear previous gradients before backward pass
            optimizer.zero_grad()
            # run backward pass
            current_Q_values.backward(d_error.unsqueeze(2).data)

            # Perfom the update
            optimizer.step()

            # Periodically update the target network by Q network to target Q network
            if t % target_update_freq == 0:
                target_Q.load_state_dict(Q.state_dict())


        ### 4. Log progress and keep track of statistics
        df_res.loc[t, :] = [t, env.num_episode, reward, explor_value]
        time_used = time.time() - time_point

        if t % log_every_n_steps == 0:
            time_point = time.time()
            lr = optimizer.param_groups[0]['lr']
            # 输出信息
            print(f"Timestep: {t}")
            print(f"Episodes: {env.num_episode}")
            print(f"Time Consumption: {time_used:.2f} s")
            if t > learning_starts:
                mean_steps_episode, mean_reward_episode = env.cal_epi_reward(df_res, env.num_episode)
                best_mean_reward = max(best_mean_reward, mean_reward_episode)
                print(f"Mean Reward ({log_every_n_steps} episodes): {mean_reward_episode:.2f}")
                print(f"Best Mean Reward: {best_mean_reward:.2f}")
                print(f"Exploration: {explor_value:.2f}")
                print(f"Learning Rate: {lr}")
            print("\n")

    # 结束
    print("Learning Finished")
    df_res.to_csv("res.csv", index=False)
    epi_x = range(100, df_res.iloc[-1]["Episode"], 100)
    epi_reward = [env.cal_epi_reward(df_res, ele)[1] for ele in epi_x]
    plt.plot(epi_x, epi_reward)
    plt.show()