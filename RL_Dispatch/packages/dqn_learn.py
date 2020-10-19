"""
用来设定dqn的学习超参，执行学习过程
主要的bug来自于Q网络的输出是二维的
"""
import sys
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

from .lib.utils.replay_buffer import ReplayBuffer



USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)

"""
    OptimizerSpec containing following attributes
        constructor: The optimizer constructor ex: RMSprop
        kwargs: {Dict} arguments for constructing optimizer
"""
OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])


def dqn_learing(
    env,
    q_func,
    optimizer_spec,
    exploration,
    d,
    ):

    """Run Deep Q-learning algorithm.

    You can specify your own convnet using q_func.

    All schedules are w.r.t. total number of steps taken in the environment.

    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    q_func: function
        Model to use for computing the q function. It should accept the
        following named arguments:
            input_channel: int
                number of channel of input.
            array_action: int
                number of actions
    optimizer_spec: OptimizerSpec
        Specifying the constructor and kwargs, as well as learning rate schedule
        for the optimizer
    exploration: Schedule (defined in utils.schedule)
        schedule for probability of chosing random action.
    replay_buffer_size: int
        How many memories to store in the replay buffer.
    batch_size: int
        How many transitions to sample each time experience is replayed.
    gamma: float
        Discount Factor
    learning_starts: int
        After how many environment steps to start replaying experiences
    learning_freq: int
        How many steps of environment to take between every experience replay
    target_update_freq: int
        How many experience replay rounds (not steps!) to perform between
        each update to the target Q network
    """

    ###############
    # BUILD MODEL #
    ###############
    batch_size = d["batch_size"]
    gamma = d["gamma"]
    replay_buffer_size = d["replay_buffer_size"]
    learning_starts = d["learning_starts"]
    learning_freq = d["learning_freq"]
    target_update_freq = d["target_update_freq"]
    num_actor = d["num_actor"]
    action_enum = d["action_enum"]
    num_observer = d["num_observer"]
    observe_attribute = d["observe_attribute"]
    num_layer = d["num_layer"]
    layer_size = d["layer_size"]
    log_every_n_steps = d["log_every_n_steps"]

    # 这里×1是因为只观测一个属性
    num_observation = num_observer * 1

    # Construct an epilson greedy policy with given exploration schedule
    def select_epilson_greedy_action(model, obs, t):
        sample = random.random()
        eps_threshold = exploration.value(t)
        if sample > eps_threshold:
            obs = torch.from_numpy(obs).type(dtype)
            with torch.no_grad():
                # 这里不记录梯度信息
                # 这是因为之后会在batch的阶段重新计算
                output = model(obs).data
            action_buffer = env.onehot_encode(output.max(1)[1].cpu(), len(action_enum))
        else:
            action_buffer = env.rand_action(num_actor, len(action_enum))
        return action_buffer

    # Initialize target q function and q function
    Q = q_func(num_observation, len(action_enum), num_actor, num_layer, layer_size).type(dtype)
    target_Q = q_func(num_observation, len(action_enum), num_actor, num_layer, layer_size).type(dtype)

    # Construct Q network optimizer function
    optimizer = optimizer_spec.constructor(Q.parameters(), **optimizer_spec.kwargs)

    # log info
    best_mean_episode_reward = -1000
    cols = ["Timestep", "Episode", "Reward", "Exploration"]
    df_res = pd.DataFrame(columns=cols)

    # Construct the replay buffer
    replay_buffer = ReplayBuffer(
                                replay_buffer_size, 
                                num_actor,
                                len(action_enum),
                                num_observation
                                )

    ###############
    # RUN ENV     #
    ###############
    num_param_updates = 0
    mean_episode_reward = -float('nan')
    best_mean_episode_reward = -float('inf')
    last_obs = env.initial_run()

    for t in count():
        env.num_step = t

        ### Check stopping criterion
        if env.stopping_criterion():
            break

        ### Step the env and store the transition
        # Store lastest observation in replay memory and last_idx can be used to store action, reward, done
        last_idx = replay_buffer.store_obs(last_obs)
        # encode_recent_observation will take the latest observation
        # that you pushed into the buffer and compute the corresponding
        # input that should be given to a Q network by appending some
        # previous frames.
        recent_observations = replay_buffer.encode_recent_observation()

        # Choose random action if not yet start learning
        if t > learning_starts:
            action = select_epilson_greedy_action(Q, recent_observations, t)
        else:
            action = env.rand_action(num_actor, len(action_enum))
        # Advance one step
        obs, reward, done = env.step(action)
        # Store other info in replay memory
        replay_buffer.store_effect(last_idx, action.squeeze(), reward, done)
        # Resets the environment when reaching an episode boundary.
        if done:
            obs = env.reset()
        last_obs = obs

        ### Perform experience replay and train the network.
        # Note that this is only done if the replay buffer contains enough samples
        # for us to learn something useful -- until then, the model will not be
        # initialized and random actions should be taken
        if (t > learning_starts and
                t % learning_freq == 0 and
                replay_buffer.can_sample(batch_size)):
            # Use the replay buffer to sample a batch of transitions
            # Note: done_mask[i] is 1 if the next state corresponds to the end of an episode,
            # in which case there is no Q-value at the next state; at the end of an
            # episode, only the current state reward contributes to the target
            obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = replay_buffer.sample(batch_size)
            # print(rew_batch)
            # Convert numpy nd_array to torch variables for calculation
            obs_batch = torch.from_numpy(obs_batch).type(dtype)
            act_batch = torch.from_numpy(act_batch).int()
            rew_batch = torch.from_numpy(rew_batch)
            next_obs_batch = torch.from_numpy(next_obs_batch).type(dtype)
            not_done_mask = torch.from_numpy(1 - done_mask).type(dtype)

            if USE_CUDA:
                act_batch = act_batch.cuda()
                rew_batch = rew_batch.cuda()

            # Compute current Q value, q_func takes only state and output value for every state-action pair
            # We choose Q based on action taken.

            # 把Q value中对应实际动作的那部分挑出来，并整理shape
            current_Q_values = Q(obs_batch).gather(-1, env.onehot_decode(act_batch).unsqueeze(-1))
            # Compute next Q value based on which action gives max Q values
            # Detach variable from the current graph since we don't want gradients for next Q to propagated
            
            # 挑出q值中最大的那个，映射到没有结束的调度序列中
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
            num_param_updates += 1

            # Periodically update the target network by Q network to target Q network
            if num_param_updates % target_update_freq == 0:
                target_Q.load_state_dict(Q.state_dict())


        ### 4. Log progress and keep track of statistics
        if t > learning_starts:
            exploration_value = exploration.value(t)
        else:
            exploration_value = 'None'
        df_res.loc[t, :] = [t, env.num_episode, replay_buffer.reward[t], exploration_value]

        if t % log_every_n_steps == 0:
            # 输出信息
            print("Timestep: %d" % (t,))
            print("Episodes: %d" % env.num_episode)
            if t > learning_starts:
                mean_steps_per_episode, mean_episode_reward = env.cal_epi_reward(df_res, env.num_episode)
                best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)
                print("Mean reward (100 episodes): %f" % mean_episode_reward)
                print("Best mean reward: %f" % best_mean_episode_reward)
                print("Exploration: %f" % exploration_value)
            print("\n")

    # 结束
    df_res.to_csv("res.csv", index=False)
    epi_x = list(range(100, df_res.iloc[-1]["Episode"]))
    epi_reward = [env.cal_epi_reward(df_res, ele)[1] for ele in epi_x]
    plt.plot(epi_x, epi_reward)
    plt.show()