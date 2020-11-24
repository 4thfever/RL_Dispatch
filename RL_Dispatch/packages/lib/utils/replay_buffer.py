"""
储存各类data，用于之后的强化学习训练
"""
import numpy as np
import random

def sample_n_unique(sampling_f, n):
    """Helper function. Given a function `sampling_f` that returns
    comparable objects, sample n such unique objects.
    """
    res = []
    while len(res) < n:
        candidate = sampling_f()
        if candidate not in res:
            res.append(candidate)
    return res

class ReplayBuffer(object):
    def __init__(self, size, num_actor, num_action, num_observation):
        """This is a memory efficient implementation of the replay buffer."""
        self.size = size
        self.num_actor = num_actor
        self.num_action = num_action
        self.num_observation = num_observation

        self.next_idx      = 0
        self.num_in_buffer = 0

        self.obs      = None
        self.action   = None
        self.reward   = None
        self.done     = None

    def can_sample(self, batch_size):
        """Returns true if `batch_size` different transitions can be sampled from the buffer."""
        return batch_size + 1 <= self.num_in_buffer

    def _encode_sample(self, idxes):
        obs_batch      = self.obs[idxes]
        act_batch      = self.action[idxes]
        rew_batch      = self.reward[idxes]
        next_obs_batch = self.obs[[idx +1 for idx in idxes]]
        done_mask      = np.array([1.0 if self.done[idx] else 0.0 for idx in idxes], dtype=np.float32)

        return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask


    def sample(self, batch_size):
        """
        从库中取出batch_size个对象
        when observing `obs_batch[i]`, action `act_batch[i]` was taken,
        after which reward `rew_batch[i]` was received and subsequent
        observation  next_obs_batch[i] was observed, unless the epsiode
        was done which is represented by `done_mask[i]` which is equal
        to 1 if episode has ended as a result of that action.
        """
        assert self.can_sample(batch_size)
        idxes = sample_n_unique(lambda: random.randint(0, self.num_in_buffer - 2), batch_size)
        return self._encode_sample(idxes)

    def store_obs(self, input_obs):
        if self.obs is None:
            self.obs      = np.zeros([self.size, self.num_observation], dtype=np.float32)
            self.action   = np.zeros([self.size, self.num_actor, self.num_action], dtype=np.int32)
            self.reward   = np.zeros([self.size], dtype=np.float32)
            self.done     = np.zeros([self.size], dtype=np.bool)

        self.obs[self.next_idx] = input_obs

        ret = self.next_idx
        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)
        return ret

    def store_result(self, idx, action, reward, done):
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx]   = done
