import gym
import yaml
import torch.optim as optim

from dqn_model import DQN
from dqn_learn import OptimizerSpec, dqn_learing
from utils.gym import get_env, get_wrapper_by_name
from utils.schedule import LinearSchedule

with open('../config.yaml') as file:
    d = yaml.load(file)
    BATCH_SIZE = d["batch_size"]
    GAMMA = d["gamma"]
    REPLAY_BUFFER_SIZE = d["replay_buffer_size"]
    LEARNING_STARTS = d["learning_starts"]
    LEARNING_FREQ = d["learning_freq"]
    FRAME_HISTORY_LEN = d["frame_history_len"]
    TARGER_UPDATE_FREQ = d["targer_update_freq"]
    LEARNING_RATE = d["learning_rate"]
    ALPHA = d["alpha"]
    EPS = d["eps"]

def main(env):
    optimizer_spec = OptimizerSpec(
        constructor=optim.RMSprop,
        kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS),
    )

    exploration_schedule = LinearSchedule(1000000, 0.1)

    dqn_learing(
        env=env,
        q_func=DQN,
        optimizer_spec=optimizer_spec,
        exploration=exploration_schedule,
        replay_buffer_size=REPLAY_BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        learning_starts=LEARNING_STARTS,
        learning_freq=LEARNING_FREQ,
        frame_history_len=FRAME_HISTORY_LEN,
        target_update_freq=TARGER_UPDATE_FREQ,
    )

if __name__ == '__main__':
    # Get Atari games.
    benchmark = gym.benchmark_spec('Atari40M')

    # Change the index to select a different game.
    task = benchmark.tasks[3]

    # Run training
    seed = 0 # Use a seed of zero (you may want to randomize the seed!)
    env = get_env(task, seed)

    main(env)
