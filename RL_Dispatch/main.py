import yaml
import torch.optim as optim

from dqn_model import DQN
from dqn_learn import OptimizerSpec, dqn_learing
from lib.utils.pp_wrapper import Wrapper
from lib.utils.env import Env
from lib.utils.schedule import LinearSchedule

with open('config.yaml') as file:
    d = yaml.load(file)
    batch_size = d["batch_size"]
    gamma = d["gamma"]
    replay_buffer_size = d["replay_buffer_size"]
    learning_starts = d["learning_starts"]
    learning_freq = d["learning_freq"]
    frame_history_len = d["frame_history_len"]
    target_update_freq = d["target_update_freq"]
    learning_rate = d["learning_rate"]
    alpha = d["alpha"]
    eps = d["eps"]
    num_actor = d["num_actor"]
    action_enum = d["action_enum"]
    num_observer = d["num_observer"]
    observe_attribute = d["observe_attribute"]

def main(env):
    optimizer_spec = OptimizerSpec(
        constructor=optim.RMSprop,
        kwargs=dict(lr=learning_rate, alpha=alpha, eps=eps),
    )

    exploration_schedule = LinearSchedule(1000000, 0.1)

    dqn_learing(
        env=env,
        q_func=DQN,
        optimizer_spec=optimizer_spec,
        exploration=exploration_schedule,
        # replay_buffer_size=REPLAY_BUFFER_SIZE,
        replay_buffer_size=10000,
        batch_size=batch_size,
        gamma=gamma,
        learning_starts=1000,
        # learning_starts=learning_starts,
        learning_freq=learning_freq,
        frame_history_len=frame_history_len,
        target_update_freq=100,
        # target_update_freq=targer_update_freq,
        num_actor = num_actor,
        action_enum = action_enum,
        num_observer = num_observer,
        observe_attribute = observe_attribute,
    )

if __name__ == '__main__':
    seed = 0 # 需要随机数吗?
    env = Env()
    main(env)
