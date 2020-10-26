import torch.optim as optim
import yaml

from packages.dqn_model import DQN
from packages.dqn_learn import dqn_learing
from packages.lib.utils.pp_wrapper import Wrapper
from packages.lib.utils.env import Env
from packages.lib.utils.schedule import LinearSchedule
from packages.lib.generator.case_generator import case_generate

def main():
    with open('config.yaml') as file:
        d = yaml.load(file)

    if d["schedule_type"] == "linear":
        schedule_timesteps = d["schedule_timesteps"]
        final_p = d["final_p"]
        exploration_schedule = LinearSchedule(schedule_timesteps, final_p)

    env = Env(d)
    dqn_learing(
        env=env,
        q_func=DQN,
        exploration=exploration_schedule,
        d=d,
    )


if __name__ == '__main__':
    # case_generate(d)
    seed = 0 # 需要随机数吗?
    main()
