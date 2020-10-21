import torch.optim as optim
import yaml

from packages.dqn_model import DQN
from packages.dqn_learn import OptimizerSpec, dqn_learing
from packages.lib.utils.pp_wrapper import Wrapper
from packages.lib.utils.env import Env
from packages.lib.utils.schedule import LinearSchedule
from packages.lib.generator.case_generator import case_generate

def main(env):
    optimizer_spec = OptimizerSpec(
        constructor=optim.RMSprop,
        kwargs=dict(lr=learning_rate, alpha=alpha, eps=eps),
    )

    if d["schedule_type"] == "linear":
        schedule_timesteps = d["schedule_timesteps"]
        final_p = d["final_p"]
        exploration_schedule = LinearSchedule(schedule_timesteps, final_p)

    dqn_learing(
        env=env,
        q_func=DQN,
        optimizer_spec=optimizer_spec,
        exploration=exploration_schedule,
        d=d,
    )


if __name__ == '__main__':
    with open('config.yaml') as file:
        d = yaml.load(file)
        learning_rate = d["learning_rate"]
        alpha = d["alpha"]
        eps = d["eps"]

    # case_generate(d)
    seed = 0 # 需要随机数吗?
    env = Env(d)
    main(env)
