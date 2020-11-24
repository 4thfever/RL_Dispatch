import yaml
import copy

from packages.dqn_learn import dqn_learing

list_modification = [
        {"Optimizer":"Adam"},
        {"learning_rate":0.01},
        {"learning_rate":0.005},
        {"learning_rate":0.001},
        {"schedule_timesteps":10000},
        {"schedule_timesteps":20000},
        {"target_update_freq":100},
        {"target_update_freq":200},
        {"target_update_freq":1000},
        {"use_batchnorm":False},
        {"learning_freq":8},
        {"learning_freq":16},
        {"learning_freq":32},
        {
         "observer":["res_bus", "res_bus"], 
         "observe_attribute":["vm_pu", "va_degree"]
         },
        {
         "observer":["res_bus", "res_bus", "res_bus", "res_line", "res_line"], 
         "observe_attribute":["vm_pu", "va_degree", "p_mw", "p_from_mw", "q_from_mvar"]
         },
        {"batch_size":64},
        {"batch_size":128},
        {"batch_size":256},
        {"num_layer":2, "layer_size":[64,32]},
        {"num_layer":2, "layer_size":[32,32]},
        {"num_layer":3, "layer_size":[128,64,32]},
        {"num_layer":3, "layer_size":[64,64,32]},
        {"num_layer":3, "layer_size":[64,32,32]},
        {"num_layer":4, "layer_size":[64,64,32,32]},
        {"num_layer":4, "layer_size":[128,64,64,32]},
        {"num_layer":4, "layer_size":[128,64,32,32]},
        {"replay_buffer_size":20000},
        {"replay_buffer_size":30000},
        {"replay_buffer_size":50000},
        {"dropout":0.1},
        {"dropout":0.2},
        {"dropout":0.5},
    ]

def main():
    times_per_config = 5
    with open('config.yaml') as file:
        d = yaml.load(file)

    for i, mod in enumerate(list_modification):
        d_temp = copy.deepcopy(d)
        # 改动

        for key_, value_ in mod.items():
            d_temp[key_] = value_
        # 运行
        for j in range(times_per_config):
            print("modification :")
            print(mod)
            print(f"No.{j} times")
            path = f"exp_result/{i}_{j}"
            dqn_learing(d_temp, output_path=path, save_or_plot="save")


if __name__ == '__main__':
    main()
