'''
Power flow case generator
'''
import yaml
import copy
import numpy as np
import pandapower as pp

GRID_NAME = "14nodes"
grid_path = f"grid\\{GRID_NAME}\\{GRID_NAME}.json"
data_path = f"data\\{GRID_NAME}"

# read infos
with open('config.yaml') as file:
    d = yaml.load(file)

load_fluc = d['LOAD_FLUCTUATION']
if load_fluc:
    load_fluc_bot = d['LOAD_FLUCTUATION_BOTTOM']
    load_fluc_top = d['LOAD_FLUCTUATION_TOP']
net = pp.from_json(grid_path)


# generating cases
for num in range(d['CASE_NUM']):
    print(num)
    # set network's deepcopy
    net_copy = copy.deepcopy(net)
    # load_fluc = False
    if load_fluc:
        # creatifng new load values
        loads = net_copy['load']['p_mw']
        ratios = np.random.random_sample(size=loads.shape[0])
        ratios = load_fluc_bot + ratios * (load_fluc_top - load_fluc_bot)
        net_copy['load']['p_mw'] = loads * ratios

    net_copy_path = f"{data_path}\\{num}.json"
    pp.to_json(net_copy, net_copy_path)
