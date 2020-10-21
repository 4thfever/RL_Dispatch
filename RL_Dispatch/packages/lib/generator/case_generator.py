'''
Power flow case generator
'''
import copy
import numpy as np
import pandapower as pp

def fluc_value(_net, _obj, _attr, area):
    values = _net[_obj][_attr]
    ratios = np.random.random_sample(size=values.shape[0])
    ratios = area[0] + ratios * (area[1] - area[0])
    _net[_obj][_attr] = values * ratios
    return _net

def case_generate(d, output=True):
    base_grid = d["base_grid"]
    data_folder = d["data_folder"]
    num_case = d["num_case"]
    obj2change = d["obj2change"]
    attr2change = d["attr2change"]
    fluc_area = d["fluc_area"]

    net = pp.from_json(base_grid)

    # generating cases
    for num in range(num_case):
        print(num)
        # set network's deepcopy
        net_copy = copy.deepcopy(net)

        # creatifng new values
        for i in range(len(obj2change)):
            _obj = obj2change[i]
            _attr = attr2change[i]
            area = fluc_area[i]
            net_copy = fluc_value(net_copy, _obj, _attr, area)            

        net_copy_path = f"{data_folder}\\{num}.json"
        if output == True:
            pp.to_json(net_copy, net_copy_path)

