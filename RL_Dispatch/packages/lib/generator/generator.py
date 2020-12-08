'''
Power flow case generator
'''
import os
import copy
import numpy as np
import pandapower as pp
import pandapower.networks as pn

def grid_generate(net_name, file_name):
    net = getattr(pn, net_name)()
    folder = "packages/grid/" 
    folder += file_name
    if not os.path.exists(folder):
        os.makedirs(folder)

    json_path = folder + '/' + file_name + ".json"
    html_path = folder + '/' + file_name + ".html"

    pp.plotting.to_html(net, html_path)
    pp.to_json(net, json_path)

def fluc_value(_net, _obj, _attr, area):
    values = _net[_obj][_attr]
    ratios = np.random.random_sample(size=values.shape[0])
    # 从过大范围和过小范围中选一个
    subarea = area[np.random.randint(len(area))]
    ratios = subarea[0] + ratios * (subarea[1] - subarea[0])
    _net[_obj][_attr] = values * ratios
    return _net

def case_generate(d, case, output=True):
    base_grid = d["base_grid"]
    if case == 'train':
        data_folder = d["train_folder"]
        num_case = d["num_train_case"]
    elif case == 'test':
        data_folder = d["test_folder"]
        num_case = d["num_test_case"]
    obj2change = d["obj2change"]
    attr2change = d["attr2change"]
    fluc_area = d["fluc_area"]

    net = pp.from_json(base_grid)

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    # generating cases
    for num in range(num_case):
        # set network's deepcopy
        net_copy = copy.deepcopy(net)

        # creatifng new values
        for i in range(len(obj2change)):
            _obj = obj2change[i]
            _attr = attr2change[i]
            area = fluc_area[i]
            net_copy = fluc_value(net_copy, _obj, _attr, area)            

        # 删除外部网络并且置slack节点
        net_copy['ext_grid']['in_service'] = False
        net_copy['gen']['slack'][0] = True

        net_copy_path = f"{data_folder}\\{num}.json"
        if output == True:
            pp.to_json(net_copy, net_copy_path)

