'''
The main program for RL+dispatch
'''
import pandapower as pp


GRID_NAME = "14nodes"
grid_path = f"grid\\{GRID_NAME}\\{GRID_NAME}.json"
net = pp.from_json(grid_path)
pp.runpp(net)
print(net['res_bus'].to_string())