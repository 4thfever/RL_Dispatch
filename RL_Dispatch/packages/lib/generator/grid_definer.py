'''
Define, save & load power grid
'''
import yaml
import pandapower as pp
import pandapower.networks as pn
import pandapower.plotting as ppl

with open('grid.yaml') as file:
    d = yaml.load(file)

if d['EXISTED_IN_PP']:
	sentence = f"pn.{d['PP_NAME']}()"
	print(sentence)
	net = eval(sentence)

print(net)
pp.to_json(net, filename=f"grid/{d['GRID_NAME']}/{d['FILE_NAME']}")
ppl.to_html(net, filename=f"grid/{d['GRID_NAME']}/info.html") 