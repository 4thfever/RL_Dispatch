import os
import pandapower as pp
import pandapower.networks as pn

from lib.utils.yaml_helper import serializable
from lib.data.grid_constructor import grid_constructor
from .sample_transform import *

@serializable
class powerFlowLoader(object):
    def __init__(self,
                 base_grid = None,
                 sample_construct = None,
                 input_def = None):
        super(powerFlowLoader, self).__init__()
        self._sample_constructor = sample_construct
        self.sample = dict()
        self.sample['__base_grid__'] = [self.__construct_base_grid(base_grid)]

    def __construct_base_grid(self, base_grid_cfg):
        # get case from pandapower.network
        if base_grid_cfg['existed_in_pp']:
            assert base_grid_cfg['name'] in dir(pn), '{} is not found in pandapower.networks!'.format(base_grid_cfg['name'])
            return getattr(pn, base_grid_cfg['name'])()

        # if network is not existed in pandapower.network, construct network from yaml in self.grid_path
        return grid_constructor(base_grid_cfg['grid_path'])

    def construct_sample(self):
        for sample_transformer in self._sample_constructor:
            print('!',sample_transformer.__class__.__name__)
            output_labels, output_samples = sample_transformer.transform(self.sample)
            print('!!!',output_labels,output_samples)

            for idx,_ in enumerate(output_labels):
                self.sample[output_labels[idx]] = output_samples[idx]
