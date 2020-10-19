import os
import numpy as np
import re
import tqdm
import copy
import random
import collections

import pandapower as pp

from concurrent.futures import ThreadPoolExecutor as tpool
from lib.utils.load_config import serializable

__all__ = ['sampleTransformer', 'caseDuplicator', 'pfCaseInitializer', 'pfSolver', 'pfActionGenerator']

@serializable
class sampleTransformer(object):
    def __init__(self,
                 input = None,
                 output = None):
        super(sampleTransformer, self).__init__()
        self.input_labels = input
        self.output_def = output
        self.output_labels = [odef['name'] for odef in self.output_def]
        self.output_fields = [odef['fields'] for odef in self.output_def]
        self.pool = tpool(8)

    def transform(self, sample):
        if isinstance(self.input_labels, list):
            input_samples = copy.deepcopy(zip(*[sample[input_label] for input_label in self.input_labels]))

        elif isinstance(self.input_labels, str):
            input_samples = copy.deepcopy(sample[self.input_labels])
        samples = list()
        records = collections.OrderedDict()
        results = list()

        for idx,input_sample in enumerate(input_samples):
            results.append(self.pool.submit(self.worker, idx, input_sample))

        for r in results:
            idc,recs = r.result()
            for idx, rec in zip(idc, recs):
                records[idx] = rec

        records = collections.OrderedDict(sorted(records.items(), key=lambda t:t[0]))
        for record in records:
            samples.append(records[record])

        if len(self.output_fields)==1 and self.output_fields[0]=='__all__':
            return self.output_labels,[samples]
        else:
            return self.output_samples(samples)

    def worker(self, idx, sample):
        raise NotImplementedError("{}.worker is not implemented.".format(self.__class__.__name__))

    def output_samples(self, samples):
        raise NotImplementedError("{}.output_samples is not implemented.".format(self.__class__.__name__))

@serializable
class caseDuplicator(sampleTransformer):
    def __init__(self,
               case_num=None,
               input = None,
               output = None):
        super(caseDuplicator, self).__init__(input=input,output=output)
        self.case_num = case_num

    def worker(self, idx, sample):
        idx_dup = list(idx * self.case_num + np.array([i for i in range(self.case_num)]))
        sample_dup = [sample for _ in range(self.case_num)]
        return idx_dup, sample_dup


@serializable
class pfCaseInitializer(sampleTransformer):
    def __init__(self,
                 case_transform = None,
                 input = None,
                 output = None
                 ):
        super(pfCaseInitializer, self).__init__(input=input,output=output)
        self.case_transform_pipeline = case_transform

    def worker(self, idx, sample):
        if not isinstance(sample, pp.auxiliary.pandapowerNet):
            raise NotImplementedError("pfCaseGenerator only support pandapowerNet samples.")

        for case_transform in self.case_transform_pipeline:
            case_transformer = getattr(self, case_transform['name'])
            case_transform_args = copy.deepcopy(case_transform)
            case_transform_args.pop('name')
            sample = case_transformer(network=sample, **case_transform_args)

        return [idx], [sample]

    def fluctuation_abs(self, network, fluc_states, fluc_vals):
        for idx, state_label in enumerate(fluc_states):
            l = state_label.split(':')
            val = fluc_vals[idx]
            for idx, state in enumerate(network[l[0]][l[1]]):
                fluc = random.random() * (val[1]-val[0]) + val[0]
                network[l[0]][l[1]][idx] = max(state + fluc, 0)
        return network

    def fluctuation_rel(self, network, fluc_states, fluc_ratios):
        for idx, state_label in enumerate(fluc_states):
            l = state_label.split(':')
            ratio = fluc_ratios[idx]
            for idx, state in enumerate(network[l[0]][l[1]]):
                fluc = random.random() * (ratio[1]-ratio[0]) + ratio[0]
                network[l[0]][l[1]][idx] = max(state * fluc, 0)
        return network

@serializable
class pfSolver(sampleTransformer):
    def __init__(self,
                 numba = False,
                 input = None,
                 output = None):
        super(pfSolver, self).__init__(input=input,output=output)
        self.numba = numba

    def worker(self, idx, sample):
        pp.runpp(sample, numba=self.numba)
        return [idx], [sample]

@serializable
class pfActionGenerator(sampleTransformer):
    def __init__(self,
                 action_state = None,
                 action_type = None,
                 action_enum = None,
                 action_range = None,
                 input = None,
                 output = None):
        super(actionGenerator, self).__init__(input=input,output=output)
        self.action_state = action_state
        self.action_type = action_type
        self.action_enum = action_enum
        self.action_range = action_range

    def worker(self, idx, sample):
        action_dict = {}
        for state_idx, state in enumerate(self.action_state):
            if action_type[state_idx]=='discrete':
                action_dict[state] = action_enum[state_idx][random.randint(0,len(action_enum[state_idx]))]
            else:
                raise NotImplementedError("pfActionGenerator only support discrete action now.")
        return [idx], [action_dict]


@serializable
class pfRewardCalculator(sampleTransformer):
    def __init__(self,
               input = None,
               output = None):
        super(rewardCalculator, self).__init__(input=input,output=output)