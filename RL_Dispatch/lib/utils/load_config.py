import importlib
import os
import sys

import yaml
import copy
import collections

from . import yaml_helper
from .yaml_helper import serializable

__all__ = [
    'global_config',
    'load_config',
    'merge_config',
    'serializable',
]

class AttrDict(dict):
    """Single level attribute dict, NOT recursive"""

    def __init__(self, **kwargs):
        super(AttrDict, self).__init__()
        super(AttrDict, self).update(kwargs)

    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError("object has no attribute '{}'".format(key))

global_config = AttrDict()

def merge_config(config, another_cfg=None):
    """
    Merge config into global config or another_cfg.

    Args:
        config (dict): Config to be merged.

    Returns: global config
    """
    global global_config
    dct = another_cfg if another_cfg is not None else global_config
    return dict_merge(dct, config)

def dict_merge(dct, merge_dct):
    """ Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.

    Args:
        dct: dict onto which the merge is executed
        merge_dct: dct merged into dct

    Returns: dct
    """
    for k, v in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict) and
                isinstance(merge_dct[k], collections.Mapping)):
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]
    return dct

def load_config(file_path):
    """
    Load config from file.

    Args:
        file_path (str): Path of the config file to be loaded.

    Returns: global config
    """
    _, ext = os.path.splitext(file_path)
    assert ext in ['.yml', '.yaml'], "only support yaml files for now"

    with open(file_path) as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
        merge_config(cfg)

    return global_config
