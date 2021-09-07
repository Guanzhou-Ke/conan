import argparse
import sys

from .emnist import *
from .fmnist import *
from .coil20 import *
from .coil100 import *
from .voc import *


def parse_config_name_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", dest="config_name", required=True)
    return parser.parse_known_args()[0].config_name


def set_cfg_value(cfg, key_list, value):
    sub_cfg = cfg
    for key in key_list[:-1]:
        sub_cfg = getattr(sub_cfg, key)
    setattr(sub_cfg, key_list[-1], value)


def get_config_by_name(name):
    try:
        cfg = getattr(sys.modules[__name__], name)
    except Exception as err:
        print(err)
        raise RuntimeError(f"Config not found: {name}") from err
    return cfg


def get_experiment_config():
    name = parse_config_name_arg()
    cfg = get_config_by_name(name)
    return name, cfg
