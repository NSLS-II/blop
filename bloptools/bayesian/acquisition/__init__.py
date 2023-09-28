import os

import yaml

from .analytic import *  # noqa F401
from .monte_carlo import *  # noqa F401

here, this_filename = os.path.split(__file__)

with open(f"{here}/config.yml", "r") as f:
    config = yaml.safe_load(f)


def parse_acq_func(acq_func_identifier):
    acq_func_name = None
    for _acq_func_name in config.keys():
        if acq_func_identifier.lower() in config[_acq_func_name]["identifiers"]:
            acq_func_name = _acq_func_name

    if acq_func_name is None:
        raise ValueError(f'Unrecognized acquisition function identifier "{acq_func_identifier}".')

    return acq_func_name
