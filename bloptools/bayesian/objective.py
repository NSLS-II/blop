from collections.abc import Sequence
from typing import Tuple, Union

import numpy as np
import pandas as pd

numeric = Union[float, int]

DEFAULT_MINIMUM_SNR = 1e1
OBJ_FIELDS = ["name", "key", "limits", "weight", "minimize", "log"]


class DuplicateKeyError(ValueError):
    ...


def _validate_objectives(objectives):
    keys = [obj.key for obj in objectives]
    unique_keys, counts = np.unique(keys, return_counts=True)
    duplicate_keys = unique_keys[counts > 1]
    if len(duplicate_keys) > 0:
        raise DuplicateKeyError(f'Duplicate key(s) in supplied objectives: "{duplicate_keys}"')


class Objective:
    def __init__(
        self,
        key: str,
        name: str = None,
        minimize: bool = False,
        target: bool = None,
        log: bool = False,
        weight: numeric = 1.0,
        limits: Tuple[numeric, numeric] = None,
        min_snr: numeric = DEFAULT_MINIMUM_SNR,
    ):
        self.name = name if name is not None else key
        self.key = key
        self.target = target

        if target is not None:
            self.mode = "target"
            self.minimize = True  # this overwrites the passed value

        self.minimize = minimize
        self.log = log
        self.weight = weight
        self.min_snr = min_snr

        if limits is not None:
            self.limits = limits
        elif self.log:
            self.limits = (0, np.inf)
        else:
            self.limits = (-np.inf, np.inf)

    @property
    def label(self):
        return f"{'neg ' if self.minimize else ''}{'log ' if self.log else ''}{self.name}"

    @property
    def summary(self):
        series = pd.Series()
        for col in OBJ_FIELDS:
            series[col] = getattr(self, col)
        return series

    def __repr__(self):
        return self.summary.__repr__()

    @property
    def noise(self):
        return self.model.likelihood.noise.item() if hasattr(self, "model") else None


class ObjectiveList(Sequence):
    def __init__(self, objectives: list = []):
        _validate_objectives(objectives)
        self.objectives = objectives

    def __getitem__(self, i):
        return self.objectives[i]

    def __len__(self):
        return len(self.objectives)

    @property
    def summary(self):
        summary = pd.DataFrame(columns=OBJ_FIELDS)
        for i, obj in enumerate(self.objectives):
            for col in summary.columns:
                summary.loc[i, col] = getattr(obj, col)

        # convert dtypes
        for attr in ["minimize", "log"]:
            summary[attr] = summary[attr].astype(bool)

        return summary

    def __repr__(self):
        return self.summary.__repr__()

    @property
    def names(self) -> list:
        return [obj.name for obj in self.objectives]

    @property
    def weights(self) -> np.array:
        """
        Returns an array of the objective weights.
        """
        return np.array([obj.weight for obj in self.objectives])

    def add(self, objective):
        _validate_objectives([*self.objectives, objective])
        self.objectives.append(objective)
