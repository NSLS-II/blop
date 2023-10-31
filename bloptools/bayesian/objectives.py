from collections.abc import Sequence
from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np
import pandas as pd
from ophyd import Signal

numeric = Union[float, int]

DEFAULT_MINIMUM_SNR = 1e1
OBJ_FIELDS = ["description", "key", "limits", "weight", "minimize", "log"]


class DuplicateKeyError(ValueError):
    ...


def _validate_objectives(objectives):
    keys = [obj.key for obj in objectives]
    unique_keys, counts = np.unique(keys, return_counts=True)
    duplicate_keys = unique_keys[counts > 1]
    if len(duplicate_keys) > 0:
        raise DuplicateKeyError(f'Duplicate key(s) in supplied objectives: "{duplicate_keys}"')


@dataclass
class Objective:
    key: str
    name: str = None
    target: Union[float, str] = "max"
    log: bool = False
    weight: numeric = 1.0
    limits: Tuple[numeric, numeric] = None
    min_snr: numeric = DEFAULT_MINIMUM_SNR
    units: str = None

    def __post_init__(self):
        if self.name is None:
            self.name = self.key

        if self.limits is None:
            if self.log:
                self.limits = (0, np.inf)
            else:
                self.limits = (-np.inf, np.inf)

        if type(self.target) is str:
            if self.target not in ["min", "max"]:
                raise ValueError("'target' must be either 'min', 'max', or a number.")

        self.device = Signal(name=self.name)

    @property
    def label(self):
        return f"{'neg ' if self.target == 'min' else ''}{'log ' if self.log else ''}{self.name}"

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
        for attr in ["log"]:
            summary[attr] = summary[attr].astype(bool)

        return summary

    def __repr__(self):
        return self.summary.__repr__()

    # @property
    # def descriptions(self) -> list:
    #     return [obj.description for obj in self.objectives]

    @property
    def keys(self) -> list:
        """
        Returns an array of the objective weights.
        """
        return [obj.key for obj in self.objectives]

    @property
    def weights(self) -> np.array:
        """
        Returns an array of the objective weights.
        """
        return np.array([obj.weight for obj in self.objectives])

    def add(self, objective):
        _validate_objectives([*self.objectives, objective])
        self.objectives.append(objective)
