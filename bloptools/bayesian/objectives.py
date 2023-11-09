from collections.abc import Sequence
from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np
import pandas as pd

numeric = Union[float, int]

DEFAULT_MINIMUM_SNR = 1e1
OBJ_FIELDS = ["description", "target", "limits", "weight", "log", "n", "snr", "min_snr"]


class DuplicateNameError(ValueError):
    ...


def _validate_objectives(objectives):
    names = [obj.name for obj in objectives]
    unique_names, counts = np.unique(names, return_counts=True)
    duplicate_names = unique_names[counts > 1]
    if len(duplicate_names) > 0:
        raise DuplicateNameError(f'Duplicate name(s) in supplied objectives: "{duplicate_names}"')


@dataclass
class Objective:
    name: str
    description: str = None
    target: Union[float, str] = "max"
    log: bool = False
    weight: numeric = 1.0
    limits: Tuple[numeric, numeric] = None
    min_snr: numeric = DEFAULT_MINIMUM_SNR
    units: str = None

    def __post_init__(self):
        if self.limits is None:
            if self.log:
                self.limits = (0, np.inf)
            else:
                self.limits = (-np.inf, np.inf)

        if type(self.target) is str:
            if self.target not in ["min", "max"]:
                raise ValueError("'target' must be either 'min', 'max', or a number.")

        # self.device = Signal(name=self.name)

    @property
    def label(self):
        return f"{'log ' if self.log else ''}{self.description}"

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

    @property
    def snr(self):
        return np.round(1 / self.model.likelihood.noise.sqrt().item(), 1) if hasattr(self, "model") else None

    @property
    def n(self):
        return self.model.train_targets.shape[0] if hasattr(self, "model") else 0


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
    def names(self) -> list:
        """
        Returns an array of the objective weights.
        """
        return [obj.name for obj in self.objectives]

    @property
    def targets(self) -> np.array:
        """
        Returns an array of the objective weights.
        """
        return [obj.target for obj in self.objectives]

    @property
    def weights(self) -> np.array:
        """
        Returns an array of the objective weights.
        """
        return np.array([obj.weight for obj in self.objectives])

    @property
    def signed_weights(self) -> np.array:
        """
        Returns a signed array of the objective weights.
        """
        return np.array([(1 if obj.target == "max" else -1) * obj.weight for obj in self.objectives])

    def add(self, objective):
        _validate_objectives([*self.objectives, objective])
        self.objectives.append(objective)
