from collections.abc import Sequence
from typing import Tuple, Union

import numpy as np
import pandas as pd

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


class Objective:
    """A degree of freedom (DOF), to be used by an agent.

    Parameters
    ----------
    description: str
        The description of the DOF. This is used as a key.
    description: str
        A longer description for the DOF.
    device: Signal, optional
        An ophyd device. If None, a dummy ophyd device is generated.
    limits: tuple, optional
        A tuple of the lower and upper limit of the DOF. If the DOF is not read-only, the agent
        will not explore outside the limits. If the DOF is read-only, the agent will reject all
        sampled data where the DOF is outside the limits.
    read_only: bool
        If True, the agent will not try to set the DOF. Must be set to True if the supplied ophyd
        device is read-only.
    active: bool
        If True, the agent will try to use the DOF in its optimization. If False, the agent will
        still read the DOF but not include it any model or acquisition function.
    units: str
        The units of the DOF (e.g. mm or deg). This is only for plotting and general housekeeping.
    tags: list
        A list of tags. These make it easier to subset large groups of dofs.
    latent_group: optional
        An agent will fit latent dimensions to all DOFs with the same latent_group. If None, the
        DOF will be modeled independently.
    """

    def __init__(
        self,
        key: str,
        description: str = "",
        minimize: bool = False,
        log: bool = False,
        weight: numeric = 1.0,
        limits: Tuple[numeric, numeric] = None,
        min_snr: numeric = DEFAULT_MINIMUM_SNR,
    ):
        self.description = description if description is not None else key
        self.key = key
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
        return f"{'neg ' if self.minimize else ''}{'log ' if self.log else ''}{self.description}"

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
