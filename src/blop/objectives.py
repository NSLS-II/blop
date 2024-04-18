from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import torch

from .utils.functions import approximate_erf

DEFAULT_MIN_NOISE_LEVEL = 1e-6
DEFAULT_MAX_NOISE_LEVEL = 1e0

OBJ_FIELD_TYPES = {
    "description": "object",
    "type": "str",
    "target": "object",
    "active": "bool",
    "trust_domain": "object",
    "active": "bool",
    "weight": "bool",
    "units": "object",
    "log": "bool",
    "min_noise": "float",
    "max_noise": "float",
    "noise": "float",
    "n": "int",
    "latent_groups": "object",
}

ALLOWED_OBJ_TYPES = ["continuous", "binary", "ordinal", "categorical"]


class DuplicateNameError(ValueError):
    ...


def _validate_objectives(objectives):
    names = [obj.name for obj in objectives]
    unique_names, counts = np.unique(names, return_counts=True)
    duplicate_names = unique_names[counts > 1]
    if len(duplicate_names) > 0:
        raise DuplicateNameError(f"Duplicate name(s) in supplied objectives: {duplicate_names}")


@dataclass
class Objective:
    """An objective to be used by an agent.

    Parameters
    ----------
    name: str
        The name of the objective. This is used as a key to index observed data.
    description: str
        A longer description for the objective.
    target: str or float or tuple
        One of "min", "max" , a float, or a tuple of floats. The agent will respectively minimize or maximize the
        objective, target the supplied number, or target the interval of the tuple of numbers.
    log: bool
        Whether to apply a log to the objective, i.e. to make the process more stationary.
    weight: float
        The relative importance of this objective, to be used when scalarizing in multi-objective optimization.
    active: bool
        If True, the agent will care about this objective during optimization.
    limits: tuple of floats
        The range of reliable measurements for the objective. Outside of this, data points will be ignored.
    min_noise: float
        The minimum noise level of the fitted model.
    max_noise: float
        The maximum noise level of the fitted model.
    units: str
        A label representing the units of the objective.
    latent_groups: list of tuples of strs, optional
        An agent will fit latent dimensions to all DOFs with the same latent_group. All other
        DOFs will be modeled independently.
    """

    name: str
    description: str = ""
    type: str = None
    target: Union[Tuple[float, float], float, str] = "max"
    log: bool = False
    weight: float = 1.0
    active: bool = True
    trust_domain: Tuple[float, float] or None = None
    min_noise: float = DEFAULT_MIN_NOISE_LEVEL
    max_noise: float = DEFAULT_MAX_NOISE_LEVEL
    units: str = None
    latent_groups: List[Tuple[str, ...]] = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.target, str):
            if self.target not in ["min", "max"]:
                raise ValueError("'target' must be either 'min', 'max', a number, or a tuple of numbers.")

        if isinstance(self.target, float):
            if self.log and not self.target > 0:
                return ValueError("'target' must strictly positive if log=True.")

        self.type = "fitness" if self.target in ["min", "max"] else "constraint"

    @property
    def _trust_domain(self):
        if self.trust_domain is None:
            return (0, np.inf) if self.log else (-np.inf, np.inf)
        return self.trust_domain

    @property
    def label(self) -> str:
        return f"{'log ' if self.log else ''}{self.description}"

    @property
    def summary(self) -> pd.Series:
        series = pd.Series(index=list(OBJ_FIELD_TYPES.keys()), dtype="object")
        for attr in series.index:
            value = getattr(self, attr)
            if attr == "trust_domain":
                if value is None:
                    value = (0, np.inf) if self.log else (-np.inf, np.inf)
            series[attr] = value
        return series

    @property
    def trust_lower_bound(self):
        if self.trust_domain is None:
            return 0 if self.log else -np.inf
        return float(self.trust_domain[0])

    @property
    def trust_upper_bound(self):
        if self.trust_domain is None:
            return np.inf
        return float(self.trust_domain[1])

    @property
    def noise(self) -> float:
        return self.model.likelihood.noise.item() if hasattr(self, "model") else None

    @property
    def snr(self) -> float:
        return np.round(1 / self.model.likelihood.noise.sqrt().item(), 3) if hasattr(self, "model") else None

    @property
    def n(self) -> int:
        return self.model.train_targets.shape[0] if hasattr(self, "model") else 0

    def targeting_constraint(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(self.target, tuple):
            return None

        a, b = self.target
        p = self.model.posterior(x)
        m = p.mean
        s = p.variance.sqrt()

        return 0.5 * (approximate_erf((b - m) / (np.sqrt(2) * s)) - approximate_erf((a - m) / (np.sqrt(2) * s)))[..., -1]

    def fitness_forward(self, y):
        f = y
        if self.log:
            f = np.log(f)
        if self.target == "min":
            f = -f
        return f

    def fitness_inverse(self, f):
        y = f
        if self.target == "min":
            y = -y
        if self.log:
            y = np.exp(y)
        return y

    @property
    def is_fitness(self):
        return self.target in ["min", "max"]

    def value_prediction(self, X):
        p = self.model.posterior(X)

        if self.is_fitness:
            return self.fitness_inverse(p.mean)

        if isinstance(self.target, tuple):
            return p.mean

    def fitness_prediction(self, X):
        p = self.model.posterior(X)

        if self.is_fitness:
            return self.fitness_inverse(p.mean)

        if isinstance(self.target, tuple):
            return self.targeting_constraint(X).log().clamp(min=-16)


class ObjectiveList(Sequence):
    def __init__(self, objectives: list = []):
        _validate_objectives(objectives)
        self.objectives = objectives

    def __getattr__(self, attr):
        # This is called if we can't find the attribute in the normal way.
        if attr in OBJ_FIELD_TYPES.keys():
            return np.array([getattr(obj, attr) for obj in self.objectives])
        if attr in self.names:
            return self.__getitem__(attr)

        raise AttributeError(f"ObjectiveList object has no attribute named '{attr}'.")

    def __getitem__(self, key):
        if isinstance(key, str):
            if key not in self.names:
                raise ValueError(f"ObjectiveList has no objective named {key}.")
            return self.objectives[self.names.index(key)]
        elif isinstance(key, Iterable):
            return [self.objectives[_key] for _key in key]
        elif isinstance(key, slice):
            return [self.objectives[i] for i in range(*key.indices(len(self)))]
        elif isinstance(key, int):
            return self.objectives[key]
        else:
            raise ValueError(f"Invalid index {key}.")

    def __len__(self):
        return len(self.objectives)

    @property
    def summary(self) -> pd.DataFrame:
        table = pd.DataFrame(columns=list(OBJ_FIELD_TYPES.keys()), index=self.names)

        for obj in self.objectives:
            for attr, value in obj.summary.items():
                table.at[obj.name, attr] = value

        for attr, dtype in OBJ_FIELD_TYPES.items():
            table[attr] = table[attr].astype(dtype)

        return table.T

    def __repr__(self):
        return self.summary.__repr__()

    def _repr_html_(self):
        return self.summary._repr_html_()

    @property
    def descriptions(self) -> list:
        """
        Returns an array of the objective names.
        """
        return [obj.description for obj in self.objectives]

    @property
    def names(self) -> list:
        """
        Returns an array of the objective names.
        """
        return [obj.name for obj in self.objectives]

    @property
    def targets(self) -> list:
        """
        Returns an array of the objective targets.
        """
        return [obj.target for obj in self.objectives]

    @property
    def weights(self) -> np.array:
        """
        Returns an array of the objective weights.
        """
        return np.array([obj.weight for obj in self.objectives])

    @property
    def is_fitness(self) -> np.array:
        """
        Returns an array of the objective weights.
        """
        return np.array([obj.target in ["min", "max"] for obj in self.objectives])

    @property
    def signed_weights(self) -> np.array:
        """
        Returns a signed array of the objective weights.
        """
        return np.array([(1 if obj.target == "max" else -1) * obj.weight for obj in self.objectives])

    def add(self, objective):
        _validate_objectives([*self.objectives, objective])
        self.objectives.append(objective)

    @staticmethod
    def _test_obj(obj, active=None):
        if active is not None:
            if obj.active != active:
                return False
        return True

    def subset(self, active=None):
        return ObjectiveList([obj for obj in self.objectives if self._test_obj(obj, active=active)])
