from collections.abc import Iterable, Sequence
from typing import Union

import numpy as np
import pandas as pd
import torch

from .utils.functions import approximate_erf

DEFAULT_MIN_NOISE_LEVEL = 1e-6
DEFAULT_MAX_NOISE_LEVEL = 1e0

OBJ_FIELD_TYPES = {
    "name": str,
    "description": object,
    "active": bool,
    "type": str,
    "units": object,
    "target": object,
    "constraint": object,
    "transform": str,
    "domain": str,
    "trust_domain": object,
    "weight": float,
    "noise_bounds": object,
    "noise": float,
    "n_valid": int,
    "latent_groups": object,
}

SUPPORTED_OBJ_TYPES = ["continuous", "binary", "ordinal", "categorical"]
TRANSFORM_DOMAINS = {"log": (0.0, np.inf), "logit": (0.0, 1.0), "arctanh": (-1.0, 1.0)}


class DuplicateNameError(ValueError):
    pass


domains = {"log"}


def _validate_obj_transform(transform):
    if transform not in TRANSFORM_DOMAINS:
        raise ValueError(f"'transform' must be a callable with one argument, or one of {TRANSFORM_DOMAINS}")


def _validate_continuous_domains(trust_domain, domain):
    """
    A DOF MUST have a search domain, and it MIGHT have a trust domain or a transform domain

    Check that all the domains are kosher by enforcing that:
    search_domain \\subseteq trust_domain \\subseteq domain
    """

    if (trust_domain is not None) and (domain is not None):
        if (trust_domain[0] < domain[0]) or (trust_domain[1] > domain[1]):
            raise ValueError(f"The trust domain {trust_domain} is outside the transform domain {domain}.")


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

    def __init__(
        self,
        name: str,
        description: str = "",
        type: str = "continuous",
        target: Union[float, str, None] = None,
        constraint: Union[tuple[float, float], set, None] = None,
        transform: str = None,
        weight: float = 1.0,
        active: bool = True,
        trust_domain: Union[tuple[float, float], None] = None,
        min_noise: float = DEFAULT_MIN_NOISE_LEVEL,
        max_noise: float = DEFAULT_MAX_NOISE_LEVEL,
        units: str = None,
        latent_groups: list[tuple[str, ...]] = {},
    ):
        self.name = name
        self.units = units
        self.description = description
        self.type = type
        self.active = active

        if (target is None) and (constraint is None):
            raise ValueError("You must supply either a 'target' or a 'constraint'.")

        self.target = target
        self.constraint = constraint

        if transform is not None:
            _validate_obj_transform(transform)

        self.transform = transform

        if self.type == "continuous":
            _validate_continuous_domains(trust_domain, self.domain)
        else:
            raise NotImplementedError()

        self.trust_domain = trust_domain
        self.weight = weight if target is not None else None
        self.min_noise = min_noise
        self.max_noise = max_noise
        self.latent_groups = latent_groups

        if isinstance(self.target, str):
            # eventually we will be able to target other strings, as outputs of a discrete objective
            if self.target not in ["min", "max"]:
                raise ValueError("'target' must be either 'min', 'max', a number, or a tuple of numbers.")

    @property
    def domain(self):
        """
        The total domain of the objective.
        """
        if self.transform is None:
            if self.type == "continuous":
                return (-np.inf, np.inf)
        return TRANSFORM_DOMAINS[self.transform]

    def constrain(self, y):
        if self.constraint is None:
            raise RuntimeError("Cannot call 'constrain' with a non-constraint objective.")
        elif isinstance(self.constraint, tuple):
            return (y > self.constraint[0]) & (y < self.constraint[1])
        else:
            return np.array([value in self.constraint for value in np.atleast_1d(y)])

    def log_total_constraint(self, x):

        log_p = 0
        # if you have a constraint
        if self.constraint is not None:
            log_p += self.constraint_probability(x).log()

        # if the validity constaint is non-trivial
        if self.validity_conjugate_model is not None:
            log_p += self.validity_constraint(x).log()

        return log_p

    @property
    def _trust_domain(self):
        if self.trust_domain is None:
            return self.domain
        return self.trust_domain

    def _transform(self, y):
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.double)

        y = torch.where((y > self.domain[0]) & (y < self.domain[1]), y, np.nan)

        if self.transform == "log":
            y = y.log()
        if self.transform == "logit":
            y = (y / (1 - y)).log()
        if self.transform == "arctanh":
            y = torch.arctanh(y)

        if self.target == "min":
            y = -y

        return y

    def _untransform(self, y):
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.double)

        if self.target == "min":
            y = -y

        if self.transform == "log":
            y = y.exp()
        if self.transform == "logit":
            y = 1 / (1 + torch.exp(-y))
        if self.transform == "arctanh":
            y = torch.tanh(y)

        return y

    @property
    def label_with_units(self) -> str:
        return f"{self.description}{f' [{self.units}]' if self.units else ''}"

    @property
    def noise_bounds(self) -> tuple:
        return (self.min_noise, self.max_noise)

    @property
    def summary(self) -> pd.Series:
        series = pd.Series(index=list(OBJ_FIELD_TYPES.keys()), dtype=object)
        for attr in series.index:
            value = getattr(self, attr)

            if attr in ["search_domain", "trust_domain"]:
                if self.type == "continuous":
                    if value is not None:
                        value = f"({value[0]:.02e}, {value[1]:.02e})"

            if attr in ["noise_bounds"]:
                if value is not None:
                    value = f"({value[0]:.01e}, {value[1]:.01e})"

            series[attr] = value if value is not None else ""
        return series

    @property
    def noise(self) -> float:
        return self.model.likelihood.noise.item() if hasattr(self, "model") else np.nan

    @property
    def snr(self) -> float:
        return np.round(1 / self.model.likelihood.noise.sqrt().item(), 3) if hasattr(self, "model") else None

    @property
    def n_valid(self) -> int:
        return int((~self.model.train_targets.isnan()).sum()) if hasattr(self, "model") else 0

    def constraint_probability(self, x: torch.Tensor) -> torch.Tensor:
        if self.constraint is None:
            raise RuntimeError("Cannot call 'constrain' with a non-constraint objective.")

        a, b = self.constraint
        p = self.model.posterior(x)
        m = p.mean
        s = p.variance.sqrt()

        sish = s + 0.1 * m.std()  # for numerical stability

        p = (
            0.5 * (approximate_erf((b - m) / (np.sqrt(2) * sish)) - approximate_erf((a - m) / (np.sqrt(2) * sish)))[..., -1]
        )  # noqa

        return p.detach()

    def pseudofitness(self, x: torch.tensor) -> torch.tensor:
        """
        When the optimization problem consists only of constraints, the
        """
        p = self.model.posterior(x)

        if self.is_fitness:
            return self.fitness_inverse(p.mean)

        if isinstance(self.target, tuple):
            return self.constraint_probability(x).log().clamp(min=-16)

    @property
    def model(self):
        return self._model.eval()


class ObjectiveList(Sequence):
    def __init__(self, objectives: list = []):
        self.objectives = objectives

    def __call__(self, *args, **kwargs):
        return self.subset(*args, **kwargs)

    @property
    def names(self):
        return [obj.name for obj in self.objectives]

    def __getattr__(self, attr):
        # This is called if we can't find the attribute in the normal way.
        if all([hasattr(obj, attr) for obj in self.objectives]):
            if OBJ_FIELD_TYPES.get(attr) in [float, "int", "bool"]:
                return np.array([getattr(obj, attr) for obj in self.objectives])
            return [getattr(obj, attr) for obj in self.objectives]
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
        # table = pd.DataFrame(columns=list(OBJ_FIELD_TYPES.keys()), index=np.arange(len(self)))

        # for index, obj in enumerate(self.objectives):
        #     for attr, value in obj.summary.items():
        #         table.at[index, attr] = value

        # for attr, dtype in OBJ_FIELD_TYPES.items():
        #     table[attr] = table[attr].astype(dtype)

        return pd.concat([objective.summary for objective in self.objectives], axis=1)

    def __repr__(self):
        return self.summary.__repr__()

    def _repr_html_(self):
        return self.summary._repr_html_()

    def add(self, objective):
        self.objectives.append(objective)

    @staticmethod
    def _test_obj(obj, active=None, fitness=None, constraint=None):
        if active is not None:
            if obj.active != active:
                return False
        if fitness is not None:
            if fitness != (obj.target is not None):
                return False
        if constraint is not None:
            if constraint != (obj.constraint is not None):
                return False
        return True

    def subset(self, **kwargs):
        return ObjectiveList([obj for obj in self.objectives if self._test_obj(obj, **kwargs)])

    def transform(self, Y):
        """
        Transform the experiment space to the model space.
        """
        if Y.shape[-1] != len(self):
            raise ValueError(f"Cannot transform points with shape {Y.shape} using DOFs with dimension {len(self)}.")

        if not isinstance(Y, torch.Tensor):
            Y = torch.tensor(Y, dtype=torch.double)

        return torch.cat([obj._transform(Y[..., i]).unsqueeze(-1) for i, obj in enumerate(self.objectives)], dim=-1)

    def untransform(self, Y):
        """
        Transform the model space to the experiment space.
        """
        if Y.shape[-1] != len(self):
            raise ValueError(f"Cannot untransform points with shape {Y.shape} using DOFs with dimension {len(self)}.")

        if not isinstance(Y, torch.Tensor):
            Y = torch.tensor(Y, dtype=torch.double)

        return torch.cat([obj._untransform(Y[..., i]).unsqueeze(-1) for i, obj in enumerate(self.objectives)], dim=-1)
