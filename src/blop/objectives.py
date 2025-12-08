import warnings
from collections.abc import Iterable, Sequence
from typing import Any, Literal, cast, overload

import numpy as np
import pandas as pd
import torch
from botorch.models.model import Model  # type: ignore[import-untyped]

from .utils.functions import approximate_erf
from .utils.sets import element_of, is_subset, validate_set

OBJ_FIELD_TYPES: dict[str, type] = {
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

OBJ_TYPES = ["continuous", "binary", "ordinal", "categorical"]
TRANSFORM_DOMAINS: dict[str, tuple[float, float]] = {"log": (0.0, np.inf), "logit": (0.0, 1.0), "arctanh": (-1.0, 1.0)}


class DuplicateNameError(ValueError):
    pass


def _validate_obj_transform(transform: str) -> None:
    if transform not in TRANSFORM_DOMAINS:
        raise ValueError(f"'transform' must be one of {TRANSFORM_DOMAINS}")


def _validate_continuous_domains(trust_domain: tuple[float, float] | None, domain: tuple[float, float] | None) -> None:
    """
    A DOF MUST have a search domain, and it MIGHT have a trust domain or a transform domain

    Check that all the domains are kosher by enforcing that:
    search_domain \\subseteq trust_domain \\subseteq domain
    """

    if trust_domain and domain:
        if (trust_domain[0] < domain[0]) or (trust_domain[1] > domain[1]):
            raise ValueError(f"The trust domain {trust_domain} is outside the transform domain {domain}.")


class Objective:
    def __init__(
        self,
        name: str,
        description: str = "",
        type: Literal["continuous", "binary", "ordinal", "categorical"] = "continuous",
        target: float | str | None = None,
        constraint: tuple[float | Literal["baseline"], float | Literal["baseline"]] | set[Any] | None = None,
        transform: Literal["log", "logit", "arctanh"] | None = None,
        weight: float = 1.0,
        active: bool = True,
        trust_domain: tuple[float, float] | set[Any] | None = None,
        min_noise: float = 1e-6,
        max_noise: float = 1e0,
        units: str | None = None,
        latent_groups: dict[str, Any] | None = None,
        min_points_to_train: int = 4,
    ) -> None:
        """An objective to be used by an agent.

        .. deprecated:: v0.9.0
            This class is deprecated and will be removed in Blop v1.0.0. Use ``blop.ax.Objective`` instead.

        Parameters
        ----------
        name: str
            The name of the objective to optimize. This is used as a key to index observed data.
        description: str
            A longer description for the objective.

            .. deprecated:: v0.8.0
                This argument is deprecated and will be removed in Blop v1.0.0.
        type: Literal["continuous", "binary", "ordinal", "categorical"]
            Describes the type of the outcome to be optimized. An outcome can be:

            - Continuous, meaning any real number.
            - Binary, meaning that it can take one of two values (e.g. [on, off])
            - Ordinal, meaning ordered categories (e.g. [low, medium, high])
            - Categorical, meaning non-ordered categories (e.g. [mango, banana, papaya])

            .. deprecated:: v0.8.0
                This argument is deprecated and will be removed in Blop v1.0.0. Only DOFs will have types.

            Default: "continuous"
        target: str
            One of "min" or "max". The agent will respectively minimize or maximize the outcome. Each Objective
            must have either a target or a constraint.
            Default: "max"
        constraint: Optional[Union[tuple[float, float], set[int], set[str]]]
            A tuple of floats for continuous outcomes, or a set of outcomes for discrete outcomes. An Objective will
            only be 'satisfied' if it lies within the constraint. Each Objective must have either a target or a constraint.
            Default: None
        transform: Optional[Literal["log", "logit", "arctanh"]]
            One of "log", "logit", or "arctanh", to transform the outcomes and make them more Gaussian.
            Default: None

            .. deprecated:: v0.8.0
                This argument is deprecated and will be removed in Blop v1.0.0. Only DOFs will have transforms. Use
                digestion functions to transform your objectives.
        weight: float
            The relative importance of this Objective, to be used when scalarizing in multi-objective optimization.
            Default: 1.

            .. deprecated:: v0.8.0
                This argument is deprecated and will be removed in Blop v1.0.0.
                Use a digestion function to weight your objectives.
        active: bool
            If True, optimize this objective. Otherwise, monitor the objective, only.
            Default: True
        trust_domain: Union[tuple[float, float], set[int], set[str]]
            A tuple of floats for continuous outcomes, or a set of outcomes for discrete outcomes. An outcome outside
            the trust_domain will not be trusted and will be ignored as 'invalid'. By default, all values are trusted.
            Default: None

            .. deprecated:: v0.8.0
                This argument is deprecated and will be removed in Blop v1.0.0. Use constraints instead.
        min_noise: float
            The minimum relative noise level of the fitted model.
            Default: 1e-6

            .. deprecated:: v0.8.0
                This argument is deprecated and will be removed in Blop v1.0.0.
        max_noise: float
            The maximum relative noise level of the fitted model.
            Default: 1e0

            .. deprecated:: v0.8.0
                This argument is deprecated and will be removed in Blop v1.0.0.
        units: str
            A label representing the units of the outcome (e.g., millimeters or counts)
            Default: None

            .. deprecated:: v0.8.0
                This argument is deprecated and will be removed in Blop v1.0.0.
        latent_groups: list of tuples of strs, optional
            An agent will fit latent dimensions to all DOFs with the same latent_group. All other DOFs will be modeled
            independently. Only used for LatentGPs.
            Default: None
        min_points_to_train: int
            How many new points to wait for before retraining model hyperparameters.
            Default: 4

            .. deprecated:: v0.8.0
                This argument is deprecated and will be removed in Blop v1.0.0.
        """
        if description:
            warnings.warn(
                "The 'description' argument is deprecated and will be removed in Blop v1.0.0.",
                DeprecationWarning,
                stacklevel=2,
            )
        if type is not None and type != "continuous":
            warnings.warn(
                "The 'type' argument is deprecated and will be removed in Blop v1.0.0. Only DOFs will have types.",
                DeprecationWarning,
                stacklevel=2,
            )
        if transform:
            warnings.warn(
                (
                    "The 'transform' argument is deprecated and will be removed in Blop v1.0.0. "
                    "Only DOFs will have transforms. Use digestion functions to transform your objectives."
                ),
                DeprecationWarning,
                stacklevel=2,
            )
        if weight is not None and weight != 1:
            warnings.warn(
                (
                    "The 'weight' argument is deprecated and will be removed in Blop v1.0.0. "
                    "Use a digestion function to weight your objectives."
                ),
                DeprecationWarning,
                stacklevel=2,
            )
        if trust_domain:
            warnings.warn(
                "The 'trust_domain' argument is deprecated and will be removed in Blop v1.0.0. Use constraints instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        if min_noise is not None and min_noise != 1e-6:
            warnings.warn(
                "The 'min_noise' argument is deprecated and will be removed in Blop v1.0.0.",
                DeprecationWarning,
                stacklevel=2,
            )
        if max_noise is not None and max_noise != 1e0:
            warnings.warn(
                "The 'max_noise' argument is deprecated and will be removed in Blop v1.0.0.",
                DeprecationWarning,
                stacklevel=2,
            )
        if units:
            warnings.warn(
                "The 'units' argument is deprecated and will be removed in Blop v1.0.0.",
                DeprecationWarning,
                stacklevel=2,
            )
        if min_points_to_train is not None and min_points_to_train != 4:
            warnings.warn(
                "The 'min_points_to_train' argument is deprecated and will be removed in Blop v1.0.0.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.name = name
        self.units = units
        self.description = description
        self.type = type
        self.active = active

        # TODO: These are currently set outside of the class, in agent.py.
        #       We should move them inside the class, and make them private.
        #       Or reconsider the design of the class.
        self._model: Model | None = None
        self.validity_conjugate_model: Model | None = None

        if not target and not constraint:
            raise ValueError("You must supply either a 'target' or a 'constraint'.")

        self.target = target
        self.constraint = constraint

        if transform is not None:
            _validate_obj_transform(transform)

        self.transform = transform

        if self.type == "continuous":
            _validate_continuous_domains(trust_domain, self.domain)
        else:
            raise NotImplementedError("Non-continuous objectives are not supported yet.")

        self.trust_domain = trust_domain
        self.weight = weight if target else None
        self.min_noise = min_noise
        self.max_noise = max_noise
        self.latent_groups = latent_groups or {}
        self.min_points_to_train = min_points_to_train

        if isinstance(self.target, str):
            # eventually we will be able to target other strings, as outputs of a discrete objective
            if self.target not in ["min", "max"]:
                raise ValueError("'target' must be either 'min', 'max', a number, or a tuple of numbers.")

    @property
    def search_domain(self) -> tuple[float, float] | set[int] | set[str] | set[bool]:
        return self._search_domain

    @search_domain.setter
    def search_domain(self, value: tuple[float, float] | set[int] | set[str] | set[bool]):
        """
        Make sure that the search domain is within the trust domain before setting it.
        """
        value = validate_set(value, type=self.type)
        trust_domain = self.trust_domain
        if is_subset(value, trust_domain, type=self.type, proper=False):
            self._search_domain = cast(tuple[float, float] | set[int] | set[str] | set[bool], value)
        else:
            raise ValueError(
                f"Cannot set search domain to {value} as it is not a subset of the trust domain {trust_domain}."
            )

    @property
    def trust_domain(self) -> tuple[float, float] | set[int] | set[str] | set[bool]:
        """
        If _trust_domain is None, then we trust the entire domain (so we return the domain).
        """
        return self._trust_domain or self.domain

    @trust_domain.setter
    def trust_domain(self, value):
        """
        Make sure that the trust domain is a subset of the domain before setting it.
        """
        if value is not None:
            value = validate_set(value, type=self.type)
            domain = self.domain

            if not is_subset(value, domain, type=self.type, proper=False):
                raise ValueError(f"Cannot set trust domain to {value} as it is not a subset of the domain {domain}.")

        self._trust_domain = value

    @property
    def domain(self) -> tuple[float, float] | set[int] | set[str] | set[bool]:
        """
        The total domain of the objective.
        """
        if self.type == "continuous":
            if not self.transform:
                return (-np.inf, np.inf)
            else:
                return TRANSFORM_DOMAINS[self.transform]
        else:
            return self._domain

    def evaluate_constraint(self, y: torch.Tensor) -> torch.Tensor:
        """
        Evaluate whether each value in y satisfies the constraint of the Objective.
        """
        if self.constraint is None:
            raise RuntimeError("Cannot call 'constrain' with a non-constraint objective.")
        elif isinstance(self.constraint, tuple):
            return (y > self.constraint[0]) & (y < self.constraint[1])
        else:
            return torch.tensor([value in self.constraint for value in np.atleast_1d(y)])

    @property
    def all_valid(self) -> bool:
        return not getattr(self, "validity_conjugate_model", None)

    def log_total_constraint(self, x: torch.Tensor) -> torch.Tensor:
        """
        What is the log probability that a sample at x will be valid and satisfy the constraint of the Objective?
        """
        log_p = torch.zeros(x.shape[:-1])
        if self.constraint:
            log_p += self.constraint_probability(x).log()

        # if the validity constaint is non-trivial
        if not self.all_valid:
            log_p += self.validity_probability(x).log()

        return log_p

    def _transform(self, y: torch.Tensor) -> torch.Tensor:
        y = torch.where(element_of(y, self.trust_domain), y, torch.nan)

        if self.transform == "log":
            y = y.log()
        elif self.transform == "logit":
            y = (y / (1 - y)).log()
        elif self.transform == "arctanh":
            y = torch.arctanh(y)

        return y

    def _untransform(self, y: torch.Tensor) -> torch.Tensor:
        if self.transform == "log":
            y = y.exp()
        elif self.transform == "logit":
            y = 1 / (1 + torch.exp(-y))
        elif self.transform == "arctanh":
            y = torch.tanh(y)

        return y

    @property
    def label_with_units(self) -> str:
        return f"{self.description}{f' [{self.units}]' if self.units else ''}"

    @property
    def noise_bounds(self) -> tuple[float, float]:
        return (self.min_noise, self.max_noise)

    @property
    def summary(self) -> pd.Series:
        """
        Return a Series summarizing the state of the Objectives.

        .. deprecated:: v0.8.0
            This method is deprecated and will be removed in Blop v1.0.0. Objectives will not have a summary.
        """
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
        return self.model.likelihood.noise.item() if self.model else np.nan

    @property
    def snr(self) -> int | None:
        return np.round(1 / self.model.likelihood.noise.sqrt().item(), 3) if self.model else None

    @property
    def n_valid(self) -> int:
        return int((~self.model.train_targets.isnan()).sum()) if self.model else 0

    def constraint_probability(self, x: torch.Tensor) -> torch.Tensor:
        """
        How much of the posterior on the outcome at x satisfies the constraint?
        """
        if not self.constraint:
            raise RuntimeError("Cannot call 'constrain' with a non-constraint objective.")
        if not self.model:
            raise RuntimeError("Cannot call 'constrain' with an untrained objective.")

        a, b = self.constraint
        p = self.model.posterior(x)
        m = p.mean
        s = p.variance.sqrt()

        sish = s + 0.1 * m.std()  # for numerical stability

        p = 0.5 * (approximate_erf((b - m) / (np.sqrt(2) * sish)) - approximate_erf((a - m) / (np.sqrt(2) * sish)))[..., -1]  # noqa

        return p.detach()

    def validity_probability(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "validity_conjugate_model"):
            return self.validity_conjugate_model.probabilities(x)[..., -1]

        return torch.ones(x.shape[:-1])

    @property
    def model(self) -> Model | None:
        """
        .. deprecated:: v0.8.0
            This method is deprecated and will be removed in Blop v1.0.0. Models will not be stored in individaul Objectives.
        """
        return self._model.eval() if self._model else None

    @property
    def sign(self) -> int:
        return (-1 if self.target == "min" else 1) if self.target is not None else 0


class ObjectiveList(Sequence[Objective]):
    def __init__(self, objectives: list[Objective] | None = None) -> None:
        self.objectives: list[Objective] = objectives or []

    def __call__(self, *args: Any, **kwargs: Any) -> "ObjectiveList":
        return self.subset(*args, **kwargs)

    @property
    def names(self) -> list[str]:
        return [obj.name for obj in self.objectives]

    @property
    def signs(self) -> torch.Tensor:
        return torch.tensor([obj.sign for obj in self.objectives])

    def __getattr__(self, attr: str) -> Objective | list[Any] | np.ndarray:
        # This is called if we can't find the attribute in the normal way.
        if all(hasattr(obj, attr) for obj in self.objectives):
            if OBJ_FIELD_TYPES.get(attr) in [float, int, bool]:
                return np.array([getattr(obj, attr) for obj in self.objectives])
            return [getattr(obj, attr) for obj in self.objectives]
        if attr in self.names:
            return self.__getitem__(attr)

        raise AttributeError(f"ObjectiveList object has no attribute named '{attr}'.")

    @overload
    def __getitem__(self, key: int) -> Objective: ...

    @overload
    def __getitem__(self, key: str) -> Objective: ...

    @overload
    def __getitem__(self, s: slice) -> Sequence[Objective]: ...

    @overload
    def __getitem__(self, key: Iterable) -> Sequence[Objective]: ...

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

    def __len__(self) -> int:
        return len(self.objectives)

    @property
    def summary(self) -> pd.DataFrame:
        """
        Return a DataFrame summarizing the state of the Objectives.
        """
        return pd.concat([objective.summary for objective in self.objectives], axis=1)

    def __repr__(self) -> str:
        return self.summary.__repr__()

    def _repr_html_(self) -> str:
        return self.summary._repr_html_()  # type: ignore

    def add(self, objective: Objective) -> None:
        self.objectives.append(objective)

    @staticmethod
    def _test_obj(
        obj: Objective, active: bool | None = None, fitness: bool | None = None, constraint: bool | None = None
    ) -> bool:
        if active:
            if obj.active != active:
                return False
        if fitness:
            if fitness != (obj.target is not None):
                return False
        if constraint:
            if constraint != (obj.constraint is not None):
                return False
        return True

    def subset(self, **kwargs: Any) -> "ObjectiveList":
        return self.__class__([obj for obj in self.objectives if self._test_obj(obj, **kwargs)])

    def transform(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Transform the experiment space to the model space.
        """
        if Y.shape[-1] != len(self):
            raise ValueError(f"Cannot transform points with shape {Y.shape} using Objectives with dimension {len(self)}.")

        return torch.cat([obj._transform(Y[..., i]).unsqueeze(-1) for i, obj in enumerate(self.objectives)], dim=-1)

    def untransform(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Transform the model space to the experiment space.
        """
        if Y.shape[-1] != len(self):
            raise ValueError(f"Cannot untransform points with shape {Y.shape} using Objectives with dimension {len(self)}.")

        return torch.cat([obj._untransform(Y[..., i]).unsqueeze(-1) for i, obj in enumerate(self.objectives)], dim=-1)
