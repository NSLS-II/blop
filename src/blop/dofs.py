import time as ttime
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, fields
from operator import attrgetter
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from ophyd import Signal, SignalRO

DOF_FIELD_TYPES = {
    "description": "str",
    "readback": "object",
    "type": "str",
    "transform": "str",
    "search_domain": "object",
    "trust_domain": "object",
    "units": "str",
    "active": "bool",
    "read_only": "bool",
    "tags": "object",
}

DOF_TYPES = ["continuous", "binary", "ordinal", "categorical"]
TRANSFORM_DOMAINS = {"log": (0.0, np.inf), "sigmoid": (0.0, 1.0), "tanh": (-1.0, 1.0)}


class ReadOnlyError(Exception):
    ...


def _validate_dofs(dofs):
    dof_names = [dof.name for dof in dofs]

    # check that dof names are unique
    unique_dof_names, counts = np.unique(dof_names, return_counts=True)
    duplicate_dof_names = unique_dof_names[counts > 1]
    if len(duplicate_dof_names) > 0:
        raise ValueError(f"Duplicate name(s) in supplied dofs: {duplicate_dof_names}")

    return list(dofs)


def _validate_dof_transform(transform):
    if transform is None:
        return (-np.inf, np.inf)

    if transform not in TRANSFORM_DOMAINS:
        raise ValueError(f"'transform' must be a callable with one argument, or one of {TRANSFORM_DOMAINS}")


def _validate_continuous_dof_domains(search_domain, trust_domain, domain):
    """
    A DOF MUST have a search domain, and it MIGHT have a trust domain or a transform domain.

    Check that all the domains are kosher by enforcing that:
    search_domain \\subseteq trust_domain \\subseteq domain
    """

    if len(search_domain) != 2:
        raise ValueError("'search_domain' must be a 2-tuple of numbers.")

    if search_domain[0] >= search_domain[1]:
        raise ValueError("The lower search bound must be strictly less than the upper search bound.")

    if domain is not None:
        if (search_domain[0] < domain[0]) or (search_domain[1] > domain[1]):
            raise ValueError(f"The search domain {search_domain} is outside the transform domain {domain}.")

    if trust_domain is not None:
        if (search_domain[0] < trust_domain[0]) or (search_domain[1] > trust_domain[1]):
            raise ValueError(f"The search domain {search_domain} is outside the trust domain {trust_domain}.")

    if (trust_domain is not None) and (domain is not None):
        if (trust_domain[0] < domain[0]) or (trust_domain[1] > domain[1]):
            raise ValueError(f"The trust domain {trust_domain} is outside the transform domain {domain}.")


def _validate_discrete_dof_domains(search_domain, trust_domain, domain):
    """
    A DOF MUST have a search domain, and it MIGHT have a trust domain or a transform domain

    Check that all the domains are kosher by enforcing that:
    search_domain \\subseteq trust_domain \\subseteq domain
    """
    ...


@dataclass
class DOF:
    """A degree of freedom (DOF), to be used by an agent.

    Parameters
    ----------
    name: str
        The name of the DOF. This is used as a key to index observed data.
    description: str, optional
        A longer, more descriptive name for the DOF.
    type: str
        What kind of DOF it is. A DOF can be:
        - Continuous, meaning that it can vary to any point between a lower and upper bound.
        - Binary, meaning that it can take one of two values (e.g. [on, off])
        - Ordinal, meaning ordered categories (e.g. [low, medium, high])
        - Categorical, meaning non-ordered categories (e.g. [mango, banana, papaya])
    search_domain: tuple
        A tuple of the lower and upper limit of the DOF for the agent to search.
    trust_domain: tuple, optional
        The agent will reject all data where the DOF value is outside the trust domain. Must be larger than search domain.
    units: str
        The units of the DOF (e.g. mm or deg). This is just for plotting and general sanity checking.
    read_only: bool
        If True, the agent will not try to set the DOF. Must be set to True if the supplied ophyd
        device is read-only.
    active: bool
        If True, the agent will try to use the DOF in its optimization. If False, the agent will
        still read the DOF but not include it any model or acquisition function.
    transform: Callable
        A transform to apply to the objective, to make the process outputs more Gaussian.
    tags: list
        A list of tags. These make it easier to subset large groups of dofs.
    device: Signal, optional
        An ophyd device. If not supplied, a dummy ophyd device will be generated.
    """

    name: str = None
    description: str = ""
    type: str = "continuous"
    search_domain: Tuple[float, float] = None
    trust_domain: Tuple[float, float] = None
    units: str = None
    read_only: bool = False
    active: bool = True
    transform: str = None
    tags: list = field(default_factory=list)
    device: Signal = None

    def __repr__(self):
        nodef_f_vals = ((f.name, attrgetter(f.name)(self)) for f in fields(self))

        nodef_f_repr = []
        for name, value in nodef_f_vals:
            if (name == "search_domain") and (self.type == "continuous"):
                search_min, search_max = self.search_domain
                nodef_f_repr.append(f"search_domain=({search_min:.03e}, {search_max:.03e})")
            else:
                nodef_f_repr.append(f"{name}={value}")

        return f"{self.__class__.__name__}({', '.join(nodef_f_repr)})"

    # Some post-processing. This is specific to dataclasses
    def __post_init__(self):
        if self.type not in DOF_TYPES:
            raise ValueError(f"'type' must be one of {DOF_TYPES}")

        if (self.name is None) ^ (self.device is None):
            if self.name is None:
                self.name = self.device.name
        else:
            raise ValueError("DOF() accepts exactly one of either a name or an ophyd device.")

        # if our input is continuous
        if self.type == "continuous":
            _validate_dof_transform(self.transform)

            if self.trust_domain is None:
                self.trust_domain = TRANSFORM_DOMAINS[self.transform] if self.transform is not None else (-np.inf, np.inf)

            if self.search_domain is None:
                if not self.read_only:
                    raise ValueError("You must specify search_domain if the device is not read-only.")
            else:
                _validate_continuous_dof_domains(self.search_domain, self.trust_domain, self.domain)

            if self.device is None:
                center = float(self._untransform(np.mean([self._transform(np.array(self.search_domain))])))
                self.device = Signal(name=self.name, value=center)

        # otherwise it must be discrete
        else:
            _validate_discrete_dof_domains(self.search_domain, self.trust_domain, self.domain)

            if self.type == "binary":
                if self.search_domain is None:
                    self.search_domain = [False, True]
                if len(self.search_domain) != 2:
                    raise ValueError("A binary DOF must have a domain of 2.")
            else:
                if self.search_domain is None:
                    raise ValueError("Discrete domain must be supplied for ordinal and categorical degrees of freedom.")

            self.search_domain = set(self.search_domain)

            self.device = Signal(name=self.name, value=list(self.search_domain)[0])

        if not self.read_only:
            # check that the device has a put method
            if isinstance(self.device, SignalRO):
                raise ValueError("You must specify read_only=True for a read-only device.")

        # all dof degrees of freedom are hinted
        self.device.kind = "hinted"

    @property
    def domain(self):
        """
        The total domain of the DOF.
        """
        if self.transform is None:
            if self.type == "continuous":
                return (-np.inf, np.inf)
            else:
                return self.search_domain
        return TRANSFORM_DOMAINS[self.transform]

    @property
    def _search_domain(self):
        if self.read_only:
            _readback = self.readback
            return np.array([_readback, _readback])
        return np.array(self.search_domain)

    def _trust(self, x):
        return (self.trust_domain[0] <= x) & (x <= self.trust_domain[1])

    @property
    def _trust_domain(self):
        if self.trust_domain is None:
            return self.domain
        return self.trust_domain

    def _transform(self, x, normalize=True):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.double)

        x = torch.where((x > self.domain[0]) & (x < self.domain[1]), x, torch.nan)

        if self.transform == "log":
            x = torch.log(x)
        if self.transform == "sigmoid":
            x = (x / (1 - x)).log()
        if self.transform == "tanh":
            x = torch.arctanh(x)

        if normalize and not self.read_only:
            min, max = self._transform(self._search_domain, normalize=False)
            x = (x - min) / (max - min)

        return x

    def _untransform(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.double)

        if not self.read_only:
            min, max = self._transform(self._search_domain, normalize=False)
            x = x * (max - min) + min

        if self.transform is None:
            return x
        if self.transform == "log":
            return torch.exp(x)
        if self.transform == "sigmoid":
            return 1 / (1 + torch.exp(-x))
        if self.transform == "tanh":
            return torch.tanh(x)

    # @property
    # def _transformed_search_domain(self):
    #     return self._transform(np.array(self._search_domain), normalize=False)

    # @property
    # def _transformed_trust_domain(self):
    #     return self._transform(np.array(self._trust_domain), normalize=False)

    @property
    def readback(self):
        # there is probably a better way to do this
        return self.device.read()[self.device.name]["value"]

    @property
    def summary(self) -> pd.Series:
        series = pd.Series(index=list(DOF_FIELD_TYPES.keys()), dtype="object")
        for attr in series.index:
            value = getattr(self, attr)
            if attr in ["search_domain", "trust_domain"]:
                if (self.type == "continuous") and not self.read_only:
                    if value is not None:
                        value = f"({value[0]:.02e}, {value[1]:.02e})"
            series[attr] = value if value is not None else ""
        return series

    @property
    def label_with_units(self) -> str:
        return f"{self.description}{f' [{self.units}]' if self.units else ''}"

    @property
    def has_model(self):
        return hasattr(self, "model")

    def activate(self):
        self.active = True

    def deactivate(self):
        self.active = False


class DOFList(Sequence):
    def __init__(self, dofs: list = []):
        _validate_dofs(dofs)
        self.dofs = dofs

    def __getattr__(self, attr):
        # This is called if we can't find the attribute in the normal way.
        if attr in DOF_FIELD_TYPES.keys():
            return np.array([getattr(dof, attr) for dof in self.dofs])
        if attr in self.names:
            return self.__getitem__(attr)

        raise AttributeError(f"DOFList object has no attribute named '{attr}'.")

    def __getitem__(self, key):
        if isinstance(key, str):
            if key not in self.names:
                raise ValueError(f"DOFList has no DOF named {key}.")
            return self.dofs[self.names.index(key)]
        elif isinstance(key, Iterable):
            return [self.dofs[_key] for _key in key]
        elif isinstance(key, slice):
            return [self.dofs[i] for i in range(*key.indices(len(self)))]
        elif isinstance(key, int):
            return self.dofs[key]
        else:
            raise ValueError(f"Invalid index {key}.")

    def __len__(self):
        return len(self.dofs)

    def __repr__(self):
        return self.summary.T.__repr__()

    def _repr_html_(self):
        return self.summary.T._repr_html_()

    def transform(self, X):
        """
        Transform X to the transformed unit hypercube.
        """

        if X.shape[-1] != len(self.subset(active=True)):
            raise ValueError()

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.double)

        return torch.cat([dof._transform(X[..., i]).unsqueeze(-1) for i, dof in enumerate(self.subset(active=True))], dim=-1)

    def untransform(self, X):
        """
        Transform the transformed unit hypercube to the search domain.
        """
        if X.shape[-1] != len(self.subset(active=True)):
            raise ValueError()

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.double)

        return torch.cat(
            [dof._untransform(X[..., i]).unsqueeze(-1) for i, dof in enumerate(self.subset(active=True))], dim=-1
        )

    @property
    def readback(self):
        """
        Return the readback from each DOF as a list. It is a list because they might be different types.
        """
        return [dof.readback for dof in self.dofs]

    @property
    def summary(self) -> pd.DataFrame:
        table = pd.DataFrame(columns=list(DOF_FIELD_TYPES.keys()), index=self.names)

        for dof in self.dofs:
            for attr, value in dof.summary.items():
                table.at[dof.name, attr] = value

        for attr, dtype in DOF_FIELD_TYPES.items():
            table[attr] = table[attr].astype(dtype)

        return table

    @property
    def names(self) -> list:
        return [dof.name for dof in self.dofs]

    @property
    def devices(self) -> list:
        return [dof.device for dof in self.dofs]

    @property
    def search_domain(self) -> np.array:
        """
        Returns a (n_dof, 2) array of domain.
        """
        return np.array([dof._search_domain for dof in self.dofs])

    @property
    def trust_domain(self) -> np.array:
        """
        Returns a (n_dof, 2) array of domain.
        """
        return np.array([dof._trust_domain for dof in self.dofs])

    def add(self, dof):
        _validate_dofs([*self.dofs, dof])
        self.dofs.append(dof)

    @staticmethod
    def _test_dof(dof, type=None, active=None, read_only=None, tag=None):
        if type is not None:
            if dof.type != type:
                return False
        if active is not None:
            if dof.active != active:
                return False
        if read_only is not None:
            if dof.read_only != read_only:
                return False
        if tag is not None:
            if not np.isin(np.atleast_1d(tag), dof.tags).any():
                return False
        return True

    def subset(self, type=None, active=None, read_only=None, tag=None):
        return DOFList(
            [dof for dof in self.dofs if self._test_dof(dof, type=type, active=active, read_only=read_only, tag=tag)]
        )

    def activate(self, **subset_kwargs):
        for dof in self.dofs:
            if self._test_dof(dof, **subset_kwargs):
                dof.active = True

    def deactivate(self, **subset_kwargs):
        for dof in self.dofs:
            if self._test_dof(dof, **subset_kwargs):
                dof.active = False

    def activate_only(self, **subset_kwargs):
        for dof in self.dofs:
            if self._test_dof(dof, **subset_kwargs):
                dof.active = True
            else:
                dof.active = False

    def deactivate_only(self, **subset_kwargs):
        for dof in self.dofs:
            if self._test_dof(dof, **subset_kwargs):
                dof.active = False
            else:
                dof.active = True


class BrownianMotion(SignalRO):
    """
    Read-only degree of freedom simulating brownian motion
    """

    def __init__(self, name=None, theta=0.95, *args, **kwargs):
        name = name if name is not None else str(uuid.uuid4())

        super().__init__(name=name, *args, **kwargs)

        self.theta = theta
        self.old_t = ttime.monotonic()
        self.old_y = 0.0

    def get(self):
        new_t = ttime.monotonic()
        alpha = self.theta ** (new_t - self.old_t)
        new_y = alpha * self.old_y + np.sqrt(1 - alpha**2) * np.random.standard_normal()

        self.old_t = new_t
        self.old_y = new_y
        return new_y


class TimeReadback(SignalRO):
    """
    Returns the current timestamp.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get(self):
        return ttime.time()


class ConstantReadback(SignalRO):
    """
    Returns a constant every time you read it (more useful than you'd think).
    """

    def __init__(self, constant=1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.constant = constant

    def get(self):
        return self.constant
