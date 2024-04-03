import time as ttime
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import pandas as pd
from ophyd import Signal, SignalRO

DOF_FIELD_TYPES = {
    "description": "str",
    "readback": "object",
    "type": "str",
    "search_domain": "object",
    "trust_domain": "object",
    "units": "str",
    "active": "bool",
    "read_only": "bool",
    "log": "bool",
    "tags": "object",
}

SUPPORTED_DOF_TYPES = ["continuous", "binary", "ordinal", "categorical"]


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


@dataclass
class DOF:
    """A degree of freedom (DOF), to be used by an agent.

    Parameters
    ----------
    name: str
        The name of the DOF. This is used as a key to index observed data.
    description: str, optional
        A longer name for the DOF.
    units: str
        The units of the DOF (e.g. mm or deg). This is just for plotting and general sanity checking.
    search_domain: tuple
        A tuple of the lower and upper limit of the DOF for the agent to search.
    trust_domain: tuple, optional
        The agent will reject all data where the DOF value is outside the trust domain. Must be larger than search domain.
    read_only: bool
        If True, the agent will not try to set the DOF. Must be set to True if the supplied ophyd
        device is read-only.
    active: bool
        If True, the agent will try to use the DOF in its optimization. If False, the agent will
        still read the DOF but not include it any model or acquisition function.
    log: bool
        Whether to apply a log to the objective, i.e. to make the process outputs more Gaussian.
    tags: list
        A list of tags. These make it easier to subset large groups of dofs.
    device: Signal, optional
        An ophyd device. If not supplied, a dummy ophyd device will be generated.
    type: str
        What kind of DOF it is. A DOF can be:
        - Continuous, meaning that it can vary to any point between two domain.
        - Binary, meaning that it can take one of two values (e.g. [on, off])
        - Ordinal, meaning ordered categories (e.g. [low, medium, high])
        - Categorical, meaning non-ordered categories (e.g. )
    """

    name: str = None
    description: str = ""
    type: bool = "continuous"
    search_domain: Tuple[float, float] = None
    trust_domain: Tuple[float, float] = None
    units: str = ""
    read_only: bool = False
    active: bool = True
    log: bool = False
    tags: list = field(default_factory=list)
    device: Signal = None

    # Some post-processing. This is specific to dataclasses
    def __post_init__(self):
        if self.type not in SUPPORTED_DOF_TYPES:
            raise ValueError(f"'type' must be one of {SUPPORTED_DOF_TYPES}")

        if (self.name is None) ^ (self.device is None):
            if self.name is None:
                self.name = self.device.name
        else:
            raise ValueError("DOF() accepts exactly one of either a name or an ophyd device.")

        # if our input is continuous
        if self.type == "continuous":
            if self.search_domain is None:
                if not self.read_only:
                    raise ValueError("You must specify search_domain if the device is not read-only.")
            else:
                self.search_domain = tuple(self.search_domain)
                if len(self.search_domain) != 2:
                    raise ValueError("'search_domain' must be a 2-tuple of floats.")
                if self.search_domain[0] > self.search_domain[1]:
                    raise ValueError("The lower search bound must be less than the upper search bound.")

            if self.trust_domain is not None:
                self.trust_domain = tuple(self.trust_domain)
                if not self.read_only:
                    if (self.search_domain[0] < self.trust_domain[0]) or (self.search_domain[1] > self.trust_domain[1]):
                        raise ValueError("Trust domain must be larger than search domain.")

            if self.log:
                if not self.search_domain[0] > 0:
                    raise ValueError("Search domain must be strictly positive if log=True.")

            if self.device is None:
                center_value = np.mean(np.log(self.search_domain)) if self.log else np.mean(self.search_domain)
                self.device = Signal(name=self.name, value=center_value)

        # otherwise it must be discrete
        else:
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
    def _search_domain(self):
        if self.read_only:
            _readback = self.readback
            return (_readback, _readback)
        return self.search_domain

    @property
    def _trust_domain(self):
        if self.trust_domain is None:
            return (0, np.inf) if self.log else (-np.inf, np.inf)
        return self.trust_domain

    @property
    def readback(self):
        # there is probably a better way to do this
        return self.device.read()[self.device.name]["value"]

    @property
    def summary(self) -> pd.Series:
        series = pd.Series(index=list(DOF_FIELD_TYPES.keys()), dtype="object")
        for attr in series.index:
            series[attr] = getattr(self, attr)
        return series

    @property
    def label(self) -> str:
        return f"{self.description}{f' [{self.units}]' if len(self.units) > 0 else ''}"

    @property
    def has_model(self):
        return hasattr(self, "model")


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
    def _test_dof(dof, active=None, read_only=None, tag=None):
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

    def subset(self, active=None, read_only=None, tag=None):
        return DOFList([dof for dof in self.dofs if self._test_dof(dof, active=active, read_only=read_only, tag=tag)])

    def activate(self, active=None, read_only=None, tag=None):
        for dof in self.dofs:
            if self._test_dof(dof, active=active, read_only=read_only, tag=tag):
                dof.active = True

    def deactivate(self, active=None, read_only=None, tag=None):
        for dof in self.dofs:
            if self._test_dof(dof, active=active, read_only=read_only, tag=tag):
                dof.active = False


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
