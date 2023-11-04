import time as ttime
import uuid
from collections.abc import Sequence
from typing import Tuple, Union

import numpy as np
import pandas as pd
from ophyd import Signal, SignalRO

DEFAULT_BOUNDS = (-5.0, +5.0)
DOF_FIELDS = ["description", "readback", "lower_limit", "upper_limit", "units", "active", "read_only", "tags"]

numeric = Union[float, int]


class ReadOnlyError(Exception):
    ...


def _validate_dofs(dofs):
    """Check that a list of DOFs can be combined into a DOFList."""

    # check that dof names are unique
    unique_dof_names, counts = np.unique([dof.name for dof in dofs], return_counts=True)
    duplicate_dof_names = unique_dof_names[counts > 1]
    if len(duplicate_dof_names) > 0:
        raise ValueError(f'Duplicate name(s) in supplied dofs: "{duplicate_dof_names}"')

    return list(dofs)


class DOF:
    """A degree of freedom (DOF), to be used by an agent.

    Parameters
    ----------
    name: str
        The name of the DOF. This is used as a key.
    description: str
        A longer name for the DOF.
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
        device: Signal = None,
        name: str = None,
        description: str = "",
        limits: Tuple[float, float] = (-10.0, 10.0),
        units: str = None,
        read_only: bool = False,
        active: bool = True,
        tags: list = [],
        latent_group=None,
    ):
        self.uuid = str(uuid.uuid4())
        self.description = description

        self.name = name if name is not None else device.name if hasattr(device, "name") else self.uuid

        self.device = device if device is not None else Signal(name=self.name)
        self.limits = limits
        self.read_only = read_only if read_only is not None else True if isinstance(device, SignalRO) else False
        self.units = units
        self.tags = tags
        self.active = active
        self.latent_group = latent_group if latent_group is not None else str(uuid.uuid4())

        self.device.kind = "hinted"

    @property
    def lower_limit(self):
        """The lower limit of the DOF."""
        return float(self.limits[0])

    @property
    def upper_limit(self):
        """The upper limit of the DOF."""
        return float(self.limits[1])

    @property
    def readback(self):
        return self.device.read()[self.device.name]["value"]

    @property
    def summary(self) -> pd.Series:
        """A pandas Series representing the current state of the DOF."""
        series = pd.Series(index=DOF_FIELDS)
        for attr in series.index:
            series[attr] = getattr(self, attr)
        return series

    @property
    def label(self) -> str:
        """A formal label for plotting."""
        return f"{self.name}{f' [{self.units}]' if self.units is not None else ''}"


class DOFList(Sequence):
    """A class for handling a list of DOFs."""

    def __init__(self, dofs: list = []):
        _validate_dofs(dofs)
        self.dofs = dofs

    def __getitem__(self, index):
        """Get a DOF either by name or its position in the list."""
        if type(index) is str:
            return self.dofs[self.names.index(index)]
        if type(index) is int:
            return self.dofs[index]

    def __len__(self):
        """Number of DOFs in the list."""
        return len(self.dofs)

    def __repr__(self):
        """A table showing the state of each DOF."""
        return self.summary.__repr__()

    @property
    def summary(self) -> pd.DataFrame:
        table = pd.DataFrame(columns=DOF_FIELDS)
        for dof in self.dofs:
            for attr in table.columns:
                table.loc[dof.name, attr] = getattr(dof, attr)

        # convert dtypes
        for attr in ["readback", "lower_limit", "upper_limit"]:
            table[attr] = table[attr].astype(float)

        for attr in ["read_only", "active"]:
            table[attr] = table[attr].astype(bool)

        return table

    @property
    def names(self) -> list:
        return [dof.name for dof in self.dofs]

    @property
    def devices(self) -> list:
        return [dof.device for dof in self.dofs]

    @property
    def lower_limits(self) -> np.array:
        return np.array([dof.lower_limit for dof in self.dofs])

    @property
    def upper_limits(self) -> np.array:
        return np.array([dof.upper_limit for dof in self.dofs])

    @property
    def limits(self) -> np.array:
        """
        Returns a (n_dof, 2) array of bounds.
        """
        return np.c_[self.lower_limits, self.upper_limits]

    @property
    def readback(self) -> np.array:
        return np.array([dof.readback for dof in self.dofs])

    def add(self, dof):
        _validate_dofs([*self.dofs, dof])
        self.dofs.append(dof)

    def _dof_read_only_mask(self, read_only=None):
        return [dof.read_only == read_only if read_only is not None else True for dof in self.dofs]

    def _dof_active_mask(self, active=None):
        return [dof.active == active if active is not None else True for dof in self.dofs]

    def _dof_tags_mask(self, tags=[]):
        return [np.isin(dof["tags"], tags).any() if tags else True for dof in self.dofs]

    def _dof_mask(self, active=None, read_only=None, tags=[]):
        return [
            (k and m and t)
            for k, m, t in zip(self._dof_read_only_mask(read_only), self._dof_active_mask(active), self._dof_tags_mask(tags))
        ]

    def subset(self, active=None, read_only=None, tags=[]):
        return DOFList([dof for dof, m in zip(self.dofs, self._dof_mask(active, read_only, tags)) if m])

    def activate(self, read_only=None, active=None, tags=[]):
        """Activate all degrees of freedom with a given tag, active status or read-only status.

        For example, `dofs.activate(tag='kb')` will turn off all dofs which contain the tag 'kb'.
        """
        for dof in self._subset_dofs(read_only, active, tags):
            dof.active = True

    def deactivate(self, read_only=None, active=None, tags=[]):
        """The same as .activate(), only in reverse."""
        for dof in self._subset_dofs(read_only, active, tags):
            dof.active = False


class BrownianMotion(SignalRO):
    """Read-only degree of freedom simulating Brownian motion.

    Parameters
    ----------
    theta : float
        Determines the autocorrelation of the process; smaller values correspond to faster variation.
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
