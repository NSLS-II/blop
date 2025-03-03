import time as ttime
import uuid
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, fields
from operator import attrgetter
from typing import Any, Literal, cast, overload

import numpy as np
import pandas as pd
import torch
from ophyd import Signal, SignalRO  # type: ignore[import-untyped]

DOF_FIELD_TYPES: dict[str, type] = {
    "description": str,
    "units": str,
    "readback": object,
    "type": str,
    "active": bool,
    "read_only": bool,
    "tags": object,
    "transform": str,
    "search_domain": object,
    "trust_domain": object,
    "domain": object,
}

DOF_TYPES = ["continuous", "binary", "ordinal", "categorical"]
TRANSFORM_DOMAINS = {"log": (0.0, np.inf), "logit": (0.0, 1.0), "arctanh": (-1.0, 1.0)}


class ReadOnlyError(Exception):
    pass


@dataclass
class DOF:
    """A degree of freedom (DOF), to be used by an agent.

    Parameters
    ----------
    name: str
        The name of the DOF. This is used as a key to index observed data.
    description: str, optional
        A longer, more descriptive name for the DOF.
    type: Literal["continuous", "binary", "ordinal", "categorical"]
        What kind of DOF it is. A DOF can be:
        - Continuous, meaning that it can vary to any point between a lower and upper bound.
        - Binary, meaning that it can take one of two values (e.g. [on, off])
        - Ordinal, meaning ordered categories (e.g. [low, medium, high])
        - Categorical, meaning non-ordered categories (e.g. [mango, banana, papaya])
        Default: "continuous"
    search_domain: Union[tuple[float, float], set[int], set[str]]
        If continuous, a tuple of the lower and upper limit of the DOF for the agent to search.
        If discrete, a set of the possible values for the DOF.
        Default: (-np.inf, np.inf)
    trust_domain: Union[tuple[float, float], set[int], set[str]]
        The agent will reject all data where the DOF value is outside this domain.
        Must span a equal or larger range than the search domain.
        Default: (-np.inf, np.inf)
    active: bool
        If True, the agent will try to use the DOF in its optimization. If False, the agent will
        still read the DOF but not include it any model or acquisition function.
        Default: True
    read_only: bool
        If True, the agent will not try to set the DOF. Must be set to True if the supplied ophyd
        device is read-only.
        Default: False
    transform: Optional[Literal["log", "logit", "arctanh"]]
        A transform to apply to the objective, to make the process outputs more Gaussian.
        Default: None
    device: Optional[Signal]
        An `ophyd.Signal`. If not supplied, a dummy `ophyd.Signal` will be generated.
        Default: None
    tags: list[str]
        A list of tags. These make it easier to subset large groups of dofs.
        Default: []
    travel_expense: float
        The cost of moving the DOF from the current position to the new position.
        Default: 1
    units: Optional[str]
        The units of the DOF (e.g. mm or deg). This is just for plotting and general sanity checking.
        Default: None
    """

    name: str = ""
    description: str = ""
    type: Literal["continuous", "binary", "ordinal", "categorical"] = "continuous"
    search_domain: tuple[float, float] | set[int] | set[str] | set[bool] = (-np.inf, np.inf)
    trust_domain: tuple[float, float] | set[int] | set[str] | set[bool] | None = None
    active: bool = True
    read_only: bool = False
    transform: Literal["log", "logit", "arctanh"] | None = None
    device: Signal | None = None
    tags: list[str] = field(default_factory=list)
    travel_expense: float = 1
    units: str | None = None

    def __repr__(self) -> str:
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
    def __post_init__(self) -> None:
        # Either name or device must be provided
        if (not self.name) != (not self.device):
            if self.device:
                self.name = self.device.name
        else:
            raise ValueError("You must specify exactly one of 'name' or 'device'. Not both.")
        if self.read_only:
            if not self.type:
                if isinstance(self.readback, float):
                    self.type = "continuous"
                else:
                    self.type = "categorical"
                warnings.warn(f"No type was specified for DOF {self.name}. Assuming type={self.type}.", stacklevel=2)
        else:
            if not self.search_domain:
                raise ValueError("You must specify the search domain if read_only=False.")
            # if there is no type, infer it from the search_domain
            if not self.type:
                if isinstance(self.search_domain, tuple):
                    self.type = "continuous"
                elif isinstance(self.search_domain, set):
                    if len(self.search_domain) == 2:
                        self.type = "binary"
                    else:
                        self.type = "categorical"
                else:
                    raise TypeError("'search_domain' must be either a 2-tuple of numbers or a set.")

        if self.type not in DOF_TYPES:
            raise ValueError(f"Invalid DOF type '{self.type}'. 'type' must be one of {DOF_TYPES}.")

        if self.type == "continuous":
            if not self.read_only:
                continuous_search_domain = cast(tuple[float, float], self.search_domain)
                continuous_trust_domain = cast(tuple[float, float], self.trust_domain)
                continuous_domain = cast(tuple[float, float], self.domain)

                _validate_continuous_dof_domains(
                    search_domain=continuous_search_domain,
                    trust_domain=continuous_trust_domain,
                    domain=continuous_domain,
                    read_only=self.read_only,
                )

                self.search_domain = (float(continuous_search_domain[0]), float(continuous_search_domain[1]))

                if not self.device:
                    search_tensor = torch.tensor(continuous_search_domain, dtype=torch.float)
                    transformed = self._transform(search_tensor)
                    center = float(self._untransform(torch.mean(transformed)))
                    self.device = Signal(name=self.name, value=center)
        else:
            discrete_search_domain = cast(set[str] | set[int], self._search_domain)
            discrete_trust_domain = cast(set[str] | set[int], self._trust_domain)

            _validate_discrete_dof_domains(search_domain=discrete_search_domain, trust_domain=discrete_trust_domain)

            if self.type == "binary":
                if not self.search_domain:
                    self.search_domain = {False, True}
                if len(self.search_domain) != 2:
                    raise ValueError("A binary DOF must have a domain of 2.")
            else:
                if not self.search_domain:
                    raise ValueError("Discrete domain must be supplied for ordinal and categorical degrees of freedom.")

            self.device = Signal(name=self.name, value=list(self.search_domain)[0])

        if not self.device:
            raise ValueError("Expected device to be set. Please check that the DOF has a name or device.")

        if not self.read_only:
            # check that the device has a put method
            if isinstance(self.device, SignalRO):
                raise ValueError("You must specify read_only=True for a read-only device.")

        # all dof degrees of freedom are hinted
        self.device.kind = "hinted"

    @property
    def _search_domain(self) -> tuple[float, float] | set[int] | set[str] | set[bool]:
        """
        Compute the search domain of the DOF.
        """
        if self.read_only:
            value = self.readback
            if self.type == "continuous":
                return (value, value)
            else:
                return {value}
        else:
            return self.search_domain

    @property
    def _trust_domain(self) -> tuple[float, float] | set[int] | set[str] | set[bool]:
        """
        If trust_domain is None, then we return the total domain.
        """
        return self.trust_domain or self.domain

    @property
    def domain(self) -> tuple[float, float] | set[int] | set[str] | set[bool]:
        """
        The total domain; the user can't control this. This is what we fall back on as the trust_domain if none is supplied.
        If the DOF is continuous:
            If there is a transform, return the domain of the transform
            Else, return (-inf, inf)
        If the DOF is discrete:
            If there is a trust domain, return the trust domain
            Else, return the search domain
        """
        if self.type == "continuous":
            if not self.transform:
                return (-np.inf, np.inf)
            else:
                return TRANSFORM_DOMAINS[self.transform]
        else:
            return self.trust_domain or self.search_domain

    def _transform(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        if self.type != "continuous":
            raise ValueError("Cannot transform non-continuous DOFs.")

        # Since the DOF is continuous, we can safely assume the domains are tuples
        domain = cast(tuple[float, float], self.domain)
        x = torch.where((x > domain[0]) & (x < domain[1]), x, torch.nan)

        if self.transform == "log":
            x = torch.log(x)
        if self.transform == "logit":
            x = (x / (1 - x)).log()
        if self.transform == "arctanh":
            x = torch.arctanh(x)

        # If we are normalizing, we also need to transform the search domain
        if normalize and not self.read_only:
            min, max = self._transform(torch.tensor(self._search_domain, dtype=torch.double), normalize=False)
            x = (x - min) / (max - min)

        return x

    def _untransform(self, x: torch.Tensor) -> torch.Tensor:
        if self.type != "continuous":
            raise ValueError("Cannot untransform non-continuous DOFs.")

        if not self.read_only:
            min, max = self._transform(torch.tensor(self._search_domain, dtype=torch.double), normalize=False)
            x = x * (max - min) + min

        if self.transform is None:
            return x
        if self.transform == "log":
            return torch.exp(x)
        if self.transform == "logit":
            return 1 / (1 + torch.exp(-x))
        if self.transform == "arctanh":
            return torch.tanh(x)

    @property
    def readback(self) -> Any:
        # there is probably a better way to do this
        if not self.device:
            raise ValueError("DOF has no device.")
        return self.device.read()[self.device.name]["value"]

    @property
    def summary(self) -> pd.Series:
        series = pd.Series(index=list(DOF_FIELD_TYPES.keys()), dtype="object")
        for attr in series.index:
            value = getattr(self, attr)
            if attr in ["search_domain", "trust_domain", "domain"]:
                if (self.type == "continuous") and not self.read_only and value is not None:
                    if attr in ["search_domain", "trust_domain"]:
                        value = f"[{value[0]:.02e}, {value[1]:.02e}]"
                    else:
                        value = f"({value[0]:.02e}, {value[1]:.02e})"
            series[attr] = value if value is not None else ""
        return series

    @property
    def label_with_units(self) -> str:
        return f"{self.description}{f' [{self.units}]' if self.units else ''}"

    @property
    def has_model(self) -> bool:
        return hasattr(self, "model")

    def activate(self) -> None:
        self.active = True

    def deactivate(self) -> None:
        self.active = False


class DOFList(Sequence[DOF]):
    def __init__(self, dofs: list[DOF] | None = None) -> None:
        dofs = dofs or []
        _validate_dofs(dofs)
        self.dofs = dofs

    @property
    def names(self) -> list[str]:
        return [dof.name for dof in self.dofs]

    @property
    def devices(self) -> list[Signal]:
        return [dof.device for dof in self.dofs]

    def __call__(self, *args: Any, **kwargs: Any) -> "DOFList":
        return self.subset(*args, **kwargs)

    def __getattr__(self, attr: str) -> DOF | list[Any] | torch.Tensor:
        # This is called if we can't find the attribute in the normal way.
        if all(hasattr(dof, attr) for dof in self.dofs):
            if DOF_FIELD_TYPES.get(attr) in [float, int, bool]:
                return torch.tensor([getattr(dof, attr) for dof in self.dofs])
            return [getattr(dof, attr) for dof in self.dofs]
        if attr in self.names:
            return self.__getitem__(attr)

        raise AttributeError(f"DOFList object has no attribute named '{attr}'.")

    @overload
    def __getitem__(self, key: int) -> DOF: ...

    @overload
    def __getitem__(self, key: str) -> DOF: ...

    @overload
    def __getitem__(self, key: slice) -> Sequence[DOF]: ...

    @overload
    def __getitem__(self, key: list[int]) -> Sequence[DOF]: ...

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

    def __len__(self) -> int:
        return len(self.dofs)

    def __repr__(self) -> str:
        return self.summary.T.__repr__()

    def _repr_html_(self) -> str | None:
        return self.summary.T._repr_html_()  # type: ignore

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Transform X to the transformed unit hypercube.
        """
        if X.shape[-1] != len(self):
            raise ValueError(f"Cannot transform points with shape {X.shape} using DOFs with dimension {len(self)}.")

        return torch.cat([dof._transform(X[..., i]).unsqueeze(-1) for i, dof in enumerate(self.dofs)], dim=-1)

    def untransform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Transform the transformed unit hypercube to the search domain.
        """
        if X.shape[-1] != len(self):
            raise ValueError(f"Cannot untransform points with shape {X.shape} using DOFs with dimension {len(self)}.")

        return torch.cat(
            [dof._untransform(X[..., i]).unsqueeze(-1) for i, dof in enumerate(self.subset(active=True))], dim=-1
        )

    @property
    def readback(self) -> list[Any]:
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
    def search_domain(self) -> torch.Tensor:
        """
        Returns a (n_dof, 2) array of domain.
        """
        return torch.tensor([dof._search_domain for dof in self.dofs])

    @property
    def trust_domain(self) -> torch.Tensor:
        """
        Returns a (n_dof, 2) array of domain.
        """
        return torch.tensor([dof._trust_domain for dof in self.dofs])

    def add(self, dof: DOF) -> None:
        _validate_dofs([*self.dofs, dof])
        self.dofs.append(dof)

    @staticmethod
    def _test_dof(
        dof: DOF,
        type: Literal["continuous", "binary", "ordinal", "categorical"] | None = None,
        active: bool | None = None,
        read_only: bool | None = None,
        tag: str | None = None,
    ) -> bool:
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

    def subset(
        self,
        type: Literal["continuous", "binary", "ordinal", "categorical"] | None = None,
        active: bool | None = None,
        read_only: bool | None = None,
        tag: str | None = None,
    ) -> "DOFList":
        return DOFList(
            [dof for dof in self.dofs if self._test_dof(dof, type=type, active=active, read_only=read_only, tag=tag)]
        )

    def activate(self, **subset_kwargs: Any) -> None:
        for dof in self.dofs:
            if self._test_dof(dof, **subset_kwargs):
                dof.active = True

    def deactivate(self, **subset_kwargs: Any) -> None:
        for dof in self.dofs:
            if self._test_dof(dof, **subset_kwargs):
                dof.active = False

    def activate_only(self, **subset_kwargs: Any) -> None:
        for dof in self.dofs:
            if self._test_dof(dof, **subset_kwargs):
                dof.active = True
            else:
                dof.active = False

    def deactivate_only(self, **subset_kwargs: Any) -> None:
        for dof in self.dofs:
            if self._test_dof(dof, **subset_kwargs):
                dof.active = False
            else:
                dof.active = True


class BrownianMotion(SignalRO):
    """
    Read-only degree of freedom simulating brownian motion
    """

    def __init__(self, name: str | None = None, theta: float = 0.95, *args: Any, **kwargs: Any) -> None:
        name = name if name is not None else str(uuid.uuid4())

        super().__init__(*args, name=name, **kwargs)

        self.theta = theta
        self.old_t = ttime.monotonic()
        self.old_y = 0.0

    def get(self) -> float:
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

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def get(self) -> float:
        return ttime.time()


class ConstantReadback(SignalRO):
    """
    Returns a constant every time you read it (more useful than you'd think).
    """

    def __init__(self, constant: float = 1, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.constant = constant

    def get(self) -> float:
        return self.constant


def _validate_dofs(dofs: Sequence[DOF]) -> Sequence[DOF]:
    """Validate a sequence of DOFs."""
    dof_names = [dof.name for dof in dofs]

    # check that dof names are unique
    unique_dof_names, counts = np.unique(dof_names, return_counts=True)
    duplicate_dof_names = unique_dof_names[counts > 1]
    if len(duplicate_dof_names) > 0:
        raise ValueError(f"Duplicate name(s) in supplied dofs: {duplicate_dof_names}")

    return list(dofs)


def _validate_continuous_dof_domains(
    search_domain: tuple[float, float], trust_domain: tuple[float, float], domain: tuple[float, float], read_only: bool
) -> None:
    """
    A DOF MUST have a search domain, and it MIGHT have a trust domain or a transform domain

    Check that all the domains are kosher by enforcing that:
    search_domain \\subseteq trust_domain \\subseteq domain
    """
    if not read_only:
        if len(search_domain) != 2:
            raise ValueError(f"Bad search domain {search_domain}. The search domain must have length 2.")
        search_domain = (float(search_domain[0]), float(search_domain[1]))

        if search_domain[0] >= search_domain[1]:
            raise ValueError("The lower search bound must be strictly less than the upper search bound.")

        if domain is not None:
            if (search_domain[0] <= domain[0]) or (search_domain[1] >= domain[1]):
                raise ValueError(f"The search domain {search_domain} must be a strict subset of the domain {domain}.")

        if trust_domain is not None:
            if (search_domain[0] < trust_domain[0]) or (search_domain[1] > trust_domain[1]):
                raise ValueError(f"The search domain {search_domain} must be a subset of the trust domain {trust_domain}.")

    if (trust_domain is not None) and (domain is not None):
        if (trust_domain[0] < domain[0]) or (trust_domain[1] > domain[1]):
            raise ValueError(f"The trust domain {trust_domain} must be a subset of the domain {domain}.")


def _validate_discrete_dof_domains(search_domain: set[int] | set[str], trust_domain: set[int] | set[str]) -> None:
    """
    Check that all the domains are kosher by enforcing that:
    search_domain \\subseteq trust_domain
    """
    if not trust_domain.issuperset(search_domain):
        raise ValueError(f"The trust domain {trust_domain} not a superset of the search domain {search_domain}.")
