import logging
import time as ttime
import uuid
import warnings
from collections.abc import Iterable, Sequence
from typing import Any, Literal, cast, overload

import numpy as np
import pandas as pd
import torch
from bluesky.protocols import NamedMovable
from ophyd import Signal, SignalRO  # type: ignore[import-untyped]

from .utils.sets import element_of, intersection, is_subset, validate_set

logger = logging.getLogger("blop")

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


class DOF:
    def __init__(
        self,
        name: str = None,
        description: str = "",
        type: Literal["continuous", "binary", "ordinal", "categorical"] = "continuous",
        search_domain: tuple[float, float] | set[int] | set[str] | set[bool] = (-np.inf, np.inf),
        trust_domain: tuple[float, float] | set[int] | set[str] | set[bool] | None = None,
        domain: tuple[float, float] | set[int] | set[str] | set[bool] | None = None,
        active: bool = True,
        read_only: bool = False,
        transform: Literal["log", "logit", "arctanh"] | None = None,
        movable: NamedMovable | None = None,
        device: Signal | None = None,
        tags: list[str] = None,
        travel_expense: float = 1,
        units: str | None = None,
    ):
        """A degree of freedom (DOF), to be used by an agent.

        .. deprecated:: v0.9.0
            This class is deprecated and will be removed in Blop v1.0.0. Use ``blop.ax.DOF`` instead.

        Parameters
        ----------
        name: str
            The name of the input. This is used as a key to index observed data.

            .. deprecated:: v0.8.0
                This argument is deprecated and will be removed in Blop v1.0.0. The `movable.name` will be used instead.
        description: str, optional
            A longer, more descriptive name for the DOF.

            .. deprecated:: v0.8.0
                This argument is deprecated and will be removed in Blop v1.0.0.
        type: Literal["continuous", "binary", "ordinal", "categorical"]
            Describes the type of the input to be optimized. An outcome can be
            - Continuous, meaning any real number.
            - Binary, meaning that it can take one of two values (e.g. [on, off])
            - Ordinal, meaning ordered categories (e.g. [low, medium, high])
            - Categorical, meaning non-ordered categories (e.g. [mango, banana, papaya])
            Default: "continuous"
        search_domain: Optional[Union[tuple[float, float], set[int], set[str]]]
            The range of value for the agent to search. Must be supplied for a non read-only DOF.
            - if continuous, a tuple of the lower and upper limit of the input for the agent to search.
            - if discrete, a set of the possible values for the input.
            Default: (-np.inf, np.inf)
        trust_domain: Optional[Union[tuple[float, float], set[int], set[str]]]
            The agent will reject all data where the DOF value is outside this domain.
            Must span a equal or larger range than the search domain.
            Default: (-np.inf, np.inf)
        domain: Optional[Union[tuple[float, float], set[int], set[str]]]
            The total domain of the input. This is inferred from the transform, unless the input is discrete.
            Must span a equal or larger range than the trust domain.
            Default: (-np.inf, np.inf)
        active: Optional[bool]
            If True, the agent will try to use the DOF in its optimization. If False, the agent will
            still read the DOF but not include it any model or acquisition function.
            Default: True

            .. deprecated:: v0.8.0
                This attribute is deprecated and will be removed in Blop v1.0.0. Inactive DOFs are no longer supported.
        read_only: Optional[bool]
            If True, the agent will not try to set the DOF. Must be set to True if the supplied ophyd
            device is read-only. The behavior of the DOF on each sample for read only/not read-only are:

            - 'read': the agent will read the input on every acquisition (all dofs are always read)
            - 'move': the agent will try to set and optimize over these (there must be at least one of these)
            - 'input' means that the agent will use the value to make its posterior

            Default: False
        transform: Optional[Literal["log", "logit", "arctanh"]]
            A transform to apply to the objective, to make the process outputs more Gaussian.
            Default: None
        movable: Optional[NamedMovable]
            A `bluesky.protocols.NamedMovable`. If not supplied, a dummy `bluesky.protocols.NamedMovable` will be generated.
            Default: None
        device: Optional[Signal]
            An `ophyd.Signal`. If not supplied, a dummy `ophyd.Signal` will be generated.
            Default: None

            .. deprecated:: v0.8.0
                This attribute is deprecated and will be removed in Blop v1.0.0. Ophyd will no longer be a direct dependency.
                Use `movable` which must be a `bluesky.protocols.NamedMovable` instead.
        tags: Optional[list[str]]
            A list of tags. These make it easier to subset large groups of DOFs.
            Default: []

            .. deprecated:: v0.8.0
                This attribute is deprecated and will be removed in Blop v1.0.0.
        travel_expense: Optional[float]
            The relative cost of moving the DOF from the current position to the new position.
            Default: 1

            .. deprecated:: v0.8.0
                This attribute is deprecated and will be removed in Blop v1.0.0. It may resurface in a future version.
        units: Optional[str]
            The units of the DOF (e.g. mm or deg). This is just for plotting and general sanity checking.
            Default: None

            .. deprecated:: v0.8.0
                This attribute is deprecated and will be removed in Blop v1.0.0.
        """
        if name:
            warnings.warn(
                (
                    "The 'name' argument is deprecated and will be removed in Blop v1.0.0. "
                    "The `movable.name` will be used instead."
                ),
                DeprecationWarning,
                stacklevel=2,
            )
        if description:
            warnings.warn(
                "The 'description' argument is deprecated and will be removed in Blop v1.0.0.",
                DeprecationWarning,
                stacklevel=2,
            )
        if device is not None:
            warnings.warn(
                "The 'device' argument is deprecated and will be removed in Blop v1.0.0. Use 'movable' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        if not active:
            warnings.warn(
                "Inactive DOFs are deprecated and will be removed in Blop v1.0.0. DOFs will always be active.",
                DeprecationWarning,
                stacklevel=2,
            )
        if tags:
            warnings.warn(
                "The 'tags' argument is deprecated and will be removed in Blop v1.0.0.",
                DeprecationWarning,
                stacklevel=2,
            )
        if travel_expense is not None and travel_expense != 1:
            warnings.warn(
                (
                    "The 'travel_expense' argument is deprecated and will be removed in Blop v1.0.0. "
                    "It may resurface in a future version."
                ),
                DeprecationWarning,
                stacklevel=2,
            )
        if units:
            warnings.warn(
                "The 'units' argument is deprecated and will be removed in Blop v1.0.0.",
                DeprecationWarning,
                stacklevel=2,
            )

        # these should be set first, as they are just variables
        self.name = name
        self.description = description
        self.type = type
        self.active = active
        self.read_only = read_only
        self.transform = transform
        self.movable: NamedMovable = movable or cast(NamedMovable, device)
        self.tags = tags or []
        self.travel_expense = travel_expense
        self.units = units

        # Either name or device must be provided
        if (not self.name) != (not self.movable):
            if self.movable:
                self.name = self.movable.name
        else:
            raise ValueError("You must specify exactly one of 'name' or 'device'.")

        if self.read_only:
            if not self.type:
                if isinstance(self.readback, float):
                    self.type = "continuous"
                else:
                    self.type = "categorical"
                logger.warning(f"No type was specified for DOF '{self.name}'. Assuming type={self.type}.")
        else:
            if not search_domain:
                raise ValueError("You must specify the search domain if read_only=False.")
            # if there is no type, infer it from the search_domain
            if not self.type:
                if isinstance(search_domain, tuple):
                    self.type = "continuous"
                elif isinstance(search_domain, set):
                    if len(search_domain) == 2:
                        self.type = "binary"
                    else:
                        self.type = "categorical"
                else:
                    raise TypeError("'search_domain' must be either a 2-tuple of numbers or a set.")

        if self.type not in DOF_TYPES:
            raise ValueError(f"Invalid DOF type '{self.type}'. 'type' must be one of {DOF_TYPES}.")

        if self.type == "continuous":
            self.trust_domain = trust_domain
            self.search_domain = search_domain

            if not self.read_only:
                if not self.movable:
                    search_tensor = torch.tensor(search_domain, dtype=torch.float)
                    transformed = self._transform(search_tensor)
                    center = float(self._untransform(torch.mean(transformed)))
                    self.movable = Signal(name=self.name, value=center)

        else:
            if search_domain is None:
                raise ValueError("You must supply a search domain for binary, ordinal, or categorical DOFs.")

            self._domain = domain or trust_domain or search_domain
            self._trust_domain = trust_domain or search_domain
            self._search_domain = search_domain

            if not is_subset(self.search_domain, self.trust_domain, type=self.type):
                raise ValueError(f"The search domain must be a subset of trust domain for DOF '{self.name}'.")

            self.movable = Signal(name=self.name, value=list(self.search_domain)[0])

        if not self.movable:
            raise ValueError("Expected device to be set. Please check that the DOF has a name or device.")

        if not self.read_only:
            # check that the device has a put method
            if isinstance(self.movable, SignalRO):
                raise ValueError("You must specify read_only=True for a read-only device.")

        # all dof degrees of freedom are hinted
        self.movable.kind = "hinted"

    def __repr__(self) -> str:
        filling = ", ".join([f"{k}={repr(v)}" for k, v in self.summary.to_dict().items()])
        return f"{self.__class__.__name__}({filling})"

    @property
    def search_domain(self) -> tuple[float, float] | set[int] | set[str] | set[bool]:
        """
        A writable DOF always has a search domain, and a read-only DOF will return its current value.
        """
        if self.read_only:
            value = self.readback
            if self.type == "continuous":
                return (value, value)
            else:
                return {value}
        else:
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
            search_domain = self.search_domain

            if not is_subset(value, domain, type=self.type, proper=False):
                raise ValueError(f"Cannot set trust domain to {value} as it is not a subset of the domain {domain}.")

            if not is_subset(search_domain, value):
                # The search domain must stay a subset of the trust domain, so set it as the intersection.
                self.search_domain = intersection(self.search_domain, value)

        self._trust_domain = value

    @property
    def domain(self) -> tuple[float, float] | set[int] | set[str] | set[bool]:
        """
        The total domain, determined by the DOF type and transform; the user can't control this.
        This is what we fall back on as the trust_domain if None is supplied.

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
            return self._domain

    def _transform(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        if self.type != "continuous":
            raise ValueError("Cannot transform non-continuous DOFs.")

        # Since the DOF is continuous, we can safely assume the domains are tuples
        x = torch.where(element_of(x, self.trust_domain), x, torch.nan)

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
        """
        The current value of the DOF.

        .. deprecated:: v0.8.0
            This method is deprecated and will be removed in Blop v1.0.0. DOFs will not have a readback
            since Ophyd will no longer be a direct dependency. Instead, use `bluesky.plan_stubs.rd` on your `movable`.
        """
        if not self.movable:
            raise ValueError("DOF has no device.")
        return self.movable.read()[self.movable.name]["value"]

    @property
    def summary(self) -> pd.Series:
        """
        Return a Series summarizing the state of the DOF.

        .. deprecated:: v0.8.0
            This method is deprecated and will be removed in Blop v1.0.0. DOFs will not have a summary.
        """
        series = pd.Series(index=list(DOF_FIELD_TYPES.keys()), dtype="object")
        for attr in series.index:
            value = getattr(self, attr)
            series[attr] = value if value is not None else ""
        return series

    @property
    def label_with_units(self) -> str:
        """
        A label for a plot, perhaps.

        .. deprecated:: v0.8.0
            This method is deprecated and will be removed in Blop v1.0.0. DOFs will not have a label with units.
        """
        return f"{self.description}{f' [{self.units}]' if self.units else ''}"

    @property
    def has_model(self) -> bool:
        """
        .. deprecated:: v0.8.0
            This method is deprecated and will be removed in Blop v1.0.0. Models will not be stored in individaul DOFs.
        """
        return hasattr(self, "model")

    def activate(self) -> None:
        """
        Activate the DOF

        .. deprecated:: v0.8.0
            This method is deprecated and will be removed in Blop v1.0.0. DOFs will always be active.
        """
        self.active = True

    def deactivate(self) -> None:
        """
        Deactivate the DOF

        .. deprecated:: v0.8.0
            This method is deprecated and will be removed in Blop v1.0.0. DOFs will always be active.
        """
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
        return [dof.movable for dof in self.dofs]

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
        active_dofs = self(active=True)
        if X.shape[-1] != len(active_dofs):
            raise ValueError(
                f"Cannot transform points with shape {X.shape} using DOFs with active dimension {len(active_dofs)}."
            )

        return torch.cat([dof._transform(X[..., i]).unsqueeze(-1) for i, dof in enumerate(active_dofs)], dim=-1)

    def untransform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Transform the transformed unit hypercube to the search domain.
        """
        active_dofs = self(active=True)
        if X.shape[-1] != len(active_dofs):
            raise ValueError(
                f"Cannot untransform points with shape {X.shape} using DOFs with active dimension {len(active_dofs)}."
            )

        return torch.cat([dof._untransform(X[..., i]).unsqueeze(-1) for i, dof in enumerate(active_dofs)], dim=-1)

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
        return torch.tensor([dof.search_domain for dof in self.dofs])

    @property
    def trust_domain(self) -> torch.Tensor:
        """
        Returns a (n_dof, 2) array of domain.
        """
        return torch.tensor([dof.trust_domain for dof in self.dofs])

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
        """
        Return all DOFs that
        """
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

    .. deprecated:: v0.8.0
        This class is deprecated and will be removed in Blop v1.0.0. Ophyd will no longer be a direct dependency.
    """

    def __init__(self, name: str | None = None, theta: float = 0.95, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "This class is deprecated and will be removed in Blop v1.0.0. Ophyd will no longer be a direct dependency.",
            DeprecationWarning,
            stacklevel=2,
        )
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

    .. deprecated:: v0.8.0
        This class is deprecated and will be removed in Blop v1.0.0. Ophyd will no longer be a direct dependency.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "This class is deprecated and will be removed in Blop v1.0.0. Ophyd will no longer be a direct dependency.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)

    def get(self) -> float:
        return ttime.time()


class ConstantReadback(SignalRO):
    """
    Returns a constant every time you read it (more useful than you'd think).

    .. deprecated:: v0.8.0
        This class is deprecated and will be removed in Blop v1.0.0. Ophyd will no longer be a direct dependency.
    """

    def __init__(self, constant: float = 1, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "This class is deprecated and will be removed in Blop v1.0.0. Ophyd will no longer be a direct dependency.",
            DeprecationWarning,
            stacklevel=2,
        )
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
