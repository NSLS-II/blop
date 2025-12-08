import re
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

from ax import ChoiceParameterConfig, RangeParameterConfig
from ax.api.types import TParameterValue
from bluesky.protocols import NamedMovable


@dataclass(frozen=True, kw_only=True)
class DOF(ABC):
    """
    Base class for a degree of freedom (DOF) to optimize.

    Attributes
    ----------
    name : str | None
        The name of the DOF. Provide a name if the DOF is not movable.
    movable : NamedMovable | None
        The movable to use for the DOF. Provide a movable if the DOF is movable by Bluesky.
    """

    name: str | None = None
    movable: NamedMovable | None = None

    def __post_init__(self) -> None:
        if not (bool(self.name) ^ bool(self.movable)):
            raise ValueError("Either name or movable must be provided, but not both or neither.")

    @property
    def parameter_name(self) -> str:
        return self.name or cast(NamedMovable, self.movable).name

    @abstractmethod
    def to_ax_parameter_config(self) -> RangeParameterConfig | ChoiceParameterConfig: ...


@dataclass(frozen=True, kw_only=True)
class RangeDOF(DOF):
    """
    A degree of freedom that is a continuous range.

    Attributes
    ----------
    bounds : tuple[float, float]
        The search domain of the DOF.
    parameter_type : Literal["float", "int"]
        The data type of the DOF.
    step_size : float | None, optional
        The step size of the DOF.
    scaling : Literal["linear", "log"] | None, optional
        The scaling of the DOF.
    """

    bounds: tuple[float, float]
    parameter_type: Literal["float", "int"]
    step_size: float | None = None
    scaling: Literal["linear", "log"] | None = None

    def to_ax_parameter_config(self) -> RangeParameterConfig:
        """Convert the DOF to the Ax parameter configuration equivalent."""
        return RangeParameterConfig(
            name=self.parameter_name,
            bounds=self.bounds,
            parameter_type=self.parameter_type,
            step_size=self.step_size,
            scaling=self.scaling,
        )


@dataclass(frozen=True, kw_only=True)
class ChoiceDOF(DOF):
    """
    A degree of freedom that is a discrete choice.

    Attributes
    ----------
    values : list[float] | list[int] | list[str] | list[bool]
        The possible discrete values of the DOF.
    parameter_type : Literal["float", "int", "str", "bool"]
        The data type of the DOF.
    is_ordered : bool | None, optional
        Whether the values are ordered. If not provided, it will be inferred from the values.
    dependent_parameters : Mapping[TParameterValue, Sequence[str]] | None, optional
        Specify which other DOFs are active dependent on specific values of this DOF.
    """

    values: list[float] | list[int] | list[str] | list[bool]
    parameter_type: Literal["float", "int", "str", "bool"]
    is_ordered: bool | None = None
    dependent_parameters: Mapping[TParameterValue, Sequence[str]] | None = None

    def to_ax_parameter_config(self) -> ChoiceParameterConfig:
        """Convert the DOF to the Ax parameter configuration equivalent."""
        return ChoiceParameterConfig(
            name=self.parameter_name,
            values=self.values,
            parameter_type=self.parameter_type,
            is_ordered=self.is_ordered,
            dependent_parameters=self.dependent_parameters,
        )


class DOFConstraint:
    """
    A constraint on DOFs to refine the search space.

    Parameters
    ----------
    constraint : str
        The constraint expression to evaluate.
    **dofs : DOF
        Keyword arguments mapping variables in the constraint to DOFs.
    """

    def __init__(self, constraint: str, **dofs: DOF) -> None:
        self._constraint = constraint
        self._dofs: dict[str, DOF] = dofs
        self._validate_dofs()

    def _validate_dofs(self) -> None:
        if not self._dofs:
            raise ValueError(
                "DOFConstraint requires at least one movable to be specified.\n"
                "Use keyword arguments to map template variables to movables:\n"
                "  DOFConstraint('x + y <= 12', x=motor_x, y=motor_y)\n\n"
                "The variable names (x, y) are your choice and make the constraint readable."
            )
        invalidated: list[tuple[str, DOF]] = []
        for name, dof in self._dofs.items():
            if name not in self._constraint:
                invalidated.append((name, dof))

        if len(invalidated) > 0:
            msg = (
                "The following DOFs did not have matching names in the constraint "
                f"'{self._constraint}': {', '.join([f'{name}={dof.parameter_name}' for name, dof in invalidated])}"
            )
            raise ValueError(msg)

    @property
    def ax_constraint(self) -> str:
        """Convert the constraint to a string that can be used by Ax."""
        template = self._constraint
        for key in self._dofs.keys():
            template = re.sub(f"\\b{key}\\b", f"{{{key}}}", template)
        return template.format(**{key: dof.parameter_name for key, dof in self._dofs.items()})

    def __str__(self) -> str:
        return self.ax_constraint

    def __repr__(self) -> str:
        dofs_str = ", ".join(f"{name}={dof.parameter_name}" for name, dof in self._dofs.items())
        return f"DOFConstraint('{self._constraint}', {dofs_str})"
