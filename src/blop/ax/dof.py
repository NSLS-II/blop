from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import re
from typing import Literal, cast

from ax import ChoiceParameterConfig, RangeParameterConfig
from ax.api.types import TParameterValue
from bluesky.protocols import NamedMovable


@dataclass(frozen=True, kw_only=True)
class DOF(ABC):
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
    bounds: tuple[float, float]
    parameter_type: Literal["float", "int"]
    step_size: float | None = None
    scaling: Literal["linear", "log"] | None = None

    def to_ax_parameter_config(self) -> RangeParameterConfig:
        return RangeParameterConfig(
            name=self.parameter_name,
            bounds=self.bounds,
            parameter_type=self.parameter_type,
            step_size=self.step_size,
            scaling=self.scaling,
        )


@dataclass(frozen=True, kw_only=True)
class ChoiceDOF(DOF):
    values: list[float] | list[int] | list[str] | list[bool]
    parameter_type: Literal["float", "int", "str", "bool"]
    is_ordered: bool | None = None
    dependent_parameters: Mapping[TParameterValue, Sequence[str]] | None = None

    def to_ax_parameter_config(self) -> ChoiceParameterConfig:
        return ChoiceParameterConfig(
            name=self.parameter_name,
            values=self.values,
            parameter_type=self.parameter_type,
            is_ordered=self.is_ordered,
            dependent_parameters=self.dependent_parameters,
        )


class DOFConstraint:
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
