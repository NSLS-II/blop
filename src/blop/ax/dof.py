from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

from ax import ChoiceParameterConfig, RangeParameterConfig
from ax.api.types import TParameterValue
from bluesky.protocols import NamedMovable


@dataclass(frozen=True, kw_only=True)
class DOF:
    name: str | None = None
    movable: NamedMovable | None = None

    def __post_init__(self) -> None:
        if not (bool(self.name) ^ bool(self.movable)):
            raise ValueError("Either name or movable must be provided, but not both or neither.")

    @property
    def parameter_name(self) -> str:
        return self.name or cast(NamedMovable, self.movable).name


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
