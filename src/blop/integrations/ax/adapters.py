import logging

from ax import ChoiceParameterConfig, RangeParameterConfig
from ax.api.protocols import IMetric

from ...dofs import DOF
from ...objectives import Objective

logger = logging.getLogger(__name__)


def _to_ax_parameter_config(dof: DOF) -> RangeParameterConfig | ChoiceParameterConfig:
    if dof.read_only:
        parameter_config = ChoiceParameterConfig(
            name=dof.name,
            values=[dof.readback],
            parameter_type="float",
            is_ordered=False,
            dependent_parameters=None,
        )
    elif dof.type == "continuous":
        parameter_config = RangeParameterConfig(
            name=dof.name,
            bounds=dof.search_domain,
            parameter_type="float",
            step_size=None,
            scaling="log" if dof.transform == "log" else "linear",
        )
    else:
        parameter_config = ChoiceParameterConfig(
            name=dof.name,
            values=list(dof.search_domain),
            parameter_type="int",
            is_ordered=dof.type == "ordinal",
            dependent_parameters=None,
        )

    return parameter_config


def configure_parameters(dofs: list[DOF]) -> list[RangeParameterConfig | ChoiceParameterConfig]:
    return [_to_ax_parameter_config(dof) for dof in dofs if dof.active]


def _unpack_objectives(objectives: list[Objective]) -> tuple[str, list[str]]:
    if not all(o.active for o in objectives):
        msg = (
            "Found inactive objectives while configuring the optimization: "
            f"{', '.join([o.name for o in objectives if not o.active])}"
        )
        raise ValueError(msg)

    objective_specs = []
    outcome_contraint_specs = []
    for o in objectives:
        objective_specs.append(o.name if o.target == "max" else f"-{o.name}")
        if o.constraint is not None:
            if isinstance(o.constraint, tuple):
                if o.constraint[0] is not None:
                    outcome_contraint_specs.append(f"{o.name} >= {o.constraint[0]}")
                if o.constraint[1] is not None:
                    outcome_contraint_specs.append(f"{o.name} <= {o.constraint[1]}")
            elif isinstance(o.constraint, set):
                outcome_contraint_specs.append(f"{o.name} in {o.constraint}")
            else:
                raise ValueError(f"Invalid constraint type: {type(o.constraint)}")

    objective_str = ", ".join(objective_specs)
    return objective_str, outcome_contraint_specs


def configure_objectives(objectives: list[Objective]) -> tuple[str, list[str]]:
    active_objectives = [o for o in objectives if o.active]
    objective_str, outcome_constraint_strs = _unpack_objectives(active_objectives)
    logger.info(
        f"Configuring optimization with objective: {objective_str} and outcome constraints: {outcome_constraint_strs}"
    )
    return objective_str, outcome_constraint_strs


def _unpack_inactive_objectives(objectives: list[Objective]) -> list[IMetric]:
    if any(o.active for o in objectives):
        msg = f"Found active objectives while configuring the metrics: {', '.join([o.name for o in objectives if o.active])}"
        raise ValueError(msg)

    return [IMetric(o.name) for o in objectives]


def configure_metrics(objectives: list[Objective]) -> list[IMetric]:
    inactive_objectives = [o for o in objectives if not o.active]
    inactive_metrics = _unpack_inactive_objectives(inactive_objectives)
    return inactive_metrics
