import re
from collections.abc import Sequence
from dataclasses import dataclass

from ax.api.protocols import IMetric


@dataclass(frozen=True, kw_only=True)
class Objective:
    name: str
    minimize: bool


class ScalarizedObjective:
    def __init__(self, expression: str, *, minimize: bool, **objectives: Objective):
        self._expression = expression
        self._objectives = objectives
        self._minimize = minimize

        if not self._objectives:
            raise ValueError("ScalarizedObjective requires at least one objective.")

        if any(o.minimize for o in self._objectives.values()):
            raise ValueError(
                "ScalarizedObjective does not support minimizing individual objectives. "
                "You can minimize the scalarized objective instead."
            )

        invalidated = [name for name in self._objectives if name not in self._expression]
        if invalidated:
            raise ValueError(f"Objectives {invalidated} not found in expression '{self._expression}'.")

    @property
    def ax_expression(self) -> str:
        template = self._expression
        for key in self._objectives:
            template = re.sub(f"\\b{key}\\b", f"{{{key}}}", template)

        objective_str = template.format(**{key: objective.name for key, objective in self._objectives.items()})
        return f"-({objective_str})" if self._minimize else objective_str

    def __repr__(self) -> str:
        return (
            f"ScalarizedObjective('{self._expression}', minimize={self._minimize}, "
            f"{', '.join([f'{k}={v.name}' for k, v in self._objectives.items()])})"
        )


class OutcomeConstraint:
    def __init__(self, constraint: str, **outcomes: Objective | IMetric):
        self._constraint = constraint
        self._outcomes = outcomes
        self._validate_outcomes()

    def _validate_outcomes(self) -> None:
        if not self._outcomes:
            raise ValueError("OutcomeConstraint requires at least one outcome.")

        invalidated = [name for name in self._outcomes if name not in self._constraint]
        if invalidated:
            raise ValueError(f"Outcomes {invalidated} not found in constraint '{self._constraint}'.")

    @property
    def ax_constraint(self) -> str:
        template = self._constraint
        for key in self._outcomes:
            template = re.sub(f"\\b{key}\\b", f"{{{key}}}", template)

        return template.format(**{key: outcome.name for key, outcome in self._outcomes.items()})

    def __repr__(self) -> str:
        outcomes_str = ", ".join(f"{k}={v.name}" for k, v in self._outcomes.items())
        return f"OutcomeConstraint('{self._constraint}', {outcomes_str})"

    def __str__(self) -> str:
        return self.ax_constraint


def to_ax_objective_str(objectives: Sequence[Objective]) -> str:
    return ", ".join([o.name if not o.minimize else f"-{o.name}" for o in objectives])
