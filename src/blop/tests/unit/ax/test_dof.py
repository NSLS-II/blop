import pytest
from ax import ChoiceParameterConfig, RangeParameterConfig

from blop.ax.dof import DOF, ChoiceDOF, RangeDOF

from ..conftest import MovableSignal


def test_dof_movable():
    movable = MovableSignal(name="test_movable")
    dof = DOF(movable=movable)
    assert dof.parameter_name == "test_movable"


def test_dof_name():
    dof = DOF(name="test_name")
    assert dof.parameter_name == "test_name"


def test_dof_invalid():
    movable = MovableSignal(name="test_movable")
    with pytest.raises(ValueError):
        DOF(movable=movable, name="test_movable")

    with pytest.raises(ValueError):
        DOF(movable=movable, name="test_name")


def test_range_dof():
    dof1 = RangeDOF(name="test_name", bounds=(0, 1), parameter_type="float", step_size=0.1, scaling="linear")

    assert dof1.to_ax_parameter_config() == RangeParameterConfig(
        name="test_name", bounds=(0, 1), parameter_type="float", step_size=0.1, scaling="linear"
    )

    movable = MovableSignal(name="test_movable")
    dof2 = RangeDOF(movable=movable, bounds=(0, 1), parameter_type="float", step_size=0.1, scaling="linear")

    assert dof2.to_ax_parameter_config() == RangeParameterConfig(
        name="test_movable", bounds=(0, 1), parameter_type="float", step_size=0.1, scaling="linear"
    )


def test_choice_dof():
    dof1 = ChoiceDOF(
        name="test_name", values=[0, 1, 2, 3, 4, 5], parameter_type="int", is_ordered=True, dependent_parameters=None
    )

    assert dof1.to_ax_parameter_config() == ChoiceParameterConfig(
        name="test_name", values=[0, 1, 2, 3, 4, 5], parameter_type="int", is_ordered=True, dependent_parameters=None
    )

    movable = MovableSignal(name="test_movable")
    dof2 = ChoiceDOF(
        movable=movable, values=[0, 1, 2, 3, 4, 5], parameter_type="int", is_ordered=True, dependent_parameters=None
    )

    assert dof2.to_ax_parameter_config() == ChoiceParameterConfig(
        name="test_movable", values=[0, 1, 2, 3, 4, 5], parameter_type="int", is_ordered=True, dependent_parameters=None
    )
