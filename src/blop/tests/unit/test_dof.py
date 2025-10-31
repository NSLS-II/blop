import pytest

from blop.dofs import DOF, DOFConstraint

from .conftest import MovableSignal


def test_dof():
    movable = MovableSignal(name="test_movable")
    dof = DOF(movable=movable, search_domain=(0, 1))
    assert dof.movable == movable
    assert dof.type == "continuous"
    assert dof.search_domain == (0, 1)
    assert not dof.read_only


def test_readonly_dof():
    movable = MovableSignal(name="test_movable")
    dof = DOF(movable=movable, search_domain=(0, 1), read_only=True)
    assert dof.read_only


def test_dof_constraint():
    movable1 = MovableSignal(name="test_movable1")
    movable2 = MovableSignal(name="test_movable2")
    constraint = DOFConstraint(constraint="x1 + x2 <= 10", x1=movable1, x2=movable2)
    assert constraint.to_ax_constraint() == "test_movable1 + test_movable2 <= 10"
    assert str(constraint) == "test_movable1 + test_movable2 <= 10"


def test_invalid_dof_constraint():
    movable1 = MovableSignal(name="test_movable1")
    movable2 = MovableSignal(name="test_movable2")
    with pytest.raises(ValueError):
        DOFConstraint(constraint="x1 + x2 <= 10", x1=movable1, x3=movable2)
