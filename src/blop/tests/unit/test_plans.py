from unittest.mock import patch

import bluesky.plan_stubs as bps
import pytest
from bluesky.run_engine import RunEngine

from blop.dofs import DOF
from blop.plans import acquire, acquire_with_background

from .conftest import MovableSignal, ReadableSignal


@pytest.fixture(scope="function")
def RE():
    return RunEngine({})


def test_acquire_single_dof(RE):
    dof = DOF(movable=MovableSignal("x1", initial_value=-1.0), search_domain=(-5.0, 5.0))
    readable = ReadableSignal("objective")
    with patch.object(readable, "read", wraps=readable.read) as mock_read:
        RE(
            acquire(
                readables=[readable],
                dofs={"x1": dof},
                trials={0: {"x1": 0.0}},
            )
        )
        assert mock_read.call_count == 1

    assert dof.movable.read()["x1"]["value"] == 0.0


def test_acquire_with_background(RE):
    def block_beam():
        yield from bps.null()

    def unblock_beam():
        yield from bps.null()

    dof = DOF(movable=MovableSignal("x1", initial_value=-1.0), search_domain=(-5.0, 5.0))
    readable = ReadableSignal("objective")

    with patch.object(readable, "read", wraps=readable.read) as mock_read:
        RE(
            acquire_with_background(
                readables=[readable],
                dofs={"x1": dof},
                trials={0: {"x1": 0.0}},
                block_beam=block_beam,
                unblock_beam=unblock_beam,
            )
        )
        assert mock_read.call_count == 2
    assert dof.movable.read()["x1"]["value"] == 0.0
