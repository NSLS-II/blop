from unittest.mock import patch

import bluesky.plan_stubs as bps
import pytest
from bluesky.run_engine import RunEngine

from blop.dofs import DOF
from blop.plans import acquire_with_background, default_acquire

from .conftest import MovableSignal, ReadableSignal


@pytest.fixture(scope="function")
def RE():
    return RunEngine({})


def test_default_acquire_single_movable_readable(RE):
    movable = MovableSignal("x1", initial_value=-1.0)
    readable = ReadableSignal("objective")
    movable_and_input = {movable: [0.0]}
    with patch.object(readable, "read", wraps=readable.read) as mock_read:
        RE(
            default_acquire(
                movable_and_input,
                [readable],
            )
        )
        assert mock_read.call_count == 1

    assert movable.read()["x1"]["value"] == 0.0


def test_acquire_with_background(RE):
    def block_beam():
        yield from bps.null()

    def unblock_beam():
        yield from bps.null()

    movable = MovableSignal("x1", initial_value=-1.0)
    readable = ReadableSignal("objective")
    movable_and_input = {movable: [0.0]}
    with patch.object(readable, "read", wraps=readable.read) as mock_read:
        RE(
            acquire_with_background(
                movables=movable_and_input,
                readables=[readable],
                block_beam=block_beam,
                unblock_beam=unblock_beam,
            )
        )
        assert mock_read.call_count == 2
    assert movable.read()["x1"]["value"] == 0.0
