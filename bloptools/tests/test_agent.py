import os

import pytest


@pytest.mark.test_func
def test_writing_hypers(RE, agent):
    RE(agent.initialize("qr", n_init=32))

    agent.save_hypers("hypers.h5")

    RE(agent.initialize("qr", n_init=8, hypers="hypers.h5"))

    os.remove("hypers.h5")


@pytest.mark.test_func
def test_writing_hypers_multitask(RE, multitask_agent):
    RE(multitask_agent.initialize("qr", n_init=32))

    multitask_agent.save_hypers("hypers.h5")

    RE(multitask_agent.initialize("qr", n_init=8, hypers="hypers.h5"))

    os.remove("hypers.h5")
