import pytest  # noqa F401


def test_agent(agent_2dof_1fit, RE):
    RE(agent_2dof_1fit.learn("qr", n=4))

    agent_2dof_1fit.dofs
    agent_2dof_1fit.objectives
    agent_2dof_1fit.best
    agent_2dof_1fit.best_inputs


def test_forget(agent_2dof_1fit, RE):
    RE(agent_2dof_1fit.learn("qr", n=4))
    agent_2dof_1fit.forget(last=2)


def test_benchmark(agent_2dof_1fit, RE):
    per_iter_learn_kwargs_list = [{"acqf": "qr", "n": 32}, {"acqf": "qei", "n": 2, "iterations": 2}]
    RE(agent_2dof_1fit.benchmark(output_dir="/tmp/blop", iterations=1, per_iter_learn_kwargs_list=per_iter_learn_kwargs_list))
