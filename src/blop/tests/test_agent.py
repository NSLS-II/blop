import pytest  # noqa F401


def test_agent(agent, RE):
    RE(agent.learn("qr", n=4))

    assert [dof.name in agent.best for dof in agent.dofs]
    assert [obj.name in agent.best for obj in agent.objectives]


def test_forget(agent, RE):
    RE(agent.learn("qr", n=4))
    agent.forget(last=2)


def test_benchmark(agent, RE):
    per_iter_learn_kwargs_list = [{"acq_func": "qr", "n": 64}, {"acq_func": "qei", "n": 2, "iterations": 2}]
    RE(agent.benchmark(output_dir="/tmp/blop", iterations=1, per_iter_learn_kwargs_list=per_iter_learn_kwargs_list))
