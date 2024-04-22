import pytest  # noqa F401


def test_agent(agent_2d_1f, RE):
    RE(agent_2d_1f.learn("qr", n=4))

    best = agent_2d_1f.best

    assert [dof.name in best for dof in agent_2d_1f.dofs]
    assert [obj.name in best for obj in agent_2d_1f.objectives]

    print(agent_2d_1f.dofs)
    print(agent_2d_1f.objectives)


def test_forget(agent_2d_1f, RE):
    RE(agent_2d_1f.learn("qr", n=4))
    agent_2d_1f.forget(last=2)


def test_benchmark(agent_2d_1f, RE):
    per_iter_learn_kwargs_list = [{"acqf": "qr", "n": 32}, {"acqf": "qei", "n": 2, "iterations": 2}]
    RE(agent_2d_1f.benchmark(output_dir="/tmp/blop", iterations=1, per_iter_learn_kwargs_list=per_iter_learn_kwargs_list))
