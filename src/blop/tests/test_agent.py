import pytest  # noqa F401


def test_agent(agent_2d_1f, RE):
    a = agent_2d_1f
    RE(a.learn("qr", n=4))

    best = a.best
    assert [dof.name in best for dof in a.dofs]
    assert [obj.name in best for obj in a.objectives]
    assert a.dofs.x1 is a.dofs[0]

    print(a.dofs)
    print(a.objectives)


def test_refresh(agent_2d_1f, RE):
    """
    Test that the agent can determine when the number of DOFs has changed, and adjust.
    """
    a = agent_2d_1f
    RE(a.learn("qr", n=4))

    RE(a.learn("qei", n=1))
    a.dofs[2].activate()
    RE(a.learn("qei", n=1))


def test_forget(agent_2d_1f, RE):
    a = agent_2d_1f
    RE(a.learn("qr", n=4))
    a.forget(last=2)


def test_benchmark(agent_2d_1f, RE):
    a = agent_2d_1f
    per_iter_learn_kwargs_list = [{"acqf": "qr", "n": 32}, {"acqf": "qei", "n": 2, "iterations": 2}]
    RE(a.benchmark(output_dir="/tmp/blop", iterations=1, per_iter_learn_kwargs_list=per_iter_learn_kwargs_list))
