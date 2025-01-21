import pytest  # noqa F401

from .conftest import all_agents


@pytest.mark.parametrize("agent", all_agents, indirect=True)
def test_agent(agent, RE, db):
    """
    All agents should be able to do these things.
    """

    agent.db = db
    RE(agent.learn("qr", n=64))

    best = agent.best
    assert [dof.name in best for dof in agent.dofs]
    assert [obj.name in best for obj in agent.objectives]
    assert agent.dofs.x1 is agent.dofs[0]

    print(agent.dofs)
    print(agent.objectives)

    # test refreshing
    RE(agent.learn("qei", n=1))
    agent.dofs.activate()
    RE(agent.learn("qei", n=1))

    # test forgetting
    RE(agent.learn("qr", n=4))
    agent.forget(last=2)

    # test some functions
    agent.refresh()
    agent.redigest()

    # save the data, reset the agent, and get the data back
    agent.save_data("/tmp/test_save_data.h5")
    agent.reset()
    agent.load_data("/tmp/test_save_data.h5")

    RE(agent.learn("qei", n=2))

    # test plots
    agent.plot_objectives()
    agent.plot_acquisition()
    agent.plot_validity()
    agent.plot_history()


@pytest.mark.parametrize("agent", all_agents, indirect=True)
def test_benchmark(agent, RE, db):
    agent.db = db
    per_iter_learn_kwargs_list = [{"acqf": "qr", "n": 32}, {"acqf": "qei", "n": 2, "iterations": 2}]
    RE(agent.benchmark(output_dir="/tmp/blop", iterations=1, per_iter_learn_kwargs_list=per_iter_learn_kwargs_list))
