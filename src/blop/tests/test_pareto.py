from blop import DOF

from .conftest import create_agent_from_config


def test_pareto_with_multiobjective_agent(RE, setup):
    """Test Pareto front functionality with multi-objective agent."""
    agent = create_agent_from_config("multiobjective_2d", db=setup)
    agent.dofs.add(DOF(name="dummy", search_domain=(0, 1), active=False))

    RE(agent.learn("qr", n=16))
    agent.plot_pareto_front()


def test_pareto_with_complex_agent(RE, setup):
    """Test Pareto front functionality with complex agent."""
    agent = create_agent_from_config("complex_3d", db=setup)
    agent.dofs.add(DOF(name="dummy", search_domain=(0, 1), active=False))

    RE(agent.learn("qr", n=16))
    agent.plot_pareto_front()


def test_qnehvi_acquisition_with_multiobjective_agent(RE, setup):
    """Test qNEHVI acquisition function with multi-objective agent."""
    agent = create_agent_from_config("multiobjective_2d", db=setup)
    agent.dofs.add(DOF(name="dummy", search_domain=(0, 1), active=False))

    RE(agent.learn("qr", n=16))
    RE(agent.learn("qnehvi", n=2))

    # Test deactivating a DOF
    agent.dofs[0].deactivate()
    RE(agent.learn("qnehvi", n=2))

    # Verify qnehvi attribute exists
    assert hasattr(agent, "qnehvi")


def test_qnehvi_acquisition_with_complex_agent(RE, setup):
    """Test qNEHVI acquisition function with complex agent."""
    agent = create_agent_from_config("complex_3d", db=setup)
    agent.dofs.add(DOF(name="dummy", search_domain=(0, 1), active=False))

    RE(agent.learn("qr", n=16))
    RE(agent.learn("qnehvi", n=2))

    # Test deactivating a DOF
    agent.dofs[0].deactivate()
    RE(agent.learn("qnehvi", n=2))

    # Verify qnehvi attribute exists
    assert hasattr(agent, "qnehvi")
