import pytest

from blop import DOF

from .conftest import SIMPLE_AGENTS, create_agent_from_config


@pytest.mark.parametrize("acqf", ["ei", "pi", "em", "ucb"])
@pytest.mark.parametrize("agent", SIMPLE_AGENTS[:1], indirect=True)
def test_analytic_acqfs_simple_agent(agent, RE, acqf):
    """Test analytic acquisition functions with a simple agent."""
    RE(agent.learn("qr", n=16))
    RE(agent.learn(acqf, n=1))
    # Verify the acquisition function attribute exists
    assert hasattr(agent, acqf)


@pytest.mark.parametrize("acqf", ["qei", "qpi", "qem", "qucb"])
@pytest.mark.parametrize("agent", SIMPLE_AGENTS[:1], indirect=True)
def test_monte_carlo_acqfs_simple_agent(agent, RE, acqf):
    """Test Monte Carlo acquisition functions with a simple agent."""
    RE(agent.learn("qr", n=16))
    RE(agent.learn(acqf, n=1))
    # Verify the acquisition function attribute exists
    assert hasattr(agent, acqf)


def test_acquisition_functions_with_different_agents(RE, setup):
    """Test a few key acquisition functions across different agent types."""
    test_cases = [
        ("simple_1d", "ei"),
        ("simple_2d", "qei"),
        ("constrained_2d", "qei"),
    ]

    for agent_config, acqf in test_cases:
        agent = create_agent_from_config(agent_config, db=setup)
        agent.dofs.add(DOF(name="dummy", search_domain=(0, 1), active=False))

        RE(agent.learn("qr", n=8))
        RE(agent.learn(acqf, n=1))
        assert hasattr(agent, acqf)


def test_acquisition_function_availability(RE, setup):
    """Test that all expected acquisition functions are available."""
    agent = create_agent_from_config("simple_2d", db=setup)
    agent.dofs.add(DOF(name="dummy", search_domain=(0, 1), active=False))

    RE(agent.learn("qr", n=8))

    # Test that key acquisition functions are accessible
    analytic_acqfs = ["ei", "pi", "em", "ucb"]
    monte_carlo_acqfs = ["qei", "qpi", "qem", "qucb"]

    for acqf in analytic_acqfs + monte_carlo_acqfs:
        assert hasattr(agent, acqf), f"Acquisition function {acqf} should be available"
