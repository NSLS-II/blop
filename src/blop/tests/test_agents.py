import pytest  # noqa F401
import numpy as np

from .conftest import SIMPLE_AGENTS, CONSTRAINED_AGENTS


# Core functionality test - test basic agent operations with a representative agent
@pytest.mark.parametrize("agent", ["simple_2d"], indirect=True)
def test_agent_core_functionality(agent, RE):
    """Test core agent functionality with a simple 2D agent."""
    RE(agent.learn("qr", n=32))

    best = agent.best
    assert [dof.name in best for dof in agent.dofs]
    assert [obj.name in best for obj in agent.objectives]
    assert agent.dofs.x1 is agent.dofs[0]

    RE(agent.learn("qei", n=2))

    # Test basic functions
    agent.refresh()
    agent.redigest()


@pytest.mark.parametrize("agent", SIMPLE_AGENTS, indirect=True)
def test_simple_agents(agent, RE):
    """Test simple agents (1D and 2D unconstrained)."""
    RE(agent.learn("qr", n=16))
    RE(agent.learn("qei", n=2))
    assert len(agent.dofs) >= 1
    assert len(agent.objectives) >= 1


@pytest.mark.parametrize("agent", CONSTRAINED_AGENTS, indirect=True)
def test_constrained_agents(agent, RE):
    """Test constrained and multi-objective agents."""
    RE(agent.learn("qr", n=16))
    RE(agent.learn("qei", n=2))
    # Constrained agents should have constraints or multiple objectives
    has_constraints = any(obj.constraint is not None for obj in agent.objectives)
    has_multiple_objectives = len(agent.objectives) > 1
    assert has_constraints or has_multiple_objectives


def test_complex_agent(RE, setup):
    """Test the complex 3D agent with read-only DOFs."""
    from blop import DOF

    from .conftest import create_agent_from_config

    agent = create_agent_from_config("complex_3d", db=setup)
    agent.dofs.add(DOF(name="dummy", search_domain=(0, 1), active=False))

    RE(agent.learn("qr", n=16))

    # Check that read-only DOFs exist
    readonly_dofs = [dof for dof in agent.dofs if dof.read_only]
    assert len(readonly_dofs) >= 1

    # Test learning with deactivated DOFs
    RE(agent.learn("qei", n=2))


def test_agent_state_management(RE, setup):
    """Test agent state management (save/load/reset) with one agent."""
    from blop import DOF

    from .conftest import create_agent_from_config

    agent = create_agent_from_config("simple_2d", db=setup)
    agent.dofs.add(DOF(name="dummy", search_domain=(0, 1), active=False))

    RE(agent.learn("qr", n=8))

    # Test trust domains
    dof = agent.dofs(active=True)[0]
    raw_x = agent.raw_inputs(dof.name).numpy()
    dof.trust_domain = tuple(np.nanquantile(raw_x, q=[0.2, 0.8]))

    obj = agent.objectives(active=True)[0]
    raw_y = agent.raw_targets(index=obj.name).numpy()
    obj.trust_domain = tuple(np.nanquantile(raw_y, q=[0.2, 0.8]))

    RE(agent.learn("qei", n=2))

    # Test save/load/reset
    agent.save_data("/tmp/test_save_data.h5")
    agent.reset()
    agent.load_data("/tmp/test_save_data.h5")
    RE(agent.learn("qei", n=2))


def test_dummy_dof_handling(RE, setup):
    """Test handling of dummy/inactive DOFs."""
    from blop import DOF

    from .conftest import create_agent_from_config

    agent = create_agent_from_config("simple_2d", db=setup)
    dummy_dof = DOF(name="dummy", search_domain=(0, 1), active=False)
    agent.dofs.add(dummy_dof)

    RE(agent.learn("qr", n=8))

    # Test activating dummy DOF
    dummy_dof.activate()
    RE(agent.learn("qei", n=2))
    dummy_dof.deactivate()


def test_agent_plotting(RE, setup):
    """Test plotting functionality with one agent."""
    from blop import DOF

    from .conftest import create_agent_from_config

    agent = create_agent_from_config("simple_2d", db=setup)
    agent.dofs.add(DOF(name="dummy", search_domain=(0, 1), active=False))

    RE(agent.learn("qr", n=16))
    RE(agent.learn("qei", n=2))

    # Test plots (these should not raise exceptions)
    agent.plot_objectives()
    agent.plot_acquisition()
    agent.plot_validity()
    agent.plot_history()


def test_forgetting_mechanism(RE, setup):
    """Test the forgetting mechanism."""
    from blop import DOF

    from .conftest import create_agent_from_config

    agent = create_agent_from_config("simple_2d", db=setup)
    agent.dofs.add(DOF(name="dummy", search_domain=(0, 1), active=False))

    RE(agent.learn("qr", n=8))
    initial_data_points = len(agent.raw_inputs())

    agent.forget(last=2)

    # Should have fewer data points after forgetting
    assert len(agent.raw_inputs()) < initial_data_points


def test_benchmark(RE, setup):
    """Test benchmarking functionality."""
    from blop import DOF

    from .conftest import create_agent_from_config

    agent = create_agent_from_config("simple_2d", db=setup)
    agent.dofs.add(DOF(name="dummy", search_domain=(0, 1), active=False))

    per_iter_learn_kwargs_list = [{"acqf": "qr", "n": 16}, {"acqf": "qei", "n": 2, "iterations": 2}]
    RE(agent.benchmark(output_dir="/tmp/blop", iterations=1, per_iter_learn_kwargs_list=per_iter_learn_kwargs_list))
