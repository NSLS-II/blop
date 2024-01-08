import pytest  # noqa F401


@pytest.mark.parametrize("item", ["mean", "error", "qei"])
def test_napari_viewer(agent, RE, item):
    RE(agent.learn("qr", n=4))
    agent.view(item)
