import logging

from . import utils  # noqa F401
from ._version import __version__, __version_tuple__  # noqa: F401
from .agent import Agent  # noqa F401
from .dofs import DOF  # noqa F401
from .objectives import Objective  # noqa F401

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("maria")
