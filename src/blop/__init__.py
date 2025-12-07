import logging

from . import utils  # noqa F401
from .agent import Agent  # noqa F401
from .dofs import DOF as DOF
from .objectives import Objective  # noqa F401

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("blop")
