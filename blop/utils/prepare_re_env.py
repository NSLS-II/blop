import argparse
import datetime
import json  # noqa F401

import bluesky.plan_stubs as bps  # noqa F401
import bluesky.plans as bp  # noqa F401
import databroker
import matplotlib.pyplot as plt
import numpy as np  # noqa F401
from bluesky.callbacks import best_effort
from bluesky.run_engine import RunEngine
from databroker import Broker
from ophyd.utils import make_dir_tree

from blop.sim import HDF5Handler

DEFAULT_DB_TYPE = "local"
DEFAULT_ROOT_DIR = "/tmp/sirepo-bluesky-data"
DEFAULT_ENV_TYPE = "stepper"
DEFAULT_USE_SIREPO = False


def re_env(db_type=DEFAULT_DB_TYPE, root_dir=DEFAULT_ROOT_DIR):
    RE = RunEngine({})
    bec = best_effort.BestEffortCallback()
    RE.subscribe(bec)

    db = Broker.named(db_type)
    db.reg.register_handler("HDF5", HDF5Handler, overwrite=True)
    try:
        databroker.assets.utils.install_sentinels(db.reg.config, version=1)
    except Exception:
        pass
    RE.subscribe(db.insert)

    _ = make_dir_tree(datetime.datetime.now().year, base_path=root_dir)

    return dict(RE=RE, db=db, bec=bec)


def register_handlers(db, handlers):
    for handler_spec, handler_class in handlers.items():
        db.reg.register_handler(handler_spec, handler_class, overwrite=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare bluesky environment")
    parser.add_argument(
        "-d",
        "--db-type",
        dest="db_type",
        default=DEFAULT_DB_TYPE,
        help="Type of databroker ('local', 'temp', etc.)",
    )
    parser.add_argument(
        "-r",
        "--root-dir",
        dest="root_dir",
        default=DEFAULT_ROOT_DIR,
        help="The root dir to create YYYY/MM/DD dir structure.",
    )

    parser.add_argument(
        "-s",
        "--use-sirepo",
        dest="use_sirepo",
        default=DEFAULT_USE_SIREPO,
        help="The root dir to create YYYY/MM/DD dir structure.",
    )

    env_choices = ["stepper", "flyer"]
    parser.add_argument(
        "-e",
        "--env-type",
        dest="env_type",
        choices=env_choices,
        default=DEFAULT_ENV_TYPE,
        help="Type of RE environment.",
    )

    parser.add_argument(
        "-f",
        "--file",
        dest="file",
        default="",
    )

    args = parser.parse_args()
    kwargs_re = dict(db_type=args.db_type, root_dir=args.root_dir)
    ret = re_env(**kwargs_re)
    globals().update(**ret)

    if args.use_sirepo:
        from sirepo_bluesky.srw_handler import SRWFileHandler

        if args.env_type == "stepper":
            from sirepo_bluesky.shadow_handler import ShadowFileHandler

            handlers = {"srw": SRWFileHandler, "SIREPO_FLYER": SRWFileHandler, "shadow": ShadowFileHandler}
            plt.ion()
        elif args.env_type == "flyer":
            from sirepo_bluesky.madx_handler import MADXFileHandler

            handlers = {"srw": SRWFileHandler, "SIREPO_FLYER": SRWFileHandler, "madx": MADXFileHandler}
            bec.disable_plots()  # noqa: F821
        else:
            raise RuntimeError(f"Unknown environment type: {args.env_type}.\nAvailable environment types: {env_choices}")

        register_handlers(db, handlers)  # noqa: F821
