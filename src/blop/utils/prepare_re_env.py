import argparse
import datetime
import json  # noqa F401
from bluesky.callbacks.zmq import Publisher
import bluesky.plan_stubs as bps  # noqa F401
import bluesky.plans as bp  # noqa F401
from tiled.client import from_uri, from_profile ##
import time as ttime
from bluesky.callbacks.tiled_writer import TiledWriter
from nslsii import configure_base
import matplotlib.pyplot as plt
import numpy as np  # noqa F401
from bluesky.callbacks import best_effort
from bluesky.run_engine import RunEngine
from ophyd.utils import make_dir_tree
from tiled.client.utils import handle_error
from tiled.utils import safe_json_dump

from blop.sim import HDF5Handler

DEFAULT_DB_TYPE = "local"
DEFAULT_ROOT_DIR = "/tmp/sirepo-bluesky-data"
DEFAULT_ENV_TYPE = "stepper"
DEFAULT_USE_SIREPO = False

SERVER_HOST_LOCATION = "http://localhost:8000"


class TiledInserter:
    def __init__(self, client):
        self.client = client  # Store the Tiled client for later use    
    def insert(self, name, doc):
        ATTEMPTS = 20
        error = None
        for attempt in range(ATTEMPTS):
            try:
                self.tiled_post_document(name, doc)
                return  # Success
            except Exception as exc:
                print(f"Attempt {attempt + 1} failed: {repr(exc)}")
                error = exc
                ttime.sleep(2)
        raise error  # If all attempts fail  
      
    def tiled_post_document(self, name, doc):
        """Handles the actual document posting to the Tiled server."""
        link = self.client.item["links"]["self"].replace("/metadata", "/documents", 1)
        response = self.client.context.http_client.post(
            link, content=safe_json_dump({"name": name, "doc": doc})
        )
        self.handle_error(response)    

    def handle_error(self, response):
        """Handle potential errors from the Tiled server."""
        if response.status_code != 200:
            raise RuntimeError(f"Tiled server error: {response.status_code} {response.text}")
    
def re_env(db_type="default", root_dir="/default/path"):
    # Initialize RunEngine
    RE = RunEngine({})    # BestEffortCallback for live visualization
    bec = best_effort.BestEffortCallback()
    RE.subscribe(bec)    # Setup Tiled Client

    tiled_client = from_uri(SERVER_HOST_LOCATION)["local"]["raw"]
    tiled_client.login()  # Authenticate if needed    # Instantiate TiledInserter with the client
    tiled_inserter = TiledInserter()    # Subscribe RunEngine to TiledInserter
    RE.subscribe(tiled_inserter.insert)    # Ensure directory structure exists

    _ = make_dir_tree(datetime.datetime.now().year, base_path=root_dir)    
    return dict(RE=RE, db=tiled_client, bec=bec)
    

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
