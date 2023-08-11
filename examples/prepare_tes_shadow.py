import sirepo_bluesky
from ophyd.utils import make_dir_tree
from sirepo_bluesky.shadow_handler import ShadowFileHandler
from sirepo_bluesky.sirepo_bluesky import SirepoBluesky
from sirepo_bluesky.sirepo_ophyd import create_classes
from sirepo_bluesky.srw_handler import SRWFileHandler

db.reg.register_handler("shadow", ShadowFileHandler, overwrite=True)
db.reg.register_handler("SIREPO_FLYER", SRWFileHandler, overwrite=True)

plt.ion()

root_dir = "/tmp/sirepo-bluesky-data"
_ = make_dir_tree(datetime.datetime.now().year, base_path=root_dir)

connection = SirepoBluesky("http://localhost:8000")

data, schema = connection.auth("shadow", "00000002")
classes, objects = create_classes(connection=connection)
globals().update(**objects)

data["models"]["simulation"]["npoint"] = 100000
data["models"]["watchpointReport12"]["histogramBins"] = 32
# w9.duration.kind = "hinted"

bec.disable_baseline()
bec.disable_heading()
# bec.disable_table()

import warnings

warnings.filterwarnings("ignore", module="sirepo_bluesky")
