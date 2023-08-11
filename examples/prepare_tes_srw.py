db.reg.register_handler("srw", SRWFileHandler, overwrite=True)
db.reg.register_handler("SIREPO_FLYER", SRWFileHandler, overwrite=True)

plt.ion()

root_dir = "/tmp/sirepo-bluesky-data"
_ = make_dir_tree(datetime.datetime.now().year, base_path=root_dir)

connection = SirepoBluesky("http://localhost:8000")

data, schema = connection.auth("srw", "00000002")
classes, objects = create_classes(connection=connection)
globals().update(**objects)

# w9.duration.kind = "hinted"

bec.disable_baseline()
bec.disable_heading()
bec.disable_table()

# This should be done by installing the package with `pip install -e .` or something similar.
# import sys
# sys.path.insert(0, "../")

kb_dofs = [kbv.grazingAngle, kbv.verticalOffset, kbh.grazingAngle, kbh.horizontalOffset]
kb_bounds = np.array([[3.5, 3.7], [-0.10, +0.10], [3.5, 3.7], [-0.10, +0.10]])
