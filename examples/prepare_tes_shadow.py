db.reg.register_handler("srw", SRWFileHandler, overwrite=True)
db.reg.register_handler("shadow", ShadowFileHandler, overwrite=True)
db.reg.register_handler("SIREPO_FLYER", SRWFileHandler, overwrite=True)

plt.ion()

root_dir = "/tmp/sirepo-bluesky-data"
_ = make_dir_tree(datetime.datetime.now().year, base_path=root_dir)

connection = SirepoBluesky("http://localhost:8001")

data, schema = connection.auth("shadow", "00000002")
classes, objects = create_classes(connection.data, connection=connection)
globals().update(**objects)

data["models"]["simulation"]["npoint"] = 100000
data["models"]["watchpointReport12"]["histogramBins"] = 32
# w9.duration.kind = "hinted"

bec.disable_baseline()
bec.disable_heading()
bec.disable_table()

# This should be done by installing the package with `pip install -e .` or something similar.
# import sys
# sys.path.insert(0, "../")

mi = np.array([0, 1])

dofs = [[kbv.x_rot, kbv.offz, kbh.x_rot, kbh.offz][i] for i in mi]  # noqa F821

hard_bounds = np.array([[-0.20, +0.20], [-1.00, +1.00], [-0.20, +0.20], [-1.00, +1.00]])[mi] * 5e-1

# for dof in dofs:
#    dof.kind = "hinted"
