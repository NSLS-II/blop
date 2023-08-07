db.reg.register_handler("srw", SRWFileHandler, overwrite=True)
db.reg.register_handler("shadow", ShadowFileHandler, overwrite=True)
db.reg.register_handler("SIREPO_FLYER", SRWFileHandler, overwrite=True)

plt.ion()

root_dir = "/tmp/sirepo-bluesky-data"
_ = make_dir_tree(datetime.datetime.now().year, base_path=root_dir)

connection = SirepoBluesky("http://localhost:8000")

data, schema = connection.auth("shadow", "I1Flcbdw")
classes, objects = create_classes(connection=connection)
globals().update(**objects)

bec.disable_baseline()
bec.disable_heading()
bec.disable_table()
