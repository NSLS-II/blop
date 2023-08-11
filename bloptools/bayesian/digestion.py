def default_digestion_function(db, uid):
    return db[uid].table(fill=True)
