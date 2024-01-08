def default_digestion_function(db, uid):
    products = db[uid].table(fill=True)
    return products
