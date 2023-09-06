def default_digestion_function(db, uid):
    products = db[uid].table(fill=True)
    print(products)
    return products
