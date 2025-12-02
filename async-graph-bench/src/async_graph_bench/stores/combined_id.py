def get_combined_id(item):
    id = item["id"]
    iter = item.get("iter", 0)
    return (iter, *id) if isinstance(id, tuple) else (iter, id)

def get_combined_id_from_parts(id, iter=0):
    return (iter, *id) if isinstance(id, tuple) else (iter, id)