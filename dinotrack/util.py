def multi_getattr(objs: list[object], key: str):
    """
    Get the attribute of an object by chaining the attributes.

    Args:
        objs (list[object]): The list of objects to get the attribute from.
        key (str): The attribute to chain.

    Returns:
        Any: The attribute of the object.
    """

    for obj in objs:
        attr = getattr(obj, key)
        if attr is not None:
            break
    return attr
