from typing import Mapping


def lazyconfig_to_dict(obj):
    """
    Recursively convert a LazyConfig object (which acts like nested
    namespaces/dicts) into a pure Python dict.
    """
    if isinstance(obj, Mapping):
        return {k: lazyconfig_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [lazyconfig_to_dict(v) for v in obj]
    else:
        return obj
