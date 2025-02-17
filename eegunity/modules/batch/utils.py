import numpy as np


def __convert_to_serializable__(value):
    if hasattr(value, 'isoformat'):
        return value.isoformat()
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, bytes):
        return value.decode('utf-8')
    return value