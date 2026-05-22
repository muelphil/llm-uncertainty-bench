"""
JSON serialisation helpers for numpy types.

``NumpyEncoder`` extends ``json.JSONEncoder`` so that numpy scalars, arrays,
and booleans are automatically converted to plain Python types when
serialising analysis results to JSON files.

``numpy_decoder`` is the complementary object_hook for ``json.loads`` / 
``json.load`` that reconstructs lists back to numpy arrays.
"""

import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that transparently handles common numpy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)


def numpy_decoder(dct):
    """
    JSON object_hook that converts every list value back to a numpy array.

    Nested dicts are handled recursively.
    """
    for key, value in dct.items():
        if isinstance(value, list):
            dct[key] = np.array(value)
        elif isinstance(value, dict):
            dct[key] = numpy_decoder(value)
    return dct
