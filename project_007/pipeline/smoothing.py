"""
PROJECT 007 — EMA Smoothing Filter
Applies Exponential Moving Average filtering to noisy raw features.
"""

from typing import Any
import numpy as np


class EMASmoother:
    """
    Maintains Exponential Moving Averages for arbitrary scalar and vector streams.
    """
    def __init__(self, alpha: float = 0.3):
        self._alpha = alpha
        self._states = {}

    def update(self, key: Any, current_val: Any) -> Any:
        """
        Update the EMA for `key` with `current_val`.
        Handles floats and numeric tuples/lists.
        """
        if key not in self._states:
            self._states[key] = current_val
            return current_val

        prev_val = self._states[key]

        if isinstance(current_val, (float, int)):
            new_val = self._alpha * current_val + (1.0 - self._alpha) * prev_val
            self._states[key] = float(new_val)
            return float(new_val)

        if isinstance(current_val, (tuple, list, np.ndarray)):
            # Assuming numpy-compatible types
            curr_arr = np.array(current_val, dtype=float)
            prev_arr = np.array(prev_val, dtype=float)
            new_arr = self._alpha * curr_arr + (1.0 - self._alpha) * prev_arr
            self._states[key] = new_arr
            # Return same type as input if tuple
            if isinstance(current_val, tuple):
                return tuple(float(x) for x in new_arr)
            elif isinstance(current_val, list):
                return [float(x) for x in new_arr]
            return new_arr

        return current_val

    def get(self, key: Any, default: Any = None) -> Any:
        return self._states.get(key, default)

    def cleanup(self, active_keys: set) -> None:
        """Remove state for keys no longer active."""
        keys_to_delete = [k for k in self._states if k not in active_keys]
        for k in keys_to_delete:
            del self._states[k]
