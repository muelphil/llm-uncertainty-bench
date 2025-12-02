import os
from contextlib import contextmanager
from typing import Dict, Any

@contextmanager
def temporary_env(env: Dict[str, Any]):
    """
    Temporarily set environment variables within a context.

    Replaces the specified environment variables for the duration of the
    context block and restores their previous values afterward.

    Args:
        env: Mapping of environment variable names to temporary values.

    Example:
        >>> with temporary_env({"MODE": "test"}):
        ...     run_tests()
        # Environment restored after context exit.
    """
    old_env = {}
    try:
        for key, value in env.items():
            if key in os.environ:
                old_env[key] = os.environ[key]
            os.environ[key] = str(value)  # env must be str
        yield
    finally:
        # Restore previous values
        for key in env:
            if key in old_env:
                os.environ[key] = old_env[key]
            else:
                del os.environ[key]