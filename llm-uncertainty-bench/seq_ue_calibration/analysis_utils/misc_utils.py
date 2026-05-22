"""General-purpose utility functions: timestamped printing and string parsing."""

import re
from datetime import datetime

_number_pattern = re.compile(r"\d+([,\.]\d+)*")


def print_with_time(*args, **kwargs):
    """Print a message prefixed with the current wall-clock time (HH:MM:SS).

    Drop-in replacement for ``print``; all positional arguments are joined
    with spaces and all keyword arguments are forwarded to ``print``.

    Args:
        *args: Objects to print (converted via ``str``).
        **kwargs: Keyword arguments forwarded to the built-in ``print``.
    """
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"[{current_time}] " + " ".join(map(str, args)), **kwargs)


def extract_number(s):
    """Extract the first number (int or float) from a string.

    Recognises comma-as-thousands-separator (``1,234``) and decimal dots
    (``3.14``).  Commas are stripped before conversion.

    Args:
        s: Input string to search.

    Returns:
        int | float | None: The parsed number, or ``None`` if no number was found.
    """
    match = _number_pattern.search(s)
    if match:
        num_str = match.group(0).replace(",", "")
        try:
            return float(num_str) if "." in num_str else int(num_str)
        except ValueError:
            pass
    return None
