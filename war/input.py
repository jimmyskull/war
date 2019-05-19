"""Read a byte from the standard input without waiting for a newline."""
# pylint: disable=C0103
import logging
import select
import sys
import termios
import tty


def getch():
    """
    Get a single character from standard input.

    Does not echo to the screen.
    """
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        if select.select([sys.stdin], [], [], 0) != ([sys.stdin], [], []):
            return None
        char = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return char


def input_int(message, bounds=None):
    """
    Read an integer from the input.

    Parameters
    ----------
    message : str
        A message to print before reading the value.
    bounds : tuple, optional
        Optional inclusive lower and upper bounds.

    Returns
    -------
    int
        The read value.

    Raises
    ------
    ValueError
        If the input is not an integer.
    """
    logger = logging.getLogger('war.input')
    level = logger.level
    logger.setLevel(logging.CRITICAL)
    try:
        value = int(input(message))
        if bounds:
            if bounds[0] > value or bounds[1] < value:
                raise ValueError(
                    'value %d is not in {%d, ..., %d}' % (value, *bounds))
        return value
    finally:
        logger.setLevel(level)


def input_float(message, bounds=None):
    """
    Read a float from the input.

    Parameters
    ----------
    message : str
        A message to print before reading the value.
    bounds : tuple, optional
        Optional inclusive lower and upper bounds.

    Returns
    -------
    float
        The read value.

    Raises
    ------
    ValueError
        If the input is not a float.
    """
    logger = logging.getLogger('war.input')
    level = logger.level
    logger.setLevel(logging.CRITICAL)
    try:
        value = float(input(message))
        if bounds:
            if bounds[0] > value or bounds[1] < value:
                raise ValueError(
                    'value %f is not in [%.1f, %.1f]' % (value, *bounds))
        return value
    finally:
        logger.setLevel(level)
