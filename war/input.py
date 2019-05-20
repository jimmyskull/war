"""Read a byte from the standard input without waiting for a newline."""
# pylint: disable=C0103
import logging
import select
import sys
import termios
import tty
import time


def getch():
    """
    Get a single character from standard input.

    Does not echo to the screen.
    """
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        char = None
        for _ in range(100):
            if select.select([sys.stdin], [], [], 0) != ([sys.stdin], [], []):
                time.sleep(0.01)
                continue
            char = sys.stdin.read(1)
            if char:
                break
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


def input_from_list(elements, list_name=''):
    """
    Read an integer from the input giving a list of values.

    Parameters
    ----------
    elements : list of str

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
        if list_name:
            print(list_name + ':')
        for idx, elem in enumerate(elements, 1):
            print(f'{idx:3d} - {elem}')
        max_value = len(elements)
        message = 'Select an element from the list above ({}-{}): '.format(
            1, max_value)
        value = int(input(message))
        if value < 1 or max_value < value:
            raise ValueError(
                'value %d is not in {%d, ..., %d}' % (value, 1, max_value))
        return value
    finally:
        logger.setLevel(level)
