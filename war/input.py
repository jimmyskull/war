"""Read a byte from the standard input without waiting for a newline."""
# pylint: disable=C0103
import select
import sys
import termios
import tty


def getch():
    """
    Gets a single character from standard input.
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
