# pylint: disable=C0103
import pprint

from pygments import highlight
from pygments.formatters import TerminalFormatter
from pygments.lexers import PythonLexer

import war

import config


pp = pprint.PrettyPrinter()

for strategy in config.STRATEGIES:
    strategy.load_cache()
    print('\n\033[1m{}\033[0m'.format(strategy.name))
    code = pp.pformat(strategy.cache)
    print(highlight(code, PythonLexer(), TerminalFormatter()))
