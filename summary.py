# pylint: disable=C0103
import pprint

import war

import config


pp = pprint.PrettyPrinter()

for strategy in config.STRATEGIES:
    strategy.load_cache()
    print('\n\033[1m{}\033[0m'.format(strategy.name))
    pp.pprint(strategy.cache)
