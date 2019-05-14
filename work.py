import logging
import warnings

from pandas import read_csv
import coloredlogs

import war

import config


def main():
    # pylint: disable=C0103
    coloredlogs.install(
        level=logging.DEBUG,
        #fmt='%(asctime)s %(levelname)s %(module)s %(message)s',
        fmt='%(asctime)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    train = read_csv('/home/jimmy/dev/chef/GiveMeSomeCredit/cs-training.csv',
                     index_col=0)
    # test = read_csv('/home/jimmy/dev/chef/GiveMeSomeCredit/cs-test.csv',
    #                 index_col=0)
    X, y = train.drop('SeriousDlqin2yrs', axis=1), train['SeriousDlqin2yrs']


    engine = war.Engine()
    engine.set_data(X, y)
    engine.set_cv(3) # roc_auc by default
    engine.set_slots(-1)
    engine.add(config.STRATEGIES)
    engine.start()


if __name__ == '__main__':
    main()
