# pylint: disable=C0103,C0111
from pandas import read_csv

import war

import config


def main():
    train = read_csv('/home/jimmy/dev/chef/GiveMeSomeCredit/cs-training.csv',
                     index_col=0)
    X, y = train.drop('SeriousDlqin2yrs', axis=1), train['SeriousDlqin2yrs']

    engine = war.Engine()
    engine.set_data(X, y)
    engine.set_cv(3) # roc_auc by default
    engine.set_slots(8)
    engine.add(config.STRATEGIES)
    engine.start()


if __name__ == '__main__':
    main()
