# pylint: disable=C0103
from collections import OrderedDict

from pandas import DataFrame, concat
from tqdm import tqdm

import war

import config


rows = list()

for strategy in tqdm(config.STRATEGIES):
    strategy.load_cache()

    for idx, (oid, result) in enumerate(strategy.database.iterate(), 1):
        if result['type'] != 'result':
            continue
        result = result['data']
        if result['status'] != 'OK':
            continue
        agg = result['agg']
        df = DataFrame(OrderedDict(
            name=strategy.name,
            mean=f'{agg["avg"]:.4f}',
            std=f'{agg["std"]:.4f}',
            min=f'{agg["min"]:.4f}',
            max=f'{agg["max"]:.4f}',
            status=result['status'],
            params=str(result['params']),
            oid=f'{strategy.__class__.__name__}/{oid}',
        ), index=[idx])
        rows.append(df)

FNAME = 'results.xlsx'
print('Writing {:,d} results to {!r}.'.format(len(rows), FNAME))
concat(rows).to_excel(FNAME)

