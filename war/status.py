"""Manager the renderization of the status table."""
import logging

import numpy

from war.format import (format_count, format_probability, format_score,
                        format_weight, sec2time)
from war.input import input_from_list
from war.table import Table, Cell


class StatusTable:
    """Status table of strategies."""

    def __init__(self, engine, scheduler):
        self.logger = logging.getLogger('war.status')
        self.engine = engine
        self.scheduler = scheduler
        self.sort_column = 7
        self.sort_order = 'descending'
        self._header = ['ID', 'Name', 'Total Time', 'T', 'S', 'TSLI', 'Ended',
                        'Best', '95% CI', 'Min', 'Max', 'Prob', 'Weight']

    def _get_header(self):
        header = self._header[:]
        symbol = '↓' if self.sort_order == 'asceding' else '↑'
        header[self.sort_column] = symbol + header[self.sort_column]
        return header

    def _table_data(self):
        strategies = self.scheduler.strategies
        scores = list()
        times = list()
        for info in strategies.values():
            scores.append(info['best']['agg']['avg'])
            times.append(info['cumulative_time'])
        total_time = sum(times) + 1e-4
        probs = self.scheduler.probabilities()
        rows = list()
        for idx, (strat, info) in enumerate(strategies.items(), 1):
            agg = info['best']['agg']
            to_ci = 2.0 / (numpy.sqrt(len(info['best']['scores'])) + 1e-4)
            rows.append([
                idx, strat.name, info['cumulative_time'], info['running'],
                info['slots'], info['tasks_since_last_improvement'],
                info['finished'], agg['avg'],
                agg['std'] * to_ci, agg['min'], agg['max'],
                probs[idx - 1], strat.weight
            ])
        rows = sorted(rows, key=lambda cols: cols[self.sort_column],
                      reverse=self.sort_order == 'descending')
        best_score = max(scores)
        return (best_score, total_time, rows)

    def report(self, table_style):
        """Report the status table."""
        table = Table(table_style)
        table.set_header(self._get_header())
        best_score, total_time, rows = self._table_data()
        for row in rows:
            formatted_row = [
                str(row[0]),
                Cell(row[1], attr=['ljust']),
                Cell(('{} ({:5.1%})'.format(sec2time(row[2], 0),
                                            row[2] / total_time)),
                     attr=['rjust']),
                format_count(row[3]),   # Tasks Running
                format_count(row[4]),   # Allocated Slots
                format_count(row[5]),   # Tasks since last improvement
                format_count(row[6]),   # Ended tasks
                format_score(row[7]),   # Best Average
                format_score(row[8]),   # Best 95% CI
                format_score(row[9]),   # Best Min
                format_score(row[10]),  # Best Max
                format_probability(row[11]),
                format_weight(row[12]),
            ]
            attr = ['bold', 'blue'] if row[6] == best_score else None
            table.add_row(formatted_row, attr)
        for table_row in table.format():
            self.logger.info(table_row)
        # sched.report_counters()

    def set_sort_status(self):
        sorting_order = ['asceding', 'descending']
        try:
            value = input_from_list(self._header, 'Columns to sort')
            order = input_from_list(sorting_order, 'Sorting order')
        except ValueError:
            self.logger.info('Sorting column has not been changed.')
        else:
            self.sort_column = value - 1
            self.sort_order = sorting_order[order - 1]
            self.report()
