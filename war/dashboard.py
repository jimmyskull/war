"""Terminal dashboard control"""
# pylint: disable=C0111
from collections import OrderedDict
import logging
import sys

from war.cformat import ColorFormat as CF
from war.input import getch, input_int, input_float
from war.table import Table, Cell, NO_BOX_DRAWING


class Dashboard:
    """Dashboard is the terminal UI Controller."""

    def __init__(self, engine, scheduler):
        self.logger = logging.getLogger('war.dashboard')
        self.engine = engine
        self.scheduler = scheduler
        self.handlers = OrderedDict(
            s=(self.status, 'Show the engine status.'),
            e=(self.show_error, 'Show last task error information.'),
            t=(self.toggle_cooperate, 'Toggle cooperation mode.'),
            c=(self.cooperate, 'Force execution of cooperation procedure.'),
            l=(self.toggle_log_level, 'Toggle logging level.'),
            p=(self.show_strategy, 'Show strategy information.'),
            w=(self.set_weight, 'Set weight of a strategy.'),
            m=(self.set_max_slots, 'Set maximum slots.'),
            u=(self.resource_usage, 'Show resource usage.'),
            h=(self.help, 'Show help information.'),
            q=(self.quit, 'Quit.'),
        )
        self.handlers['\x03'] = (self.quit, None)

    def update(self):
        char = getch()
        if not char:
            return
        sys.stderr.write('\r')
        sys.stderr.flush()
        if char in self.handlers:
            handler, _ = self.handlers[char]
            handler()
        else:
            self.logger.warning('Command not recognized: %s', repr(char))

    def toggle_cooperate(self):
        self.scheduler.toggle_cooperate()

    def cooperate(self):
        self.scheduler.cooperate(force=True)

    def status(self):
        self.scheduler.report_results()
        # sched.report_counters()

    def toggle_log_level(self):
        # Toggle global logging between info and debug.
        # pylint: disable=R0201
        logger = logging.getLogger()
        if logger.level == logging.DEBUG:
            logger.setLevel(logging.INFO)
            logger.info('Changed logging level to info')
        elif logger.level == logging.INFO:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.info('Changed logging level to debug')
        else:
            logger.info('Logging level not recognized')

    def help(self):
        table = Table(NO_BOX_DRAWING)
        table.set_header(['Command', 'Description'])
        for command, (_, description) in self.handlers.items():
            if description:
                table.add_row([Cell(command, attr=['center']), description])
        for line in table.format().split('\n'):
            self.logger.info(line)

    def quit(self):
        # pylint: disable=R0201
        raise StopIteration()

    def show_error(self):
        self.scheduler.report_last_error()

    def show_strategy(self):
        bounds = (1, len(self.engine.strategies))
        msg = 'Select a strategy by ID (1-{}): '.format(bounds[1])
        try:
            st_id = input_int(msg, bounds=bounds)
            self.scheduler.report_best(st_id)
        except ValueError as err:
            self.logger.error('Could not get strategy: %s', err)

    def set_weight(self):
        bounds = (1, len(self.engine.strategies))
        msg = 'Select a strategy by ID (1-{}): '.format(bounds[1])
        st_id = -1
        try:
            st_id = input_int(msg, bounds=bounds)
        except ValueError as err:
            self.logger.error('Could not get strategy: %s', err)
        if st_id > 0:
            st_ob = self.scheduler.strategy_by_id(st_id)
            msg = 'Set a weight (current={:.4f}): '.format(st_ob.weight)
            weight = input_float(msg)
            self.scheduler.set_weight(st_id, weight)

    def set_max_slots(self):
        bounds = (2, self.engine.num_consumers)
        msg = (
            'Select maximum number of slots '
            '(current={}, min={}, max={}): ').format(
            self.scheduler.max_slots, bounds[0], bounds[1])
        try:
            new_max_slots = input_int(msg, bounds=bounds)
            self.scheduler.set_max_slots(new_max_slots)
        except ValueError as err:
            self.logger.error('Could not change max. slots: %s', err)

    def resource_usage(self):
        self.scheduler.report_worker_usage()
        self.logger.info(CF('Main thread CPU usage: %s%%').cyan,
                         self.engine.proc.cpu_percent())
