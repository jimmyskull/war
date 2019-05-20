"""Terminal dashboard control"""
# pylint: disable=C0111
from collections import OrderedDict
import logging
import pprint
import sys
import time

from pygments import highlight
from pygments.formatters import TerminalFormatter  # pylint: disable=E0611
from pygments.lexers import PythonLexer            # pylint: disable=E0611

from war.cformat import ColorFormat as CF
from war.input import getch, input_int, input_float, input_from_list
from war.status import StatusTable
from war.table import (Table, Cell, ASCII_BOX_DRAWING, NO_BOX_DRAWING,
                       UNICODE_BOX_DRAWING, BAR_BOX_DRAWING)


class Dashboard:
    """Dashboard is the terminal UI Controller."""

    def __init__(self, engine, scheduler):
        self.logger = logging.getLogger('war.dashboard')
        self.table_box = UNICODE_BOX_DRAWING
        self.engine = engine
        self.scheduler = scheduler
        self.handlers = OrderedDict(
            s=(self.status, 'Show the status table.'),
            S=(self.sort_status, 'Set sorting column of the status table.'),
            b=(self.show_strategy, 'Show strategy info and the best candidate.'),
            T=(self.set_table_theme, 'Set theme for tables.'),
            e=(self.show_error, 'Show last task error information.'),
            C=(self.toggle_cooperate, 'Toggle cooperation mode.'),
            c=(self.cooperate, 'Run the cooperation procedure.'),
            L=(self.toggle_log_level, 'Toggle logging level.'),
            w=(self.set_weight, 'Set weight of a strategy.'),
            m=(self.set_max_slots, 'Set maximum slots.'),
            u=(self.resource_usage, 'Show resource usage.'),
            z=(self.show_counters, 'Show engine internal counters.'),
            H=(self.status_help, 'Show help for the status table.'),
            h=(self.help, 'Show help information.'),
            q=(self.quit, 'Quit.'),
        )
        self.handlers['\x03'] = (self.quit, None)
        self._status = StatusTable(engine, scheduler)

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
        if self.scheduler.cooperation_mode:
            self.scheduler.cooperation_mode = False
            self.logger.info(CF('Cooperation has been disabled.').cyan.bold)
        else:
            self.scheduler.cooperation_mode = True
            msg = (str(CF('Cooperation has been enabled:').cyan.bold)
                   + str(CF(' the current number of slots is %d.').cyan))
            self.logger.info(msg, self.scheduler.max_slots)

    def cooperate(self):
        self.scheduler.cooperate(force=True)

    def status(self):
        self._status.report(self.table_box)

    def sort_status(self):
        self._status.set_sort_status()

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
        table = Table(self.table_box)
        table.set_header(['Command', 'Description'])
        for command, (_, description) in self.handlers.items():
            if description:
                table.add_row([Cell(command, attr=['center']), description])
        for line in table.format():
            self.logger.info(line)

    def quit(self):
        # pylint: disable=R0201
        raise StopIteration()

    def show_error(self):
        error = self.scheduler.last_error
        if not error:
            self.logger.info('No error occurred during this session.')
            return
        message = error.error_info['message']
        traceback_msg = error.error_info['traceback']
        print('\n')
        print(str(CF('Traceback:').bold))
        print(traceback_msg)
        print(str(CF('Message:').bold))
        print('\t' + message)

    def show_strategy(self):
        strategies = self.engine.strategies
        names = [strat.name for strat in strategies]
        try:
            stid = input_from_list(names, 'Strategies')
        except ValueError:
            pass
        else:
            pp = pprint.PrettyPrinter()
            strategy = strategies[stid - 1]
            print(CF(strategy.name).bold)
            code = pp.pformat(self.scheduler.strategies[strategy])
            fmt = highlight(code, PythonLexer(), TerminalFormatter())
            print(fmt)

    def show_counters(self):
        sched = self.scheduler
        table = Table(self.table_box)
        table.set_header(['Counter', 'Current Value'])
        table.add_row(['Scheduler thread CPU usage',
                       '%.f%%' % sched.proc.cpu_percent()])
        table.add_row(['CPU count', str(sched.cpu_count)])
        table.add_row(['Workers', str(sched.nconsumers)])
        table.add_row(['Maximum slots for validation',
                       str(sched.max_slots_per_evaluation)])
        table.add_row(['Maximum slots', str(sched.max_slots)])
        table.add_row(['Running slots', str(sched.slots_running)])
        table.add_row(['Tasks ended in this session',
                       str(sched.tasks_finished)])
        table.add_row(['Cooperation mode', str(sched.cooperation_mode)])
        table.add_row([
            'Cooperation resting',
            '%ds, since %s' % (
                time.time() - sched.last_coop_time,
                time.strftime('%c', time.localtime(sched.last_coop_time))),
        ])
        for line in table.format():
            self.logger.info(line)

    def set_weight(self):
        strategies = self.engine.strategies
        names = [strat.name for strat in strategies]
        try:
            stid = input_from_list(names, 'Strategies')
        except ValueError:
            pass
        else:
            st_ob = self.scheduler.strategy_by_id(stid)
            msg = 'Set a weight (current={:.4f}): '.format(st_ob.weight)
            weight = input_float(msg)
            self.scheduler.set_weight(stid, weight)

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

    def set_table_theme(self):
        themes = {
            'ASCII box': ASCII_BOX_DRAWING,
            'Bar box': BAR_BOX_DRAWING,
            'Unicode box': UNICODE_BOX_DRAWING,
            'No box': NO_BOX_DRAWING,
        }
        try:
            value = input_from_list(themes.keys(), 'Themes for status table')
        except ValueError:
            self.logger.info('Theme has not been changed.')
        else:
            name, self.table_box = list(themes.items())[value - 1]
            self.logger.info(f'Theme has been changed to {name!r}.')

    def status_help(self):
        self._status.help(self.table_box)
