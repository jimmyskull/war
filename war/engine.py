import multiprocessing
import logging
import time

from war.cformat import ColorFormat
from war.input import getch
from war.scheduler import Scheduler


def input_int(message, bounds=None):
    logger = logging.getLogger()
    level = logger.level
    logger.setLevel(logging.CRITICAL)
    try:
        value = int(input(message))
        if bounds:
            if bounds[0] > value or bounds[1] < value:
                raise ValueError('value %d is not in {%d, ..., %d}' % (value, *bounds))
        return value
    finally:
        logger.setLevel(level)


def input_float(message, bounds=None):
    logger = logging.getLogger()
    level = logger.level
    logger.setLevel(logging.CRITICAL)
    try:
        value = float(input(message))
        if bounds:
            if bounds[0] > value or bounds[1] < value:
                raise ValueError('value %f is not in [%.1f, %.1f]' % (value, *bounds))
        return value
    finally:
        logger.setLevel(level)


class Worker(multiprocessing.Process):

    def __init__(self, task_queue, result_queue):
        super().__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        try:
            while True:
                next_task = self.task_queue.get()
                try:
                    if next_task:
                        result = next_task()
                        self.result_queue.put(result)
                finally:
                    self.task_queue.task_done()
        except KeyboardInterrupt:
            logger = logging.getLogger('war.engine')
            logger.debug('%s stopped execution.', self.name)


class Engine:

    def __init__(self):
        self.strategies = list()
        self.features = None
        self.target = None
        self.cv = 3
        self.slots = -1
        self.cooperate = False

    def add(self, strategies):
        """Add strategies to compete."""
        assert isinstance(strategies, list)
        self.strategies += strategies

    def set_data(self, features, target):
        """Classification dataset."""
        self.features = features
        self.target = target

    def set_cv(self, cv, scoring='roc_auc'):
        """CV configuration."""
        self.cv = cv
        self.scoring = scoring

    def set_slots(self, slots, cooperate=False):
        """Slots should be up to the number of CPU cores."""
        assert slots > 1
        self.slots = slots
        self.cooperate = cooperate

    def start(self):
        """Start engine work in blocking mode."""
        try:
            self._start()
        except KeyboardInterrupt:
            print(end='\r')
            logger = logging.getLogger('war.engine')
            logger.info('Bye.')

    def _start(self):
        logger = logging.getLogger('war.engine')
        num_consumers = multiprocessing.cpu_count()
        if self.slots < 0:
            num_consumers -= (self.slots + 1)
        else:
            if self.slots > num_consumers:
                logger.warning('Using %d slots of %d cpu cores.',
                               self.slots, num_consumers)
            num_consumers = self.slots
        assert num_consumers > 0

        # Establish communication queues
        tasks = multiprocessing.JoinableQueue()
        results = multiprocessing.Queue()

        # Start consumers
        logger.info('Creating %d consumers', num_consumers)

        consumers = [Worker(tasks, results) for i in range(num_consumers)]
        for worker in consumers:
            worker.start()

        nfolds = self.cv
        sched = Scheduler(self.strategies, num_consumers, nfolds,
                          self.cooperate)

        # Some strategy may want to define its search space based on
        # the data information, such as the number of features.
        for strategy in self.strategies:
            if not hasattr(strategy, 'init'):
                continue
            strategy.init(info={
                'features': self.features,
                'target': self.target,
                'cv': self.cv,
            })

        sched.report_results()
        import psutil
        proc = psutil.Process()
        proc.cpu_percent()

        if self.cooperate:
            logger.info(ColorFormat('Working in cooperative mode').cyan)

        keep_running = True
        while keep_running:
            # Get new tasks
            new_tasks = sched.next()
            # Add tasks to the queue
            for task in new_tasks:
                task.features = self.features
                task.target = self.target
                task.cv = nfolds
                task.scoring = self.scoring
                tasks.put(task)
            for _ in range(100):
                if sched.available_slots() > num_consumers // 2:
                    break
                char = getch()
                if char:
                    import sys
                    sys.stderr.write('\r')
                    sys.stderr.flush()
                if char == 'r':
                    sched.report_results()
                elif char == 't':
                    sched.toggle_cooperate()
                elif char == 'c':
                    sched.cooperate(force=True)
                elif char == 'p':
                    sched.report_counters()
                elif char == 'd':
                    logging.getLogger().setLevel(logging.DEBUG)
                    logger.info('Changed log level to debug')
                elif char == 'i':
                    logging.getLogger().setLevel(logging.INFO)
                    logger.info('Changed log level to info')
                elif char in ['q', '\x03']:
                    keep_running = False
                    break
                elif char == 'e':
                    sched.report_last_error()
                elif char == 'o':
                    bounds = (1, len(self.strategies))
                    msg = 'Select a strategy by ID (1-{}): '.format(bounds[1])
                    try:
                        st_id = input_int(msg, bounds=bounds)
                        sched.report_best(st_id)
                    except ValueError as err:
                        logger.error('Could not get strategy: %s', err)
                elif char == 'a':
                    bounds = (1, len(self.strategies))
                    msg = 'Select a strategy by ID (1-{}): '.format(bounds[1])
                    st_id = -1
                    try:
                        st_id = input_int(msg, bounds=bounds)
                    except ValueError as err:
                        logger.error('Could not get strategy: %s', err)
                    if st_id > 0:
                        st_ob = sched.strategy_by_id(st_id)
                        msg = 'Set a weight (current={:.4f}): '.format(st_ob.weight)
                        weight = input_float(msg)
                        sched.set_weight(st_id, weight)
                elif char == 'm':
                    bounds = (2, num_consumers)
                    msg = ('Select maximum number of slots '
                           '(current={}, min={}, max={}): ').format(
                           sched.max_slots, bounds[0], bounds[1])
                    try:
                        new_max_slots = input_int(msg, bounds=bounds)
                        sched.set_max_slots(new_max_slots)
                    except ValueError as err:
                        logger.error('Could not change max. slots: %s', err)
                elif char == 'w':
                    sched.report_worker_usage()
                elif char == 'u':
                    logger.info(
                        ColorFormat('Main thread CPU usage: %s%%').cyan,
                        proc.cpu_percent())
                elif char == 'h':
                    logger.info(ColorFormat('Commands:').bold)
                    logger.info('   r    Report scheduler information.')
                    logger.info('   o    Show scheduler information of a strategy.')
                    logger.info('   p    Show scheduler counters.')
                    logger.info('   m    Set maximum slots.')
                    logger.info('   t    Toggle cooperation mode.')
                    logger.info(
                               ('   c    Force execution of cooperation procedure '
                                        '(can run outside of cooperation mode).'))
                    logger.info('   e    Show last task error information.')
                    logger.info('   a    Set weight of a strategy.')
                    logger.info('   u    Report CPU usage of main thread (engine + UI + scheduler).')
                    logger.info('   w    Report CPU usage of worker processes.')
                    logger.info('   d    Set debugging log level.')
                    logger.info('   i    Set information log level.')
                    logger.info('   h    Show help.')
                    logger.info('   q    Quit.')
                elif char == '\r':
                    pass
                elif char is not None:
                    logger.warning('Command not recognized: %s', repr(char))
                if results.empty():
                    continue
                while not results.empty():
                    result = results.get()
                    sched.collect(result)

            if sched.improved_since_last_report:
                sched.report_results()

        for worker in consumers:
            worker.terminate()
        logger.info('Bye.')
