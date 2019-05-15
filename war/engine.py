import multiprocessing
import logging
import time

from war.cformat import ColorFormat
from war.input import getch
from war.scheduler import Scheduler


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

    def add(self, strategies):
        """Add strategies to compete."""
        assert isinstance(strategies, list)
        self.strategies += strategies

    def set_data(self, features, target):
        """Classification dataset."""
        self.features = features
        self.target = target

    def set_cv(self, cv):
        """CV configuration."""
        self.cv = cv

    def set_slots(self, slots):
        """Slots should be up to the number of CPU cores."""
        assert slots > 1
        self.slots = slots

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
        sched = Scheduler(self.strategies, num_consumers, nfolds)

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

        keep_running = True
        while keep_running:
            # Get new tasks
            new_tasks = sched.next()
            # Add tasks to the queue
            for task in new_tasks:
                task.features = self.features
                task.target = self.target
                task.cv = nfolds
                tasks.put(task)
            # Collect results for some time.
            import select
            import sys
            for _ in range(60):
                if sched.available_slots() > num_consumers // 2:
                    break
                # while sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                char = getch()
                if char == 'r':
                    sched.report_results()
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
                elif char in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
                    sched.report_best(int(char))
                elif char == 'h':
                    logger.info(ColorFormat('Commands:').bold)
                    logger.info('1-9 Best results for a strategy.')
                    logger.info(' d  Set debugging log level.')
                    logger.info(' e  Show last error information')
                    logger.info(' h  Show help.')
                    logger.info(' i  Set information log level.')
                    logger.info(' q  Quit.')
                    logger.info(' r  Report scheduler information.')
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
