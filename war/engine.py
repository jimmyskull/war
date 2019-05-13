import multiprocessing
import logging
import time

from war.scheduler import Scheduler


def gently_stop(run):
    tr


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

        while True:

            # Get new tasks
            new_tasks = sched.next()
            # Add tasks to the queue
            for task in new_tasks:
                task.features = self.features
                task.target = self.target
                task.cv = nfolds
                tasks.put(task)

            for _ in range(10):

                # Collect available results
                if not results.empty():
                    # logger.debug('Collecting results')
                    # Start printing results
                    while not results.empty():
                        result = results.get()
                        # logger.debug('Result: %s', result)
                        sched.collect(result)
                else:
                    time.sleep(1)

                if sched.available_slots() > num_consumers // 2:
                    break

            if sched.improved_since_last_report:
                sched.report_results()
