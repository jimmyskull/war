import logging
import multiprocessing
import psutil
import time

from sklearn.metrics import get_scorer

from war.cformat import ColorFormat
from war.dashboard import Dashboard
from war.scheduler import Scheduler
from war.worker import Worker


class Engine:

    def __init__(self):
        self.strategies = list()
        self.features = None
        self.target = None
        self.cv = 3
        self.slots = -1
        self.cooperate = False
        self.proc = psutil.Process()
        self.proc.cpu_percent()

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
        self.scoring = get_scorer(scoring)

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

        if self.cooperate:
            logger.info(ColorFormat('Working in cooperative mode').cyan)

        dashboard = Dashboard(self, sched)

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

            try:
                dashboard.update()
            except StopIteration:
                keep_running = False

            while not results.empty():
                result = results.get()
                sched.collect(result)

            if sched.improved_since_last_report:
                sched.report_results()

        for worker in consumers:
            worker.terminate()
        logger.info('Bye.')
