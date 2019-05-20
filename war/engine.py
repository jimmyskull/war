"""War engine."""
import logging
import multiprocessing

from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_val_score
import psutil

from war import metrics
from war.cformat import ColorFormat as CF
from war.dashboard import Dashboard
from war.scheduler import Scheduler
from war.worker import Worker


class Engine:
    """Main controller of the War engine."""

    def __init__(self):
        self.strategies = list()
        self.features = None
        self.target = None
        self.trials = 3
        self.validator = cross_val_score
        self.scoring = None
        self.slots = -1
        self.cooperate = True
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

    def set_validation(self, trials=3, scoring='roc_auc',
                       validator=cross_val_score):
        """
        Validator configuration.

        Each task is defined by a validator and an estimator.
        The validator train the estimator `trials` times to get a
        good estimate of the score.

        Parameters
        ----------
        trials : int
            The number of score result from trials that the validator
            generates.
        scoring : str or callable
            The name or a scoring callable created using
            `sklearn.metrices.make_scorer`. Currently, it must be
            a the-greater-the-better scorer.
        validator : callable
            A function like `sklearn.model_selection.cross_val_score`.
        """
        assert isinstance(trials, int), 'trials must be an int'
        self.trials = trials
        if scoring == 'gini':
            self.scoring = metrics.gini
        else:
            self.scoring = get_scorer(scoring)
        self.validator = validator

    def set_slots(self, slots, cooperate=True):
        """
        Set the number of processing slots to manage.

        This sets up the maximum number of active processing slots
        to use.  The engine may create up to 2*slots threads/processes,
        but will work hard to keep `slots` of these threads/processes
        active, running tasks.

        Parameters
        ----------
        slots : int
            The number of slots to manage.  This should be up to the
            number of CPU cores.
        """
        assert slots > 1, 'at least 2 slots are necessary'
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

        self.num_consumers = num_consumers

        # Establish communication queues
        tasks = multiprocessing.JoinableQueue()
        results = multiprocessing.Queue()

        # Start consumers
        logger.info('Creating %d consumers', num_consumers)

        consumers = [Worker(tasks, results) for i in range(num_consumers)]
        for worker in consumers:
            worker.start()

        sched = Scheduler(self.strategies, num_consumers, self.trials,
                          self.cooperate)

        # Some strategy may want to define its search space based on
        # the data information, such as the number of features.
        for strategy in self.strategies:
            if not hasattr(strategy, 'init'):
                continue
            strategy.init(info={
                'features': self.features,
                'target': self.target,
                'trials': self.trials,
            })

        if self.cooperate:
            logger.info(CF('Working in cooperative mode').cyan)

        dashboard = Dashboard(self, sched)

        keep_running = True
        while keep_running:
            # Get new tasks
            new_tasks = sched.next()
            # Add tasks to the queue
            for task in new_tasks:
                task.features = self.features
                task.target = self.target
                task.trials = self.trials
                task.scoring = self.scoring
                task.validator = self.validator
                tasks.put(task)
            # Update UI
            try:
                dashboard.update()
            except StopIteration:
                keep_running = False
            # Fetch results
            while not results.empty():
                result = results.get()
                logger.debug(CF('New result %s').dark_gray,
                             result.task.full_id())
                sched.collect(result)

        for worker in consumers:
            worker.terminate()
        logger.info('Bye.')
