"""Process worker."""
from multiprocessing import Process
import logging


class Worker(Process):
    """Worker is the execution of a process task consumer."""

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
