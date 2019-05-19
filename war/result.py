"""Store a task's result."""


class Result:
    """
    Store a task's result, that is copied between processes.

    This class provide a `data()` method to provide the python object
    to be written to the database.  I have chose to store a dict
    instead of a result to consume fewer bytes in the database file.
    """

    def __init__(self, task, begin_time, elapsed_time, total_time, status,
                 error_info, agg, scores, scoring, jobs):
        self.task = task
        self.begin_time = begin_time
        self.elapsed_time = elapsed_time
        self.total_time = total_time
        self.status = status
        self.error_info = error_info
        self.agg = agg
        self.scores = scores
        self.scoring = scoring
        self.jobs = jobs

    def __repr__(self):
        name = self.task.estimator.__class__.__name__
        if self.status == 'OK':
            info = f'mean={self.agg["mean"]:.4f}'
        else:
            info = self.error_info['message']
        return f'<Result name={name} {info}>'

    def data(self):
        """
        Return the object suitable to store into the database.

        Not all parameters are in o good format. For example, the params
        value may be a ConfigSpace sample, which would increase quite
        a lot the size of the object.

        Returns
        -------
        dict
            Dictionary with task's result information.
        """
        data = {
            'status': self.status,
            'begin_time': self.begin_time,
            'elapsed_time': self.elapsed_time,
            'total_time': self.total_time,
            'error_info': self.error_info,
            'agg': self.agg,
            'scores': self.scores,
            'scoring': self.scoring,
            'params': dict(**self.task.params),
        }
        return data
