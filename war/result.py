"""Store a task's result."""


class Result:
    """
    Store a task's result, that is copied between processes.

    This class provide a `data()` method to provide the python object
    to be written to the database.  I have chose to store a dict
    instead of a result to consume fewer bytes in the database file.
    """

    def __init__(self, task, begin_time, real_times, cpu_times,
                 status, error_info, agg, scores, njobs):
        self.task = task
        self.begin_time = begin_time
        self.real_times = real_times
        self.cpu_times = cpu_times
        self.status = status
        self.error_info = error_info
        self.agg = agg
        self.scores = scores
        self.njobs = njobs
        self.slots = njobs['valid'] * njobs['fit']

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
            'real_times': self.real_times,
            'cpu_times': self.cpu_times,
            'error_info': self.error_info,
            'agg': self.agg,
            'scores': self.scores,
            'scoring': self.task.scoring_name(),
            'validator': self.task.validator_name(),
            'params': dict(**self.task.params),
            'data': self.task.data_id,
            'njobs': self.njobs,
        }
        return data
