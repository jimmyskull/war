
class Result:

    def __init__(self, task, begin_time, elapsed_time, total_time, status,
                 error_info, agg, scores, jobs):
        self.task = task
        self.begin_time = begin_time
        self.elapsed_time = elapsed_time
        self.total_time = total_time
        self.status = status
        self.error_info = error_info
        self.agg = agg
        self.scores = scores
        self.jobs = jobs

    def __repr__(self):
        name = self.task.estimator.__class__.__name__
        if self.status == 'OK':
            info = f'mean={self.agg["mean"]:.4f}'
        else:
            info = self.error_info['message']
        return f'<Result name={name} {info}>'

    def data(self):
        data = {
            'status': self.status,
            'begin_time': self.begin_time,
            'elapsed_time': self.elapsed_time,
            'total_time': self.total_time,
            'error_info': self.error_info,
            'agg': self.agg,
            'scores': self.scores,
            'params': dict(**self.task.params),
        }
        return data
