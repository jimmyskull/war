"""Formatters."""


def sec2time(sec, n_msec=3):
    """Convert seconds to 'Dd, HH:MM:SS.FFF'."""
    # pylint: disable=C0103
    #if hasattr(sec, '__len__'):
    #    return [sec2time(s) for s in sec]
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    if n_msec > 0:
        pattern = '%%02d:%%02d:%%0%d.%df' % (n_msec + 3, n_msec)
    else:
        pattern = r'%02d:%02d:%02d'
    if d == 0:
        return pattern % (h, m, s)
    return ('%dd, ' + pattern) % (d, h, m, s)


def format_count(x):
    return f'{x:,d}'


def format_probability(x):
    return f'{x:.0%}'


def format_score(x):
    return f'{x:.4f}'


def format_weight(x):
    return f'{x:.2f}'
