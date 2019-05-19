"""Optimizer of slots allocation."""
from itertools import product

from numpy import prod, floor, ceil, stack
from scipy.optimize import minimize


def optimize_slots_config(available_slots, tasks_bounds,
                          validation_bounds, fit_bounds):
    """
    Return the best slots configuration the the given bounds.

    We want to maximize |T*V*F - S| such that:
    - T is Tasks, integer and bounded,
    - V is Validation, integer and bounded,
    - F is Fit, integer and bounded,
    - S is the available number of slots.

    This might be solved by integer programming, though I could not
    find an algorithm that deals with the product of variables being
    optimized.

    Current implementation works in a two-step solution:
    1. Sequential Least SQuares Programming optimization (SLQSP)
        Find approximate best solution as a continuous variables.
    2. Brute force
        Given the best continuous solution, brute force over one-plus
        and one-minus of ceil and floor values of each variable.

    This clearly is not a good solution. We should replace it by a
    MINLP solver.

    Parameters
    ----------
    available_slots : int
        A positive integer with the maximum number of slots to occupy.
    tasks_bounds, validation_bounds, fit_bounds : tuple
        A tuple with minimum and maximum positive integers defining
        the constraints of the optimization problem.

    Returns
    -------
    dict
        A dictionary with the best configuration.

    Raises
    ------
    RuntimeError
        When no configuration could be found to satisfy the bounds.
    """

    def _evaluate_task_config(x):
        return abs(prod(x) - available_slots)

    # Optimize as a continuous probl
    opt_res = minimize(
        fun=_evaluate_task_config,
        x0=(tasks_bounds[0], validation_bounds[1], fit_bounds[1]),
        method='SLSQP',
        bounds=(tasks_bounds, validation_bounds, fit_bounds))

    xfloor, xceil = floor(opt_res.x), ceil(opt_res.x)
    borders = stack([xfloor - 1, xfloor, xceil, xceil + 1])

    def in_bounds(x, bounds):
        """Return whether x is in inclusive bounds."""
        return bounds[0] <= x <= bounds[1]

    best_p, best = None, 0
    for task, cv, fit in product(*borders.transpose().tolist()):
        total = task * cv * fit
        if not in_bounds(task, tasks_bounds):
            continue
        if not in_bounds(cv, validation_bounds):
            continue
        if not in_bounds(fit, fit_bounds):
            continue
        if total == 0 or total < best or available_slots < total:
            continue
        best_p = dict(
            tasks=int(task),
            njobs_on_validation=int(cv),
            njobs_on_estimator=int(fit))
        best = total

    if not best_p:
        raise RuntimeError('Could not solve optimization problem')
    return best_p
