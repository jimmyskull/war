from datetime import datetime
from math import sqrt
import logging
import time
import warnings

from IPython import display
import matplotlib.pyplot as plt
from numpy import array
import numpy as np
import pylab as pl
import seaborn

import war

import config


def make_plot(strategy):
    seaborn.set_style('whitegrid')
    fig, ax = plt.subplots(1, figsize=(10, 5))
    #ax.set_ylim(0.3, 0.75)
    first_date = None
    best_scores = list()
    # Get best scores and first task date
    best_scores.append(strategy.cache['best']['agg']['avg'])
    for _, result in strategy.database.iterate():
        if result['type'] != 'result':
            continue
        result = result['data']
        if result['status'] != 'OK':
            continue
        date = datetime.strptime(result['begin_time'], '%Y-%m-%d %H:%M:%S')
        if first_date is None or first_date > date:
            first_date = date
    best_rank = np.argsort(-array(best_scores))
    best_global = max(best_scores)
    best_strat_stid = np.argmax(best_scores)
    # Make plots
    min_score, max_score = None, None
    for stid, strat in enumerate(config.STRATEGIES):
        history = list()
        for (oid, result) in strategy.database.iterate():
            if result['type'] != 'result':
                continue
            result = result['data']
            if result['status'] != 'OK':
                continue
            date = datetime.strptime(result['begin_time'], '%Y-%m-%d %H:%M:%S')
            avg = result['agg']['avg']
            ci = result['agg']['std'] * 2 / sqrt(len(result['scores']))
            history.append((date, avg, ci))

        history = sorted(history, key=lambda x: x[0])

        for idx in range(1, len(history)):
            if history[idx][1] < history[idx - 1][1]:
                history[idx] = (
                    history[idx][0],
                    history[idx - 1][1],
                    history[idx - 1][2])

        if stid == best_strat_stid and history:
            min_score = history[0][1]
            max_score = history[0][1]
            for item in history:
                min_score = min(min_score, item[1] - item[2])
                max_score = max(max_score, item[1] + item[2])

        date = array([(item[0] - first_date).total_seconds() / 60
                      for item in history])
        avg = array([item[1] for item in history])
        ci = array([item[2] for item in history])

        color = COLORS[stid % len(COLORS)]
        if best_strat_stid == stid:
            dot_alpha, fill_alpha = 1, 0.1
        else:
            dot_alpha, fill_alpha = 0.1, 0.01
        ax.plot(date, avg, lw=2, label=strategy.name, color=color,
                alpha=dot_alpha)
        ax.scatter(date, avg, s=7, color=color, alpha=dot_alpha)
        ax.fill_between(date, avg + ci, avg - ci, alpha=fill_alpha,
                        color=color)
    legend = ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.5),
                       ncol=3, fancybox=True, shadow=True)
    plt.title('War database history')
    ax.set_ylabel('Gini')
    ax.set_xlabel('Minutes since first task')
    if min_score is not None:
        ax.set_ylim(max(0, min_score), min(1, max_score))
    return (fig, legend)


def save_plot():
    fig, legend = make_plot(config.STRATEGIES[6])
    fig.savefig('by_complexity.png', quality=100, dpi=100,
                bbox_extra_artists=(legend,),
                bbox_inches='tight')


if __name__ == '__main__':
    save_plot()
