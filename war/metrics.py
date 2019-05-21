"""Metrics for task evaluation."""
# pylint: disable=C0103
from sklearn import metrics


def gini_score(expected, predicted):
    """Compute the Gini normalized score."""
    auc = metrics.roc_auc_score(expected, predicted)
    return (auc - 0.5) * 2


gini = metrics.make_scorer(gini_score, needs_proba=True)
