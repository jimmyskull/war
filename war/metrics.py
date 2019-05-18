from sklearn import metrics


def gini_score(expected, predicted):
    auc = metrics.roc_auc_score(expected, predicted[:, 1])
    return (auc - 0.5) * 2


gini = metrics.make_scorer(gini_score, needs_proba=True)

