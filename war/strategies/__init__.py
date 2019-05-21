import warnings

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')

    from . import (
        ada,
        bagging,
        catboost,
        extra,
        gb,
        gp,
        keras,
        knn,
        lda,
        lgb,
        linear,
        nb,
        perceptron,
        qda,
        rf,
        svm,
        tree,
        xgb,
    )
