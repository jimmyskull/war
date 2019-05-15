import logging

import coloredlogs

import war


coloredlogs.install(
    level=logging.INFO,
    fmt='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

STRATEGIES = [
    war.strategies.gb.RandomSearchGradientBoosting(),
    war.strategies.lda.LDA(),
    war.strategies.lda.PCALDA(),
    war.strategies.linear.RandomSearchLogisticRegressionL1(),
    war.strategies.linear.RandomSearchLogisticRegressionL2(),
    war.strategies.nb.NaiveBayes(),
    war.strategies.nb.PCANaiveBayes(),
    war.strategies.rf.RandomSearchRandomForest(),
    war.strategies.svm.RandomSearchPCASVMLinear(),
    # war.strategies.keras.RandomSearchKerasMLP(),
    war.strategies.lgb.RandomSearchLGBM(),
]
