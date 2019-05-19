import logging

import coloredlogs

import war


coloredlogs.install(
    level=logging.DEBUG,
    fmt='%(asctime)s %(message)s',
    datefmt='%H:%M:%S'
)
logging.getLogger().setLevel(logging.INFO)

STRATEGIES = [
    war.strategies.ada.RandomSearchAdaBoost(),
    war.strategies.gb.RandomSearchGradientBoosting(),
    war.strategies.keras.RandomSearchKerasMLP(),
    war.strategies.keras.RandomSearchKerasPCAMLP(),
    war.strategies.lda.LDA(),
    war.strategies.lda.PCALDA(),
    war.strategies.lgb.RandomSearchLGBM(),
    war.strategies.linear.RandomSearchLogisticRegressionL1(),
    war.strategies.linear.RandomSearchLogisticRegressionL2(),
    war.strategies.nb.NaiveBayes(),
    war.strategies.nb.PCANaiveBayes(),
    war.strategies.qda.PCAQDA(),
    war.strategies.qda.QDA(),
    war.strategies.rf.RandomSearchRandomForest(),
    war.strategies.svm.RandomSearchPCASVMLinear(),
    war.strategies.xgb.RandomSearchXGB(),
]
