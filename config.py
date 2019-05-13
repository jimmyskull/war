import war


STRATEGIES = [
    war.strategies.gb.RandomSearchGradientBoosting(),
    war.strategies.lda.LDA(),
    war.strategies.lda.PCALDA(),
    war.strategies.linear.RandomSearchLogisticRegressionL1(),
    war.strategies.linear.RandomSearchLogisticRegressionL2(),
    war.strategies.mlp.RandomSearchPCAMLP(),
    war.strategies.nb.NaiveBayes(),
    war.strategies.nb.PCANaiveBayes(),
    war.strategies.rf.RandomSearchRandomForest(),
    war.strategies.svm.RandomSearchPCASVMLinear(),
]
