import graphAttack as ga
import numpy as np
import scipy.optimize
"""Control script"""


def run():
    """Run the model"""
    N, T, D, H1, H2 = 2, 3, 4, 5, 4

    trainData = np.linspace(- 0.1, 0.3, num=N * T * D).reshape(N, T, D)
    trainLabels = np.random.random((N, T, D))

    mainGraph = ga.Graph(False)
    xop = mainGraph.addOperation(ga.Variable(trainData), feederOperation=True)

    hactivations0, cStates0 = ga.addInitialLSTMLayer(mainGraph,
                                                     inputOperation=xop,
                                                     nHidden=H1)
    hactivations1, cStates1 = ga.appendLSTMLayer(mainGraph,
                                                 previousActivations=hactivations0,
                                                 nHidden=H2)

    # hactivations0 = ga.addInitialRNNLayer(mainGraph,
    #                                       inputOperation=xop,
    #                                       activation=ga.TanhActivation,
    #                                       nHidden=H1)

    # hactivations1 = ga.appendRNNLayer(mainGraph,
    #                                  previousActivations=hactivations0,
    #                                  activation=ga.TanhActivation,
    #                                  nHidden=H2)

    finalCost, costOperationsList = ga.addRNNCost(mainGraph,
                                                  hactivations1,
                                                  costActivation=ga.SoftmaxActivation,
                                                  costOperation=ga.CrossEntropyCostSoftmax,
                                                  nHidden=H2,
                                                  labelsShape=xop.shape,
                                                  labels=None)

    def f(p, costOperationsList=costOperationsList, mainGraph=mainGraph):
        data = trainData
        labels = trainLabels
        mainGraph.feederOperation.assignData(data)
        mainGraph.resetAll()
        for index, cop in enumerate(costOperationsList):
            cop.assignLabels(labels[:, index, :])
        mainGraph.attachParameters(p)
        c = mainGraph.feedForward()
        return c

    hactivations = [hactivations0, hactivations1]
    cStates = [cStates0, cStates1]

    def fprime(p, data, labels, costOperationsList=costOperationsList, mainGraph=mainGraph):
        mainGraph.feederOperation.assignData(data)
        mainGraph.resetAll()
        for index, cop in enumerate(costOperationsList):
            cop.assignLabels(labels[:, index, :])
        mainGraph.attachParameters(p)
        c = mainGraph.feedForward()
        mainGraph.feedBackward()
        g = mainGraph.unrollGradients()

        nLayers = len(hactivations)
        for i in range(nLayers):
            hactivations[i][0].assignData(hactivations[i][-1].getValue())
            cStates[i][0].assignData(cStates[i][-1].getValue())

        return c, g

    params = mainGraph.unrollGradientParameters()
    numGrad = scipy.optimize.approx_fprime(params, f, 1e-8)
    analCostGraph, analGradientGraph = fprime(params, trainData, trainLabels)
    return numGrad, analGradientGraph, analCostGraph, mainGraph


if (__name__ == "__main__"):
    nGrad, aGrad, aCost, mainGraph = run()
    params = mainGraph.unrollGradientParameters()
    print(mainGraph)

    print("\n%-16.16s %-16.16s" % ("Grad difference", "Total Gradient"))
    print("%-16.8e %-16.8e" % (np.sum(np.abs(aGrad - nGrad)), np.sum(np.abs(aGrad))))
