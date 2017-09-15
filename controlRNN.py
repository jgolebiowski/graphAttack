import graphAttack as ga
import numpy as np
import pickle
"""Control script"""


N, T, D, H = 2, 3, 4, 5

x = np.linspace(- 0.1, 0.3, num=N * T * D).reshape(N, T, D)
testLabels = np.random.random((N, T, D))

mainGraph = ga.Graph(False)
xop = mainGraph.addOperation(ga.Variable(x), feederOperation=True)


hactivations0 = ga.addInitialRNNLayer(mainGraph,
                                      inputOperation=xop,
                                      activation=ga.TanhActivation,
                                      nHidden=5)

hactivations = ga.appendRNNLayer(mainGraph,
                                 previousActivations=hactivations0,
                                 activation=ga.TanhActivation,
                                 nHidden=5)

finalCost, costOperationsList = ga.addRNNCost(mainGraph,
                                              hactivations,
                                              costActivation=ga.SoftmaxActivation,
                                              costOperation=ga.CrossEntropyCostSoftmax,
                                              nHidden=5,
                                              labelsShape=xop.shape,
                                              labels=None)


def f(p, costOperationsList=costOperationsList, mainGraph=mainGraph):
    data = x
    labels = testLabels
    mainGraph.feederOperation.assignData(data)
    mainGraph.resetAll()
    for index, cop in enumerate(costOperationsList):
        cop.assignLabels(labels[:, index, :])
    mainGraph.attachParameters(p)
    c = mainGraph.feedForward()
    return c


def fprime(p, data, labels, costOperationsList=costOperationsList, mainGraph=mainGraph):
    mainGraph.feederOperation.assignData(data)
    mainGraph.resetAll()
    for index, cop in enumerate(costOperationsList):
        cop.assignLabels(labels[:, index, :])
    mainGraph.attachParameters(p)
    c = mainGraph.feedForward()
    mainGraph.feedBackward()
    g = mainGraph.unrollGradients()
    return c, g


def sampleCharacter(previousX, previousH,
                    hactivations=hactivations,
                    costOperationsList=costOperationsList,
                    mainGraph=mainGraph):
    N, T, D = mainGraph.feederOperation.shape
    preiousData = np.zeros((1, T, D))
    preiousData[:, 0, :] = previousX

    mainGraph.resetAll()
    mainGraph.feederOperation.assignData(preiousData)
    hactivations[0].assignData(previousH)

    newH = hactivations[1].getValue()
    # newX = costOperationsList[0].makePredictions()
    newX = costOperationsList[0].inputA.getValue()

    return newX, newH


import scipy.optimize
params = mainGraph.unrollGradientParameters()

numGrad = scipy.optimize.approx_fprime(params, f, 1e-8)
analCostGraph, analGradientGraph = fprime(params, x, testLabels)
print(analGradientGraph, analCostGraph)
print(analGradientGraph - numGrad)
print(np.sum(np.abs(analGradientGraph - numGrad)), np.sum(np.abs(analGradientGraph)))
