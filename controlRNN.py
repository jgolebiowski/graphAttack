import graphAttack as ga
import numpy as np
import pickle
"""Control script"""


N, T, D, H = 2, 3, 4, 5

x = np.linspace(- 0.1, 0.3, num=N * T * D).reshape(N, T, D)
testLabels = np.random.random((N, T, D))

mainGraph = ga.Graph(False)
xop = mainGraph.addOperation(ga.Variable(x), feederOperation=True)

hactivations0, cStates0 = ga.addInitialLSTMLayer(mainGraph,
                                                 inputOperation=xop,
                                                 nHidden=5)
hactivations1, cStates1 = ga.appendLSTMLayer(mainGraph,
                                             previousActivations=hactivations0,
                                             nHidden=4)


# hactivations0 = ga.addInitialRNNLayer(mainGraph,
#                                       inputOperation=xop,
#                                       activation=ga.TanhActivation,
#                                       nHidden=5)

# hactivations1 = ga.appendRNNLayer(mainGraph,
#                                  previousActivations=hactivations0,
#                                  activation=ga.TanhActivation,
#                                  nHidden=5)

finalCost, costOperationsList = ga.addRNNCost(mainGraph,
                                              hactivations1,
                                              costActivation=ga.SoftmaxActivation,
                                              costOperation=ga.CrossEntropyCostSoftmax,
                                              nHidden=4,
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


import scipy.optimize
params = mainGraph.unrollGradientParameters()

numGrad = scipy.optimize.approx_fprime(params, f, 1e-3)
analCostGraph, analGradientGraph = fprime(params, x, testLabels)
# print(analGradientGraph, analCostGraph)
# print(analGradientGraph - numGrad)
print(analCostGraph)
print(np.sum(np.abs(analGradientGraph - numGrad)), np.sum(np.abs(analGradientGraph)))
