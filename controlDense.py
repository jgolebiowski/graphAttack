import graphAttack as ga
import numpy as np
"""Control script"""


sizes = np.array([3, 4, 2])

Xcheck2 = np.arange(1, 7).reshape((2, 3)).astype(np.float)
Ycheck2 = np.array([[1, 0], [0, 1]]).astype(np.float)

mainGraph = ga.Graph()

ffeed = mainGraph.addOperation(ga.Variable(Xcheck2), doGradient=False, feederOperation=True)
feedDrop = mainGraph.addOperation(ga.DropoutOperation(ffeed, 0.0), doGradient=False, finalOperation=False)

l1 = ga.addDenseLayer(mainGraph, 4,
                      inputOperation=feedDrop,
                      activation=ga.ReLUActivation,
                      dropoutRate=0.0,
                      w=None,
                      b=None)
l2 = ga.addDenseLayer(mainGraph, 2,
                      inputOperation=l1,
                      activation=ga.SoftmaxActivation,
                      dropoutRate=0.0,
                      w=None,
                      b=None)
fcost = mainGraph.addOperation(
    ga.CrossEntropyCostSoftmax(l2, Ycheck2),
    doGradient=False,
    finalOperation=True)


print(mainGraph)
import scipy.optimize
params = mainGraph.unrollGradientParameters()


def f(x):
    mainGraph.attachParameters(x)
    return mainGraph.getValue()


def fprime(p, data, labels):
    mainGraph.feederOperation.assignData(data)
    mainGraph.costOperation.assignLabels(labels)
    mainGraph.attachParameters(p)
    mainGraph.resetAll()
    c = mainGraph.feedForward()
    mainGraph.feedBackward()
    g = mainGraph.unrollGradients()
    return c, g


numGrad = scipy.optimize.approx_fprime(params, f, 1e-6)
analCostGraph, analGradientGraph = fprime(params, Xcheck2, Ycheck2)
print(numGrad)
print(analGradientGraph - numGrad, analCostGraph)
