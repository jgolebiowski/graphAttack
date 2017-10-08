import graphAttack as ga
import numpy as np
"""Control script"""

N, D, H1, H2 = 10, 3, 4, 2

Xcheck2 = np.arange(0, N * D).reshape(N, D).astype(np.float)
Ycheck2 = np.arange(0, N * H2).reshape(N, H2).astype(np.float)
mainGraph = ga.Graph()

ffeed = mainGraph.addOperation(ga.Variable(Xcheck2), doGradient=False, feederOperation=True)
feedDrop = mainGraph.addOperation(ga.DropoutOperation(ffeed, 0.0), doGradient=False, finalOperation=False)

l1 = ga.addDenseLayer(mainGraph, H1,
                      inputOperation=feedDrop,
                      activation=ga.ReLUActivation,
                      dropoutRate=0.0,
                      batchNormalisation=True,
                      w=None,
                      b=None)
l2 = ga.addDenseLayer(mainGraph, H2,
                      inputOperation=l1,
                      activation=ga.SoftmaxActivation,
                      dropoutRate=0.0,
                      batchNormalisation=True,
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
    mainGraph.resetAll()
    mainGraph.finalOperation.assignLabels(labels)
    mainGraph.attachParameters(p)
    c = mainGraph.feedForward()
    mainGraph.feedBackward()
    g = mainGraph.unrollGradients()
    return c, g


numGrad = scipy.optimize.approx_fprime(params, f, 1e-4)
analCostGraph, analGradientGraph = fprime(params, Xcheck2, Ycheck2)
print("analGrad     numGrad     diff")
for index in range(len(params)):
    print("%12.5e %12.5e %12.5e" %
          (analGradientGraph[index], numGrad[index], analGradientGraph[index] - numGrad[index]))
print(np.sum(np.abs(analGradientGraph - numGrad)), np.sum(np.abs(analGradientGraph)))
