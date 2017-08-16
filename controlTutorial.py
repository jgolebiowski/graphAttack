import graphAttack as ga
import numpy as np
"""Control script"""

Xcheck = np.arange(7).reshape((1, 7))
Ycheck = np.array([92]).astype(np.float)

mainGraph = ga.Graph()

feed = mainGraph.addOperation(ga.Variable(Xcheck), doGradient=False, feederOperation=True)
w = np.arange(7).reshape((1, 7)).T
b = np.array([2])

wop = mainGraph.addOperation(ga.Variable(w), doGradient=True)
mmop = mainGraph.addOperation(ga.MatMatmulOperation(feed, wop))

bop = mainGraph.addOperation(ga.Variable(b), doGradient=True)
addop = mainGraph.addOperation(ga.AddOperation(mmop, bop))

costop = mainGraph.addOperation(ga.QuadratiCcostOperation(addop, Ycheck), finalOperation=True)

print(mainGraph)
import scipy.optimize
params = mainGraph.unrollGradientParameters()


def fprime(p, data, labels):
    mainGraph.feederOperation.assignData(data)
    mainGraph.costOperation.assignLabels(labels)
    mainGraph.attachParameters(p)
    mainGraph.resetAll()
    c = mainGraph.feedForward()
    mainGraph.feedBackward()
    g = mainGraph.unrollGradients()
    return c, g


def f(p):
    data = Xcheck
    labels = Ycheck
    mainGraph.feederOperation.assignData(data)
    mainGraph.costOperation.assignLabels(labels)
    mainGraph.attachParameters(p)
    mainGraph.resetAll()
    c = mainGraph.feedForward()
    return c


numGrad = scipy.optimize.approx_fprime(params, f, 1e-6)
analCostGraph, analGradientGraph = fprime(params, Xcheck, Ycheck)
print(numGrad)
print(analGradientGraph - numGrad, analCostGraph)
