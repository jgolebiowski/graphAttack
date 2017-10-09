import graphAttack as ga
import numpy as np
import pickle
"""Control script"""

Xt = np.random.random((5, 1, 10, 10))
Yt = np.random.random((5, 10))

# ------ conv2D operation testing
mainGraph = ga.Graph()
feed = mainGraph.addOperation(ga.Variable(Xt), doGradient=False, feederOperation=True)

cnn1 = ga.addConv2dLayer(mainGraph,
                         inputOperation=feed,
                         nFilters=3,
                         filterHeigth=5,
                         filterWidth=5,
                         padding="SAME",
                         convStride=1,
                         activation=ga.ReLUActivation,
                         batchNormalisation=True,
                         pooling=ga.MaxPoolOperation,
                         poolHeight=2,
                         poolWidth=2,
                         poolStride=2)

flattenOp = mainGraph.addOperation(ga.FlattenFeaturesOperation(cnn1))
flattenDrop = mainGraph.addOperation(ga.DropoutOperation(
    flattenOp, 0.0), doGradient=False, finalOperation=False)

l1 = ga.addDenseLayer(mainGraph, 20,
                      inputOperation=flattenDrop,
                      activation=ga.ReLUActivation,
                      dropoutRate=0.0,
                      batchNormalisation=False)
l2 = ga.addDenseLayer(mainGraph, 10,
                      inputOperation=l1,
                      activation=ga.SoftmaxActivation,
                      dropoutRate=0.0,
                      batchNormalisation=False)
fcost = mainGraph.addOperation(
    ga.CrossEntropyCostSoftmax(l2, Yt),
    doGradient=False,
    finalOperation=True)


def fprime(p, data, labels):
    mainGraph.feederOperation.assignData(data)
    mainGraph.resetAll()
    mainGraph.finalOperation.assignLabels(labels)
    mainGraph.attachParameters(p)
    c = mainGraph.feedForward()
    mainGraph.feedBackward()
    g = mainGraph.unrollGradients()
    return c, g


def f(p):
    data = Xt
    labels = Yt
    mainGraph.feederOperation.assignData(data)
    mainGraph.resetAll()
    mainGraph.finalOperation.assignLabels(labels)
    mainGraph.attachParameters(p)
    c = mainGraph.feedForward()
    return c


print(mainGraph)
import scipy.optimize
params = mainGraph.unrollGradientParameters()


numGrad = scipy.optimize.approx_fprime(params, f, 1e-6)
analCostGraph, analGradientGraph = fprime(params, Xt, Yt)
print(sum(abs(numGrad)))
print(sum(abs(analGradientGraph - numGrad)), analCostGraph)

print(analGradientGraph - numGrad)
