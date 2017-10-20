import graphAttack as ga
import numpy as np
import scipy.optimize

"""Control script"""


def run():
    """Run the model"""
    N, D, H1, H2 = 10, 3, 4, 2

    trainData = np.arange(0, N * D).reshape(N, D).astype(np.float)
    trainLabels = np.arange(0, N * H2).reshape(N, H2).astype(np.float)
    mainGraph = ga.Graph()

    ffeed = mainGraph.addOperation(ga.Variable(trainData), doGradient=False, feederOperation=True)
    feedDrop = mainGraph.addOperation(ga.DropoutOperation(ffeed, 0.0), doGradient=False, finalOperation=False)

    l1 = ga.addDenseLayer(mainGraph, H1,
                          inputOperation=feedDrop,
                          activation=ga.ReLUActivation,
                          dropoutRate=0.0,
                          batchNormalisation=True)
    l2 = ga.addDenseLayer(mainGraph, H2,
                          inputOperation=l1,
                          activation=ga.SoftmaxActivation,
                          dropoutRate=0.0,
                          batchNormalisation=False)
    fcost = mainGraph.addOperation(
        ga.CrossEntropyCostSoftmax(l2, trainLabels),
        doGradient=False,
        finalOperation=True)

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

    params = mainGraph.unrollGradientParameters()
    numGrad = scipy.optimize.approx_fprime(params, f, 1e-8)
    analCostGraph, analGradientGraph = fprime(params, trainData, trainLabels)
    return numGrad, analGradientGraph, analCostGraph, mainGraph


if (__name__ == "__main__"):
    nGrad, aGrad, aCost, mainGraph = run()
    params = mainGraph.unrollGradientParameters()
    print(mainGraph)

    print("analGrad     numGrad     diff")
    for index in range(len(params)):
        print("%12.5e %12.5e %12.5e" %
              (aGrad[index], nGrad[index], aGrad[index] - nGrad[index]))
    print("\n%-16.16s %-16.16s" % ("Grad difference", "Total Gradient"))
    print("%-16.8e %-16.8e" % (np.sum(np.abs(aGrad - nGrad)), np.sum(np.abs(aGrad))))
