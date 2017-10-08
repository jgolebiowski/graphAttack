import graphAttack as ga
import numpy as np
import pickle
"""Control script"""

# ------ This is a very limited dataset, load a lrger one for better results
pickleFilename = "dataSet/notMNISTreformatted_small.pkl"
with open(pickleFilename, "rb") as fp:
    allDatasets = pickle.load(fp)

X = allDatasets["trainDataset"]
Y = allDatasets["trainLabels"]

Xtest = allDatasets["testDataset"]
Ytest = allDatasets["testLabels"]

Xvalid = allDatasets["validDataset"]
Yvalid = allDatasets["validLabels"]


simulationIndex = 0
print(("Training with:", simulationIndex))

mainGraph = ga.Graph()
ffeed = mainGraph.addOperation(ga.Variable(X), doGradient=False, feederOperation=True)
feedDrop = mainGraph.addOperation(ga.DropoutOperation(
    ffeed, 0.0), doGradient=False, finalOperation=False)

l1 = ga.addDenseLayer(mainGraph, 100,
                      inputOperation=feedDrop,
                      activation=ga.ReLUActivation,
                      dropoutRate=0.0,
                      batchNormalisation=True,
                      w=None,
                      b=None)
l2 = ga.addDenseLayer(mainGraph, 10,
                      inputOperation=l1,
                      activation=ga.SoftmaxActivation,
                      dropoutRate=0,
                      batchNormalisation=False,
                      w=None,
                      b=None)
fcost = mainGraph.addOperation(
    ga.CrossEntropyCostSoftmax(l2, Y),
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


param0 = mainGraph.unrollGradientParameters()
adamGrad = ga.adaptiveSGD(trainingData=X,
                          trainingLabels=Y,
                          param0=param0,
                          epochs=1e2,
                          miniBatchSize=20,
                          initialLearningRate=1e-3,
                          beta1=0.9,
                          beta2=0.999,
                          epsilon=1e-8,
                          testFrequency=1e2,
                          function=fprime)

pickleFilename = "minimizerParamsDense_" + str(simulationIndex) + ".pkl"

# with open(pickleFilename, "rb") as fp:
#     adamParams = pickle.load(fp)
#     adamGrad.restoreState(adamParams)
#     params = adamParams["params"]

params = adamGrad.minimize(printTrainigCost=True, printUpdateRate=False,
                           dumpParameters=pickleFilename)
mainGraph.attachParameters(params)

print("train: Trained with:", simulationIndex)
print("train: Accuracy on train set:", ga.calculateAccuracy(mainGraph, X, Y))
print("train: Accuracy on cv set:", ga.calculateAccuracy(mainGraph, Xvalid, Yvalid))
print("train: Accuracy on test set:", ga.calculateAccuracy(mainGraph, Xtest, Ytest))
