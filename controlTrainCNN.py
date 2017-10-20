import graphAttack as ga
import numpy as np
import pickle
import sys
"""Control script"""

# ------ This is a very limited dataset, load a lrger one for better results
# ------ The convolution net is quute slow to train, be aware.
pickleFilename = "dataSet/notMNIST_small.pkl"
with open(pickleFilename, "rb") as fp:
    allDatasets = pickle.load(fp)

X = allDatasets["train_dataset"]
Y = allDatasets["train_labels"]

Xtest = allDatasets["test_dataset"]
Ytest = allDatasets["test_labels"]

Xvalid = allDatasets["valid_dataset"]
Yvalid = allDatasets["valid_labels"]

# index = int(sys.argv[1])
simulationIndex = 0
print("Training with:", simulationIndex)
dropValueL = 0.1
dropValueS = 0.05
# ------ Build a LeNet archicture CNN

mainGraph = ga.Graph()
feed = mainGraph.addOperation(ga.Variable(X), doGradient=False, feederOperation=True)
feedDrop = mainGraph.addOperation(ga.DropoutOperation(
    feed, dropValueS), doGradient=False, finalOperation=False)

cnn1 = ga.addConv2dLayer(mainGraph,
                         inputOperation=feedDrop,
                         nFilters=20,
                         filterHeigth=5,
                         filterWidth=5,
                         padding="SAME",
                         convStride=1,
                         activation=ga.ReLUActivation,
                         batchNormalisation=False,
                         pooling=ga.MaxPoolOperation,
                         poolHeight=2,
                         poolWidth=2,
                         poolStride=2)

cnn2 = ga.addConv2dLayer(mainGraph,
                         inputOperation=cnn1,
                         nFilters=50,
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

flattenOp = mainGraph.addOperation(ga.FlattenFeaturesOperation(cnn2))
flattenDrop = mainGraph.addOperation(ga.DropoutOperation(
    flattenOp, dropValueL), doGradient=False, finalOperation=False)

l1 = ga.addDenseLayer(mainGraph, 500,
                      inputOperation=flattenDrop,
                      activation=ga.ReLUActivation,
                      dropoutRate=dropValueL,
                      batchNormalisation=True)
l2 = ga.addDenseLayer(mainGraph, 10,
                      inputOperation=l1,
                      activation=ga.SoftmaxActivation,
                      dropoutRate=0.0,
                      batchNormalisation=False)
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
                          epochs=10,
                          miniBatchSize=10,
                          initialLearningRate=1e-2,
                          beta1=0.9,
                          beta2=0.999,
                          epsilon=1e-8,
                          testFrequency=1e1,
                          function=fprime)

pickleFilename = "minimizerParamsCNN_" + str(simulationIndex) + ".pkl"

# with open(pickleFilename, "rb") as fp:
#     adamParams = pickle.load(fp)
#     adamGrad.restoreState(adamParams)
#     params = adamParams["params"]

params = adamGrad.minimize(printTrainigCost=True, printUpdateRate=False,
                           dumpParameters=pickleFilename)


mainGraph.attachParameters(params)

print("train: Trained with:", simulationIndex)
print("train: Accuracy on the train set:", ga.calculateAccuracy(mainGraph, X, Y))
print("train: Accuracy on cv set:", ga.calculateAccuracy(mainGraph, Xvalid, Yvalid))
print("train: Accuracy on test set:", ga.calculateAccuracy(mainGraph, Xtest, Ytest))
