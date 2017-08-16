import graphAttack as ga
import numpy as np
import pickle
"""Control script"""


pickleFilename = "dataSet/notMNIST.pickle"
with open(pickleFilename, "rb") as fp:
    allDatasets = pickle.load(fp)

X = allDatasets["train_dataset"]
Y = allDatasets["train_labels"]

Xtest = allDatasets["test_dataset"]
Ytest = allDatasets["test_labels"]

Xvalid = allDatasets["valid_dataset"]
Yvalid = allDatasets["valid_labels"]


for index in [0]:
    print("Training with:", index)

    # ------ Build a LeNet archicture CNN

    mainGraph = ga.Graph()
    feed = mainGraph.addOperation(ga.Variable(X), doGradient=False, feederOperation=True)
    feedDrop = mainGraph.addOperation(ga.DropoutOperation(
        feed, 0.0), doGradient=False, finalOperation=False)

    cnn1 = ga.addConv2dLayer(mainGraph,
                             inputOperation=feedDrop,
                             nFilters=20,
                             filterHeigth=5,
                             filterWidth=5,
                             padding="SAME",
                             convStride=1,
                             activation=ga.ReLUActivation,
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
                             pooling=ga.MaxPoolOperation,
                             poolHeight=2,
                             poolWidth=2,
                             poolStride=2)

    flattenOp = mainGraph.addOperation(ga.FlattenFeaturesOperation(cnn2))
    flattenDrop = mainGraph.addOperation(ga.DropoutOperation(
        flattenOp, 0.25), doGradient=False, finalOperation=False)

    l1 = ga.addDenseLayer(mainGraph, 500,
                          inputOperation=flattenDrop,
                          activation=ga.ReLUActivation,
                          dropoutRate=0.5,
                          w=None,
                          b=None)
    l2 = ga.addDenseLayer(mainGraph, 10,
                          inputOperation=l1,
                          activation=ga.SoftmaxActivation,
                          dropoutRate=0.0,
                          w=None,
                          b=None)
    fcost = mainGraph.addOperation(
        ga.CrossEntropyCostSoftmax(l2, Y),
        doGradient=False,
        finalOperation=True)

    def fprime(p, data, labels):
        mainGraph.feederOperation.assignData(data)
        mainGraph.costOperation.assignLabels(labels)
        mainGraph.attachParameters(p)
        mainGraph.resetAll()
        c = mainGraph.feedForward()
        mainGraph.feedBackward()
        g = mainGraph.unrollGradients()
        return c, g

    param0 = mainGraph.unrollGradientParameters()
    adaGrad = ga.adaptiveSGD(trainingData=X,
                             trainingLabels=Y,
                             param0=param0,
                             epochs=20,
                             miniBatchSize=100,
                             initialLearningRate=1e-3,
                             beta1=0.9,
                             beta2=0.999,
                             epsilon=1e-8,
                             testFrequency=1e3,
                             function=fprime)

    params = adaGrad.minimize(printTrainigCost=True, printUpdateRate=False,
                              dumpParameters="paramsCNN" + str(index) + ".pkl")
    mainGraph.attachParameters(params)

    pickleFileName = "graphSGD_" + str(index) + ".pkl"
    with open(pickleFileName, "wb") as fp:
        mainGraph.resetAll()
        pickle.dump(mainGraph, fp)
    with open(pickleFileName, "rb") as fp:
        mainGraph = pickle.load(fp)

    print("train: Trained with:", index)
    if (len(X) <= 10000):
        print("train: Accuracy on part of the train set:",
              ga.calculateAccuracy(mainGraph, X, Y))
    else:
        print("train: Accuracy on part of the train set:",
              ga.calculateAccuracy(mainGraph, X[0:10000], Y[0:10000]))
    print("train: Accuracy on cv set:", ga.calculateAccuracy(mainGraph, Xvalid, Yvalid))
    print("train: Accuracy on test set:", ga.calculateAccuracy(mainGraph, Xtest, Ytest))
