import graphAttack as ga
import numpy as np
import pickle
"""Control script"""


pickleFilename = "dataSet/notMNISTreformatted.pkl"
with open(pickleFilename, "rb") as fp:
    allDatasets = pickle.load(fp)

X = allDatasets["trainDataset"]
Y = allDatasets["trainLabels"]

Xtest = allDatasets["testDataset"]
Ytest = allDatasets["testLabels"]

Xvalid = allDatasets["validDataset"]
Yvalid = allDatasets["validLabels"]

# pickleFilename = "testData.pkl"
# with open(pickleFilename, "rb") as fp:
#     allDatasets = pickle.load(fp)

# X = allDatasets["X"][0:1000]
# Y = allDatasets["Y"]


for index in [0]:
    print(("Training with:", index))

    mainGraph = ga.Graph()
    ffeed = mainGraph.addOperation(ga.Variable(X), doGradient=False, feederOperation=True)
    feedDrop = mainGraph.addOperation(ga.DropoutOperation(
        ffeed, 0.25), doGradient=False, finalOperation=False)

    l1 = ga.addDenseLayer(mainGraph, 800,
                          inputOperation=feedDrop,
                          activation=ga.ReLUActivation,
                          dropoutRate=0.5,
                          w=None,
                          b=None)
    l2 = ga.addDenseLayer(mainGraph, 10,
                          inputOperation=l1,
                          activation=ga.SoftmaxActivation,
                          dropoutRate=0,
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
                             epochs=1e2,
                             miniBatchSize=200,
                             initialLearningRate=1e-3,
                             beta1=0.9,
                             beta2=0.999,
                             epsilon=1e-8,
                             testFrequency=1e2,
                             function=fprime)

    params = adaGrad.minimize(printTrainigCost=True, printUpdateRate=False,
                              dumpParameters="paramsDense" + str(index) + ".pkl")
    mainGraph.attachParameters(params)

    pickleFileName = "graphSGD_" + str(index) + ".pkl"
    with open(pickleFileName, "wb") as fp:
        mainGraph.resetAll()
        pickle.dump(mainGraph, fp)
    with open(pickleFileName, "rb") as fp:
        mainGraph = pickle.load(fp)

    print("train: Trained with:", index)
    print("train: Accuracy on train set:", ga.calculateAccuracy(mainGraph, X, Y))
    print("train: Accuracy on cv set:", ga.calculateAccuracy(mainGraph, Xvalid, Yvalid))
    print("train: Accuracy on test set:", ga.calculateAccuracy(mainGraph, Xtest, Ytest))
