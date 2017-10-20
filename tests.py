import numpy as np
import graphAttack as ga
import pickle

from controlDense import run as runDense
from controlTrainDense import run as runTrainDense
from controlCNN import run as runCNN
from controlTrainCNN import run as runTrainCNN
from controlRNN import run as runRNN
from controlTrainRNN import run as runTrainRNN


def test_dense():
    nGrad, aGrad, aCost, mainGraph = runDense()
    diff = np.sum(np.abs(aGrad - nGrad)) / np.sum(np.abs(aGrad))
    assert(diff < 1e-4)


def test_CNN():
    nGrad, aGrad, aCost, mainGraph = runCNN()
    diff = np.sum(np.abs(aGrad - nGrad)) / np.sum(np.abs(aGrad))
    assert(diff < 1e-4)


def test_RNN():
    nGrad, aGrad, aCost, mainGraph = runRNN()
    diff = np.sum(np.abs(aGrad - nGrad)) / np.sum(np.abs(aGrad))
    assert(diff < 1e-4)


def test_trainDense():
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
    mainGraph = runTrainDense(simulationIndex, X, Y)

    trainAcc = ga.calculateAccuracy(mainGraph, X, Y)
    validAcc = ga.calculateAccuracy(mainGraph, Xvalid, Yvalid)
    testAcc = ga.calculateAccuracy(mainGraph, Xtest, Ytest)

    assert(trainAcc > 0.9)
    assert(validAcc > 0.75)
    assert(testAcc > 0.75)


def test_trainCNN():
    pickleFilename = "dataSet/notMNIST_small.pkl"
    with open(pickleFilename, "rb") as fp:
        allDatasets = pickle.load(fp)

    X = allDatasets["train_dataset"]
    Y = allDatasets["train_labels"]

    Xtest = allDatasets["test_dataset"]
    Ytest = allDatasets["test_labels"]

    Xvalid = allDatasets["valid_dataset"]
    Yvalid = allDatasets["valid_labels"]

    simulationIndex = 0
    mainGraph = runTrainCNN(simulationIndex, X, Y)

    trainAcc = ga.calculateAccuracy(mainGraph, X, Y)
    validAcc = ga.calculateAccuracy(mainGraph, Xvalid, Yvalid)
    testAcc = ga.calculateAccuracy(mainGraph, Xtest, Ytest)

    assert(trainAcc > 0.9)
    assert(validAcc > 0.75)
    assert(testAcc > 0.75)


def test_trainRNN():
    # ------ This is a very limited dataset, load a lrger one for better results
    pickleFilename = "dataSet/singleSentence.pkl"
    with open(pickleFilename, "rb") as fp:
        x, index_to_word, word_to_index = pickle.load(fp)

    simulationIndex = 0
    mainGraph, cache, finalCost = runTrainRNN(simulationIndex, x)

    assert(finalCost < 1e-3)
