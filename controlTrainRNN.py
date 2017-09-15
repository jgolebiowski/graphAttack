import graphAttack as ga
import numpy as np
import pickle
import sys
"""Control script"""
simulationIndex = 0
# simulationIndex = int(sys.argv[1])


pickleFilename = "dataSet/trump_campaign.pkl"
# pickleFilename = "dataSet/singleSentence.pkl"
with open(pickleFilename, "rb") as fp:
    x, index_to_word, word_to_index = pickle.load(fp)

seriesLength, nFeatures = x.shape
nExamples = simulationIndex
exampleLength = 20
nHidden = 80
nHidden2 = 100

# nExamples = 2
# exampleLength = 15
# nHidden = 35
# nHidden2 = 40

mainGraph = ga.Graph(False)
dummyX = np.zeros((nExamples, exampleLength, nFeatures))
feed = mainGraph.addOperation(ga.Variable(dummyX), feederOperation=True)

hactivations0 = ga.addInitialRNNLayer(mainGraph,
                                      inputOperation=feed,
                                      activation=ga.TanhActivation,
                                      nHidden=nHidden)

hactivations = ga.appendRNNLayer(mainGraph,
                                 previousActivations=hactivations0,
                                 activation=ga.TanhActivation,
                                 nHidden=nHidden2)

finalCost, costOperationsList = ga.addRNNCost(mainGraph,
                                              hactivations,
                                              costActivation=ga.SoftmaxActivation,
                                              costOperation=ga.CrossEntropyCostSoftmax,
                                              nHidden=nHidden2,
                                              labelsShape=feed.shape,
                                              labels=None)


def fprime(p, data, labels, costOperationsList=costOperationsList, mainGraph=mainGraph):
    mainGraph.feederOperation.assignData(data)
    mainGraph.resetAll()
    for index, cop in enumerate(costOperationsList):
        cop.assignLabels(labels[:, index, :])
    mainGraph.attachParameters(p)
    c = mainGraph.feedForward()
    mainGraph.feedBackward()
    g = mainGraph.unrollGradients()
    return c, g


param0 = mainGraph.unrollGradientParameters()
print("Number of parameters to train:", len(param0))
1/0
adaGrad = ga.adaptiveSGDrecurrent(trainingData=x,
                                  param0=param0,
                                  epochs=1e3,
                                  miniBatchSize=nExamples,
                                  exampleLength=exampleLength,
                                  initialLearningRate=1e-3,
                                  beta1=0.9,
                                  beta2=0.999,
                                  epsilon=1e-8,
                                  testFrequency=1e2,
                                  function=fprime)

params = adaGrad.minimize(printTrainigCost=True, printUpdateRate=False,
                          dumpParameters="paramsRNN_" + str(simulationIndex) + ".pkl")
# pickleFilename = "paramsRNN_" + str(simulationIndex) + ".pkl"
# with open(pickleFilename, "rb") as fp:
#     params = pickle.load(fp)

mainGraph.attachParameters(params)


def array2char(array):
    return index_to_word[np.argmax(array)]


def sampleSingle(previousX, previousH,
                 hactivations=hactivations,
                 costOperationsList=costOperationsList,
                 mainGraph=mainGraph):
    N, T, D = mainGraph.feederOperation.shape
    preiousData = np.zeros((1, T, D))
    preiousData[:, 0, :] = previousX

    mainGraph.resetAll()
    mainGraph.feederOperation.assignData(preiousData)
    hactivations[0].assignData(previousH)

    newH = hactivations[1].getValue()
    # newX = costOperationsList[0].makePredictions()
    newX = costOperationsList[0].inputA.getValue()

    return newX, newH


def sampleMany(n, hactivations=hactivations,
               costOperationsList=costOperationsList,
               mainGraph=mainGraph):
    string = ""
    nextH = np.zeros_like(hactivations[0].getValue())
    nextX = np.zeros_like(x[0])
    nextX[int(np.random.random() * nFeatures)] = 1

    for index in range(n):
        nextX, nextH = sampleSingle(nextX, nextH)
        idx = np.random.choice(nextX.size, p=nextX[0])
        nextX[:] = 0
        nextX[0, idx] = 1
        string += array2char(nextX) + " "
    return string


print(sampleMany(100))
