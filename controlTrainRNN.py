import graphAttack as ga
import numpy as np
import pickle
"""Control script"""


def run(simulationIndex, X, Y=None):
    """Run the model"""
    print("Training with:", simulationIndex)

    seriesLength, nFeatures = X.shape
    # ------ it is important that the exampleLength is the same as
    # ------ the number if examples in the mini batch so that
    # ------ the state of the RNN is continously passed forward
    exampleLength = 4
    nExamples = exampleLength
    nHidden0 = 25
    nHidden1 = 25

    mainGraph = ga.Graph(False)
    dummyX = np.zeros((nExamples, exampleLength, nFeatures))
    feed = mainGraph.addOperation(ga.Variable(dummyX), feederOperation=True)

    # ------ Generate the network, options are RNN and LSTM gates
    # ------ Add initial layer and then possibly append more
    hactivations0, cStates0 = ga.addInitialLSTMLayer(mainGraph,
                                                     inputOperation=feed,
                                                     nHidden=nHidden0)
    hactivations1, cStates1 = ga.appendLSTMLayer(mainGraph,
                                                 previousActivations=hactivations0,
                                                 nHidden=nHidden1)

    # hactivations0 = ga.addInitialRNNLayer(mainGraph,
    #                                       inputOperation=feed,
    #                                       activation=ga.TanhActivation,
    #                                       nHidden=nHidden1)
    # hactivations1 = ga.appendRNNLayer(mainGraph,
    #                                   previousActivations=hactivations0,
    #                                   activation=ga.TanhActivation,
    #                                   nHidden=nHidden1)

    finalCost, costOperationsList = ga.addRNNCost(mainGraph,
                                                  hactivations1,
                                                  costActivation=ga.SoftmaxActivation,
                                                  costOperation=ga.CrossEntropyCostSoftmax,
                                                  nHidden=nHidden1,
                                                  labelsShape=feed.shape,
                                                  labels=None)

    hactivations = [hactivations0, hactivations1]
    cStates = [cStates0, cStates1]
    nHiddenList = [nHidden0, nHidden1]

    def fprime(p, data, labels, costOperationsList=costOperationsList, mainGraph=mainGraph):
        mainGraph.feederOperation.assignData(data)
        mainGraph.resetAll()
        for index, cop in enumerate(costOperationsList):
            cop.assignLabels(labels[:, index, :])
        mainGraph.attachParameters(p)
        c = mainGraph.feedForward()
        mainGraph.feedBackward()
        g = mainGraph.unrollGradients()

        nLayers = len(hactivations)
        for i in range(nLayers):
            hactivations[i][0].assignData(hactivations[i][-1].getValue())
            cStates[i][0].assignData(cStates[i][-1].getValue())

        return c, g

    param0 = mainGraph.unrollGradientParameters()
    print("Number of parameters to train:", len(param0))
    adamGrad = ga.adaptiveSGDrecurrent(trainingData=X,
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

    pickleFilename = "minimizerParamsRNN_" + str(simulationIndex) + ".pkl"

    # with open(pickleFilename, "rb") as fp:
    #     adamParams = pickle.load(fp)
    #     adamGrad.restoreState(adamParams)
    #     params = adamParams["params"]

    params = adamGrad.minimize(printTrainigCost=True, printUpdateRate=False,
                               dumpParameters=pickleFilename)
    mainGraph.attachParameters(params)

    cache = (nFeatures, nHiddenList, hactivations, cStates, costOperationsList)
    return mainGraph, cache, adamGrad.costLists[-1]


if(__name__ == "__main__"):
    # ------ This is a very limited dataset, load a lrger one for better results
    pickleFilename = "dataSet/singleSentence.pkl"
    with open(pickleFilename, "rb") as fp:
        x, index_to_word, word_to_index = pickle.load(fp)

    simulationIndex = 0
    mainGraph, cache, finalCost = run(simulationIndex, x)
    nFeatures, nHiddenList, hactivations, cStates, costOperationsList = cache

    temp = ga.sampleManyLSTM(100, nFeatures, nHiddenList,
                             hactivations=hactivations,
                             cStates=cStates,
                             costOperationsList=costOperationsList,
                             mainGraph=mainGraph,
                             index_to_word=index_to_word)
    print(temp)

    # temp = ga.sampleManyRNN(100, nFeatures, nHiddenList,
    #                         hactivations=hactivations,
    #                         costOperationsList=costOperationsList,
    #                         mainGraph=mainGraph,
    #                         index_to_word=index_to_word)
    # print(temp)
