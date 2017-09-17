import numpy as np
"""Control script"""


def array2char(array, index_to_word):
    return index_to_word[np.argmax(array)]


def sampleSingleSingleRNN(previousX, previousH,
                          hactivations=None,
                          costOperationsList=None,
                          mainGraph=None,
                          index_to_word=None):
    N, T, D = mainGraph.feederOperation.shape
    preiousData = np.zeros((1, T, D))
    preiousData[0, 0, :] = previousX[0]

    mainGraph.resetAll()
    mainGraph.feederOperation.assignData(preiousData)
    hactivations[0].assignData(previousH)

    newH = hactivations[1].getValue()
    newX = costOperationsList[0].inputA.getValue()

    return newX, newH


def sampleManySingleRNN(n, nFeatures, nHidden,
                        hactivations=None,
                        costOperationsList=None,
                        mainGraph=None,
                        index_to_word=None):
    string = ""
    nextX = np.zeros((1, 1, nFeatures))
    nextX[0, 0, int(np.random.random() * nFeatures)] = 1
    nextH = np.zeros((1, nHidden))
    for index in range(n):
        nextX, nextH = sampleSingleSingleRNN(nextX, nextH,
                                             hactivations=hactivations,
                                             costOperationsList=costOperationsList,
                                             mainGraph=mainGraph,
                                             index_to_word=index_to_word)
        idx = np.random.choice(nextX[0].size, p=nextX[0])
        nextX[:] = 0
        nextX[0, idx] = 1
        string += array2char(nextX, index_to_word) + " "
    return string


def sampleSingleSingleLSTM(previousX, previousH, previousC,
                           hactivations=None,
                           cStates=None,
                           costOperationsList=None,
                           mainGraph=None,
                           index_to_word=None):
    N, T, D = mainGraph.feederOperation.shape
    preiousData = np.zeros((1, T, D))
    preiousData[0, 0, :] = previousX[0]

    mainGraph.resetAll()
    mainGraph.feederOperation.assignData(preiousData)
    hactivations[0].assignData(previousH)
    cStates[0].assignData(previousC)

    newH = hactivations[1].getValue()
    newC = cStates[1].getValue()
    newX = costOperationsList[0].inputA.getValue()

    return newX, newH, newC


def sampleManySingleLSTM(n, nFeatures, nHidden,
                         hactivations=None,
                         cStates=None,
                         costOperationsList=None,
                         mainGraph=None,
                         index_to_word=None):
    string = ""
    nextX = np.zeros((1, 1, nFeatures))
    nextX[0, 0, int(np.random.random() * nFeatures)] = 1
    nextH = np.zeros((1, nHidden))
    nextC = np.zeros((1, nHidden))
    for index in range(n):
        nextX, nextH, nextC = sampleSingleSingleLSTM(nextX, nextH, nextC,
                                                     hactivations=hactivations,
                                                     cStates=cStates,
                                                     costOperationsList=costOperationsList,
                                                     mainGraph=mainGraph,
                                                     index_to_word=index_to_word)
        idx = np.random.choice(nextX[0].size, p=nextX[0])
        nextX[:] = 0
        nextX[0, idx] = 1
        string += array2char(nextX, index_to_word) + " "
    return string
