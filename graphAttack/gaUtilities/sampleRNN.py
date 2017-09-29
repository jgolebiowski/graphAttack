import numpy as np
"""Control script"""


def array2char(array, index_to_word):
    return index_to_word[np.argmax(array)]


def sampleSingleRNN(previousX, previousH,
                    hactivations=None,
                    costOperationsList=None,
                    mainGraph=None,
                    index_to_word=None):
    """sample a single timestep from a RNN made out of LSTM gates

    Parameters
    ----------
    previousX : np.array
        input values for the previous timestep
    previousH : list of np.arrays
        hActivation values for the previous timestep
    hactivations : list of lists
        list of activation operations for each layer (as returned by ga.xxLSTMLayer)
    cStates : list of lists
        list of cState operations for each layer (as returned by ga.xxLSTMLayer)
    costOperationsList : list of lists
        list of cost operations for each layer (as returned by ga.addRNNCost)
    mainGraph : ga.graph
        main graph

    Returns
    -------
    newX
        input values sampled from this timestep
    newH
        activation values sampled from this timestep

    Raises
    ------
    ValueError
        # of hactivations must be the same as # of cStates
    """
    nLayers = len(hactivations)

    N, T, D = mainGraph.feederOperation.shape
    preiousData = np.zeros((1, T, D))
    preiousData[0, 0, :] = previousX[0]

    mainGraph.resetAll()
    mainGraph.feederOperation.assignData(preiousData)

    for i in range(nLayers):
        hactivations[i][0].assignData(previousH[i])

    newH = [None for i in range(nLayers)]
    for i in range(nLayers):
        newH[i] = hactivations[i][1].getValue()

    newX = costOperationsList[0].inputA.getValue()

    return newX, newH


def sampleManyRNN(n, nFeatures, nHidden,
                  hactivations=None,
                  costOperationsList=None,
                  mainGraph=None,
                  index_to_word=None,
                  delimiter=" "):
    """Sample a RNN made out of LSTM gates

    Parameters
    ----------
    n : int
        Number of timesteps to sample
    nFeatures : int
        number of input features
    nHidden : list of ints
        list of hidden units for each layer
    hactivations : list of lists
        list of activation operations for each layer (as returned by ga.xxLSTMLayer)
    costOperationsList : list of lists
        list of cost operations for each layer (as returned by ga.addRNNCost)
    mainGraph : ga.graph
        main graph
    index_to_word : dictionary
        dict used ot translate one-hot-encoded index to words

    Returns
    -------
    str
        string sampled from the RNN

    Raises
    ------
    ValueError
        # of hactivations must be the same as # nHidden
    """
    if not (len(hactivations) == len(nHidden)):
        raise ValueError("# of hactivations must be the same as # of cStates and as # nHidden")
    nLayers = len(hactivations)

    string = ""
    nextX = np.zeros((1, 1, nFeatures))
    nextX[0, 0, int(np.random.random() * nFeatures)] = 1

    nextH = [np.zeros((1, nHidden[i])) for i in range(nLayers)]

    for index in range(n):
        nextX, nextH = sampleSingleRNN(nextX, nextH,
                                       hactivations=hactivations,
                                       costOperationsList=costOperationsList,
                                       mainGraph=mainGraph)
        idx = np.random.choice(nextX[0].size, p=nextX[0])
        nextX[:] = 0
        nextX[0, idx] = 1
        string += array2char(nextX, index_to_word) + delimiter
    return string


def sampleSingleLSTM(previousX, previousH, previousC,
                     hactivations=None,
                     cStates=None,
                     costOperationsList=None,
                     mainGraph=None):
    """sample a single timestep from a RNN made out of LSTM gates

    Parameters
    ----------
    previousX : np.array
        input values for the previous timestep
    previousH : list of np.arrays
        hActivation values for the previous timestep
    previousC : list of np.arrays
        cState values for the previous timestep
    hactivations : list of lists
        list of activation operations for each layer (as returned by ga.xxLSTMLayer)
    cStates : list of lists
        list of cState operations for each layer (as returned by ga.xxLSTMLayer)
    costOperationsList : list of lists
        list of cost operations for each layer (as returned by ga.addRNNCost)
    mainGraph : ga.graph
        main graph

    Returns
    -------
    newX
        input values sampled from this timestep
    newH
        activation values sampled from this timestep
    newC
        cState values sampled from this timestep

    Raises
    ------
    ValueError
        # of hactivations must be the same as # of cStates
    """
    if len(hactivations) != len(cStates):
        raise ValueError("# of hactivations must be the same as # of cStates")
    nLayers = len(hactivations)

    N, T, D = mainGraph.feederOperation.shape
    preiousData = np.zeros((1, T, D))
    preiousData[0, 0, :] = previousX[0]

    mainGraph.resetAll()
    mainGraph.feederOperation.assignData(preiousData)

    for i in range(nLayers):
        hactivations[i][0].assignData(previousH[i])
        cStates[i][0].assignData(previousC[i])

    newH = [None for i in range(nLayers)]
    newC = [None for i in range(nLayers)]
    for i in range(nLayers):
        newH[i] = hactivations[i][1].getValue()
        newC[i] = cStates[i][1].getValue()

    newX = costOperationsList[0].inputA.getValue()

    return newX, newH, newC


def sampleManyLSTM(n, nFeatures, nHidden,
                   hactivations=None,
                   cStates=None,
                   costOperationsList=None,
                   mainGraph=None,
                   index_to_word=None,
                   delimiter=" "):
    """Sample a RNN made out of LSTM gates

    Parameters
    ----------
    n : int
        Number of timesteps to sample
    nFeatures : int
        number of input features
    nHidden : list of ints
        list of hidden units for each layer
    hactivations : list of lists
        list of activation operations for each layer (as returned by ga.xxLSTMLayer)
    cStates : list of lists
        list of cState operations for each layer (as returned by ga.xxLSTMLayer)
    costOperationsList : list of lists
        list of cost operations for each layer (as returned by ga.addRNNCost)
    mainGraph : ga.graph
        main graph
    index_to_word : dictionary
        dict used ot translate one-hot-encoded index to words

    Returns
    -------
    str
        string sampled from the RNN

    Raises
    ------
    ValueError
        # of hactivations must be the same as # of cStates and as # nHidden
    """
    if not (len(hactivations) == len(cStates) == len(nHidden)):
        raise ValueError("# of hactivations must be the same as # of cStates and as # nHidden")
    nLayers = len(hactivations)

    string = ""
    nextX = np.zeros((1, 1, nFeatures))
    nextX[0, 0, int(np.random.random() * nFeatures)] = 1

    nextH = [np.zeros((1, nHidden[i])) for i in range(nLayers)]
    nextC = [np.zeros((1, nHidden[i])) for i in range(nLayers)]

    for index in range(n):
        nextX, nextH, nextC = sampleSingleLSTM(nextX, nextH, nextC,
                                               hactivations=hactivations,
                                               cStates=cStates,
                                               costOperationsList=costOperationsList,
                                               mainGraph=mainGraph)
        # idx = np.argmax(nextX[0])
        idx = np.random.choice(nextX[0].size, p=nextX[0])
        nextX[:] = 0
        nextX[0, idx] = 1
        string += array2char(nextX, index_to_word) + delimiter
    return string
