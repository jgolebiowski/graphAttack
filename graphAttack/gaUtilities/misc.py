"""Additional utilities"""
import numpy as np

from ..coreDataContainers import Variable


def generateRandomVariable(shape, transpose=False, nInputs=None):
    """Generate a ga.Variable of a given shape filled with random values
    from a Gaussian distribution with mean 0.
    If the transpose flag is set, generate a Variable that is the transpose of a
    given shape

    Parameters
    ----------
    shape : tuple
        Shape of the desired variable
    transpose : bool
        If true, generate ga.Transposed variable with the shape being shape.T
    nInputs : int
        number of inputs for the variable

    Returns
    -------
    ga.Variable
        generated random variable
    """

    reduction = 2 * np.sqrt(nInputs)
    # print("Initiazing with reduction", reduction, "and shape", shape)

    X = np.random.random(shape) / reduction
    if (transpose):
        return Variable(X.T)
    else:
        return Variable(X)


def generateZeroVariable(shape, transpose=False):
    """Generate a ga.Variable of a given shape filled with zeros

    Parameters
    ----------
    shape : tuple
        Shape of the desired variable
    transpose : bool
        If true, generate ga.Transposed variable with the shape being shape.T


    Returns
    -------
    ga.Variable
        generated random variable
    """

    X = np.zeros(shape)
    if (transpose):
        return Variable(X.T)
    else:
        return Variable(X)


def generateLinspaceVariable(shape, transpose=False, nInputs=None):
    """Generate a ga.Variable of a given shape filled with values from 0 to size
    If the transpose flag is set, generate a Variable that is the transpose of a
    given shape

    Parameters
    ----------
    shape : tuple
        Shape of the desired variable
    transpose : bool
        If true, generate ga.Transposed variable with the shape being shape.T
    nInputs : int
        number of inputs for the variable

    Returns
    -------
    ga.Variable
        generated random variable
    """

    X = np.zeros(shape)
    X = np.arange(X.size).reshape(shape)
    if (transpose):
        return Variable(X.T)
    else:
        return Variable(X)


def calculateAccuracy(graph, data, labels):
    """Feed data to a graph, ask it for predictions and obtain accuracy

    Parameters
    ----------
    graph : ga.Graph
        calculation graph
    data : np.array
        Input data
    labels : np.array
        labels for the data

    Returns
    -------
    float
        accuracy as a number from 0 to 1
    """
    graph.resetAll()
    graph.feederOperation.assignData(data)
    preds = graph.makePredictions()

    if np.size(labels.shape) == 1:
        nExamples = 1
    else:
        nExamples = labels.shape[0]

    error = np.sum(np.abs(preds - labels)) / (2 * nExamples)
    return 1 - error
