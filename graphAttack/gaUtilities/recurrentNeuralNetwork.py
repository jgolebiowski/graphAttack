"""Neural networks utilities"""
import numpy as np
from ..coreDataContainers import Variable
from ..operations.activationOperations import *
from ..operations.costOperations import *
from ..operations.twoInputOperations import *
from ..operations.singleInputOperations import *
from ..operations.convolutionOperation import *
from ..operations.transformationOperations import *
from .misc import generateZeroVariable

# ------ WARNIGN - generate LinspaceVriable is for testing purposes only
from .misc import generateRandomVariable
# from .misc import generateLinspaceVariable as generateRandomVariable


def addInitialRNNLayer(mainGraph,
                       inputOperation=None,
                       activation=TanhActivation,
                       nHidden=100):
    """Add a RNN layer to input data

    Parameters
    ----------
    mainGraph : ga.Graph
        computation graph to which append the dense layer
    inputOperation : ga.Operation
        operation feeding the data to the layer, must have a shape of
        (nExamples, seriesLength, nFeatures)
    activation : ga.SingleInputOperation [class]
        activatin operation for hidden units
    nHidden : int
        number of hidden units

    Returns
    -------
    list(ga.Operation)
        List of activation operations from the RNN layer
    """

    nExamples, seriesLength, nFeatures = inputOperation.shape

    h0 = generateZeroVariable(shape=(nExamples, nHidden),
                              transpose=False)
    W = generateRandomVariable(shape=(nFeatures, nHidden),
                               transpose=False, nInputs=nFeatures * seriesLength ** 3)
    U = generateRandomVariable(shape=(nHidden, nHidden),
                               transpose=False, nInputs=nHidden * seriesLength ** 3)
    B = generateRandomVariable(shape=(1, nHidden),
                               transpose=False, nInputs=nHidden * seriesLength ** 3)

    h0op = mainGraph.addOperation(h0)
    Wop = mainGraph.addOperation(W, doGradient=True)
    Uop = mainGraph.addOperation(U, doGradient=True)
    Bop = mainGraph.addOperation(B, doGradient=True)

    hactivations = [h0op]

    # ------ append activation gates
    for indexRNN in range(seriesLength):
        xSliceop = mainGraph.addOperation(SliceOperation(inputOperation,
                                                         np.index_exp[:, indexRNN, :]))
        newHActiv = createRNNgate(mainGraph,
                                  xSliceop,
                                  hactivations[-1],
                                  Wop,
                                  Uop,
                                  Bop,
                                  activation)
        hactivations.append(newHActiv)

    return hactivations


def appendRNNLayer(mainGraph,
                   previousActivations=None,
                   activation=TanhActivation,
                   nHidden=100):
    """Append a next RNN layer to an already existing RNn layer

    Parameters
    ----------
    mainGraph : ga.Graph
        computation graph to which append the dense layer
    previousActivations : list(ga.Operation)
        List of activation operations from the previous RNN layer
    activation : ga.SingleInputOperation [class]
        activatin operation for hidden units
    nHidden : int
        number of hidden units


    Returns
    -------
    list(ga.Operation)
        List of activation operations from the RNN layer
    """

    nExamples, nFeatures = previousActivations[1].shape
    seriesLength = len(previousActivations) - 1

    h0 = generateZeroVariable(shape=(nExamples, nHidden),
                              transpose=False)
    W = generateRandomVariable(shape=(nFeatures, nHidden),
                               transpose=False, nInputs=nFeatures * seriesLength ** 3)
    U = generateRandomVariable(shape=(nHidden, nHidden),
                               transpose=False, nInputs=nHidden * seriesLength ** 3)
    B = generateRandomVariable(shape=(1, nHidden),
                               transpose=False, nInputs=nHidden * seriesLength ** 3)

    h0op = mainGraph.addOperation(h0)
    Wop = mainGraph.addOperation(W, doGradient=True)
    Uop = mainGraph.addOperation(U, doGradient=True)
    Bop = mainGraph.addOperation(B, doGradient=True)

    hactivations = [h0op]

    # ------ append activation gates
    for indexRNN in range(seriesLength):
        xSliceop = previousActivations[indexRNN + 1]
        newHActiv = createRNNgate(mainGraph,
                                  xSliceop,
                                  hactivations[-1],
                                  Wop,
                                  Uop,
                                  Bop,
                                  activation)
        hactivations.append(newHActiv)

    return hactivations


def createRNNgate(mainGraph,
                  inputOp,
                  hActiv,
                  Wop,
                  Uop,
                  Bop,
                  activation):
    """generate a new RNN gate for the network

    Parameters
    ----------
    mainGraph : ga.Graph
        main computational graph for the simulation
    inputOp : ga.operation
        operation holding inputs
    hActiv : ga.operation
        operation holding last gate's activations
    Wop : ga.Variable
        Variable holding weigths W (x @ W)
    Uop : ga.Variable
        Variable holding weigths U (h @ U)
    Bop : ga.Variable
        Variable holding biases B (x @ W + h @ U + B)
    activation : ga.SingleInputOperation [class]
        activatin operation for hidden units

    Returns
    -------
    ga.operation
        Activation op for this gate
    """
    xwop = mainGraph.addOperation(inputOp @ Wop)
    huop = mainGraph.addOperation(hActiv @ Uop)
    sumop = mainGraph.addOperation(xwop + huop)
    biasop = mainGraph.addOperation(sumop + Bop)
    newHActiv = mainGraph.addOperation(activation(biasop))

    return newHActiv


def addRNNCost(mainGraph,
               RNNactivations,
               costActivation=SoftmaxActivation,
               costOperation=CrossEntropyCostSoftmax,
               nHidden=None,
               labelsShape=None,
               labels=None):
    """Add a RNN network

    Parameters
    ----------
    mainGraph : ga.Graph
        computation graph to which append the dense layer
    RNNactivations : list(ga.Operation)
        List of activation operations from the last RNN layer
    costActivation : ga.SingleInputOperation [class]
        activatin operation for outputs (values extracted from RNN flow)
    costOperation : ga.CostOperation [class]
        cost operation to be used throughout the RNN
    nHidden : int
        number of hidden units
    labelsShape : tuple
        Shape of the labels
        (nExamples, seriesLength, nOutputFeatures)
    labels : np.array
        Potentially, labels to initialize the RNN, must be in the shape of
        (nExamples, seriesLength, nOutputFeatures)

    Returns
    -------
    ga.Operation
        Last operation of the RNN

    Raises
    ------
    ValueError
        "Labels must be in a compatible shape (nExamples, seriesLength, nOutputFeatures)"
    """

    nExamples, seriesLength, nOutputFeatures = labelsShape

    if (labels is not None):
        if labels.shape != (nExamples, seriesLength, nOutputFeatures):
            raise ValueError(
                "Labels must be in a compatible shape (nExamples, seriesLength, nOutputFeatures)")
    else:
        labels = np.zeros((nExamples, seriesLength, nOutputFeatures))

    wH = generateRandomVariable(shape=(nHidden, nOutputFeatures),
                                transpose=False, nInputs=nHidden * seriesLength ** 3)
    bH = generateRandomVariable(shape=(1, nOutputFeatures),
                                transpose=False, nInputs=nHidden * seriesLength ** 3)

    Whop = mainGraph.addOperation(wH, doGradient=True)
    Bhop = mainGraph.addOperation(bH, doGradient=True)

    totalCostop = mainGraph.addOperation(Variable([0]))
    costOperationsList = []
    totalCostsList = [totalCostop]

    # ------ append cost operation to each activation gate
    for indexRNNCost in range(seriesLength):
        lastActivation = RNNactivations[indexRNNCost + 1]
        haWhop = mainGraph.addOperation(lastActivation @ Whop)
        habiasop = mainGraph.addOperation(haWhop + Bhop)

        costActivationop = mainGraph.addOperation(costActivation(habiasop))
        localCost = mainGraph.addOperation(costOperation(costActivationop, labels[:, indexRNNCost, :]))
        costOperationsList.append(localCost)
        totalCostop = mainGraph.addOperation(localCost + totalCostsList[-1], finalOperation=True)
        totalCostsList.append(totalCostop)

    return totalCostop, costOperationsList
