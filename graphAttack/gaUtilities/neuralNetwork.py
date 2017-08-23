"""Neural networks utilities"""
import numpy as np
from ..coreDataContainers import Variable
from ..operations.activationOperations import *
from ..operations.costOperations import *
from ..operations.twoInputOperations import *
from ..operations.singleInputOperations import *
from ..operations.convolutionOperation import *
from ..operations.transformationOperations import *
from .misc import generateRandomVariable, generateZeroVariable


def addDenseLayer(mainGraph, nOutputNodes,
                  inputOperation=None,
                  activation=ReLUActivation,
                  dropoutRate=0,
                  w=None,
                  b=None):
    """Append a dense layer to the graph

    Parameters
    ----------
    mainGraph : ga.Graph
        computation graph to which append the dense layer
    nOutputNodes : int
        Number of output nodes
    inputOperation : ga.Operation
        operation feeding the data to the layer
    activation : ga.SingleInputOperation
        activatin operation of choice
    dropoutRate : float
        dropout rate at the end of this layer
    w : np.array
        weigthts in shape (nOutputNodes, nFeatures)
        if None randomly initialized
    b : np.array
        biases, in shape (nOutputNodes, )
        if None, randomly initialized

    Returns
    -------
    ga.Operation
        Last operation of the dense layer
    """
    if (inputOperation is None):
        inputOperation = mainGraph.operations[-1]

    if (w is None):
        w = generateRandomVariable(shape=(nOutputNodes, inputOperation.shape[1]),
                                   transpose=True, nInputs=inputOperation.shape[1])
    else:
        w = Variable(w.T)

    if (b is None):
        b = generateRandomVariable(shape=nOutputNodes, transpose=False, nInputs=4)
    else:
        b = Variable(b)

    wo = mainGraph.addOperation(w, doGradient=True)
    bo = mainGraph.addOperation(b, doGradient=True)

    mmo = mainGraph.addOperation(MatMatmulOperation(inputOperation, wo),
                                 doGradient=False,
                                 finalOperation=False)
    addo = mainGraph.addOperation(AddOperation(mmo, bo),
                                  doGradient=False,
                                  finalOperation=False)

    if (dropoutRate > 0):
        dpo = mainGraph.addOperation(DropoutOperation(addo, dropoutRate),
                                     doGradient=False,
                                     finalOperation=False)
    else:
        dpo = addo

    acto = mainGraph.addOperation(activation(dpo),
                                  doGradient=False,
                                  finalOperation=False)
    return acto


def addConv2dLayer(mainGraph,
                   inputOperation=None,
                   nFilters=1,
                   filterHeigth=2,
                   filterWidth=2,
                   padding="SAME",
                   convStride=1,
                   activation=ReLUActivation,
                   pooling=MaxPoolOperation,
                   poolHeight=2,
                   poolWidth=2,
                   poolStride=2):
    """Append a convolution2D layer with pooling

    Parameters
    ----------
    mainGraph : ga.Graph
        computation graph to which append the dense layer
    inputOperation : ga.Operation
        operation feeding the data to the layer
    nFilters : int
        number of filter to be applied for the convolution
    filterHeigth : int
        convolution filter heigth
    filterWidth : int
        convolution filter width
    padding: "SAME" or "VALID"
        padding method for the convolution
    convStride : int
        stride for the convolution filter
    activation : ga.SingleInputOperation
        activatin operation of choice
    pooling : ga.SingleInputOperation
        pooling operation of choice
    poolHeight : int
        heigth of the pooling filter
    poolWidth : int
        width of the pooling filter
    poolStride : int
        stride of the pooling operation

    Returns
    -------
    ga.Operation
        Last operation of the dense layer
    """

    N, C, H, W = inputOperation.shape

    w = generateRandomVariable(shape=(nFilters, C, filterHeigth, filterWidth),
                               # transpose=False, nInputs=(filterHeigth * filterHeigth * C))
                               transpose=False, nInputs=(H * W * C))
    b = generateRandomVariable(shape=(1, nFilters, 1, 1), transpose=False, nInputs=(nFilters * 4))

    filterWop = mainGraph.addOperation(w, doGradient=True, feederOperation=False)
    opConv2d = mainGraph.addOperation(Conv2dOperation(
        inputOperation, filterWop, stride=convStride, paddingMethod=padding))

    filterBop = mainGraph.addOperation(b, doGradient=True, feederOperation=False)
    addConv2d = mainGraph.addOperation(AddOperation(opConv2d, filterBop))

    actop = mainGraph.addOperation(activation(addConv2d),
                                   doGradient=False,
                                   finalOperation=False)

    poolOP = mainGraph.addOperation(pooling(inputA=actop,
                                            poolHeight=poolHeight,
                                            poolWidth=poolWidth,
                                            stride=poolStride))

    return poolOP


def addRNNnetwork(mainGraph,
                  inputOperation=None,
                  activation=TanhActivation,
                  costActivation=SoftmaxActivation,
                  costOperation=CrossEntropyCostSoftmax,
                  nHidden=100,
                  labels=None):
    """Add a RNN network

    Parameters
    ----------
    mainGraph : ga.Graph
        computation graph to which append the dense layer
    inputOperation : ga.Operation
        operation feeding the data to the layer, must have a shape of
        (nExamples, seriesLength, nFeatures)
    activation : ga.SingleInputOperation
        activatin operation for hidden units
    costActivation : ga.SingleInputOperation
        activatin operation for outputs (values extracted from RNN flow)
    costOperation : ga.CostOperation
        cost operation to be used throughout the RNN
    nHidden : int
        number of hidden units
    labels : np.array
        Potentially, labels to initialize the RNN, must be in the shape of
        (nExamples, seriesLength, nHidden)

    Returns
    -------
    ga.Operation
        Last operation of the RNN

    Raises
    ------
    ValueError
        "Labels must be in a compatible shape (nExamples, seriesLength, nHidden)"
    """

    nExamples, seriesLength, nFeatures = inputOperation.shape

    if (labels is not None):
        if labels.shape != (nExamples, seriesLength, nHidden):
            raise ValueError("Labels must be in a compatible shape (nExamples, seriesLength, nHidden)")
    else:
        labels = np.random.random((nExamples, seriesLength, nFeatures))

    h0 = generateZeroVariable(shape=(1, nHidden),
                              transpose=False)
    W = generateRandomVariable(shape=(nFeatures, nHidden),
                               transpose=False, nInputs=nFeatures)
    U = generateRandomVariable(shape=(nHidden, nHidden),
                               transpose=False, nInputs=nHidden)
    B = generateRandomVariable(shape=(1, nHidden),
                               transpose=False, nInputs=nHidden)

    wH = generateRandomVariable(shape=(nHidden, nFeatures),
                                transpose=False, nInputs=nHidden)
    bH = generateRandomVariable(shape=(1, nFeatures),
                                transpose=False, nInputs=nHidden)

    h0op = mainGraph.addOperation(h0)
    Wop = mainGraph.addOperation(W, doGradient=True)
    Uop = mainGraph.addOperation(U, doGradient=True)
    Bop = mainGraph.addOperation(B, doGradient=True)
    wHop = mainGraph.addOperation(wH, doGradient=True)
    bHop = mainGraph.addOperation(bH, doGradient=True)
    totalCostop = mainGraph.addOperation(Variable([0]))

    hactivations = [h0op]
    costOperationsList = []
    totalCostsList = [totalCostop]

    for indexRNN in range(seriesLength):
        xSliceop = mainGraph.addOperation(SliceOperation(inputOperation,
                                                         np.index_exp[:, indexRNN, :]))
        xwop = mainGraph.addOperation(xSliceop @ Wop)
        huop = mainGraph.addOperation(hactivations[-1] @ Uop)
        sumop = mainGraph.addOperation(xwop + huop)
        biasop = mainGraph.addOperation(sumop + Bop)
        tanhop = mainGraph.addOperation(activation(biasop))
        hactivations.append(tanhop)

        hawHop = mainGraph.addOperation(tanhop @ wHop)
        habiasop = mainGraph.addOperation(hawHop + bHop)

        costActivationop = mainGraph.addOperation(costActivation(habiasop))
        localCost = mainGraph.addOperation(costOperation(costActivationop, labels[:, indexRNN, :]))
        costOperationsList.append(localCost)
        totalCostop = mainGraph.addOperation(localCost + totalCostsList[-1], finalOperation=True)
        totalCostsList.append(totalCostop)

    return totalCostop, hactivations, costOperationsList
