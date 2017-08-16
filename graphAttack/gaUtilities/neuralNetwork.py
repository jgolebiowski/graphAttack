"""Neural networks utilities"""
import numpy as np
from ..coreDataContainers import Variable
from ..operations.activationOperations import *
from ..operations.twoInputOperations import *
from ..operations.singleInputOperations import *
from ..operations.convolutionOperation import *
from .misc import generateRandomVariable


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
