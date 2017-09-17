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


def addInitialLSTMLayer(mainGraph,
                        inputOperation=None,
                        nHidden=100):
    """Add a LSTM layer to input data

    Parameters
    ----------
    mainGraph : ga.Graph
        computation graph to which append the dense layer
    inputOperation : ga.Operation
        operation feeding the data to the layer, must have a shape of
        (nExamples, seriesLength, nFeatures)
    nHidden : int
        number of hidden units

    Returns
    -------
    list(ga.Operation)
        List of activation operations from the LSTM layer
    list(ga.Operation)
        List of internal state operations from the LSTM layer
    """

    nExamples, seriesLength, nFeatures = inputOperation.shape
    h0 = generateZeroVariable(shape=(nExamples, nHidden),
                              transpose=False)
    c0 = generateZeroVariable(shape=(nExamples, nHidden),
                              transpose=False)

    Wi = generateRandomVariable(shape=(nFeatures, nHidden),
                                transpose=False, nInputs=nFeatures)
    Wf = generateRandomVariable(shape=(nFeatures, nHidden),
                                transpose=False, nInputs=nFeatures)
    Wo = generateRandomVariable(shape=(nFeatures, nHidden),
                                transpose=False, nInputs=nFeatures)
    Wg = generateRandomVariable(shape=(nFeatures, nHidden),
                                transpose=False, nInputs=nFeatures)

    Ui = generateRandomVariable(shape=(nHidden, nHidden),
                                transpose=False, nInputs=nHidden)
    Uf = generateRandomVariable(shape=(nHidden, nHidden),
                                transpose=False, nInputs=nHidden)
    Uo = generateRandomVariable(shape=(nHidden, nHidden),
                                transpose=False, nInputs=nHidden)
    Ug = generateRandomVariable(shape=(nHidden, nHidden),
                                transpose=False, nInputs=nHidden)

    Bi = generateRandomVariable(shape=(1, nHidden),
                                transpose=False, nInputs=nHidden)
    Bf = generateRandomVariable(shape=(1, nHidden),
                                transpose=False, nInputs=nHidden)
    Bo = generateRandomVariable(shape=(1, nHidden),
                                transpose=False, nInputs=nHidden)
    Bg = generateRandomVariable(shape=(1, nHidden),
                                transpose=False, nInputs=nHidden)

    # N, D, H, T = 2, 5, 4, 3
    # nHidden = H
    # h0 = Variable(np.linspace(-0.4, 0.8, num=N * H).reshape(N, H))
    # c0 = generateZeroVariable(shape=(nExamples, nHidden),
    #                           transpose=False)
    # Wwx = np.linspace(-0.2, 0.9, num=4 * D * H).reshape(D, 4 * H)
    # Uuh = np.linspace(-0.3, 0.6, num=4 * H * H).reshape(H, 4 * H)
    # Bb = np.linspace(0.2, 0.7, num=4 * H)

    # Wi = Variable(Wwx[:, 0:H])
    # Wf = Variable(Wwx[:, H: 2 * H])
    # Wo = Variable(Wwx[:, 2 * H: 3 * H])
    # Wg = Variable(Wwx[:, 3 * H: 4 * H])

    # Ui = Variable(Uuh[:, 0:H])
    # Uf = Variable(Uuh[:, H: 2 * H])
    # Uo = Variable(Uuh[:, 2 * H: 3 * H])
    # Ug = Variable(Uuh[:, 3 * H: 4 * H])

    # Bi = Variable(Bb[0:H])
    # Bf = Variable(Bb[H: 2 * H])
    # Bo = Variable(Bb[2 * H: 3 * H])
    # Bg = Variable(Bb[3 * H: 4 * H])

    Wiop = mainGraph.addOperation(Wi, doGradient=True)
    Wfop = mainGraph.addOperation(Wf, doGradient=True)
    Woop = mainGraph.addOperation(Wo, doGradient=True)
    Wgop = mainGraph.addOperation(Wg, doGradient=True)
    Uiop = mainGraph.addOperation(Ui, doGradient=True)
    Ufop = mainGraph.addOperation(Uf, doGradient=True)
    Uoop = mainGraph.addOperation(Uo, doGradient=True)
    Ugop = mainGraph.addOperation(Ug, doGradient=True)
    Biop = mainGraph.addOperation(Bi, doGradient=True)
    Bfop = mainGraph.addOperation(Bf, doGradient=True)
    Boop = mainGraph.addOperation(Bo, doGradient=True)
    Bgop = mainGraph.addOperation(Bg, doGradient=True)

    h0op = mainGraph.addOperation(h0)
    c0op = mainGraph.addOperation(c0)

    hactivations = [h0op]
    cStates = [c0op]

    # ------ append activation gates
    for indexRNN in range(seriesLength):
        xSliceop = mainGraph.addOperation(SliceOperation(inputOperation,
                                                         np.index_exp[:, indexRNN, :]))
        newHActiv, newC = createLSTMgate(mainGraph,
                                         xSliceop,
                                         hactivations[-1],
                                         cStates[-1],
                                         Wiop, Wfop, Woop, Wgop,
                                         Uiop, Ufop, Uoop, Ugop,
                                         Biop, Bfop, Boop, Bgop)
        hactivations.append(newHActiv)
        cStates.append(newC)

    return hactivations, cStates


def appendLSTMLayer(mainGraph,
                    previousActivations=None,
                    nHidden=100):
    """Add a LSTM layer to input data

    Parameters
    ----------
    mainGraph : ga.Graph
        computation graph to which append the dense layer
    inputOperation : ga.Operation
        operation feeding the data to the layer, must have a shape of
        (nExamples, seriesLength, nFeatures)
    nHidden : int
        number of hidden units

    Returns
    -------
    list(ga.Operation)
        List of activation operations from the LSTM layer
    list(ga.Operation)
        List of internal state operations from the LSTM layer
    """

    nExamples, nFeatures = previousActivations[1].shape
    seriesLength = len(previousActivations) - 1

    h0 = generateZeroVariable(shape=(nExamples, nHidden),
                              transpose=False)
    c0 = generateZeroVariable(shape=(nExamples, nHidden),
                              transpose=False)

    Wi = generateRandomVariable(shape=(nFeatures, nHidden),
                                transpose=False, nInputs=nFeatures)
    Wf = generateRandomVariable(shape=(nFeatures, nHidden),
                                transpose=False, nInputs=nFeatures)
    Wo = generateRandomVariable(shape=(nFeatures, nHidden),
                                transpose=False, nInputs=nFeatures)
    Wg = generateRandomVariable(shape=(nFeatures, nHidden),
                                transpose=False, nInputs=nFeatures)

    Ui = generateRandomVariable(shape=(nHidden, nHidden),
                                transpose=False, nInputs=nHidden)
    Uf = generateRandomVariable(shape=(nHidden, nHidden),
                                transpose=False, nInputs=nHidden)
    Uo = generateRandomVariable(shape=(nHidden, nHidden),
                                transpose=False, nInputs=nHidden)
    Ug = generateRandomVariable(shape=(nHidden, nHidden),
                                transpose=False, nInputs=nHidden)

    Bi = generateRandomVariable(shape=(1, nHidden),
                                transpose=False, nInputs=nHidden)
    Bf = generateRandomVariable(shape=(1, nHidden),
                                transpose=False, nInputs=nHidden)
    Bo = generateRandomVariable(shape=(1, nHidden),
                                transpose=False, nInputs=nHidden)
    Bg = generateRandomVariable(shape=(1, nHidden),
                                transpose=False, nInputs=nHidden)

    Wiop = mainGraph.addOperation(Wi, doGradient=True)
    Wfop = mainGraph.addOperation(Wf, doGradient=True)
    Woop = mainGraph.addOperation(Wo, doGradient=True)
    Wgop = mainGraph.addOperation(Wg, doGradient=True)
    Uiop = mainGraph.addOperation(Ui, doGradient=True)
    Ufop = mainGraph.addOperation(Uf, doGradient=True)
    Uoop = mainGraph.addOperation(Uo, doGradient=True)
    Ugop = mainGraph.addOperation(Ug, doGradient=True)
    Biop = mainGraph.addOperation(Bi, doGradient=True)
    Bfop = mainGraph.addOperation(Bf, doGradient=True)
    Boop = mainGraph.addOperation(Bo, doGradient=True)
    Bgop = mainGraph.addOperation(Bg, doGradient=True)

    h0op = mainGraph.addOperation(h0)
    c0op = mainGraph.addOperation(c0)

    hactivations = [h0op]
    cStates = [c0op]

    # ------ append activation gates
    for indexRNN in range(seriesLength):
        xSliceop = previousActivations[indexRNN + 1]
        newHActiv, newC = createLSTMgate(mainGraph,
                                         xSliceop,
                                         hactivations[-1],
                                         cStates[-1],
                                         Wiop, Wfop, Woop, Wgop,
                                         Uiop, Ufop, Uoop, Ugop,
                                         Biop, Bfop, Boop, Bgop)
        hactivations.append(newHActiv)
        cStates.append(newC)

    return hactivations, cStates


def createLSTMgate(mainGraph,
                   xInput,
                   hActiv,
                   prevC,
                   Wiop, Wfop, Woop, Wgop,
                   Uiop, Ufop, Uoop, Ugop,
                   Biop, Bfop, Boop, Bgop):
    """generate a new RNN gate for the network

    Parameters
    ----------
    mainGraph : ga.Graph
        main computational graph for the simulation
    xInput : ga.operation
        operation holding inputs
    hActiv : ga.operation
        operation holding last gate's activations
    prevC : ga.operation
        State of the previous gate
    Wiop, Wfop, Woop, Wgop : ga.Variable
        Multiple Variables holding weigths W (W @ x)
    Uiop, Ufop, Uoop, Ugop : ga.Variable
        Multiple Variables holding weigths U (U @ h)
    Biop, Bfop, Boop, Bgop : ga.Variable
        Multiple Variables holding biases B (W @ x + U @ h + B)

    Returns
    -------
    newHActiv : ga.operation
        Activation op for this gate
    newState : ga.operation
        State op for this gate
    """

    aix = mainGraph.addOperation(xInput @ Wiop)
    aih = mainGraph.addOperation(hActiv @ Uiop)
    aisum = mainGraph.addOperation(aix + aih)
    ai = mainGraph.addOperation(aisum + Biop)
    i = mainGraph.addOperation(SigmoidActivation(ai))

    afx = mainGraph.addOperation(xInput @ Wfop)
    afh = mainGraph.addOperation(hActiv @ Ufop)
    afsum = mainGraph.addOperation(afx + afh)
    af = mainGraph.addOperation(afsum + Bfop)
    f = mainGraph.addOperation(SigmoidActivation(af))

    aox = mainGraph.addOperation(xInput @ Woop)
    aoh = mainGraph.addOperation(hActiv @ Uoop)
    aosum = mainGraph.addOperation(aox + aoh)
    ao = mainGraph.addOperation(aosum + Boop)
    o = mainGraph.addOperation(SigmoidActivation(ao))

    agx = mainGraph.addOperation(xInput @ Wgop)
    agh = mainGraph.addOperation(hActiv @ Ugop)
    agsum = mainGraph.addOperation(agx + agh)
    ag = mainGraph.addOperation(agsum + Bgop)
    g = mainGraph.addOperation(TanhActivation(ag))

    forget = mainGraph.addOperation(f * prevC)
    update = mainGraph.addOperation(i * g)
    newC = mainGraph.addOperation(forget + update)

    newCActiv = mainGraph.addOperation(TanhActivation(newC))
    newHActiv = mainGraph.addOperation(o * newCActiv)

    return newHActiv, newC
