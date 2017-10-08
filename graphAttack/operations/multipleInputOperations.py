"""This where implementations of individual operations live"""

from ..coreOperation import *
from ..coreNode import broadcast_shape, reduce_shape
import numpy as np


class BatchNormalisationOperation(MultipleInputOperation):
    """Perform batch normalisation in the data

    Attributes
    ----------
    initialize with thee inputs:
        inputA: dataflow
        inputB: beta parameter
        inputC: gamma parameter

    name : str
        Name of the operation
    result : np.array
        Output of the operation
    testing : bool
        Flag specifying if the operation is in testing (making prefictions: True)
        or training (optimizing parameters: False) mode

    grads : list(np.array)
        gradients with respect to inouts grads[i]: gradient iwht respect to input i
    inputs : list(ga.Operation)
        Operations feeding data into this operation
    shape : tuple
        shape of the output
    """
    name = "BatchNormalisationOperation"

    def __init__(self, inputA=None, inputB=None, inputC=None, running_param=0.1):
        super().__init__(inputA, inputB, inputC)
        self.setShape()
        self.testing = False

        self.muMean = 0
        self.varMean = 1

        self.lastaMean = 0
        self.lastVarInv = 1
        self.lastaNorm = np.zeros(inputA.shape)

        self.running_param = running_param

    def reset(self):
        """Reset the values and gradients held by this operation"""
        self.result = None
        self.gradA = None
        self.setShape()

    def setShape(self):
        """Set the output shape"""
        self.shape = self.inputs[0].shape

    def perform(self, a, b, c):
        """Perform dropout
        Parameters
        ----------
        a : np.array
            Input data
        b : np.array
            Input data
        c : np.array
            Input data

        Returns
        -------
        np.array
            Output data
        """

        if self.testing:
            self.lastaNorm = (a - self.muMean) / np.sqrt(self.varMean + 1e-8)
        else:
            mu = np.mean(a, axis=0, keepdims=True)
            var = np.var(a, axis=0, keepdims=True)
            self.lastaMean = (a - mu)
            self.lastVarInv = 1 / np.sqrt(var + 1e-8)
            self.lastaNorm = self.lastaMean * self.lastVarInv

            self.muMean = self.muMean * (1 - self.running_param) + mu * self.running_param
            self.varMean = self.varMean * (1 - self.running_param) + var * self.running_param

        out = self.lastaNorm * c + b
        # out = self.lastaNorm
        return out

    def performGradient(self, input):
        """Find out the gradient with respect to the parameter

        Parameters
        ----------
        input : int
            Specify an input operation with respect to which the
            gradient is calculated

        Returns
        -------
        np.array
            Gradient propagated through this operation

        Raises
        ------
        ValueError
            input has to be from 0 to len(self.inputs)
        """
        if (self.endNode):
            grad = np.ones(self.inputs[input].shape)
        else:
            grad = np.zeros(self.inputs[0].shape)
            for out in self.outputs:
                grad += reduce_shape(out.getGradient(self), grad)

        if (input == 0):
            nExamples = self.inputs[0].shape[0]

            daNorm = grad * self.inputs[2].getValue()
            dVar = np.sum(daNorm * self.lastaMean, axis=0) * (-0.5) * np.power(self.lastVarInv, 3)
            dMu = np.sum(daNorm * (-self.lastVarInv), axis=0) + dVar * np.mean(-2 * self.lastaMean, axis=0)

            grad = (daNorm * self.lastVarInv) + (dVar * 2 * self.lastaMean / nExamples) + (dMu / nExamples)
        elif (input == 1):
            grad = np.sum(grad, axis=0)
        elif (input == 2):
            grad = np.sum(grad * self.lastaNorm, axis=0)
        return grad
