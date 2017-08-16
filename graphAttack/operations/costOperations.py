"""This where implementations of individual operations live"""

from ..coreOperation import *
from ..coreNode import broadcast_shape, reduce_shape
import numpy as np


class QuadratiCcostOperation(CostOperation):
    """Evaliate the quadratic cost given the labels

    Attributes
    ----------
    name : str
        Name of the operation
    result : np.array
        Output of the operation
    testing : bool
        Flag specifying if the operation is in testing (making prefictions: True)
        or training (optimizing parameters: False) mode

    gradA : np.array
        gradient with respect to inputA
    inputA : ga.Operation
        Operation feeding data A into this operation
    labels : np.arrays
        Data labels to compare with hypothesis
    nExamples : int
        Number of examples in current batch
    shape : tuple
        shape of the output
    """
    
    name = "QuadratiCcostOperation"

    def perform(self, a, y):
        """Perform costOperation

        Parameters
        ----------
        a : np.array
            Predictions
        y : np.array
            Data labels


        Returns
        -------
        np.array
            Output data
        """

        return (0.5 / self.nExamples) * np.sum(np.square(a - y))

    def performGradient(self, input=None):
        """Find out the gradient with respect to the parameter

        Parameters
        ----------
        input : int
            placeholder variable since this operation has only one input

        Returns
        -------
        np.array
            Gradient propagated through this operation
    """
        if (self.endNode):
            grad = np.ones(self.inputA.shape)
        else:
            grad = np.zeros(self.inputA.shape)
            for out in self.outputs:
                grad += out.getGradient(self)
        return grad * (1.0 / self.nExamples) * (self.inputA.getValue() - self.labels)


class CrossEntropyCostSoftmax(CostOperation):
    """Evaliate the CrossEntropy cost given the labels, works with softmax activation ONLY

    Attributes
    ----------
    name : str
        Name of the operation
    result : np.array
        Output of the operation
    testing : bool
        Flag specifying if the operation is in testing (making prefictions: True)
        or training (optimizing parameters: False) mode

    gradA : np.array
        gradient with respect to inputA
    inputA : ga.Operation
        Operation feeding data A into this operation
    labels : np.arrays
        Data labels to compare with hypothesis
    nExamples : int
        Number of examples in current batch
    shape : tuple
        shape of the output
    """
    name = "CrossEntropyCostSoftmax"

    def perform(self, a, y):
        """Perform costOperation

        Parameters
        ----------
        a : np.array
            Predictions
        y : np.array
            Data labels


        Returns
        -------
        np.array
            Output data
        """
        predLog = np.nan_to_num(-np.log(a))
        cEntropyMat = np.multiply(y, predLog)
        return (1.0 / self.nExamples) * np.sum(cEntropyMat)

    def performGradient(self, input=None):
        """Find out the gradient with respect to the parameter

        Parameters
        ----------
        input : int
            placeholder variable since this operation has only one input

        Returns
        -------
        np.array
            Gradient propagated through this operation
        """
        if (self.endNode):
            grad = np.ones(self.inputA.shape)
        else:
            grad = np.zeros(self.inputA.shape)
            for out in self.outputs:
                grad += out.getGradient(self)
        return grad * (1.0 / self.nExamples) * (-self.labels / self.inputA.getValue())

    # def perform(self, a):
    #     """Perform MatMul"""
    #     return np.sum(np.square(a))

    # def performGradient(self, input=None):
    #     """Find out the gradient with respect to the parameter
    #     the key is:
    #     inputA => 0
    #     inputB => 1"""
    #     if (self.endNode):
    #         grad = np.ones(self.inputA.shape)
    #     else:
    #         grad = np.zeros(self.inputA.shape)
    #         for out in self.outputs:
    #             grad += out.getGradient(self)

    #     return grad * self.inputA.getValue() * 2
