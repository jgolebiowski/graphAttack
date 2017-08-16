"""This where implementations of individual operations live"""

from ..coreOperation import *
from ..coreNode import broadcast_shape, reduce_shape
from .twoInputOperations import DivideOperation
import numpy as np


class SumAllOperation(SingleInputOperation):
    """Sum all elements together

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
    shape : tuple
        shape of the output
    """
    name = "SumAllOperation"

    def setShape(self):
        """Set the output shape"""
        self.shape = (1, )

    def perform(self, a):
        """Summ all elements of the input

        Parameters
        ----------
        a : np.array
            Input data

        Returns
        -------
        np.array
            Output data
        """
        return np.sum(a)

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

        return grad * np.ones(self.inputA.shape)


class SumAxisOperation(SingleInputOperation):
    """Sum all elements together along a given axis

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
    shape : tuple
        shape of the output

    axis : int
        Axis over which to perform the sum
    """
    name = "SumAllOperation"

    def __init__(self, inputA=None, axis=0):
        self.axis = axis
        super().__init__(inputA)
        self.setShape()

    def setShape(self):
        """Set the output shape"""
        self.shape = np.delete(self.inputA.shape, self.axis)

    def perform(self, a):
        """Sum all elements along the given axis

        Parameters
        ----------
        a : np.array
            Input data

        Returns
        -------
        np.array
            Output data
        """
        return np.sum(a, axis=self.axis)

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
        if (self.axis == 0):
            return (grad * np.ones(self.inputA.shape))
        elif (self.axis == 1):
            return (grad * np.ones(self.inputA.shape)).T
        else:
            raise NotImplemented("Must investigate this gradient further")


class SumSquaredOperation(SingleInputOperation):
    """Sum all elements together

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
    shape : tuple
        shape of the output
    """
    name = "SumSquaresOperation"

    def setShape(self):
        """Set the output shape"""
        self.shape = (1, )

    def perform(self, a):
        """Sum all squared elements

        Parameters
        ----------
        a : np.array
            Input data

        Returns
        -------
        np.array
            Output data
        """
        return np.sum(np.square(a))

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

        return grad * self.inputA.getValue() * 2


class ExpOperation(SingleInputOperation):
    """Apply exponential function to all of the elements

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
    shape : tuple
        shape of the output
    """
    name = "ExpOperation"

    def setShape(self):
        """Set the output shape"""
        self.shape = self.inputA.shape

    def perform(self, a):
        """Calculate the exponens element-wise

        Parameters
        ----------
        a : np.array
            Input data

        Returns
        -------
        np.array
            Output data
        """
        return np.exp(a)

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

        return grad * self.getValue()
