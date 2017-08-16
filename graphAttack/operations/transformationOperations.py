"""This where implementations of individual operations live"""

from ..coreOperation import *
from ..coreNode import broadcast_shape, reduce_shape
import numpy as np


class FlattenFeaturesOperation(SingleInputOperation):
    """Flatten the axis greater than 0 to turn
    dim > 2 tensors into 2d arrays

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
    nExamples : int
        Number of examples in current batch
    shape : tuple
        shape of the output
    """
    name = "FlattenFeaturesOperation"

    def setShape(self):
        """Set the output shape"""
        inpShapeSize = len(self.inputA.shape)
        if (inpShapeSize >= 2):
            self.nExamples = self.inputA.shape[0]
            numFeatures = 1
            for index in range(inpShapeSize - 1):
                numFeatures *= self.inputA.shape[index + 1]
            self.shape = (self.nExamples, numFeatures)
        else:
            self.nExamples = 1
            self.shape = (self.nExamples, self.inputA.shape[0])

    def perform(self, a):
        """Perform the flattening

        Parameters
        ----------
        a : np.array
            Input data

        Returns
        -------
        np.array
            Output data
        """
        return a.reshape(self.shape)

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
                grad += out.getGradient(self).reshape(self.inputA.shape)

        return grad


class ReshapeFeaturesOperation(SingleInputOperation):
    """Gather features and reshape them, transform a 2d array
    (nExamples, nFeatures) into a multidim array of
    (nExamples, shape)

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
    nExamples : int
        Number of examples in current batch
    shape : tuple
        shape of the output
    exampleShape : tuple
        shape of each example, Result of this operation is a matrix
        with shape (nExamples, nFeatures in each examle)
    """
    name = "ReshapeFeaturesOperation"

    def __init__(self, inputA=None, exampleShape=0):
        self.exampleShape = exampleShape
        super().__init__(inputA)
        self.setShape()

    def setShape(self):
        """Set the output shape"""
        inpShapeSize = len(self.inputA.shape)
        if (inpShapeSize >= 2):
            self.nExamples = self.inputA.shape[0]
            self.shape = (self.nExamples, ) + self.exampleShape
        else:
            self.nExamples = 1
            self.shape = (self.nExamples, ) + self.exampleShape

    def perform(self, a):
        """Reshape the flatend array to desired shape

        Parameters
        ----------
        a : np.array
            Input data

        Returns
        -------
        np.array
            Output data
        """
        return a.reshape(self.shape)

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
                grad += out.getGradient(self).reshape(self.inputA.shape)

        return grad
