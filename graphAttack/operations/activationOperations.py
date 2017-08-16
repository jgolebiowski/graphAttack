"""This where implementations of individual operations live"""

from ..coreOperation import *
from ..coreNode import broadcast_shape, reduce_shape
import numpy as np


class ReLUActivation(SingleInputOperation):
    """ReLu activation function, zeres all negative entries

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
    name = "ReLUActivation"

    def setShape(self):
        """Set the output shape"""
        self.shape = self.inputA.shape

    def perform(self, a):
        """Perform ReLU, element wise

        Parameters
        ----------
        a : np.array
            Input data

        Returns
        -------
        np.array
            Output data
        """
        self.mask = np.greater(a, 0).astype(int)
        return np.multiply(a, self.mask)

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

        return np.multiply(grad, self.mask)


class SigmoidActivation(SingleInputOperation):
    """Sigmoid activation function    

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
    name = "SigmoidActivation"

    def setShape(self):
        """Set the output shape"""
        self.shape = self.inputA.shape

    def perform(self, a):
        """Perform Sigmoid
        Parameters
        ----------
        a : np.array
            Input data

        Returns
        -------
        np.array
            Output data
        """

        return 1.0 / (1.0 + np.exp(-a))

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

        return grad * self.getValue() * (1 - self.getValue())


class SoftmaxActivation(SingleInputOperation):
    """Perform softmax on a given axis

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
    name = "SoftmaxActivation"

    def __init__(self, inputA=None, axis=1):
        super().__init__(inputA)
        self.axis = axis

    def perform(self, X, theta=1.0):
        """
        Compute the softmax of each element along an axis of X.

        Parameters
        ----------
        X: ND-Array. Probably should be floats.
        theta (optional): float parameter, used as a multiplier
            prior to exponentiation. Default = 1.0
        axis (optional): axis to compute values along. Default is the
            first non-singleton axis.

        Returns an array the same size as X. The result will sum to 1
        along the specified axis.
        """
        axis = self.axis
        # make X at least 2d
        y = np.atleast_2d(X)

        # find axis
        if axis is None:
            axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

        # multiply y against the theta parameter,
        y = y * float(theta)

        # subtract the max for numerical stability
        y = y - np.expand_dims(np.max(y, axis=axis), axis)

        # exponentiate y
        y = np.exp(y)

        # take the sum along the specified axis
        ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

        # finally: divide elementwise
        p = y / ax_sum

        # flatten if X was 1D
        if len(X.shape) == 1:
            p = p.flatten()

        return p

    def performGradient(self, inputA=None):
        """Evaluate the gradient of softmax

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

        grad = self.getValue() *\
            np.subtract(grad, np.expand_dims(np.sum(grad * self.getValue(), axis=self.axis), axis=self.axis))
        return grad


class DropoutOperation(SingleInputOperation):
    """Drops out some of the elements to prevent overfitting
    In default, the operation is active (performing dropout).
    For testing purposes (asking for prediction) the self.testing
    flag should be set to True to disable dropout and use all the
    neurons for prediction

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
    name = "DropoutOperation"

    def __init__(self, inputA=None, dropoutRate=0):
        super().__init__(inputA)
        self.setShape()
        self.dropoutRate = dropoutRate
        self.generateMask()
        self.testing = False

    def generateMask(self):
        """Generate dropout mask"""
        if (self.testing):
            self.dropoutMask = np.ones(self.shape[1:])
        else:
            self.dropoutMask = np.ones(self.shape[1:])
            nNeurons = np.size(self.dropoutMask)
            nNeuronsToDrop = int(nNeurons * self.dropoutRate)
            weightingFactor = 1.0 / (1 - self.dropoutRate)
            neuronsToDrop = np.random.choice(nNeurons, nNeuronsToDrop, replace=False)
            self.dropoutMask.ravel()[neuronsToDrop] = 0
            self.dropoutMask.reshape(self.shape[1:])
            self.dropoutMask *= weightingFactor

    def reset(self):
        """Reset the values and gradients held by this operation"""
        self.result = None
        self.gradA = None
        self.setShape()
        self.generateMask()

    def setShape(self):
        """Set the output shape"""
        self.shape = self.inputA.shape

    def perform(self, a):
        """Perform dropout
        Parameters
        ----------
        a : np.array
            Input data

        Returns
        -------
        np.array
            Output data
        """
        return np.multiply(a, self.dropoutMask)

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

        return np.multiply(grad, self.dropoutMask)
