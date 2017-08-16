"""This where implementations of individual operations live"""

from ..coreOperation import *
from ..coreNode import broadcast_shape, reduce_shape
import numpy as np


class MultiplyOperation(TwoInputOperation):
    """Multiply two inputs

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
    gradB : np.array
        gradient with respect to inputB
    inputA : ga.Operation
        Operation feeding data A into this operation
    inputB : ga.Operation
        Operation feeding data B into this operation
    shape : tuple
        shape of the output
    """
    name = "MultiplyOperation"

    def perform(self, a, b):
        """Multiply two together

        Parameters
        ----------
        a : np.array
            first set of input data
        b : np.array
            second set pf input data

        Returns
        -------
        np.aarray
            Result of the operation
        """
        return np.multiply(a, b)

    def performGradient(self, input):
        """Find out the gradient with respect to the parameter

        Parameters
        ----------
        input : int
            Specify an input operation with respect to which the
            gradient is calculated

            the key is:
            inputA => 0
            inputB => 1

        Returns
        -------
        np.array
            Gradient propagated through this operation

        Raises
        ------
        ValueError
            input has ot be either 0 or 1
        """
        if (self.endNode):
            if (input == 0):
                grad = np.ones(self.inputA.shape)
            elif (input == 1):
                grad = np.ones(self.inputB.shape)
            else:
                raise ValueError
        else:
            if (input == 0):
                grad = np.zeros(self.inputA.shape)
            elif (input == 1):
                grad = np.zeros(self.inputB.shape)
            else:
                raise ValueError

            for out in self.outputs:
                grad += reduce_shape(out.getGradient(self), grad)

            if (input == 0):
                grad = grad * self.inputB.getValue()
            elif (input == 1):
                grad = grad * self.inputA.getValue()
        return grad


class DivideOperation(TwoInputOperation):
    """Divide two inputs

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
    gradB : np.array
        gradient with respect to inputB
    inputA : ga.Operation
        Operation feeding data A into this operation
    inputB : ga.Operation
        Operation feeding data B into this operation
    shape : tuple
        shape of the output
    """
    name = "DivideOperation"

    def perform(self, a, b):
        """Multiply two together

        Parameters
        ----------
        a : np.array
            first set of input data
        b : np.array
            second set pf input data

        Returns
        -------
        np.aarray
            Result of the operation
        """
        return np.divide(a, b)

    def performGradient(self, input):
        """Find out the gradient with respect to the parameter

        Parameters
        ----------
        input : int
            Specify an input operation with respect to which the
            gradient is calculated

            the key is:
            inputA => 0
            inputB => 1

        Returns
        -------
        np.array
            Gradient propagated through this operation

        Raises
        ------
        ValueError
            input has ot be either 0 or 1
        """
        if (self.endNode):
            if (input == 0):
                grad = np.ones(self.inputA.shape)
            elif (input == 1):
                grad = np.ones(self.inputB.shape)
            else:
                raise ValueError
        else:
            if (input == 0):
                grad = np.zeros(self.inputA.shape)
            elif (input == 1):
                grad = np.zeros(self.inputB.shape)
            else:
                raise ValueError

            for out in self.outputs:
                grad += reduce_shape(out.getGradient(self), grad)

            if (input == 0):
                grad = np.divide(grad, self.inputB.getValue())
            elif (input == 1):
                grad = np.divide(grad, np.square(self.inputA.getValue()))
        return grad


class AddOperation(TwoInputOperation):
    """add two inputs

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
    gradB : np.array
        gradient with respect to inputB
    inputA : ga.Operation
        Operation feeding data A into this operation
    inputB : ga.Operation
        Operation feeding data B into this operation
    shape : tuple
        shape of the output
    """
    name = "AddOperation"

    def perform(self, a, b):
        """add two together

        Parameters
        ----------
        a : np.array
            first set of input data
        b : np.array
            second set pf input data

        Returns
        -------
        np.aarray
            Result of the operation
        """
        return np.add(a, b)

    def performGradient(self, input):
        """Find out the gradient with respect to the parameter

        Parameters
        ----------
        input : int
            Specify an input operation with respect to which the
            gradient is calculated

            the key is:
            inputA => 0
            inputB => 1

        Returns
        -------
        np.array
            Gradient propagated through this operation

        Raises
        ------
        ValueError
            input has ot be either 0 or 1
        """
        if (self.endNode):
            if (input == 0):
                grad = np.ones(self.inputA.shape)
            elif (input == 1):
                grad = np.ones(self.inputB.shape)
            else:
                raise ValueError
        else:
            if (input == 0):
                grad = np.zeros(self.inputA.shape)
            elif (input == 1):
                grad = np.zeros(self.inputB.shape)
            else:
                raise ValueError

            for out in self.outputs:
                grad += reduce_shape(out.getGradient(self), grad)
        return grad


class MatMatmulOperation(TwoInputOperation):
    """MatrixMultiplication for 2d matrices

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
    gradB : np.array
        gradient with respect to inputB
    inputA : ga.Operation
        Operation feeding data A into this operation
    inputB : ga.Operation
        Operation feeding data B into this operation
    shape : tuple
        shape of the output
    """
    name = "MatMatmulOperation"

    def setShape(self):
        """Set the output shape"""
        if not (len(self.inputA.shape) == 2 == len(self.inputB.shape)):
            raise ValueError("This should be used on arrays of ndims == 2")

        self.shape = (self.inputA.shape[0], self.inputB.shape[1])

    def perform(self, a, b):
        """Perform MatMul

        Parameters
        ----------
        a : np.array
            first set of input data
        b : np.array
            second set pf input data

        Returns
        -------
        np.aarray
            Result of the operation
        """
        return np.matmul(a, b)

    def performGradient(self, input):
        """Find out the gradient with respect to the parameter

        Parameters
        ----------
        input : int
            Specify an input operation with respect to which the
            gradient is calculated

            the key is:
            inputA => 0
            inputB => 1

        Returns
        -------
        np.array
            Gradient propagated through this operation

        Raises
        ------
        ValueError
            input has ot be either 0 or 1
        """
        if (self.endNode):
            grad = np.ones(self.shape)
        else:
            grad = np.zeros(self.shape)
            for out in self.outputs:
                grad += out.getGradient(self)

        if (input == 0):
            grad = np.matmul(grad, self.inputB.getValue().T)
        elif (input == 1):
            grad = np.matmul(self.inputA.getValue().T, grad)

        return grad
