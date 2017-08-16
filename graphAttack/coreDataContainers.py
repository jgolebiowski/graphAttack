"""This where implementations of individual operations live"""

from .coreNode import *
import numpy as np


class Variable(Node):
    """Store some data ot feed in into the graph

    Attributes
    ----------
    gradA : np.array
        gradient with respect to inputA
    inputA : np.array
        data held by this variable, reference to self.result
    name : str
        Name of this operation
    result : np.array
        data held by this variable
    shape : tuple
        shape of ths output

    """
    name = "Variable"

    def __init__(self, data=None):
        super().__init__()

        self.result = data
        self.inputA = self.result
        self.gradA = None
        self.setShape()

    def __repr__(self):
        """Represent as a string - usefull for printing"""
        output = "<%s with outputs: (" % (self.name)
        for op in self.outputs:
            output += "%s, " % op.name
        output += ")>"
        return output

    def setShape(self):
        """Set the shape of the output of this Variable"""
        self.shape = np.shape(self.result)

    def getValue(self):
        """Return a vaue of this Variable

        Returns
        -------
        np.array
            Data stored by the variable

        Raises
        ------
        AttributeError
            A value for the variable must be set
        """
        if (self.result is None):
            raise AttributeError("A value for the variable must be set")
        return self.result

    def assignData(self, data):
        """Set the data being held by this operation

        Parameters
        ----------
        data : np.array
            Data to be held by the variable

        """

        self.result = data
        self.inputA = self.result
        self.setShape()

    def reset(self):
        """Reset the gradient of this variable"""
        self.gradA = None
        self.setShape()

    def getGradient(self, input=None):
        """Obtain gradient with respect to the input.
        parameter input added for consistancy

        Parameters
        ----------
        input : ga.Operation
            added for consistancy, this operation should have no inputs

        Returns
        -------
        np.array
            Gradient of the graphs final op with respect to the data in this varibale
        """

        if (self.gradA is None):
            self.gradA = self.performGradient()
        return self.gradA

    def performGradient(self, input=None):
        """Find out the gradient"""
        grad = 0
        for out in self.outputs:
            grad += out.getGradient(self)
        return grad
