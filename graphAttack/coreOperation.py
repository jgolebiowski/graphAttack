"""Operation definition"""
from .coreNode import Node
from .coreNode import broadcast_shape
import numpy as np


class Operation(Node):
    """Class for storing all possible operations

    Attributes
    ----------
    name : str
        Name of the operation
    result : np.array
        Output of the operation
    testing : bool
        Flag specifying if the operation is in testing (making prefictions: True)
        or training (optimizing parameters: False) mode
    """
    name = "Operation"

    def __init__(self):
        super().__init__()
        self.result = None
        self.testing = False

    def __repr__(self):
        """Represent as a string - usefull for printing"""
        output = "<%s>" % self.name
        return output

    def getValue(self, *args, **kwargs):
        """Obtain value of the oprtation"""
        raise NotImplementedError("This is not yet implemented")


    def perform(self, *args, **kwargs):
        """Return the value of the operation given inputs"""
        raise NotImplementedError("This is an abstract class, this routine should be implemented in children")

    def reset(self, *args, **kwargs):
        """Reset the values and gradients held by this operation"""
        raise NotImplementedError("This is an abstract class, this routine should be implemented in children")

    def getGradient(self, *args, **kwargs):
        """Return the derevative of this operation with respect to
        each input"""
        raise NotImplementedError("This is an abstract class, this routine should be implemented in children")



class TwoInputOperation(Operation):
    """Operation accepting two input and returning one output

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
    name = "TwoInputOperation"

    def __init__(self, inputA=None, inputB=None):
        super().__init__()
        self.inputA = inputA
        self.inputB = inputB

        self.gradA = None
        self.gradB = None

        inputA.addOutput(self)
        inputB.addOutput(self)

        self.setShape()

    def __repr__(self):
        """Represent as a string - usefull for printing"""
        output = "<%s with inputs: (%s, %s) and outputs: (" % (self.name, self.inputA.name, self.inputB.name)
        for op in self.outputs:
            output += "%s, " % op.name
        output += ")>"
        return output

    def setShape(self):
        """Set the output shape"""
        self.shape = broadcast_shape(np.shape(self.inputA), np.shape(self.inputB))

    def reset(self):
        """Reset the values and gradients held by this operation"""
        self.result = None
        self.gradA = None
        self.gradB = None
        self.setShape()

    def getValue(self):
        """Return a vaue of this operation

        Returns
        -------
        np.array
            Output value
        """
        if (self.result is None):
            self.result = self.perform(self.inputA.getValue(), self.inputB.getValue())
        return self.result

    def getGradient(self, input):
        """Obtain gradient with respect ot a chosen input

        Parameters
        ----------
        input : ga.Operation
            Operation with respect to which the graient is calculated

        Returns
        -------
        np.array
            Gradient value

        Raises
        ------
        ValueError
            Must select either gradient from inputA or inputB
        """
        if (input is self.inputA):
            if (self.gradA is None):
                self.gradA = self.performGradient(input=0)
            return self.gradA
        elif (input is self.inputB):
            if (self.gradB is None):
                self.gradB = self.performGradient(input=1)
            return self.gradB
        else:
            raise ValueError("Must select either gradient from inputA or inputB")


class SingleInputOperation(Operation):
    """Operation accepting one input and returning one output

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

    name = "OneInputOperation"

    def __init__(self, inputA=None):
        super().__init__()
        self.inputA = inputA

        self.gradA = None

        inputA.addOutput(self)

        self.setShape()

    def __repr__(self):
        """Represent as a string - usefull for printing"""
        output = "<%s with input: (%s) and outputs: (" % (self.name, self.inputA.name)
        for op in self.outputs:
            output += "%s, " % op.name
        output += ")>"
        return output

    def setShape(self):
        """Set the output shape"""
        self.shape = np.shape(self.inputA)

    def reset(self):
        """Reset the values and gradients held by this operation"""
        self.result = None
        self.gradA = None
        self.setShape()

    def getValue(self):
        """Return a vaue of this operation

        Returns
        -------
        np.array
            Output value
        """
        if (self.result is None):
            self.result = self.perform(self.inputA.getValue())
        return self.result

    def getGradient(self, input=None):
        """Obtain gradient with respect ot a chosen input
        parameter input added for consistancy

        Parameters
        ----------
        input : ga.Operation
            Operation with respect to which the graient is calculated
            Added for consistancy as those operations only have one input

        Returns
        -------
        np.array
            Gradient value

        Raises
        ------
        ValueError
            Must select either gradient from inputA or inputB
        """

        if (input is self.inputA):
            if (self.gradA is None):
                self.gradA = self.performGradient()
            return self.gradA
        else:
            raise ValueError("Must select gradient from inputA")


class CostOperation(SingleInputOperation):
    """Operation accepting one input and one label, returning the cost
    Labels are to be provided as a standard numpy array, not an operation.

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

    name = "CostOperation"

    def __init__(self, inputA, labels):
        self.labels = labels
        super().__init__(inputA)
        self.setShape()

    def setShape(self):
        """Set the output shape"""
        self.shape = (1, )
        if (np.ndim(self.labels) >= 2):
            self.nExamples = self.labels.shape[0]
        else:
            self.nExamples = 1

    def reset(self):
        """Reset the values and gradients held by this operation"""
        self.result = None
        self.gradA = None
        self.setShape()

    def assignLabels(self, labels):
        """Assign a new set of labels"""
        self.labels = labels
        self.setShape()

    def getValue(self):
        """Return a vaue of this operation

        Returns
        -------
        float
            Evaluated cost
        """
        if (self.result is None):
            self.result = self.perform(self.inputA.getValue(), self.labels)
        return self.result

    def makePredictions(self):
        """Do not evaluate the cost but instead make predictions besed on input

        Returns
        -------
        np.array
            Predictions using the current hypothesis: values fed to cost evaluation operation
        """
        shape = self.inputA.getValue().shape
        predictions = np.zeros(shape)

        if np.size(shape) == 1:
            indexMax = np.argmax(self.inputA.getValue())
            predictions[indexMax] = 1
        else:
            for i, example in enumerate(self.inputA.getValue()):
                indexMax = np.unravel_index(example.argmax(), example.shape)
                predictions[i, indexMax] = 1

        return predictions
