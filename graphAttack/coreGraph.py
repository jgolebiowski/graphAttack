"""Graph definition"""
from .coreDataContainers import Variable
from .coreOperation import CostOperation
from .operations.activationOperations import DropoutOperation
import numpy as np


class Graph(object):

    """Computational graph, main objects holding others together

    Attributes
    ----------
    costOperation : np.CostOperation
        final operation of the graph, evaluating cost. Usually the final operation
    endOperations : list
        list of operations that do not have a follow up
    feederOperation : ga.Operation
        Operation feeding the data to the graph
    finalOperation : np.Operation
        final operation of the graph, most often the cost operation
    gradientOps : list
        List of operation that need their gradients evaliated
    nOperations : int
        Total number of operations
    operations : list
        List of all operations
    """

    def __init__(self):
        self.operations = []
        self.gradientOps = []
        self.finalOperation = None
        self.costOperation = None
        self.feederOperation = None
        self.endOperations = []

        self.nOperations = 0

    def __repr__(self):
        """Represent as a string - usefull for printing

        Returns
        -------
        str
            Representation string, prints out all the operations.
        """
        output = "Computation Graph:\n"
        for op in self.operations:
            output += op.__repr__() + "\n"
        return output

    def __iter__(self):
        return iter(self.operations)

    def addOperation(self, operation, doGradient=False, finalOperation=False, feederOperation=False):
        """Add an operation ot the graph

        Parameters
        ----------
        operation : ga.Operation
            Operation to be added
        doGradient : bool
            When true, calculate the derevative of the final operation with respect to this
            operation when self.getGradients() is called
        finalOperation : bool
            When true, specify this operation as final of the graph
        feederOperation : bool
            When true, specify this operation as the data feeder operation

        Returns
        -------
        ga.Operation
            the handle for added operation

        Raises
        ------
        ValueError
            Graph can only provide gradients with respect to variables!
            Only variables can be feeders
        """
        self.operations.append(operation)
        operation.assignReferenceNumber(self.nOperations)
        self.nOperations += 1

        self.endOperations.append(operation)
        for op in self.endOperations[:]:
            if operation in op.outputs:
                self.endOperations.remove(op)

        if (doGradient):
            if not (isinstance(operation, Variable)):
                raise ValueError("Graph can only provide gradients with respect to variables!\
                    Call individual ops.getGradient(inputOperation) for individual gradients.")
            self.gradientOps.append(operation)
        if (finalOperation):
            self.finalOperation = operation
            if (isinstance(operation, CostOperation)):
                self.costOperation = operation
        if (feederOperation):
            if (isinstance(operation, Variable)):
                self.feederOperation = operation
            else:
                raise ValueError("Only variables can be feeders")
        return operation

    def unrollGradientParameters(self):
        """For each variable (NOT operation) that needs a gradient calculated
        obtain the inputs and unroll them into a nice vector

        Returns
        -------
        np.array
            A flat array of gradient parameters
        """
        params = np.empty(0)
        for op in self.gradientOps:
            if isinstance(op, Variable):
                params = np.hstack((params, np.ravel(op.getValue())))
        return params

    def attachParameters(self, params):
        """Given a params vector, attach it as data to all variables,
        NOT operations, that need a gradient evaluation

        Parameters
        ----------
        params : flat np.array
            parameters to be attached
        """
        pointer = 0
        for op in self.gradientOps:
            if isinstance(op, Variable):
                nElems = np.size(op.result)
                shaperino = op.shape
                op.assignData(np.reshape(params[pointer: pointer + nElems], shaperino))
                pointer += nElems

    def feedForward(self):
        """feed forwards through the graph obtaining the value
        of the final operation

        Returns
        -------
        np.array / float
            Value of the final operation
        """
        return self.finalOperation.getValue()

    def getValue(self):
        """Reset the graph and feed forwards through the graph obtaining the value
        of the final operation

        Returns
        -------
        np.array / float
            Value of the final operation
        """
        self.resetAll()
        return self.feedForward()

    def feedBackward(self):
        """Propagate backwards, gathering all the graients

        Returns
        -------
        tuple
            ((operation reference number, operation name, operation gradient) for each operation)
        """
        gradients = []
        for op in self.gradientOps:
            gradients.append((op.referenceNumber, op.name, op.getGradient()))
        return gradients

    def getGradients(self):
        """Reset th graph and get gradients of the specified variables

        Returns
        -------
        tuple
            ((operation reference number, operation name, operation gradient) for each operation)
        """
        self.resetAll()
        return self.feedBackward()

    def makePredictions(self):
        """Get predictions from a cost operation

        Returns
        -------
        np.array
            predictions from the model (values that go into calculating cost)

        Raises
        ------
        AttributeError
            Must add a cost operation
        """
        if self.costOperation is None:
            raise AttributeError("Must add a cost operation")

        # ------ Set all operations to testing
        for op in self.operations:
            op.testing = True

        self.resetAll()
        pred = self.costOperation.makePredictions()

        # ------ Set all operations to training
        for op in self.operations:
            op.testing = False

        return pred

    def unrollGradients(self):
        """Provide gradiens in the same form the unrolled parameters are provided

        Returns
        -------
        flat np.array
            Gradients obtained from self.getGradients() unrolled into a flat array
        """
        grads = np.empty(0)
        for op in self.gradientOps:
            if isinstance(op, Variable):
                grads = np.hstack((grads, np.ravel(op.getGradient())))
        return grads

    def resetAll(self):
        """Reset all of the operations"""
        for op in self.operations:
            op.reset()

    def printGraph(self):
        """Print out all of the operations"""
        for op in self.operations:
            print(op)
