"""Node definition"""
import numpy as np


class Node(object):
    """Node - a basic building block of the graph

    Attributes
    ----------
    endNode : bool
        Flag stating whether this is the final node of the graph
    name : str
        name of the node
    outputs : list
        list of nodes that operate on output of this node
    referenceNumber : int
        reference number of this node
    result : np.array
        output of this node
    shape : tuple
        shape
    """
    shape = None
    name = "Node"
    referenceNumber = None

    def __init__(self):
        self.outputs = []
        self.result = None
        self.endNode = True

    def __repr__(self):
        """Represent as a string - usefull for printing

        Returns
        -------
        str
            description of this node
        """
        output = "<%s>" % self.name
        return output

    def prependName(self, string):
        """Prepend name with a string

        Parameters
        ----------
        string : str
            prefix
        """
        self.name = str(string) + self.name

    def assignReferenceNumber(self, number):
        """Assign a reference number

        Parameters
        ----------
        number : int
            reference number
        """
        self.referenceNumber = number
        self.prependName("op" + str(number) + "-")

    def setShape(self):
        """Set the shape of the output of this node"""
        raise NotImplementedError("This is an abstract class, this routine should be implemented in children")

    def addOutput(self, output):
        """Attach the node that is the output of this Node

        Parameters
        ----------
        output : ga.Node
            attach an output node to this node
        """
        self.outputs.append(output)
        self.endNode = False

    def reset(self):
        """Reset the values and gradients held by this operation"""
        raise NotImplemented("This is an abstract class")

    def getValue(self):
        """Return a vaue of this operation"""
        if (self.result is None):
            raise NotImplemented("The result is not set at initialization, maybe use an operation")
        return self.result


def broadcast_shape(shp1, shp2):
    """Broadcast the shape of those arrays

    Parameters
    ----------
    shp1 : tuple
        shape of array 1
    shp2 : tuple
        shape of array 2

    Returns
    -------
    tuple
        shape resulting from broadcasting two arrays using numpy rules

    Raises
    ------
    ValueError
        Arrays cannot be broadcasted
    """
    try:
        return np.broadcast(np.empty(shp1), np.empty(shp2)).shape
    except ValueError:
        raise ValueError("Arrays cannot be broadcasted - %s and %s " % (str(shp1), str(shp2)))


def reduce_shape(inputArr, targetArr):
    """Reduce the dimensions by summing the input array over necesary axis
    to obtain the targetArray shape.

    Parameters
    ----------
    inputArr : np.array
        array 1
    targetArr : np.array
        array 2

    Returns
    -------
    np.array
        Resulting array (sum over the necessary axis)

    Raises
    ------
    ValueError
        The two arrays cannot be reduced properly
    """
    if (inputArr.shape == targetArr.shape):
        return inputArr

    if (inputArr.ndim == targetArr.ndim):
        axReduce = []
        for dimIndex in range(inputArr.ndim):
            if targetArr.shape[dimIndex] == 1:
                axReduce.append(dimIndex)
        axReduce = tuple(axReduce)
        return np.sum(inputArr, axis=axReduce, keepdims=True)

    try:
        if (inputArr.shape[1] == targetArr.shape[0]):
            return np.sum(inputArr, axis=0)
    except (IndexError):
        pass
    except (TypeError):
        pass

    try:
        if (inputArr.shape[0] == targetArr.shape[1]):
            return np.sum(inputArr, axis=1)
    except (IndexError):
        pass
    except (TypeError):
        pass

    raise ValueError("The two arrays cannot be reduced properly")
