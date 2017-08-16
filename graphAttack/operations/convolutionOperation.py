"""This where implementations of individual operations live"""

from ..coreOperation import *
from ..coreNode import broadcast_shape, reduce_shape
import numpy as np
import math


class Conv2dOperation(TwoInputOperation):
    """Convolution operation

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
        gradient with respect to inputA (data)
    gradB : np.array
        gradient with respect to inputB (weights)
    inputA : ga.Operation
        Operation feeding data A into this operation
        This shold be the input of convolution in the format of
        nExamples x nChannels x Height x Width (NCHW)
    inputB : ga.Operation
        Operation feeding data B into this operation
        This should provide the weights for this operation in the format of
        nFilters x nChannels x filterHeight x filterWidth
    shape : tuple
        shape of the output
    padding : int
        how much zero rows/cold to add to an image
        If provided at initialization stage, ovverites padding generated to match
        paddingMethod
    paddingMethod : str
        SAME: enough padding is added so that outpu image dimensions match the input's
        VALID: no padding is added and the image size will be reduced
    stride : int
        step size for scanning an image
    """
    name = "Conv2dOperation"

    def __init__(self, inputA=None, inputB=None, stride=1, paddingMethod="SAME", padding=None):
        if (stride < 1):
            raise ValueError("Stride must be at least one")
        self.stride = stride

        shapesMatch = (inputA.shape[1] == inputB.shape[1])
        if not shapesMatch:
            raise ValueError(
                """Shapes of inputs must be compatible with regard of nChannels, but %d != %d.
                see docstring for explanation of the format.""" % (inputA.shape[1], inputB.shape[1]))

        if (paddingMethod != "SAME") and (paddingMethod != "VALID"):
            raise NotImplementedError("Only SAME and VALID paddingMethod is implemented!")
        self.paddingMethod = paddingMethod

        # ------ Figure out padding value
        if padding is not None:
            self.padding = padding
        elif (self.paddingMethod == "SAME"):
            if (inputB.shape[2] != inputB.shape[3]):
                raise NotImplementedError("Only square filters are supported for same padding")
            self.padding = int((inputB.shape[2] - 1) / 2)
        elif (self.paddingMethod == "VALID"):
            self.padding = 0
        else:
            raise NotImplementedError("Only SAME and VALID paddingMethod is implemented!")

        super().__init__(inputA, inputB)

    def setShape(self):
        """Set the output shape"""

        # ------ Find the output shape
        nExamples = self.inputA.shape[0]
        nFilters = self.inputB.shape[0]
        outputHeight = int(((self.inputA.shape[2] - self.inputB.shape[2] + 2 * self.padding) /
                            self.stride) + 1)
        outputWidth = int(((self.inputA.shape[3] - self.inputB.shape[3] + 2 * self.padding) /
                           self.stride) + 1)

        self.shape = (nExamples, nFilters, outputHeight, outputWidth)

    def perform(self, a, b):
        """Perform convolution in 2D

        Parameters
        ----------
        a : np.array
            Input data
        b : np.array
            Weights

        Returns
        -------
        np.aarray
            Result of the operation
        """
        N, C, H, W = self.inputA.shape
        NF, C, FH, FW = self.inputB.shape
        N, NF, oH, oW = self.shape

        # ------ Obtain a 2d representation of image and filters
        aCol = im2col_indices(a, FH, FW, self.padding, self.stride)
        bCol = b.reshape(NF, -1)

        # ------ Store them for later gradient evaluation
        self.inputACols = aCol
        self.inputBCols = bCol

        # ------ Obtain the 2d representation of the output
        outCol = np.matmul(aCol, bCol.T)

        # ------ Convert into appropriate shape
        outMat = outCol
        outMat = outCol.reshape(oH, oW, N, NF)
        outMat = outMat.transpose(2, 3, 0, 1)

        return outMat

    def performGradient(self, input, out=None):
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
        N, C, H, W = self.inputA.shape
        NF, C, FH, FW = self.inputB.shape
        N, NF, oH, oW = self.shape

        # ------ Gather gradient
        if (self.endNode):
            grad = np.ones(self.shape)
        else:
            grad = np.zeros(self.shape)
            for out in self.outputs:
                grad += out.getGradient(self)

        # ------ Reshape the gradient into the form of 2D array in the
        # ------ format of outCol from forward pass

        gradCols = grad.transpose(2, 3, 0, 1)
        gradCols = gradCols.reshape(-1, NF)

        if (input == 0):
            gradCols = np.matmul(gradCols, self.inputBCols)
            grad = col2im_indices(gradCols, self.inputA.shape, FH, FW, padding=self.padding, stride=self.stride)
        elif (input == 1):
            gradCols = np.matmul(self.inputACols.T, gradCols)
            grad = gradCols.T.reshape(self.inputB.shape)

        return grad


class MaxPoolOperation(SingleInputOperation):
    """Apply max pooling using im2col, quite slow yet simple to understand

    Attributes
    ----------
    name : str
        Name of the operation
    result : np.array
        Output of the operation
    testing : bool
        Flag specifying if the operation is in testing (making prefictions: True)
        or training (optimizing parameters: False) mode

    inputA : ga.Operation
        Operation feeding data A into this operation
        This shold be the input of convolution in the format of
        nExamples x nChannels x Height x Width (NCHW)
    inputA : ga.Operation
        Operation feeding data A into this operation
    shape : tuple
        shape of the output
    poolHeight : int
        Heigth of the pooling filter
    poolWidth : int
        Width of the pooling filter
    stride : int
        step size for scanning an image

    NOTE: stride and filters should be compatible so that
    Image width/heigth - poolHeigth/Width % stride == 0

    axis : int
        Axis over which to perform the sum
    """
    name = "MaxPoolOperation"

    def __init__(self, inputA=None, poolHeight=2, poolWidth=2, stride=1):
        self.stride = stride
        self.poolHeight = poolHeight
        self.poolWidth = poolWidth
        super().__init__(inputA)

    def setShape(self):
        """Set the output shape"""
        if (((self.inputA.shape[2] - self.poolHeight) % self.stride != 0) or
            ((self.inputA.shape[3] - self.poolWidth) % self.stride != 0)):
            raise ValueError("""stride and padding should be compatible so that
                            Image weigth/heigth - poolHeigth/Width MODULO stride == 0""")
        # ------ Find the output shape
        nExamples = self.inputA.shape[0]
        nChannels = self.inputA.shape[1]
        outputHeight = int((self.inputA.shape[2] - self.poolHeight) / self.stride + 1)
        outputWidth = int((self.inputA.shape[3] - self.poolWidth) / self.stride + 1)

        self.shape = (nExamples, nChannels, outputHeight, outputWidth)

    def perform(self, a):
        """MaxPool each of the images, use im2cols and find argmax of each row

        Parameters
        ----------
        a : np.array
            Input data

        Returns
        -------
        np.array
            Output data
        """
        N, C, H, W = self.inputA.shape
        N, C, oH, oW = self.shape
        aReshaped = a.reshape(N * C, 1, H, W)
        self.aCols = im2col_indices(aReshaped, self.poolHeight, self.poolWidth, padding=0, stride=self.stride)
        self.aColsArgmax = np.argmax(self.aCols, axis=1)

        outCol = self.aCols[np.arange(self.aColsArgmax.size), self.aColsArgmax]
        outMat = outCol.reshape(oH, oW, N, C)
        outMat = outMat.transpose(2, 3, 0, 1)

        return outMat

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
            grad = np.ones(self.shape)
        else:
            grad = np.zeros(self.shape)
            for out in self.outputs:
                grad += out.getGradient(self)

        N, C, H, W = self.inputA.shape
        N, C, oH, oW = self.shape

        inpGradCols = grad.transpose(2, 3, 0, 1).flatten()
        gradCols = np.zeros_like(self.aCols)
        gradCols[np.arange(self.aColsArgmax.size), self.aColsArgmax] = inpGradCols

        grad = col2im_indices(gradCols, (N * C, 1, H, W), self.poolHeight, self.poolWidth, padding=0, stride=self.stride)
        grad = grad.reshape(self.inputA.shape)
        return grad




###############################################################################
# Those are im2col and col2im functions copied from the assignement for
# Stanford CS 231n class:
# http://cs231n.stanford.edu/
# https://github.com/cs231n/cs231n.github.io
#
# The only modification is that the im2col_indices returns a transposed version
# of the matrix in shape of (nelements, filterH x filterW)
# col2im_indices now accepts the transposed version of cols to be compatible with im2col
# Both functions use the same notation internally, the changes is only in intreface
###############################################################################


def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = int((H + 2 * padding - field_height) / stride + 1)
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k.astype(int), i.astype(int), j.astype(int))


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols.T


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1):
    """ An implementation of col2im based on fancy indexing and np.add.at """
    # ------ Chyange to be compatible wih im2col
    cols = cols.T
    # ------ end of change
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]

