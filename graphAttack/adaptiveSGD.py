import numpy as np
import random
import pickle
"""A module with the stocastic gradient descent"""


class adaptiveSGD(object):
    """Class that holds most of the funtionalities of the
    adaptive SGD, currently using the ADAM algorightm


    Attributes
    ----------

    trainDataset : np.array
        features for each example
    trainLabels : np.array
        labels for each example
    param0 : np.array
        Initial parameters
    epochs : int
        number of epochs to run the minimizer
    miniBatchSize : int
        size of the mini batch

    initialLearningRate : flooat
        Initial learning rate (typical choice: 1e-3)
    beta1 : float
        Beta1 Adam paramater (typical choice: 0.9)
    beta2 : float
        Beta2 Adam parameter (typical choice: 0.999)
    epsilon : float
        epsilon Adam parameter (typical choice: 1e-8)

    testFrequency : int
        How many minibatches to average the cost over for testing
    function : float
        Function to minimize that is of form
        (cost, gradient) = function(params, FeaturesMatrix, LabelsMatrix)

    """

    def __init__(self,
                 trainingData=None,
                 trainingLabels=None,
                 param0=None,
                 epochs=None,
                 miniBatchSize=None,
                 initialLearningRate=None,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 testFrequency=None,
                 function=None):

        self.trainingData = trainingData
        self.trainingLabels = trainingLabels
        self.params = param0
        self.epochs = int(epochs)
        self.testFrequency = testFrequency
        if (self.testFrequency is None):
            self.testFrequency = int(epochs)

        self.initialLearningRate = initialLearningRate
        self.updateValue = np.zeros(len(param0))
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta1T = beta1
        self.beta2T = beta2
        self.epsilon = epsilon
        self.m = 0
        self.v = 0

        self.miniBatchSize = miniBatchSize
        self.nMiniBatches = int(len(trainingData) / miniBatchSize)
        self.miniBatchIndexList = list(range(0, len(trainingData), self.miniBatchSize))

        self.trainingDataBatchesIndex = [(index, index + self.miniBatchSize)
                                         for index in self.miniBatchIndexList]
        self.trainingLabelsBatchesIndex = [(index, index + self.miniBatchSize)
                                           for index in self.miniBatchIndexList]

        self.func = function

    def minimize(self, printTrainigCost=True, printUpdateRate=False, dumpParameters=None):
        """find the minimum of the function

        Parameters
        ----------
        printTrainigCost : bool
            Flag deciding whether to print the cost information
        printUpdateRate : bool
            Flag deciding whether to print the update rate information
        dumpParameters : str or None
            if str is provided, dump parameters to a file with a specified name
            every test iteration

        Returns
        -------
        np.array
            Optimal parameters
        """

        iterationBetweenTests = int(self.epochs * self.nMiniBatches / self.testFrequency)
        self.costLists = []
        iterCost = 0
        randomMiniBatchIndexList = list(range(self.nMiniBatches))

        for indexE in range(self.epochs):
            random.shuffle(randomMiniBatchIndexList)

            for indexMB in range(self.nMiniBatches):
                iterNo = indexE * self.nMiniBatches + indexMB
                randBatchIndex = randomMiniBatchIndexList[indexMB]
                mBatchBeggining = self.trainingDataBatchesIndex[randBatchIndex][0]
                mBatchEnd = self.trainingDataBatchesIndex[randBatchIndex][1]

                cost = self.updateMiniBatch(self.params,
                                            self.trainingData[mBatchBeggining: mBatchEnd],
                                            self.trainingLabels[mBatchBeggining: mBatchEnd])

                iterCost += cost
                if ((iterNo % iterationBetweenTests == 0) and (self.testFrequency != 0)):
                    iterCost /= iterationBetweenTests
                    self.costLists.append(iterCost)
                    # TODO print out cross validation cost every time + use it to implement early stopping
                    if (printTrainigCost):
                        print("Mibatch: %d out of %d from epoch: %d out of %d, iterCost is: %e" %
                              (indexMB, self.nMiniBatches, indexE, self.epochs, iterCost))
                    if (printUpdateRate):
                        print("\tMean of the update momentum: %0.7e variance: %0.7e, beta2T: %07.e" %
                              (np.mean(np.abs(self.m)),
                               np.mean(np.abs(self.v)),
                               np.mean(np.abs(self.beta2T))))
                    if (dumpParameters is not None):
                        self.dumpParameters(dumpParameters)
                    iterCost = 0

        return self.params

    def dumpParameters(self, filename):
        """Dump parameters that can be later used ot restore the state
        The object is a dictionary with
        
        params: weigths thet were being minimized
        
        m: self.m
        v: self.v
        
        beta1: self.beta1
        beta2: self.beta2
        
        beta1T: self.beta1T
        beta2T: self.beta2T
        
        epsilon: self.epsilon
        
        Parameters
        ----------
        filename : str
            name of the file to dump the data into
        """

        p = dict(params=self.params,
                 m=self.m,
                 v=self.v,
                 beta1=self.beta1,
                 beta2=self.beta2,
                 beta1T=self.beta1T,
                 beta2T=self.beta2T,
                 epsilon=self.epsilon)

        with open(filename, "wb") as fp:
            pickle.dump(p, fp)

    def restoreState(self, p):
        """Restore the minimizer state from the data dumped by dumpParameters"""
        self.params = p["params"]
        self.m = p["m"]
        self.v = p["v"]
        self.beta1 = p["beta1"]
        self.beta2 = p["beta2"]
        self.beta1T = p["beta1T"]
        self.beta2T = p["beta2T"]
        self.epsilon = p["epsilon"]


    def updateMiniBatch(self, params, X, Y):
        """ Make an update with a small batch

        Parameters
        ----------

        params : vector of parameters for the function

        X : features for each example in a matrix form
           x.shape = (nExamples, nFeatures)

        Y : labels for each example in a matrix form and
           in a one-hot notation
           y.shape = (nExamples, nClasses)

        """

        # ------ Evaluate gradient
        cost, gradient = self.func(params, X, Y)

        # ------ Calculate moment and variance
        self.m = self.m * self.beta1 + (1 - self.beta1) * gradient
        self.v = self.v * self.beta2 + (1 - self.beta2) * np.square(gradient)

        # ------ Calculate bias-corrected moment and variance
        mHat = self.m / (1 - self.beta1T)
        vHat = self.v / (1 - self.beta2T)
        self.beta1T *= self.beta1
        self.beta2T *= self.beta2

        # ------ Calculate the pdate value
        self.updateValue = self.initialLearningRate * mHat * np.sqrt(np.reciprocal(vHat + self.epsilon))

        # ------ Update parameters
        self.params = params - self.updateValue

        return cost


class adaptiveSGDrecurrent(adaptiveSGD):
    """Class that holds most of the funtionalities of the
    adaptive SGD, for applications to time series.
    Currently uses the ADAM algorightm


    Attributes
    ----------

    trainDataset : np.array
        time series data in the format of (nDataPoints, exampleFeatures1, exampleFeatures2, ...)
    param0 : np.array
        Initial parameters
    epochs : int
        number of epochs to run the minimizer
    miniBatchSize : int
        size of the mini batch
    exampleLength : int
        Length of an individual example to be fed ot the function

    initialLearningRate : flooat
        Initial learning rate (typical choice: 1e-3)
    beta1 : float
        Beta1 Adam paramater (typical choice: 0.9)
    beta2 : float
        Beta2 Adam parameter (typical choice: 0.999)
    epsilon : float
        epsilon Adam parameter (typical choice: 1e-8)

    testFrequency : int
        How many minibatches to average the cost over for testing
    function : float
        Function to minimize that is of form
        (cost, gradient) = function(params, FeaturesMatrix, LabelsMatrix)

    """

    def __init__(self,
                 trainingData=None,
                 param0=None,
                 epochs=None,
                 miniBatchSize=None,
                 exampleLength=None,
                 initialLearningRate=None,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 testFrequency=None,
                 function=None):

        self.trainingData = trainingData
        self.params = param0
        self.epochs = int(epochs)
        self.testFrequency = testFrequency
        if (self.testFrequency is None):
            self.testFrequency = int(epochs)

        self.initialLearningRate = initialLearningRate
        self.updateValue = np.zeros(len(param0))
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta1T = beta1
        self.beta2T = beta2
        self.epsilon = epsilon
        self.m = 0
        self.v = 0

        self.miniBatchSize = miniBatchSize
        nMiniBatches = (len(trainingData) - exampleLength) / miniBatchSize
        if not float(nMiniBatches).is_integer():
            print("""      WARNING!: self.nMiniBatches =  %f is not an integer whereas it should be
                nMiniBatchs = (seriesLength - exampleLength) / miniBatchSize
                in case where exampleLength ==  miniBatchSize an appropriate value is
                exampleLength = seriesLength / (integer + 1) as long as exLength is also an integer"""
            % nMiniBatches)
            print("      Values are: seriesLength == len(trainingData =", len(trainingData))
            print("                  exampleLength =", exampleLength)
            print("                  miniBatchSize =", miniBatchSize)

        self.nMiniBatches = int(nMiniBatches)
        self.exampleLength = exampleLength
        self.nExamples = len(trainingData) - exampleLength

        self.dataShape = (self.miniBatchSize, self.exampleLength) + self.trainingData.shape[1:]
        self.trainingX = np.zeros((self.dataShape))
        self.trainingY = np.zeros((self.dataShape))

        self.func = function

    def minimize(self, printTrainigCost=True, printUpdateRate=False, dumpParameters=None):
        """find the minimum of the function

        Parameters
        ----------
        printTrainigCost : bool
            Flag deciding whether to print the cost information
        printUpdateRate : bool
            Flag deciding whether to print the update rate information
        dumpParameters : str or None
            if str is provided, dump parameters to a file with a specified name
            every test iteration

        Returns
        -------
        np.array
            Optimal parameters
        """

        iterationBetweenTests = int(self.epochs * self.nMiniBatches / self.testFrequency)
        self.costLists = []
        iterCost = 0

        for indexE in range(self.epochs):
            self.pointer = 0
            for indexMB in range(self.nMiniBatches):
                self.populateDataAndLabels()
                cost = self.updateMiniBatch(self.params, self.trainingX, self.trainingY)

                iterNo = indexE * self.nMiniBatches + indexMB
                iterCost += cost
                if ((iterNo % iterationBetweenTests == 0) and (self.testFrequency != 0)):
                    iterCost /= iterationBetweenTests
                    self.costLists.append(iterCost)
                    # TODO print out cross validation cost every time + use it to implement early stopping
                    if (printTrainigCost):
                        print("Mibatch: %d out of %d from epoch: %d out of %d, iterCost is: %e" %
                              (indexMB, self.nMiniBatches, indexE, self.epochs, iterCost))
                    if (printUpdateRate):
                        print("\tMean of the update momentum: %0.7e variance: %0.7e, beta2T: %07.e" %
                              (np.mean(np.abs(self.m)),
                               np.mean(np.abs(self.v)),
                               np.mean(np.abs(self.beta2T))))
                    if (dumpParameters is not None):
                        self.dumpParameters(dumpParameters)
                    iterCost = 0

        return self.params

    def populateDataAndLabels(self):
        """Populate
        self.trainingX by vectors of features from self.trainingData
            from the timestep given by
                pointer
            to the timestep given by
                pointer + self.exampleLength
        for pointer from (self.pointer -> self.pointer + self.pointer + self.miniBatchSize)

        self.trainingY by vectors of labels from self.trainingData
            from the timestep given by
                pointer + 1
            to the timestep given by
                pointer + self.exampleLength + 1
        for pointer from (self.pointer -> self.pointer + self.pointer + self.miniBatchSize)
        """

        for index in range(self.miniBatchSize):
            self.trainingX[index] = self.trainingData[self.pointer:
                                                      self.pointer + self.exampleLength]
            self.trainingY[index] = self.trainingData[self.pointer + 1:
                                                      self.pointer + 1 + self.exampleLength]
            self.pointer += 1
