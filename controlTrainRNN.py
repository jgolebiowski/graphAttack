import graphAttack as ga
import numpy as np
import pickle
import sys
"""Control script"""
simulationIndex = 0

# ------ This is a very limited dataset, load a lrger one for better results
# ------ This net is quute slow to train, be aware.

pickleFilename = "dataSet/singleSentence.pkl"
with open(pickleFilename, "rb") as fp:
    x, index_to_word, word_to_index = pickle.load(fp)

seriesLength, nFeatures = x.shape
nExamples = 2
exampleLength = 15
nHidden0 = 20
nHidden1 = 25

mainGraph = ga.Graph(False)
dummyX = np.zeros((nExamples, exampleLength, nFeatures))
feed = mainGraph.addOperation(ga.Variable(dummyX), feederOperation=True)


# ------ Generate the network, options are RNN and LSTM gates
# ------ Add initial layer and then possibly append more
hactivations1, cStates1 = ga.addInitialLSTMLayer(mainGraph,
                                                 inputOperation=feed,
                                                 nHidden=nHidden1)
# hactivations1, cStates1 = ga.appendLSTMLayer(mainGraph,
#                                   previousActivations=hactivations0,
#                                   nHidden=nHidden1)

# hactivations1 = ga.addInitialRNNLayer(mainGraph,
#                                       inputOperation=feed,
#                                       activation=ga.TanhActivation,
#                                       nHidden=nHidden1)
# hactivations = ga.appendRNNLayer(mainGraph,
#                                  previousActivations=hactivations0,
#                                  activation=ga.TanhActivation,
#                                  nHidden=nHidden1)

finalCost, costOperationsList = ga.addRNNCost(mainGraph,
                                              hactivations1,
                                              costActivation=ga.SoftmaxActivation,
                                              costOperation=ga.CrossEntropyCostSoftmax,
                                              nHidden=nHidden1,
                                              labelsShape=feed.shape,
                                              labels=None)


def fprime(p, data, labels, costOperationsList=costOperationsList, mainGraph=mainGraph):
    mainGraph.feederOperation.assignData(data)
    mainGraph.resetAll()
    for index, cop in enumerate(costOperationsList):
        cop.assignLabels(labels[:, index, :])
    mainGraph.attachParameters(p)
    c = mainGraph.feedForward()
    mainGraph.feedBackward()
    g = mainGraph.unrollGradients()
    return c, g


param0 = mainGraph.unrollGradientParameters()
print("Number of parameters to train:", len(param0))
adaGrad = ga.adaptiveSGDrecurrent(trainingData=x,
                                  param0=param0,
                                  epochs=5e2,
                                  miniBatchSize=nExamples,
                                  exampleLength=exampleLength,
                                  initialLearningRate=1e-3,
                                  beta1=0.9,
                                  beta2=0.999,
                                  epsilon=1e-8,
                                  testFrequency=1e2,
                                  function=fprime)

params = adaGrad.minimize(printTrainigCost=True, printUpdateRate=False,
                          dumpParameters="paramsRNN_" + str(simulationIndex) + ".pkl")
# pickleFilename = "paramsRNN_" + str(simulationIndex) + ".pkl"
# with open(pickleFilename, "rb") as fp:
#     params = pickle.load(fp)

mainGraph.attachParameters(params)

temp = ga.sampleManySingleLSTM(100, nFeatures, nHidden1,
                               hactivations=hactivations1,
                               cStates=cStates1,
                               costOperationsList=costOperationsList,
                               mainGraph=mainGraph,
                               index_to_word=index_to_word)
print(temp)

# temp = sampleManySingleRNN(100, nFeatures, nHidden1,
#                            hactivations=hactivations1,
#                            costOperationsList=costOperationsList,
#                            mainGraph=mainGraph,
#                            index_to_word=index_to_word)
# print(temp)
