import graphAttack as ga
import numpy as np
import pickle
"""Control script"""


pickleFilename = "dataSet/redditCommentFormatted.pkl"
with open(pickleFilename, "rb") as fp:
    x, y, index_to_word, word_to_index = pickle.load(fp)

vocabSize = 2000
nExamples, seriesLength, nFeatures = x.shape


mainGraph = ga.Graph(False)

feed = mainGraph.addOperation(ga.Variable(x), feederOperation=True)

finalCost,\
hactivations,\
costOperationsList = ga.addRNNnetwork(mainGraph,
                                      inputOperation=feed,
                                      activation=ga.TanhActivation,
                                      costActivation=ga.SoftmaxActivation,
                                      costOperation=ga.CrossEntropyCostSoftmax,
                                      nHidden=100,
                                      labels=None)

# progress = 0
# paramLen = len(mainGraph.unrollGradientParameters())

def f(p, costOperationsList=costOperationsList, mainGraph=mainGraph):
    data = x
    labels = y
    mainGraph.feederOperation.assignData(data)
    mainGraph.resetAll()
    for index, cop in enumerate(costOperationsList):
        cop.assignLabels(labels[:, index, :])
    mainGraph.attachParameters(p)

    mainGraph.resetAll()
    c = mainGraph.feedForward()

    # global progress
    # progress += 1
    # if not (progress % 10):
    #     print("Progress:", progress, "Out of:", paramLen)
    return c


def fprime(p, data, labels, costOperationsList=costOperationsList, mainGraph=mainGraph):
    mainGraph.feederOperation.assignData(data)
    mainGraph.resetAll()
    for index, cop in enumerate(costOperationsList):
        cop.assignLabels(labels[:, index, :])
    mainGraph.attachParameters(p)

    mainGraph.resetAll()
    c = mainGraph.feedForward()
    mainGraph.feedBackward()
    g = mainGraph.unrollGradients()
    return c, g

index = 0
param0 = mainGraph.unrollGradientParameters()
adaGrad = ga.adaptiveSGD(trainingData=x,
                         trainingLabels=y,
                         param0=param0,
                         epochs=1e3,
                         miniBatchSize=2,
                         initialLearningRate=1e-3,
                         beta1=0.9,
                         beta2=0.999,
                         epsilon=1e-8,
                         testFrequency=1e3,
                         function=fprime)

# params = adaGrad.minimize(printTrainigCost=True, printUpdateRate=False,
#                           dumpParameters="paramsCNN_" + str(index) + ".pkl")
pickleFilename = "paramsRNN_" + str(index) + ".pkl"
with open(pickleFilename, "rb") as fp:
    params = pickle.load(fp)

mainGraph.attachParameters(params)

def array2char(array):
    return index_to_word[np.argmax(array)]

def sampleCharacter(previousX, previousH,
                    hactivations=hactivations,
                    costOperationsList=costOperationsList,
                    mainGraph=mainGraph):
    N, T, D = mainGraph.feederOperation.shape
    preiousData = np.zeros((1, T, D))
    preiousData[:, 0, :] = previousX

    mainGraph.resetAll()
    mainGraph.feederOperation.assignData(preiousData)
    hactivations[0].assignData(previousH)

    newH = hactivations[1].getValue()
    # newX = costOperationsList[0].makePredictions()
    newX = costOperationsList[0].inputA.getValue()

    return newX, newH


nextH = hactivations[0].getValue().copy()
nextX = x[0,0]
# nextX = np.zeros_like(x[0,0])
# nextX[int(np.random.random() * vocabSize)] = 1

print(array2char(nextX))
length = 45
for index in range(length):
    nextX, nextH = sampleCharacter(nextX, nextH)
    print(array2char(nextX))
    # print(array2char(nextX), array2char(x[0, index + 1]))
